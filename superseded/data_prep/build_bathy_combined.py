#!/usr/bin/env python3
"""
Build combined bathymetry GeoTIFF from multiple sources.

Priority (highest first):
  1. LiDAR .bag files (5-10m res, coastal)
  2. Henderson/Perth Canyon multibeam (16m res, offshore)
  3. GMRT baseline (~200m res, full area)

Output: data/bathy_combined.tif at 0.0001° (~11m) resolution in EPSG:4326.
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

# Output grid configuration
OUT_PATH = "data/bathy_combined.tif"
OUT_CRS = "EPSG:4326"
OUT_RES = 0.0001  # ~11m at 32°S

# Bounding box (matches generate_bathy_tiles.py region + margin)
BBOX_WEST = 114.4992
BBOX_EAST = 116.0006
BBOX_SOUTH = -32.8005
BBOX_NORTH = -31.2994

BATHY_DIR = "data/Bathymetry"


def build_output_grid():
    """Create empty output grid."""
    width = int(round((BBOX_EAST - BBOX_WEST) / OUT_RES))
    height = int(round((BBOX_NORTH - BBOX_SOUTH) / OUT_RES))
    transform = from_bounds(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH, width, height)
    grid = np.full((height, width), np.nan, dtype=np.float32)
    return grid, transform, width, height


def overlay_source(grid, transform, width, height, src_path, label):
    """Reproject and overlay a source onto the grid (non-NaN pixels win)."""
    try:
        with rasterio.open(src_path) as src:
            # Create temporary array for reprojected data
            tmp = np.full((height, width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=OUT_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )

            # Count valid pixels
            valid = ~np.isnan(tmp) & (tmp != 0)
            n_valid = int(np.sum(valid))

            if n_valid > 0:
                # Overlay: source pixels replace existing where valid
                grid[valid] = tmp[valid]
                print(f"  {label}: {n_valid:,} pixels merged")
            else:
                print(f"  {label}: no valid pixels in region")

    except Exception as e:
        print(f"  {label}: ERROR - {e}")


def main():
    print("Building combined bathymetry...")
    grid, transform, width, height = build_output_grid()
    print(f"Output grid: {width}x{height} @ {OUT_RES}° ({OUT_PATH})")

    # --- Layer 1: GMRT baseline ---
    gmrt_path = "data/bathy_gmrt.tif"
    if os.path.exists(gmrt_path):
        print("\nLayer 1: GMRT baseline")
        overlay_source(grid, transform, width, height, gmrt_path, "GMRT")
    else:
        print(f"WARNING: {gmrt_path} not found!")

    # --- Layer 2: Henderson/Perth Canyon multibeam ---
    print("\nLayer 2: Henderson/Perth Canyon multibeam (16m)")
    multibeam_files = [
        # COG versions preferred (faster), fall back to raw
        "HendersonPerthCanyontoHenderson_FK150301_Bathymetry_SJ50_Depth_16m_2015_20251015_cog.tiff",
        "HendersonPerthCanyontoHenderson_FK150301_Bathymetry_SI50_Depth_16m_2015_20251015_cog.tiff",
    ]
    for f in multibeam_files:
        path = os.path.join(BATHY_DIR, f)
        if os.path.exists(path):
            overlay_source(grid, transform, width, height, path, f[:40])
        else:
            # Try non-COG version
            alt = f.replace("_cog.tiff", ".tiff").replace("Henderson", "20150012S_Henderson")
            alt_path = os.path.join(BATHY_DIR, alt)
            if os.path.exists(alt_path):
                overlay_source(grid, transform, width, height, alt_path, alt[:40])

    # --- Layer 3: LiDAR .bag files (highest priority) ---
    print("\nLayer 3: LiDAR .bag files (5-10m)")
    bag_files = sorted([f for f in os.listdir(BATHY_DIR) if f.endswith('.bag')])
    for f in bag_files:
        path = os.path.join(BATHY_DIR, f)
        overlay_source(grid, transform, width, height, path, f)

    # --- Write output ---
    valid_total = int(np.sum(~np.isnan(grid)))
    print(f"\nTotal valid pixels: {valid_total:,} / {width * height:,}")

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': OUT_CRS,
        'transform': transform,
        'nodata': np.nan,
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }

    with rasterio.open(OUT_PATH, 'w', **profile) as dst:
        dst.write(grid, 1)

    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\nWrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
