"""
Generate high-resolution bathymetry hillshade tiles from GMRT GeoTIFF.

Produces a set of z/x/y PNG tiles that MapLibre can serve as a local
raster tile source, giving sharp canyon detail at any zoom level.

Usage:
    python generate_bathy_tiles.py [--min-zoom 8] [--max-zoom 14]
"""

import os
import sys
import math
import numpy as np
from PIL import Image
import rasterio
import mercantile

# --- Config ---
GMRT_TIF = "data/bathy_merged.tif"
TILE_DIR = "data/bathy_tiles"
TILE_SIZE = 256
MIN_ZOOM = 8
MAX_ZOOM = 14  # ~10m/pixel at z14 — plenty for canyon detail

# Region of interest (slightly larger than chart area)
BBOX_WEST = 113.5
BBOX_EAST = 116.5
BBOX_SOUTH = -33.0
BBOX_NORTH = -30.5

# Hillshade parameters
AZIMUTH = 315      # light from NW
ALTITUDE = 45      # sun angle
Z_FACTOR = 3.0     # vertical exaggeration for underwater relief
OCEAN_COLOR = (20, 50, 80)     # deep ocean base RGB
SHELF_COLOR = (60, 120, 140)   # shallow shelf RGB
DEPTH_RANGE = (-2500, 0)       # color mapping range in meters


def load_bathy():
    """Load bathymetry GeoTIFF into arrays."""
    with rasterio.open(GMRT_TIF) as src:
        data = src.read(1).astype(np.float64)
        t = src.transform
        width, height = src.width, src.height
        # Build coordinate arrays
        lons = np.array([t.c + (j + 0.5) * t.a for j in range(width)])
        lats = np.array([t.f + (i + 0.5) * t.e for i in range(height)])
    return data, lons, lats


def compute_hillshade(elevation, cellsize_x, cellsize_y, azimuth=315, altitude=45, z_factor=1.0):
    """Compute hillshade from elevation grid."""
    az_rad = math.radians(360 - azimuth + 90)
    alt_rad = math.radians(altitude)

    # Gradient using numpy
    dy, dx = np.gradient(elevation * z_factor, cellsize_y, cellsize_x)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    shade = (np.sin(alt_rad) * np.cos(slope) +
             np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    shade = np.clip(shade, 0, 1)
    return shade


def depth_to_color(depth, shade):
    """Map depth + hillshade to RGB color."""
    # Normalise depth to 0-1 range
    d_min, d_max = DEPTH_RANGE
    norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)

    # Blend between deep ocean and shelf colors based on depth
    r = OCEAN_COLOR[0] + (SHELF_COLOR[0] - OCEAN_COLOR[0]) * norm
    g = OCEAN_COLOR[1] + (SHELF_COLOR[1] - OCEAN_COLOR[1]) * norm
    b = OCEAN_COLOR[2] + (SHELF_COLOR[2] - OCEAN_COLOR[2]) * norm

    # Apply hillshade as brightness multiplier
    brightness = 0.4 + 0.6 * shade  # keep some ambient light
    r = np.clip(r * brightness, 0, 255).astype(np.uint8)
    g = np.clip(g * brightness, 0, 255).astype(np.uint8)
    b = np.clip(b * brightness, 0, 255).astype(np.uint8)

    # Land (above sea level) -> transparent
    alpha = np.where(depth >= 0, 0, 220).astype(np.uint8)

    return np.stack([r, g, b, alpha], axis=-1)


def lonlat_to_pixel(lon, lat, zoom):
    """Convert lon/lat to global pixel coordinates at given zoom."""
    n = 2 ** zoom
    x = (lon + 180) / 360 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2 * n * TILE_SIZE
    return x, y


def tile_bounds_lonlat(tile):
    """Get lon/lat bounds of a tile."""
    bounds = mercantile.bounds(tile)
    return bounds.west, bounds.south, bounds.east, bounds.north


def generate_tiles(min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM):
    """Generate all tiles for the region."""
    print(f"Loading bathymetry from {GMRT_TIF}...")
    bathy, lons, lats = load_bathy()

    # Compute cellsize in meters for hillshade (approximate at 32S)
    lat_mean = np.mean(lats)
    cellsize_x = abs(lons[1] - lons[0]) * 111320 * math.cos(math.radians(lat_mean))
    cellsize_y = abs(lats[1] - lats[0]) * 110540

    print("Computing hillshade...")
    shade = compute_hillshade(bathy, cellsize_x, cellsize_y,
                              azimuth=AZIMUTH, altitude=ALTITUDE, z_factor=Z_FACTOR)

    print("Rendering colored bathymetry...")
    rgba_full = depth_to_color(bathy, shade)

    # Build interpolation arrays for fast tile rendering
    # lons/lats define the grid; we need to map tile pixel coords to grid indices
    lon_min, lon_max = lons[0], lons[-1]
    lat_min, lat_max = lats[-1], lats[0]  # lats are typically descending
    if lats[0] < lats[-1]:
        lat_min, lat_max = lats[0], lats[-1]
        rgba_full = rgba_full[::-1]  # flip to descending lat order

    total_tiles = 0
    for zoom in range(min_zoom, max_zoom + 1):
        tiles = list(mercantile.tiles(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH, zooms=zoom))
        print(f"Zoom {zoom}: {len(tiles)} tiles")

        for tile in tiles:
            west, south, east, north = tile_bounds_lonlat(tile)

            # Map tile pixel coordinates to source data indices
            # Each tile pixel maps to a lon/lat, which maps to a source array index
            tile_lons = np.linspace(west, east, TILE_SIZE, endpoint=False)
            tile_lats = np.linspace(north, south, TILE_SIZE, endpoint=False)  # top to bottom

            # Convert to fractional source indices
            col_idx = (tile_lons - lon_min) / (lon_max - lon_min) * (len(lons) - 1)
            row_idx = (lat_max - tile_lats) / (lat_max - lat_min) * (len(lats) - 1)

            # Check if tile is entirely outside our data
            if (col_idx[-1] < 0 or col_idx[0] >= len(lons) - 1 or
                row_idx[-1] < 0 or row_idx[0] >= len(lats) - 1):
                continue

            # Bilinear interpolation from source RGBA
            col_idx = np.clip(col_idx, 0, len(lons) - 2)
            row_idx = np.clip(row_idx, 0, len(lats) - 2)

            ci = col_idx.astype(int)
            ri = row_idx.astype(int)
            cf = col_idx - ci
            rf = row_idx - ri

            # Build 2D index grids
            ri2d = ri[:, np.newaxis] * np.ones(TILE_SIZE, dtype=int)[np.newaxis, :]
            ci2d = np.ones(TILE_SIZE, dtype=int)[:, np.newaxis] * ci[np.newaxis, :]
            rf2d = rf[:, np.newaxis] * np.ones(TILE_SIZE)[np.newaxis, :]
            cf2d = np.ones(TILE_SIZE)[:, np.newaxis] * cf[np.newaxis, :]

            # Bilinear sample
            tile_rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.float32)
            for dr in range(2):
                for dc in range(2):
                    wr = np.where(dr == 0, 1 - rf2d, rf2d)
                    wc = np.where(dc == 0, 1 - cf2d, cf2d)
                    w = wr * wc
                    r = np.clip(ri2d + dr, 0, rgba_full.shape[0] - 1)
                    c = np.clip(ci2d + dc, 0, rgba_full.shape[1] - 1)
                    tile_rgba += rgba_full[r, c].astype(np.float32) * w[:, :, np.newaxis]

            tile_img = np.clip(tile_rgba, 0, 255).astype(np.uint8)

            # Skip entirely transparent tiles
            if tile_img[:, :, 3].max() == 0:
                continue

            # Save tile
            tile_path = os.path.join(TILE_DIR, str(zoom), str(tile.x))
            os.makedirs(tile_path, exist_ok=True)
            img = Image.fromarray(tile_img, 'RGBA')
            img.save(os.path.join(tile_path, f"{tile.y}.png"), optimize=True)
            total_tiles += 1

    print(f"Generated {total_tiles} tiles in {TILE_DIR}/")
    return total_tiles


if __name__ == "__main__":
    min_z = MIN_ZOOM
    max_z = MAX_ZOOM
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--min-zoom" and i + 2 < len(sys.argv):
            min_z = int(sys.argv[i + 2])
        elif arg == "--max-zoom" and i + 2 < len(sys.argv):
            max_z = int(sys.argv[i + 2])

    generate_tiles(min_z, max_z)
