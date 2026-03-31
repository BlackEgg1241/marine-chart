"""
Download high-resolution bathymetry data for Spanish Mackerel reef scoring.

Sources (in preference order):
1. AusSeabed Compilation — multibeam survey data, variable resolution (10-50m)
   Served via WCS from Geoscience Australia
2. Perth Canyon Marine Park 40m grid — Geoscience Australia eCat
3. GMRT fallback — ~200m resolution (insufficient for reef detection)

Output: data/bathy_hires.tif — GeoTIFF covering the inshore SM scoring zone.

Usage:
    python fetch_hires_bathy.py                          # default Perth region
    python fetch_hires_bathy.py --bbox 114.5 -32.8 116 -31.3  # custom bbox
"""

import os
import sys
import subprocess
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Inshore Spanish Mackerel zone — shallower and closer to coast than marlin
DEFAULT_BBOX = {
    "lon_min": 114.5,
    "lon_max": 116.0,
    "lat_min": -32.8,
    "lat_max": -31.3,
}

# AusSeabed WCS endpoint for bathymetry compilation
AUSSEABED_WCS = "https://warehouse.ausseabed.gov.au/geoserver/ows"

# GMRT REST API (fallback)
GMRT_URL = "https://www.gmrt.org/services/GridServer"


def fetch_ausseabed_wcs(bbox, output_path, resolution=0.0005):
    """Download bathymetry from AusSeabed WCS service.

    Resolution ~0.0005 deg = ~50m at Perth latitude.
    """
    # Coverage ID for the national compilation
    coverage_id = "ausseabed:bathymetry"

    # Calculate grid size
    width = int((bbox["lon_max"] - bbox["lon_min"]) / resolution)
    height = int((bbox["lat_max"] - bbox["lat_min"]) / resolution)

    # WCS 2.0.1 GetCoverage request
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "CoverageId": coverage_id,
        "format": "image/tiff",
        "subset": f"Long({bbox['lon_min']},{bbox['lon_max']})",
        "subset": f"Lat({bbox['lat_min']},{bbox['lat_max']})",
        "scalesize": f"Long({width}),Lat({height})",
    }

    # Build URL with proper WCS 2.0.1 subsetting
    url = (f"{AUSSEABED_WCS}?service=WCS&version=2.0.1&request=GetCoverage"
           f"&CoverageId={coverage_id}"
           f"&format=image/tiff"
           f"&subset=Long({bbox['lon_min']},{bbox['lon_max']})"
           f"&subset=Lat({bbox['lat_min']},{bbox['lat_max']})")

    print(f"Fetching AusSeabed bathymetry...")
    print(f"  URL: {url[:120]}...")
    print(f"  Region: {bbox['lon_min']}-{bbox['lon_max']}E, "
          f"{bbox['lat_min']}-{bbox['lat_max']}S")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "tiff" in content_type or "image" in content_type:
                with open(output_path, "wb") as f:
                    f.write(resp.read())
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
                return True
            else:
                # Probably an error response
                body = resp.read().decode("utf-8", errors="replace")[:500]
                print(f"  AusSeabed returned non-TIFF: {content_type}")
                print(f"  Body: {body[:200]}")
                return False
    except Exception as e:
        print(f"  AusSeabed WCS failed: {e}")
        return False


def fetch_gmrt(bbox, output_path, resolution="high"):
    """Download bathymetry from GMRT REST API (~200m resolution)."""
    url = (f"{GMRT_URL}?north={bbox['lat_max']}&south={bbox['lat_min']}"
           f"&east={bbox['lon_max']}&west={bbox['lon_min']}"
           f"&layer=topo&format=geotiff&resolution={resolution}")

    print(f"Fetching GMRT bathymetry (fallback)...")
    print(f"  URL: {url[:120]}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(output_path, "wb") as f:
                f.write(resp.read())
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  GMRT fetch failed: {e}")
        return False


def check_resolution(tif_path):
    """Check and report the resolution of a bathymetry GeoTIFF."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            res_x = abs(src.transform.a) * 111000  # approx meters at equator
            res_y = abs(src.transform.e) * 110000
            # Correct for latitude
            import math
            lat_mid = (src.bounds.bottom + src.bounds.top) / 2
            res_x *= math.cos(math.radians(lat_mid))
            print(f"\nBathymetry resolution:")
            print(f"  File: {tif_path}")
            print(f"  Size: {src.width}x{src.height} pixels")
            print(f"  Resolution: ~{res_x:.0f}m x ~{res_y:.0f}m")
            print(f"  Bounds: {src.bounds.left:.3f}-{src.bounds.right:.3f}E, "
                  f"{src.bounds.bottom:.3f}-{src.bounds.top:.3f}N")
            depth_data = src.read(1)
            import numpy as np
            valid = depth_data[depth_data != src.nodata] if src.nodata else depth_data
            if len(valid) > 0:
                print(f"  Depth range: {float(np.nanmin(valid)):.0f} to "
                      f"{float(np.nanmax(valid)):.0f}m")
            reef_capable = res_x < 100
            print(f"  Reef detection: {'YES' if reef_capable else 'NO'} "
                  f"(need <100m, have ~{res_x:.0f}m)")
            return res_x
    except ImportError:
        print("  rasterio not installed - cannot check resolution")
        return None
    except Exception as e:
        print(f"  Error checking resolution: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch high-res bathymetry")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("W", "S", "E", "N"),
                        help="Bounding box: west south east north")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "bathy_hires.tif"),
                        help="Output GeoTIFF path")
    parser.add_argument("--gmrt-only", action="store_true",
                        help="Skip AusSeabed, use GMRT only")
    args = parser.parse_args()

    bbox = DEFAULT_BBOX.copy()
    if args.bbox:
        bbox = {
            "lon_min": args.bbox[0], "lat_min": args.bbox[1],
            "lon_max": args.bbox[2], "lat_max": args.bbox[3],
        }

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success = False

    # Try AusSeabed first (higher resolution)
    if not args.gmrt_only:
        success = fetch_ausseabed_wcs(bbox, output_path)

    # Fall back to GMRT
    if not success:
        success = fetch_gmrt(bbox, output_path)

    if success:
        check_resolution(output_path)
    else:
        print("\nFailed to download bathymetry from any source.")
        print("Manual download options:")
        print("  1. AusSeabed Portal: https://www.ausseabed.gov.au/data")
        print("  2. GA Perth Canyon 40m: https://ecat.ga.gov.au/geonetwork/srv/api/records/6a04ae7d-73f7-47c3-8fb1-f416c55ab319")
        print("  3. WA DoT Portal: https://catalogue.data.wa.gov.au/app/department-of-transport-wa-bathymetric-surveys")
        print(f"\nPlace the GeoTIFF at: {output_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
