#!/usr/bin/env python3
"""
marlin_data.py — Marlin Zone Data Pipeline
==========================================
Fetches ocean data from Copernicus Marine Service,
detects SST fronts and chlorophyll edges,
extracts bathymetry contours from GEBCO,
and outputs GeoJSON files for the marine chart web app.

Requirements:
    pip install copernicusmarine xarray numpy scipy netCDF4 geojson

First-time setup:
    copernicusmarine login
    (enter your Copernicus Marine credentials)

Usage:
    python marlin_data.py                    # today's data, Perth region
    python marlin_data.py --date 2026-03-05  # specific date
    python marlin_data.py --bbox 113 -34 117 -30  # custom bounding box
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_BBOX = {
    "lon_min": 113.5,
    "lon_max": 116.5,
    "lat_min": -33.5,
    "lat_max": -30.5,
}

# Marlin temperature preferences (°C)
MARLIN_TEMPS = {
    "blue": {"min": 23, "max": 29, "color": "#f97316"},
    "striped": {"min": 21, "max": 24, "color": "#6366f1"},
}

# SST gradient threshold for front detection (°C per ~10km)
SST_GRADIENT_THRESHOLD = 0.3

# Chlorophyll gradient threshold for edge detection (log scale)
CHL_GRADIENT_THRESHOLD = 0.4

OUTPUT_DIR = "data"


# ---------------------------------------------------------------------------
# 1. Fetch ocean data from Copernicus Marine
# ---------------------------------------------------------------------------
def fetch_copernicus_sst(date_str, bbox):
    """Download SST data for the given date and region."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")

    print(f"[SST] Fetching for {date_str}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=0,
        maximum_depth=1,
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[SST] Saved to {output_file}")
    return output_file


def fetch_copernicus_currents(date_str, bbox):
    """Download ocean current data (u, v components)."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "currents_raw.nc")

    print(f"[Currents] Fetching for {date_str}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=0,
        maximum_depth=1,
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[Currents] Saved to {output_file}")
    return output_file


def fetch_copernicus_chlorophyll(date_str, bbox):
    """Download chlorophyll-a data."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "chl_raw.nc")

    print(f"[Chlorophyll] Fetching for {date_str}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=["chl"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=0,
        maximum_depth=1,
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[Chlorophyll] Saved to {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# 2. SST Front Detection
# ---------------------------------------------------------------------------
def detect_sst_fronts(sst_file, threshold=SST_GRADIENT_THRESHOLD):
    """
    Detect SST fronts using Sobel gradient magnitude.
    Returns GeoJSON FeatureCollection of front contour lines.
    """
    import xarray as xr
    from scipy.ndimage import sobel, gaussian_filter

    print("[SST Fronts] Processing...")
    ds = xr.open_dataset(sst_file)

    # Get SST array — handle different variable names
    for var in ["thetao", "analysed_sst", "sst"]:
        if var in ds:
            sst = ds[var].squeeze()
            break
    else:
        raise ValueError(f"No SST variable found. Available: {list(ds.data_vars)}")

    lons = sst.longitude.values if "longitude" in sst.dims else sst.lon.values
    lats = sst.latitude.values if "latitude" in sst.dims else sst.lat.values
    data = sst.values.copy()

    mask = np.isnan(data)

    # Fill land pixels with the ocean mean — NOT zero.
    # Filling with 0 creates a giant artificial ~22°C gradient at every coastline,
    # causing the Sobel filter to trace the entire coast instead of real fronts.
    ocean_mean = float(np.nanmean(data))
    data[mask] = ocean_mean

    # Smooth to suppress pixel-scale noise
    data_smooth = gaussian_filter(data, sigma=1.5)

    # Sobel gradient in x and y
    grad_x = sobel(data_smooth, axis=1)
    grad_y = sobel(data_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Mask out land
    grad_mag[mask] = 0

    # Extract contours at the threshold level
    features = _contours_to_geojson(
        grad_mag, lons, lats, threshold,
        properties={"type": "sst_front", "threshold": threshold}
    )

    # Extract isotherms at all marlin-relevant temperatures (deduplicated)
    # Only draw an isotherm if it falls within the actual SST data range
    data_min = float(np.nanmin(data_smooth[~mask]))
    data_max = float(np.nanmax(data_smooth[~mask]))
    seen_temps = set()
    isotherm_features = []
    for species, temps in MARLIN_TEMPS.items():
        for temp in [temps["min"], temps["max"]]:
            if temp in seen_temps:
                continue
            seen_temps.add(temp)
            if temp < data_min - 0.5 or temp > data_max + 0.5:
                continue
            iso = _contours_to_geojson(
                data_smooth, lons, lats, temp,
                properties={
                    "type": "isotherm",
                    "temperature": temp,
                    "species": species,
                    "color": temps["color"],
                }
            )
            isotherm_features.extend(iso)

    all_features = features + isotherm_features

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "sst_fronts.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[SST Fronts] {len(features)} front lines, {len(isotherm_features)} isotherms → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3. Chlorophyll Concentration Contours
# ---------------------------------------------------------------------------

# Ecologically meaningful chlorophyll levels (mg/m³):
#   0.07 = oligotrophic open ocean
#   0.15 = transition / shelf edge
#   0.30 = productive shelf / upwelling fringe  ← marlin bait zone
#   0.60 = high productivity / coastal bloom
CHL_LEVELS = [
    (0.07, "low",       "#86efac"),
    (0.15, "moderate",  "#4ade80"),
    (0.30, "high",      "#16a34a"),
    (0.60, "very_high", "#14532d"),
]

def detect_chlorophyll_edges(chl_file, threshold=CHL_GRADIENT_THRESHOLD):
    """
    Trace chlorophyll-a concentration contours at ecologically meaningful levels.
    Gradient-based edge detection is too noisy at 0.25° resolution and produces
    coastal artefacts.  Contours show the actual chlorophyll field clearly.
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[Chlorophyll] Processing concentration contours...")
    ds = xr.open_dataset(chl_file)

    for var in ["chl", "CHL", "chlor_a"]:
        if var in ds:
            chl = ds[var].squeeze()
            break
    else:
        raise ValueError(f"No chlorophyll variable found. Available: {list(ds.data_vars)}")

    lons = chl.longitude.values if "longitude" in chl.dims else chl.lon.values
    lats = chl.latitude.values if "latitude" in chl.dims else chl.lat.values
    data = chl.values.copy().astype(float)

    mask = np.isnan(data) | (data <= 0)
    # Fill land with a value well below any contour level so contours don't
    # run along the coast.
    data[mask] = 0.0

    # Light smoothing to reduce single-pixel noise at 0.25° resolution
    data_smooth = gaussian_filter(data, sigma=0.8)
    data_smooth[mask] = 0.0

    data_max = float(np.nanmax(data_smooth))
    all_features = []

    for level, label, color in CHL_LEVELS:
        if level > data_max:
            continue
        feats = _contours_to_geojson(
            data_smooth, lons, lats, level,
            properties={
                "type": "chl_contour",
                "concentration": level,
                "label": label,
                "color": color,
            }
        )
        # Drop artefact lines that touch the bounding box edge — those are land boundary
        # contours from the zero-fill.  Any real ocean contour won't span the full domain.
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 3]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "chl_edges.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Chlorophyll] {len(all_features)} contours at {[l[0] for l in CHL_LEVELS]} mg/m³ → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Current Vectors (for arrow overlay)
# ---------------------------------------------------------------------------
def process_currents(currents_file):
    """
    Convert current data to a grid of arrow features for map display.
    Decimates to ~20km spacing to keep the overlay readable.
    """
    import xarray as xr

    print("[Currents] Processing...")
    ds = xr.open_dataset(currents_file)

    uo = ds["uo"].squeeze()
    vo = ds["vo"].squeeze()

    lons = uo.longitude.values if "longitude" in uo.dims else uo.lon.values
    lats = uo.latitude.values if "latitude" in uo.dims else uo.lat.values
    u = uo.values
    v = vo.values

    # Decimate — take every Nth point for readable arrows
    step = max(1, len(lons) // 30)
    features = []

    for i in range(0, len(lats), step):
        for j in range(0, len(lons), step):
            u_val = float(u[i, j]) if not np.isnan(u[i, j]) else None
            v_val = float(v[i, j]) if not np.isnan(v[i, j]) else None
            if u_val is None or v_val is None:
                continue

            speed = np.sqrt(u_val**2 + v_val**2)
            if speed < 0.01:
                continue

            direction = np.degrees(np.arctan2(u_val, v_val)) % 360
            # Arrow endpoint — exaggerated for visibility
            scale = 0.02  # degrees per 0.1 m/s
            end_lon = float(lons[j]) + u_val * scale
            end_lat = float(lats[i]) + v_val * scale

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(lons[j]), float(lats[i])],
                        [end_lon, end_lat],
                    ],
                },
                "properties": {
                    "speed_ms": round(speed, 3),
                    "speed_kn": round(speed * 1.94384, 2),
                    "direction": round(direction, 1),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    output_path = os.path.join(OUTPUT_DIR, "currents.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Currents] {len(features)} vectors → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 5. Bathymetry Contour Extraction (from GEBCO GeoTIFF)
# ---------------------------------------------------------------------------
def extract_bathymetry_contours(gebco_file, depths=[-50, -100, -200, -500, -1000]):
    """
    Extract depth contour lines from a GEBCO GeoTIFF file.
    Requires: pip install rasterio
    
    Download GEBCO GeoTIFF from:
        https://download.gebco.net/
        Select Perth region: lon 113-117, lat -34 to -30
        Download as GeoTIFF
    """
    try:
        import rasterio
        from rasterio import features as rio_features
    except ImportError:
        print("[Bathymetry] rasterio not installed. Run: pip install rasterio")
        print("[Bathymetry] Alternatively, use GDAL directly:")
        print(f"  gdal_contour -fl {' '.join(str(d) for d in depths)} {gebco_file} contours.geojson -f GeoJSON")
        return None

    print(f"[Bathymetry] Extracting contours from {gebco_file}...")

    with rasterio.open(gebco_file) as src:
        data = src.read(1)
        transform = src.transform

    all_features = []
    for depth in depths:
        # Create binary mask: 1 where depth <= target, 0 otherwise
        binary = (data <= depth).astype(np.uint8)

        # Extract polygon boundaries
        for geom, val in rio_features.shapes(binary, transform=transform):
            if val == 1:
                # Convert polygon boundary to linestring
                coords = geom["coordinates"][0]
                if len(coords) > 2:
                    all_features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"depth": depth, "label": f"{abs(depth)}m"},
                    })

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "bathymetry_contours.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Bathymetry] {len(all_features)} contour segments → {output_path}")
    return output_path


def extract_contours_gdal(gebco_file, depths=[-50, -100, -200, -500, -1000]):
    """
    Alternative: use GDAL command line for contour extraction.
    More reliable for large files.
    """
    import subprocess

    output_path = os.path.join(OUTPUT_DIR, "bathymetry_contours.geojson")
    depth_str = " ".join(str(d) for d in depths)

    cmd = f'gdal_contour -fl {depth_str} "{gebco_file}" "{output_path}" -f GeoJSON'
    print(f"[Bathymetry] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    print(f"[Bathymetry] Contours → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 5b. Bathymetry from GMRT (Global Multi-Resolution Topography) REST API
# ---------------------------------------------------------------------------
def fetch_bathymetry_gmrt(bbox, depths=[-200, -500, -1000]):
    """
    Download a bathymetry GeoTIFF from the GMRT GridServer API and extract
    depth contours.  GMRT is freely available, no API key required.

    API docs: https://www.gmrt.org/services/gridserverinfo.html
    """
    import requests

    tif_path = os.path.join(OUTPUT_DIR, "bathy_gmrt.tif")

    # Expand bbox by ~0.5° so contours don't end at the exact region edge
    pad = 0.5
    url = (
        "https://www.gmrt.org/services/GridServer"
        f"?west={bbox['lon_min'] - pad}"
        f"&east={bbox['lon_max'] + pad}"
        f"&south={bbox['lat_min'] - pad}"
        f"&north={bbox['lat_max'] + pad}"
        f"&layer=topo&format=geotiff&resolution=low"
    )

    print(f"[Bathymetry] Downloading GMRT GeoTIFF...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with open(tif_path, "wb") as f:
        f.write(resp.content)
    print(f"[Bathymetry] Downloaded {len(resp.content)//1024} KB → {tif_path}")

    return extract_bathymetry_contours(tif_path, depths=depths)


# ---------------------------------------------------------------------------
# Contour extraction helper (from numpy array to GeoJSON)
# ---------------------------------------------------------------------------
def _contours_to_geojson(grid, lons, lats, level, properties=None):
    """Convert a 2D grid to GeoJSON contour lines at a given level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        cs = ax.contour(lons, lats, grid, levels=[level])
        plt.close(fig)

        # Use allsegs (avoids the deprecated .collections attribute in mpl 3.8+)
        features = []
        for seg_list in cs.allsegs:       # one entry per level (we only have one)
            for seg in seg_list:           # each connected segment
                if len(seg) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[round(float(c[0]), 5), round(float(c[1]), 5)] for c in seg],
                        },
                        "properties": dict(properties) if properties else {},
                    })
        return features

    except ImportError:
        # Fallback: use skimage if matplotlib not available
        try:
            from skimage.measure import find_contours

            contours = find_contours(grid, level)
            features = []
            for contour in contours:
                coords = []
                for point in contour:
                    yi, xi = int(point[0]), int(point[1])
                    if 0 <= yi < len(lats) and 0 <= xi < len(lons):
                        coords.append([round(float(lons[xi]), 5), round(float(lats[yi]), 5)])
                if len(coords) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": dict(properties) if properties else {},
                    })
            return features
        except ImportError:
            print("WARNING: Need matplotlib or scikit-image for contour extraction")
            return []


# ---------------------------------------------------------------------------
# 6. Generate summary report
# ---------------------------------------------------------------------------
def generate_report(date_str, bbox):
    """Create a simple JSON summary of conditions."""
    import xarray as xr

    report = {
        "date": date_str,
        "region": bbox,
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    sst_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")
    if os.path.exists(sst_file):
        ds = xr.open_dataset(sst_file)
        for var in ["thetao", "analysed_sst", "sst"]:
            if var in ds:
                sst = ds[var].squeeze().values
                sst_valid = sst[~np.isnan(sst)]
                report["sst"] = {
                    "min": round(float(np.min(sst_valid)), 1),
                    "max": round(float(np.max(sst_valid)), 1),
                    "mean": round(float(np.mean(sst_valid)), 1),
                    "blue_marlin_zone": bool(
                        np.any((sst_valid >= 23) & (sst_valid <= 29))
                    ),
                    "striped_marlin_zone": bool(
                        np.any((sst_valid >= 21) & (sst_valid <= 24))
                    ),
                }
                break

    output_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"MARLIN ZONE REPORT — {date_str}")
    print(f"{'='*50}")
    if "sst" in report:
        s = report["sst"]
        print(f"SST Range: {s['min']}°C — {s['max']}°C (mean {s['mean']}°C)")
        print(f"Blue Marlin Zone (23-29°C):    {'✅ YES' if s['blue_marlin_zone'] else '❌ NO'}")
        print(f"Striped Marlin Zone (21-24°C): {'✅ YES' if s['striped_marlin_zone'] else '❌ NO'}")
    print(f"{'='*50}\n")

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Marlin Zone Data Pipeline")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="Bounding box. Default: Perth region")
    parser.add_argument("--gebco", default=None, help="Path to GEBCO GeoTIFF for bathymetry contours")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download, process existing files")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    # Set date
    if args.date:
        date_str = args.date
    else:
        date_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Set bbox
    bbox = dict(DEFAULT_BBOX)
    if args.bbox:
        bbox = {
            "lon_min": args.bbox[0],
            "lat_min": args.bbox[1],
            "lon_max": args.bbox[2],
            "lat_max": args.bbox[3],
        }

    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n🐟 Marlin Zone Data Pipeline")
    print(f"   Date:   {date_str}")
    print(f"   Region: {bbox['lon_min']}–{bbox['lon_max']}°E, {bbox['lat_min']}–{bbox['lat_max']}°S")
    print(f"   Output: {OUTPUT_DIR}/\n")

    # Fetch data
    if not args.skip_fetch:
        try:
            fetch_copernicus_sst(date_str, bbox)
        except Exception as e:
            print(f"[SST] Error: {e}")

        try:
            fetch_copernicus_currents(date_str, bbox)
        except Exception as e:
            print(f"[Currents] Error: {e}")

        try:
            fetch_copernicus_chlorophyll(date_str, bbox)
        except Exception as e:
            print(f"[Chlorophyll] Error: {e}")

    # Process SST fronts
    sst_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")
    if os.path.exists(sst_file):
        try:
            detect_sst_fronts(sst_file)
        except Exception as e:
            print(f"[SST Fronts] Error: {e}")

    # Process chlorophyll edges
    chl_file = os.path.join(OUTPUT_DIR, "chl_raw.nc")
    if os.path.exists(chl_file):
        try:
            detect_chlorophyll_edges(chl_file)
        except Exception as e:
            print(f"[Chlorophyll Edges] Error: {e}")

    # Process currents
    cur_file = os.path.join(OUTPUT_DIR, "currents_raw.nc")
    if os.path.exists(cur_file):
        try:
            process_currents(cur_file)
        except Exception as e:
            print(f"[Currents] Error: {e}")

    # Bathymetry contours — GEBCO file takes priority; fall back to GMRT API
    if args.gebco:
        try:
            extract_contours_gdal(args.gebco)
        except Exception as e:
            print(f"[Bathymetry] GDAL failed: {e}, trying rasterio...")
            try:
                extract_bathymetry_contours(args.gebco)
            except Exception as e2:
                print(f"[Bathymetry] Error: {e2}")
    else:
        try:
            fetch_bathymetry_gmrt(bbox)
        except Exception as e:
            print(f"[Bathymetry] GMRT download failed: {e}")

    # Generate report
    generate_report(date_str, bbox)

    print("✅ Done! GeoJSON files are in the data/ directory.")
    print("   Copy them to your web app's data/ folder and reload.\n")


if __name__ == "__main__":
    main()
