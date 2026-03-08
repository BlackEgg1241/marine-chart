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
# Higher value = fewer but more significant fronts (less noise)
SST_GRADIENT_THRESHOLD = 0.5

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


def fetch_copernicus_kd490(date_str, bbox):
    """
    Download KD490 (diffuse attenuation at 490nm) from Copernicus Marine.
    KD490 is a water clarity indicator:
      < 0.06 m⁻¹ = very clear, clean blue water  (no trichodesmium/turbidity)
      0.06–0.10   = moderately clear
      > 0.10 m⁻¹  = turbid / likely bloom-affected
    Dataset: cmems_obs-oc_glo_bgc-transp_nrt_l4-gapfree-multi-4km_P1D
    Note: satellite ocean colour products have a ~2-3 day latency, so we
    step back through the last 5 days until a valid day is found.
    """
    import copernicusmarine
    from datetime import datetime, timedelta

    output_file = os.path.join(OUTPUT_DIR, "kd490_raw.nc")
    base_dt = datetime.strptime(date_str, "%Y-%m-%d")

    for delta in range(0, 6):
        try_date = (base_dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        print(f"[Water Clarity] Fetching KD490 for {try_date}...")
        try:
            copernicusmarine.subset(
                dataset_id="cmems_obs-oc_glo_bgc-transp_nrt_l4-gapfree-multi-4km_P1D",
                variables=["KD490"],
                minimum_longitude=bbox["lon_min"],
                maximum_longitude=bbox["lon_max"],
                minimum_latitude=bbox["lat_min"],
                maximum_latitude=bbox["lat_max"],
                start_datetime=f"{try_date}T00:00:00",
                end_datetime=f"{try_date}T23:59:59",
                output_filename=output_file,
                output_directory=".",
                overwrite=True,
            )
            print(f"[Water Clarity] Saved to {output_file} (data date: {try_date})")
            return output_file
        except Exception as e:
            if delta < 5:
                print(f"[Water Clarity] {try_date} not available, trying earlier...")
            else:
                raise
    return None


def fetch_copernicus_ssh(date_str, bbox):
    """
    Download Sea Level Anomaly (SLA) from CMEMS satellite altimetry.
    SLA shows eddies: positive = warm-core anticyclonic eddy (marlin habitat).
    Dataset: cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D
    Satellite altimetry has ~3-5 day latency; steps back automatically.
    """
    import copernicusmarine
    from datetime import datetime, timedelta

    output_file = os.path.join(OUTPUT_DIR, "ssh_raw.nc")
    base_dt = datetime.strptime(date_str, "%Y-%m-%d")

    for delta in range(0, 8):
        try_date = (base_dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        print(f"[SSH/Eddies] Fetching SLA for {try_date}...")
        try:
            copernicusmarine.subset(
                dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
                variables=["sla"],
                minimum_longitude=bbox["lon_min"],
                maximum_longitude=bbox["lon_max"],
                minimum_latitude=bbox["lat_min"],
                maximum_latitude=bbox["lat_max"],
                start_datetime=f"{try_date}T00:00:00",
                end_datetime=f"{try_date}T23:59:59",
                output_filename=output_file,
                output_directory=".",
                overwrite=True,
            )
            print(f"[SSH/Eddies] Saved to {output_file} (data date: {try_date})")
            return output_file
        except Exception as e:
            if delta < 7:
                print(f"[SSH/Eddies] {try_date} not available, trying earlier...")
            else:
                raise
    return None


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

    # Mask out land AND a 2-pixel coastal buffer.
    # Even with ocean_mean fill, the boundary pixels still carry a residual
    # gradient artefact.  Dilating the land mask by 2 cells removes it.
    from scipy.ndimage import binary_dilation
    coast_buffer = binary_dilation(mask, iterations=2)
    grad_mag[coast_buffer] = 0

    # Extract contours at the threshold level — require >=8 pts to filter noise stubs
    features = _contours_to_geojson(
        grad_mag, lons, lats, threshold,
        properties={"type": "sst_front", "threshold": threshold},
        min_pts=8
    )

    # Extract isotherms at all marlin-relevant temperatures (deduplicated)
    # Extract isotherms from lightly-smoothed data (sigma=0.5) so they closely
    # match the visible SST gradient rather than the front-detection smooth field
    iso_data = gaussian_filter(data, sigma=0.5)
    iso_data[mask] = np.nan

    data_min = float(np.nanmin(iso_data[~mask]))
    data_max = float(np.nanmax(iso_data[~mask]))
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
                np.where(mask, float(np.nanmean(iso_data[~mask])), iso_data),
                lons, lats, temp,
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
# 3b. Water Clarity (KD490)
# ---------------------------------------------------------------------------
# Kd490 contour levels (m⁻¹): boundary between clean and turbid water
# < 0.06 = clear blue offshore water  (avoid trichodesmium)
# 0.06   = "clean water line" — the most useful boundary for anglers
# 0.10   = noticeably turbid
# > 0.15 = bloom/trichodesmium likely
KD490_LEVELS = [
    (0.06, "clean",   "#38bdf8"),   # sky blue — clear water boundary
    (0.10, "turbid",  "#fb923c"),   # orange — murky water starts here
    (0.15, "bloom",   "#dc2626"),   # red — likely bloom/trichodesmium
]

def process_water_clarity(kd490_file):
    """
    Extract KD490 contour lines to show clean vs bloom-affected water.
    Anglers want to stay in water below the 0.06 m⁻¹ contour (clear side).
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[Water Clarity] Processing KD490 contours...")
    ds = xr.open_dataset(kd490_file)
    kd = ds["KD490"].squeeze()

    lons = kd.longitude.values if "longitude" in kd.dims else kd.lon.values
    lats = kd.latitude.values if "latitude" in kd.dims else kd.lat.values
    data = kd.values.copy().astype(float)

    mask = np.isnan(data) | (data <= 0)
    data[mask] = 0.0
    data_smooth = gaussian_filter(data, sigma=0.5)
    data_smooth[mask] = 0.0

    data_max = float(np.nanmax(data_smooth))
    all_features = []

    for level, label, color in KD490_LEVELS:
        if level > data_max:
            continue
        feats = _contours_to_geojson(
            data_smooth, lons, lats, level,
            properties={
                "type": "kd490_contour",
                "kd490": level,
                "label": label,
                "color": color,
            }
        )
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 3]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "water_clarity.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Water Clarity] {len(all_features)} contours → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3c. SSH / Eddy Detection (Sea Level Anomaly)
# ---------------------------------------------------------------------------
# SLA contour levels (m): eddy structure boundaries
# Positive SLA = warm-core anticyclonic eddy (clockwise in S. hemisphere)
#   → traps warm water, marlin congregate on leading edge
# Negative SLA = cold-core cyclonic eddy → upwelling, bait but cold
SLA_LEVELS = [
    (-0.01, "cold_eddy",  "#60a5fa"),   # blue — slight cold anomaly
    ( 0.03, "neutral",    "#94a3b8"),   # gray — eddy boundary
    ( 0.07, "warm_eddy",  "#fb923c"),   # orange — warm eddy
    ( 0.11, "warm_core",  "#ef4444"),   # red — warm eddy core
]

def process_ssh(ssh_file):
    """
    Extract SLA contours to show eddy structure.
    Anglers look for the warm side of the 0.0 m contour (positive SLA).
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[SSH/Eddies] Processing SLA contours...")
    ds = xr.open_dataset(ssh_file)
    # variable may be 'sla' or 'adt' depending on product version
    varname = "sla" if "sla" in ds else "adt"
    sla = ds[varname].squeeze()

    lons = sla.longitude.values if "longitude" in sla.dims else sla.lon.values
    lats = sla.latitude.values if "latitude" in sla.dims else sla.lat.values
    data = sla.values.copy().astype(float)

    mask = np.isnan(data)
    if not mask.all():
        fill = float(np.nanmean(data))
        data[mask] = fill
        data_smooth = gaussian_filter(data, sigma=0.5)
        data_smooth[mask] = np.nan
    else:
        print("[SSH/Eddies] No valid data")
        return None

    all_features = []
    data_min = float(np.nanmin(data_smooth))
    data_max = float(np.nanmax(data_smooth))

    for level, label, color in SLA_LEVELS:
        if level < data_min - 0.05 or level > data_max + 0.05:
            continue
        feats = _contours_to_geojson(
            np.where(np.isnan(data_smooth), fill, data_smooth),
            lons, lats, level,
            properties={"type": "sla_contour", "sla": level, "label": label, "color": color}
        )
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 5]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "ssh_eddies.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[SSH/Eddies] {len(all_features)} contours → {output_path}")
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
            if speed < 0.02:
                continue

            direction = np.degrees(np.arctan2(u_val, v_val)) % 360
            # Arrow scale: 0.075 degrees per m/s  (~8 km for 1 m/s current)
            scale = 0.075
            dx = u_val * scale
            dy = v_val * scale
            ox, oy = float(lons[j]), float(lats[i])
            ex, ey = ox + dx, oy + dy

            # Build arrowhead: small V at the tip pointing in travel direction
            angle_rad = np.arctan2(dx, dy)
            head = 0.0075  # arrowhead arm length in degrees
            spread = np.radians(30)
            ax1 = ex - head * np.sin(angle_rad + spread)
            ay1 = ey - head * np.cos(angle_rad + spread)
            ax2 = ex - head * np.sin(angle_rad - spread)
            ay2 = ey - head * np.cos(angle_rad - spread)

            # Shaft + arrowhead as a single MultiLineString
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[ox, oy], [ex, ey]],
                        [[ax1, ay1], [ex, ey], [ax2, ay2]],
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
    except ImportError:
        print("[Bathymetry] rasterio not installed. Run: pip install rasterio")
        print("[Bathymetry] Alternatively, use GDAL directly:")
        print(f"  gdal_contour -fl {' '.join(str(d) for d in depths)} {gebco_file} contours.geojson -f GeoJSON")
        return None

    print(f"[Bathymetry] Extracting contours from {gebco_file}...")

    DEPTH_STYLE = {
        -100:  {"label": "100m contour",    "color": "#a3e635"},
        -200:  {"label": "200m shelf edge", "color": "#f59e0b"},
        -500:  {"label": "500m contour",    "color": "#06b6d4"},
        -1000: {"label": "1000m contour",   "color": "#3b82f6"},
    }

    with rasterio.open(gebco_file) as src:
        data = src.read(1).astype(float)
        t = src.transform
        height, width = data.shape
        lons = np.array([t.c + (j + 0.5) * t.a for j in range(width)])
        lats = np.array([t.f + (i + 0.5) * t.e for i in range(height)])
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_features = []
    for depth in depths:
        style = DEPTH_STYLE.get(depth, {"label": f"{abs(depth)}m", "color": "#94a3b8"})
        fig, ax = plt.subplots()
        cs = ax.contour(lons, lats, data, levels=[depth])
        plt.close(fig)
        segs = 0
        for seg_list in cs.allsegs:
            for seg in seg_list:
                # Filter out tiny artefact segments (< 10 points)
                if len(seg) >= 10:
                    segs += 1
                    coords = [[round(float(x), 5), round(float(y), 5)] for x, y in seg]
                    all_features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"depth": depth, "label": style["label"], "color": style["color"]},
                    })
        print(f"[Bathymetry] {depth}m: {segs} segments")

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
def fetch_bathymetry_gmrt(bbox, depths=[-100, -200, -500, -1000]):
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
        f"&layer=topo&format=geotiff&resolution=high"
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
def _contours_to_geojson(grid, lons, lats, level, properties=None, min_pts=2):
    """Convert a 2D grid to GeoJSON contour lines at a given level.
    min_pts: minimum number of coordinate points to keep a segment (filters noise)."""
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
                if len(seg) >= min_pts:
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
    parser.add_argument("--no-update-latest", action="store_true", help="Do not copy GeoJSON to base data/ dir (use for backfill)")

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

    # Always write to a dated subfolder so the UI can navigate by date,
    # and ALSO write to the base output dir as the "latest" copy.
    base_output = args.output
    dated_output = os.path.join(base_output, date_str)
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(dated_output, exist_ok=True)
    OUTPUT_DIR = dated_output

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

        try:
            fetch_copernicus_kd490(date_str, bbox)
        except Exception as e:
            print(f"[Water Clarity] Fetch error: {e}")

        try:
            fetch_copernicus_ssh(date_str, bbox)
        except Exception as e:
            print(f"[SSH/Eddies] Fetch error: {e}")

    def _nc(name):
        """Find a NetCDF file: prefer dated dir, fall back to base dir."""
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            return p
        p2 = os.path.join(base_output, name)
        return p2 if os.path.exists(p2) else None

    # Download bathymetry FIRST so we can build the depth mask for clipping
    # (reuse cached tif if it already exists in the dated or base dir)
    tif_path = os.path.join(OUTPUT_DIR, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        tif_path_base = os.path.join(base_output, "bathy_gmrt.tif")
        if os.path.exists(tif_path_base):
            import shutil as _shutil
            _shutil.copy2(tif_path_base, tif_path)

    if not os.path.exists(tif_path):
        try:
            fetch_bathymetry_gmrt(bbox)
        except Exception as e:
            print(f"[Bathymetry] Early download failed: {e}")
            tif_path = None

    deep_mask = build_deep_water_mask(tif_path) if tif_path and os.path.exists(tif_path) else None

    # Process SST fronts
    sst_file = _nc("sst_raw.nc")
    if sst_file:
        try:
            detect_sst_fronts(sst_file)
        except Exception as e:
            print(f"[SST Fronts] Error: {e}")

    # Process chlorophyll edges
    chl_file = _nc("chl_raw.nc")
    if chl_file:
        try:
            detect_chlorophyll_edges(chl_file)
        except Exception as e:
            print(f"[Chlorophyll Edges] Error: {e}")

    # Process water clarity (KD490)
    kd_file = _nc("kd490_raw.nc")
    if kd_file:
        try:
            process_water_clarity(kd_file)
        except Exception as e:
            print(f"[Water Clarity] Processing error: {e}")

    # Process SSH eddies
    ssh_file = _nc("ssh_raw.nc")
    if ssh_file:
        try:
            process_ssh(ssh_file)
        except Exception as e:
            print(f"[SSH/Eddies] Processing error: {e}")

    # Process currents
    cur_file = _nc("currents_raw.nc")
    if cur_file:
        try:
            process_currents(cur_file)
        except Exception as e:
            print(f"[Currents] Error: {e}")

    # Bathymetry contours — GEBCO file takes priority; fall back to GMRT API
    # Tif was already downloaded above for the depth mask; just extract contours
    if args.gebco:
        try:
            extract_contours_gdal(args.gebco)
        except Exception as e:
            print(f"[Bathymetry] GDAL failed: {e}, trying rasterio...")
            try:
                extract_bathymetry_contours(args.gebco)
            except Exception as e2:
                print(f"[Bathymetry] Error: {e2}")
    elif tif_path and os.path.exists(tif_path):
        try:
            extract_bathymetry_contours(tif_path)
        except Exception as e:
            print(f"[Bathymetry] Contour extraction failed: {e}")
    else:
        try:
            fetch_bathymetry_gmrt(bbox)
        except Exception as e:
            print(f"[Bathymetry] GMRT download failed: {e}")

    # Clip all analytical GeoJSON features to the >100m depth mask
    # This removes features over land and shallow coastal areas
    if deep_mask is not None:
        for fname in ["sst_fronts.geojson", "chl_edges.geojson", "currents.geojson",
                      "water_clarity.geojson", "ssh_eddies.geojson"]:
            clip_geojson_to_mask(os.path.join(OUTPUT_DIR, fname), deep_mask)

    # Generate report
    generate_report(date_str, bbox)

    # Copy GeoJSON files to base data/ dir so latest data is always at data/*.geojson
    import shutil
    geojson_files = [
        "sst_fronts.geojson", "chl_edges.geojson", "currents.geojson",
        "bathymetry_contours.geojson", "water_clarity.geojson", "ssh_eddies.geojson",
    ]
    if not args.no_update_latest:
        for fname in geojson_files:
            src = os.path.join(OUTPUT_DIR, fname)
            dst = os.path.join(base_output, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"✅ Done! GeoJSON files in {OUTPUT_DIR}/ and {base_output}/")
    else:
        print(f"✅ Done! GeoJSON files in {OUTPUT_DIR}/ (base data/ not updated)")
    print(f"   Date folder: data/{date_str}/\n")


def build_deep_water_mask(tif_path, depth_threshold=-100):
    """
    Build a Shapely (Multi)Polygon covering ocean areas deeper than depth_threshold.
    Used to clip marlin zone features so they don't appear over land or shallow water.
    Returns None on failure (features will not be clipped).
    """
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.path import Path as MplPath
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        with rasterio.open(tif_path) as ds:
            data = ds.read(1).astype(float)
            t = ds.transform
            h, w = data.shape
            lons = np.array([t.c + (j + 0.5) * t.a for j in range(w)])
            lats = np.array([t.f + (i + 0.5) * t.e for i in range(h)])
            nd = ds.nodata
            if nd is not None:
                data[data == nd] = np.nan

        # Filled contour: region where elevation < depth_threshold (i.e. deeper water)
        fig, ax = plt.subplots()
        cf = ax.contourf(lons, lats, data, levels=[-12000, depth_threshold])
        plt.close(fig)

        polys = []
        for col in cf.collections:
            for path in col.get_paths():
                verts, codes = path.vertices, path.codes
                parts, cur = [], []
                for v, c in zip(verts, codes if codes is not None else [MplPath.LINETO]*len(verts)):
                    if c == MplPath.MOVETO:
                        if cur: parts.append(cur)
                        cur = [tuple(v)]
                    elif c == MplPath.CLOSEPOLY:
                        if cur: cur.append(cur[0]); parts.append(cur)
                        cur = []
                    else:
                        cur.append(tuple(v))
                if cur: parts.append(cur)
                if not parts: continue
                try:
                    p = Polygon(parts[0], parts[1:] if len(parts) > 1 else [])
                    p = p.buffer(0)
                    if p.is_valid and p.area > 0:
                        polys.append(p)
                except Exception:
                    pass

        if not polys:
            return None
        mask = unary_union(polys).buffer(0)
        print(f"[Depth mask] Built >100m ocean mask ({len(polys)} polygons)")
        return mask
    except Exception as e:
        print(f"[Depth mask] Warning — could not build mask: {e}")
        return None


def clip_geojson_to_mask(geojson_path, mask):
    """
    Load a GeoJSON file, clip all features to mask (Shapely polygon),
    and save back to the same path. Features entirely outside mask are dropped;
    features straddling the boundary are trimmed to the mask edge.
    """
    if mask is None or not os.path.exists(geojson_path):
        return
    try:
        from shapely.geometry import shape, mapping
        with open(geojson_path) as f:
            gj = json.load(f)
        clipped = []
        for feat in gj.get("features", []):
            try:
                geom = shape(feat["geometry"])
                inter = geom.intersection(mask)
                if inter.is_empty:
                    continue
                feat = dict(feat)
                feat["geometry"] = mapping(inter)
                clipped.append(feat)
            except Exception:
                clipped.append(feat)  # keep original if shapely fails
        gj["features"] = clipped
        with open(geojson_path, "w") as f:
            json.dump(gj, f)
        print(f"[Depth mask] Clipped {geojson_path.split(os.sep)[-1]}: {len(gj['features'])} features kept")
    except Exception as e:
        print(f"[Depth mask] Clip failed for {geojson_path}: {e}")


if __name__ == "__main__":
    main()
