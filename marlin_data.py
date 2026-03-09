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

# Marlin temperature preferences (°C) — based on Dale et al. (2022) IGFA tagging study
MARLIN_TEMPS = {
    "blue_prime": {"min": 24, "max": 27, "color": "#06b6d4", "species": "blue", "tier": "prime", "min_depth": 100},
    "blue_good":  {"min": 22, "max": 30, "color": "#22d3ee", "species": "blue", "tier": "good", "min_depth": 100},
    "striped":    {"min": 21, "max": 24, "color": "#6366f1", "species": "striped", "tier": "prime", "min_depth": 100},
}

# SST gradient threshold for front detection (°C per ~10km)
# Higher value = fewer but more significant fronts (less noise)
SST_GRADIENT_THRESHOLD = 0.5

# Chlorophyll gradient threshold for edge detection (log scale)
CHL_GRADIENT_THRESHOLD = 0.4

OUTPUT_DIR = "data"


def _kelvin_to_celsius(data):
    """Convert SST from Kelvin to Celsius if needed (auto-detects)."""
    mean_val = float(np.nanmean(data))
    if mean_val > 100:  # Kelvin
        return data - 273.15
    return data


# ---------------------------------------------------------------------------
# 1. Fetch ocean data from Copernicus Marine
# ---------------------------------------------------------------------------
def fetch_copernicus_sst(date_str, bbox):
    """Download SST data — satellite observation L4 (same source as GIBS MUR tiles)."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")

    print(f"[SST] Fetching observation L4 for {date_str}...")
    copernicusmarine.subset(
        dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        variables=["analysed_sst"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[SST] Saved to {output_file}")
    return output_file


def fetch_copernicus_currents(date_str, bbox):
    """Download ocean current data — model first (0.083deg finer grid for canyon-scale
    features), observation fallback (0.25° coarser but satellite-derived)."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "currents_raw.nc")

    # Prefer model product: 0.083deg (~9km) resolves Perth Canyon currents
    # much better than the 0.25° (~28km) observation product
    try:
        print(f"[Currents] Fetching model 0.083deg for {date_str}...")
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
        print(f"[Currents] Saved model 0.083deg to {output_file}")
        return output_file
    except Exception as e:
        print(f"[Currents] Model unavailable ({str(e)[:60]}), falling back to observation...")

    copernicusmarine.subset(
        dataset_id="cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=0,
        maximum_depth=15,
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[Currents] Saved observation fallback to {output_file}")
    return output_file


def fetch_copernicus_chlorophyll(date_str, bbox):
    """Download chlorophyll-a — observation first, model fallback."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "chl_raw.nc")

    # Try satellite observation L4 (GlobColour gapfree), fall back to model
    try:
        print(f"[Chlorophyll] Fetching observation L4 for {date_str}...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
            variables=["CHL"],
            minimum_longitude=bbox["lon_min"],
            maximum_longitude=bbox["lon_max"],
            minimum_latitude=bbox["lat_min"],
            maximum_latitude=bbox["lat_max"],
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T23:59:59",
            output_filename=output_file,
            output_directory=".",
            overwrite=True,
        )
        print(f"[Chlorophyll] Saved observation to {output_file}")
        return output_file
    except Exception as e:
        print(f"[Chlorophyll] Observation unavailable ({str(e)[:60]}), falling back to model...")

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
    print(f"[Chlorophyll] Saved model fallback to {output_file}")
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


def fetch_copernicus_mld(date_str, bbox):
    """Download Mixed Layer Depth from CMEMS global physics model.
    MLD < 50m = shallow mixed layer = marlin compressed at surface = good fishing."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "mld_raw.nc")

    print(f"[MLD] Fetching for {date_str}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["mlotst"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[MLD] Saved to {output_file}")
    return output_file


def fetch_copernicus_oxygen(date_str, bbox):
    """Download dissolved oxygen at 100m depth from CMEMS biogeochemistry model.
    O2 < 150 mmol/m3 at 100m = hypoxic, limits marlin vertical habitat."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "oxygen_raw.nc")

    print(f"[Oxygen] Fetching for {date_str}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
        variables=["o2"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=95,
        maximum_depth=105,
        output_filename=output_file,
        output_directory=".",
        overwrite=True,
    )
    print(f"[Oxygen] Saved to {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# 2. SST Front Detection
# ---------------------------------------------------------------------------
def detect_sst_fronts(sst_file, threshold=SST_GRADIENT_THRESHOLD, deep_mask=None, tif_path=None):
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
    data = _kelvin_to_celsius(sst.values.copy())

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
    for key, temps in MARLIN_TEMPS.items():
        species = temps.get("species", key)
        tier = temps.get("tier", "prime")
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
                    "tier": tier,
                    "color": temps["color"],
                }
            )
            isotherm_features.extend(iso)

    all_features = features + isotherm_features

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "sst_fronts.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[SST Fronts] {len(features)} front lines, {len(isotherm_features)} isotherms ->{output_path}")

    # --- Generate filled marlin zone polygons ---
    zone_features = _generate_marlin_zones(iso_data, mask, lons, lats, deep_mask=deep_mask, tif_path=tif_path)
    zone_geojson = {"type": "FeatureCollection", "features": zone_features}
    zone_path = os.path.join(OUTPUT_DIR, "marlin_zones.geojson")
    with open(zone_path, "w") as f:
        json.dump(zone_geojson, f)
    print(f"[Marlin Zones] {len(zone_features)} zone polygons ->{zone_path}")

    return output_path


def _generate_marlin_zones(sst_data, land_mask, lons, lats, deep_mask=None, tif_path=None):
    """Generate filled polygons for marlin temperature zones using contourf.
    Uses per-species min_depth from MARLIN_TEMPS to build appropriate depth masks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fill land with NaN so contourf skips it
    data = sst_data.copy()
    data[land_mask] = np.nan

    # Build per-depth-threshold masks for species-specific clipping
    depth_masks = {}
    can_clip = False
    if tif_path and os.path.exists(tif_path):
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon, mapping
            can_clip = True
            # Collect unique depth thresholds needed
            for temps in MARLIN_TEMPS.values():
                d = temps.get("min_depth", 100)
                if d not in depth_masks:
                    mask_poly = build_deep_water_mask(tif_path, depth_threshold=-d)
                    if mask_poly is not None:
                        depth_masks[d] = mask_poly
                        print(f"[Marlin Zones] Built >{d}m depth mask for clipping")
        except ImportError:
            print("[Marlin Zones] Warning: shapely not available, skipping depth clipping")
    elif deep_mask is not None:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon, mapping
            can_clip = True
            depth_masks[100] = deep_mask
        except ImportError:
            pass

    features = []
    for key, temps in MARLIN_TEMPS.items():
        species = temps.get("species", key)
        tier = temps.get("tier", "prime")
        tmin, tmax = temps["min"], temps["max"]
        min_depth = temps.get("min_depth", 100)

        fig, ax = plt.subplots()
        cf = ax.contourf(lons, lats, data, levels=[tmin, tmax], extend="neither")
        plt.close(fig)

        # Get the appropriate depth mask for this species
        clip_mask = depth_masks.get(min_depth) if can_clip else None

        for collection_paths in cf.get_paths() if hasattr(cf, 'get_paths') else [p for col in cf.collections for p in col.get_paths()]:
            paths = [collection_paths] if not isinstance(collection_paths, list) else collection_paths
            for path in paths:
                verts = path.vertices
                codes = path.codes
                if len(verts) < 4:
                    continue
                # Split path into exterior and holes using MOVETO codes
                rings = []
                current = []
                for i, (v, c) in enumerate(zip(verts, codes)):
                    if c == 1 and current:  # MOVETO = new ring
                        if len(current) >= 4:
                            rings.append(current)
                        current = []
                    current.append([round(float(v[0]), 4), round(float(v[1]), 4)])
                if len(current) >= 4:
                    rings.append(current)
                if not rings:
                    continue

                # SST suitability: prime tier ≈ 70%, good tier ≈ 40%
                intensity = 0.70 if tier == "prime" else 0.40
                props = {
                    "species": species,
                    "tier": tier,
                    "temp_min": tmin,
                    "temp_max": tmax,
                    "color": temps["color"],
                    "label": f"{species}_{tier}",
                    "min_depth": min_depth,
                    "intensity": intensity,
                }

                # Clip to deep water if mask available
                if clip_mask is not None:
                    try:
                        exterior = [(x, y) for x, y in rings[0]]
                        holes = [[(x, y) for x, y in r] for r in rings[1:]] if len(rings) > 1 else []
                        poly = ShapelyPolygon(exterior, holes).buffer(0)
                        clipped = poly.intersection(clip_mask)
                        if clipped.is_empty:
                            continue
                        geom = mapping(clipped)
                        if geom["type"] == "Polygon":
                            features.append({"type": "Feature", "geometry": geom, "properties": props})
                        elif geom["type"] == "MultiPolygon":
                            for coords in geom["coordinates"]:
                                features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": coords}, "properties": props})
                        continue
                    except Exception:
                        pass  # Fall through to unclipped

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": props,
                })
    return features


# ---------------------------------------------------------------------------
# 2b. Blue Marlin Habitat Suitability Heatmap
# ---------------------------------------------------------------------------
# Composite score 0–1 per grid cell based on all available ocean variables.
# Weights reflect relative importance for blue marlin habitat selection.
BLUE_MARLIN_WEIGHTS = {
    # Optimized by Optuna (100 trials, 71 historical catches, 53 dates).
    # Result: mean 89%, median 92%, 97% >= 70%, min floor 68%.
    "sst":          0.29,   # SST — primary habitat driver
    "sst_front":    0.16,   # SST gradient — prey aggregation at fronts
    "sst_intrusion":0.08,   # Cross-shelf SST gradient — Leeuwin Current
    "chl":          0.06,   # Chlorophyll — bait productivity indicator
    "ssh":          0.10,   # Sea level anomaly — warm water mass + eddies
    "current":      0.10,   # Current favorability — warm water advection
    "convergence":  0.04,   # Current convergence — bait aggregation
    "mld":          0.12,   # Mixed layer depth — shallow = catchable
    "o2":           0.02,   # Dissolved oxygen at 100m (rarely limiting)
    "clarity":      0.02,   # Water clarity (rarely limiting offshore)
    # Static factors applied as MULTIPLIERS, not additive:
    # depth:       0->1 gate (zero if <100m)
    # shelf_break: 1.0->1.6 boost (canyon walls get up to +60%)
}

# Intensity bands for contourf polygon export
HOTSPOT_BANDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


def generate_blue_marlin_hotspots(bbox, tif_path=None):
    """
    Build a composite habitat suitability grid for blue marlin by scoring
    each pixel across all available ocean variables, then export as filled
    GeoJSON polygons with intensity bands.
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter, sobel
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[Hotspots] Building blue marlin habitat suitability...")

    # --- Load SST as master grid ---
    sst_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")
    if not os.path.exists(sst_file):
        print("[Hotspots] No SST data — skipping")
        return None
    ds = xr.open_dataset(sst_file)
    for var in ["thetao", "analysed_sst", "sst"]:
        if var in ds:
            sst_da = ds[var].squeeze()
            break
    else:
        print("[Hotspots] No SST variable found")
        return None

    lons = sst_da.longitude.values if "longitude" in sst_da.dims else sst_da.lon.values
    lats = sst_da.latitude.values if "latitude" in sst_da.dims else sst_da.lat.values
    sst = _kelvin_to_celsius(sst_da.values.copy().astype(float))
    land = np.isnan(sst)

    ny, nx = sst.shape
    score = np.zeros((ny, nx), dtype=float)
    weight_sum = np.zeros((ny, nx), dtype=float)
    sub_scores = {}  # store individual score arrays for per-polygon breakdown

    def _interp_to_grid(data, src_lons, src_lats):
        """Interpolate a 2D array onto the master SST grid."""
        if data.shape == (ny, nx):
            return data
        interp = RegularGridInterpolator(
            (src_lats, src_lons), data,
            method="linear", bounds_error=False, fill_value=np.nan
        )
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        return interp((lat_grid, lon_grid))

    def _maxpool_to_grid(data_hr, hr_lons, hr_lats):
        """Max-pool a high-res array to the master grid.
        For each coarse cell, take the max value from all high-res pixels
        that fall within it. Prevents shelf-edge dilution from linear interp."""
        result = np.zeros((ny, nx))
        dlat = abs(lats[1] - lats[0]) / 2 if ny > 1 else 0.04
        dlon = abs(lons[1] - lons[0]) / 2 if nx > 1 else 0.04
        for yi in range(ny):
            for xi in range(nx):
                b_row = (hr_lats >= lats[yi] - dlat) & (hr_lats <= lats[yi] + dlat)
                b_col = (hr_lons >= lons[xi] - dlon) & (hr_lons <= lons[xi] + dlon)
                if np.any(b_row) and np.any(b_col):
                    result[yi, xi] = np.max(data_hr[np.ix_(b_row, b_col)])
                else:
                    byi = np.argmin(np.abs(hr_lats - lats[yi]))
                    bxi = np.argmin(np.abs(hr_lons - lons[xi]))
                    result[yi, xi] = data_hr[byi, bxi]
        return result

    def _add_score(name, values, mask=None):
        """Add a weighted sub-score. Values should be 0–1."""
        w = BLUE_MARLIN_WEIGHTS.get(name, 0)
        if w == 0:
            return
        v = np.clip(values, 0, 1)
        valid = ~np.isnan(v) & ~land
        if mask is not None:
            valid &= ~mask
        score[valid] += w * v[valid]
        weight_sum[valid] += w
        sub_scores[name] = v.copy()

    # 1. SST score — Gaussian centered at optimal temp
    #    Validated: 81% of 71 catches at 22-24°C, median 22.9°C
    sst_filled = sst.copy()
    sst_filled[land] = np.nanmean(sst)
    sst_smooth = gaussian_filter(sst_filled, sigma=0.5)
    sst_smooth[land] = np.nan
    optimal_temp = getattr(sys.modules[__name__], '_opt_sst_optimal', 24.0)
    sst_sigma = getattr(sys.modules[__name__], '_opt_sst_sigma', 3.0)
    sst_score = np.exp(-0.5 * ((sst_smooth - optimal_temp) / sst_sigma) ** 2)
    _add_score("sst", sst_score)

    # 2. SST front score — Sobel gradient magnitude, modulated by SST suitability
    #    A front at 20°C is useless for marlin — only score fronts in warm water
    sst_for_grad = sst_filled.copy()
    sst_grad = gaussian_filter(sst_for_grad, sigma=1.5)
    gx = sobel(sst_grad, axis=1)
    gy = sobel(sst_grad, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    from scipy.ndimage import binary_dilation
    coast_buf = binary_dilation(land, iterations=2)
    grad_mag[coast_buf] = 0
    # Normalize by 90th percentile (robust to outlier pixels, more
    # temporally consistent than max-based normalization)
    ocean_grad = grad_mag[~coast_buf & ~land]
    g90 = np.nanpercentile(ocean_grad, 90) if len(ocean_grad) > 0 else 0
    if g90 > 0:
        front_score = np.clip(grad_mag / g90, 0, 1)
    else:
        front_score = np.zeros_like(grad_mag)
    # Modulate: fronts only count where SST is suitable for marlin
    front_score = front_score * sst_score
    # Floor: warm water (SST score > 0.6) gets minimum front score
    # Validated: catches in warm water without fronts still productive (54%)
    _front_floor = getattr(sys.modules[__name__], '_opt_front_floor', 0.15)
    warm_mask = sst_score > 0.6
    front_score = np.where(warm_mask, np.maximum(front_score, _front_floor), front_score)
    _add_score("sst_front", front_score)

    # 2b. Cross-shelf SST gradient — Leeuwin Current warm water intrusion
    #     Positive = warmer inshore = active warm current. Validated: partial r=0.42
    #     independent of SST score. Warm inshore catches score 88% vs 80%.
    cross_grad = np.full((ny, nx), np.nan)
    for yi in range(ny):
        transect = sst_smooth[yi, :]
        valid_mask = ~np.isnan(transect) & ~land[yi, :]
        if np.sum(valid_mask) < 5:
            continue
        for xi in range(nx):
            if land[yi, xi]:
                continue
            # Compare SST 2-3 cells east (inshore) vs 2-5 cells west (offshore)
            east_end = min(nx, xi + 4)
            west_start = max(0, xi - 5)
            nearshore = transect[xi:east_end]
            offshore = transect[west_start:xi]
            ns_valid = nearshore[~np.isnan(nearshore)]
            os_valid = offshore[~np.isnan(offshore)]
            if len(ns_valid) > 0 and len(os_valid) > 0:
                cross_grad[yi, xi] = np.mean(ns_valid) - np.mean(os_valid)
    # Score: positive gradient (warm inshore) = good
    _intrusion_thresh = getattr(sys.modules[__name__], '_opt_intrusion_threshold', 0.17)
    _intrusion_baseline = getattr(sys.modules[__name__], '_opt_intrusion_baseline', 0.18)
    intrusion_score = np.clip(cross_grad / _intrusion_thresh, 0, 1)
    # Warm-water baseline: if SST is suitable, give minimum score even without
    # a clear cross-shelf gradient (LC may be present without E-W gradient)
    warm_baseline = sst_score > 0.5
    intrusion_score = np.where(warm_baseline, np.maximum(intrusion_score, _intrusion_baseline), intrusion_score)
    intrusion_score[land] = np.nan
    _add_score("sst_intrusion", intrusion_score)

    # 3. Chlorophyll score — peaks at 0.15–0.30 mg/m³ (shelf edge bait zone)
    chl_file = os.path.join(OUTPUT_DIR, "chl_raw.nc")
    if os.path.exists(chl_file):
        try:
            cds = xr.open_dataset(chl_file)
            for cv in ["chl", "CHL", "chlor_a"]:
                if cv in cds:
                    chl_da = cds[cv].squeeze()
                    break
            chl_lons = chl_da.longitude.values if "longitude" in chl_da.dims else chl_da.lon.values
            chl_lats = chl_da.latitude.values if "latitude" in chl_da.dims else chl_da.lat.values
            chl_data = _interp_to_grid(chl_da.values.astype(float), chl_lons, chl_lats)
            # Peak score at optimal CHL, Gaussian falloff in log space
            _chl_opt = getattr(sys.modules[__name__], '_opt_chl_optimal', 0.18)
            _chl_sig = getattr(sys.modules[__name__], '_opt_chl_sigma', 0.58)
            chl_log = np.log10(np.clip(chl_data, 0.01, 10))
            optimal_chl = np.log10(_chl_opt)
            chl_score = np.exp(-0.5 * ((chl_log - optimal_chl) / _chl_sig) ** 2)
            _add_score("chl", chl_score)
        except Exception as e:
            print(f"[Hotspots] Chl scoring failed: {e}")

    # 4. Depth score — peaks at 100–2000m, zero below 100m
    # 4b. Shelf break score — steep bathymetric gradient = canyon walls, upwelling
    if tif_path and os.path.exists(tif_path):
        try:
            import rasterio
            with rasterio.open(tif_path) as src:
                bathy = src.read(1).astype(float)
                bt = src.transform
                bw, bh = bathy.shape[1], bathy.shape[0]
                b_lons = np.array([bt.c + (j + 0.5) * bt.a for j in range(bw)])
                b_lats = np.array([bt.f + (i + 0.5) * bt.e for i in range(bh)])
                nd = src.nodata
                if nd is not None:
                    bathy[bathy == nd] = np.nan

            # Shelf break: Sobel gradient on raw bathy (high res)
            bathy_filled = bathy.copy()
            bathy_filled[np.isnan(bathy_filled)] = 0
            dgx = sobel(bathy_filled, axis=1)
            dgy = sobel(bathy_filled, axis=0)
            depth_gradient = np.sqrt(dgx**2 + dgy**2)
            depth_gradient[np.isnan(bathy)] = 0
            # Interpolate gradient to master grid (linear interp is fine for
            # shelf break — we want the average steepness, not the max)
            shelf_break = _interp_to_grid(depth_gradient, b_lons, b_lats)
            shelf_score = np.clip(shelf_break / 100, 0, 1)
            shelf_score[land] = np.nan
            # Store for hover breakdown but don't add to weighted sum
            sub_scores["shelf_break"] = shelf_score.copy()
            sb_pct = np.sum(shelf_score[~np.isnan(shelf_score)] > 0.5) / np.sum(~np.isnan(shelf_score)) * 100
            print(f"[Hotspots] Shelf break: {sb_pct:.0f}% of cells >50% (applied as x1.0-1.5 multiplier)")

            # Compute depth score at native bathy resolution FIRST, then
            # max-pool to coarse grid. This prevents shelf-edge catches from
            # being scored as shallow due to linear interpolation averaging
            # deep water with nearby land/shallow cells.
            abs_depth_hr = np.where(np.isnan(bathy), 0, -bathy)
            depth_score_hr = np.where(abs_depth_hr < 50, 0,
                             np.where(abs_depth_hr < 80, (abs_depth_hr - 50) / 30,
                             np.where(abs_depth_hr < 800, 1.0,
                             np.where(abs_depth_hr < 2000, 0.85 + 0.15 * (1.0 - (abs_depth_hr - 800) / 1200),
                             0.7))))
            depth_score_hr[np.isnan(bathy)] = 0
            depth_score = _maxpool_to_grid(depth_score_hr, b_lons, b_lats)
            depth_score[land] = np.nan
            # Store for hover but don't add to weighted sum
            sub_scores["depth"] = np.clip(depth_score, 0, 1)
        except Exception as e:
            print(f"[Hotspots] Depth/shelf scoring failed: {e}")

    # 5. SSH score — blended absolute + relative SLA
    #    Absolute: high SLA (>0.10m) = warm water mass present over canyon (2025: 0.20m vs 2026: 0.06m)
    #    Relative: local highs above background = eddy edges where marlin hunt
    ssh_file = os.path.join(OUTPUT_DIR, "ssh_raw.nc")
    if os.path.exists(ssh_file):
        try:
            sds = xr.open_dataset(ssh_file)
            sv = "sla" if "sla" in sds else "adt"
            sla_da = sds[sv].squeeze()
            s_lons = sla_da.longitude.values if "longitude" in sla_da.dims else sla_da.lon.values
            s_lats = sla_da.latitude.values if "latitude" in sla_da.dims else sla_da.lat.values
            sla_data = _interp_to_grid(sla_da.values.astype(float), s_lons, s_lats)
            # Absolute SLA: warm water mass indicator
            # 0 at SLA<=0, 1.0 at SLA>=0.15m (strong warm anomaly)
            abs_score = np.clip(sla_data / 0.15, 0, 1)
            # Relative SLA: eddy structure (local highs above background)
            sla_filled = sla_data.copy()
            sla_filled[np.isnan(sla_filled)] = np.nanmean(sla_data)
            sla_bg = gaussian_filter(sla_filled, sigma=4)
            sla_relative = sla_data - sla_bg
            rel_score = np.clip(sla_relative / 0.04, 0, 1)
            # Blend: 50% absolute (warm water presence) + 50% relative (eddy edges)
            ssh_score = 0.5 * abs_score + 0.5 * rel_score
            ssh_score[land] = np.nan
            _add_score("ssh", ssh_score)
            mean_abs = np.nanmean(sla_data[~land])
            pct_high = np.sum(ssh_score[~np.isnan(ssh_score)] > 0.5) / np.sum(~np.isnan(ssh_score)) * 100
            print(f"[Hotspots] SSH scoring: {pct_high:.0f}% >50%, mean abs SLA={mean_abs:.3f}m")
        except Exception as e:
            print(f"[Hotspots] SSH scoring failed: {e}")

    # 6. MLD score — shallower = better (marlin compressed at surface)
    mld_file = os.path.join(OUTPUT_DIR, "mld_raw.nc")
    if os.path.exists(mld_file):
        try:
            mds = xr.open_dataset(mld_file)
            for mv in ["mlotst", "mld", "MLD"]:
                if mv in mds:
                    mld_da = mds[mv].squeeze()
                    break
            m_lons = mld_da.longitude.values if "longitude" in mld_da.dims else mld_da.lon.values
            m_lats = mld_da.latitude.values if "latitude" in mld_da.dims else mld_da.lat.values
            mld_data = _interp_to_grid(mld_da.values.astype(float), m_lons, m_lats)
            # Score: 1.0 at MLD<20m, 0.5 at 50m, 0 at 100m
            mld_score = np.clip(1.0 - (mld_data - 20) / 80, 0, 1)
            _add_score("mld", mld_score)
        except Exception as e:
            print(f"[Hotspots] MLD scoring failed: {e}")

    # 7. Oxygen score — O2 at 100m depth
    o2_file = os.path.join(OUTPUT_DIR, "oxygen_raw.nc")
    if os.path.exists(o2_file):
        try:
            ods = xr.open_dataset(o2_file)
            for ov in ["o2", "O2", "doxy"]:
                if ov in ods:
                    o2_da = ods[ov].squeeze()
                    break
            o_lons = o2_da.longitude.values if "longitude" in o2_da.dims else o2_da.lon.values
            o_lats = o2_da.latitude.values if "latitude" in o2_da.dims else o2_da.lat.values
            o2_data = _interp_to_grid(o2_da.values.astype(float), o_lons, o_lats)
            # Score: 0 at <100 mmol/m³, 0.5 at 150, 1.0 at >200
            o2_score = np.clip((o2_data - 100) / 100, 0, 1)
            _add_score("o2", o2_score)
        except Exception as e:
            print(f"[Hotspots] O2 scoring failed: {e}")

    # 8. Water clarity — KD490
    kd_file = os.path.join(OUTPUT_DIR, "kd490_raw.nc")
    if os.path.exists(kd_file):
        try:
            kds = xr.open_dataset(kd_file)
            kd_da = kds["KD490"].squeeze()
            k_lons = kd_da.longitude.values if "longitude" in kd_da.dims else kd_da.lon.values
            k_lats = kd_da.latitude.values if "latitude" in kd_da.dims else kd_da.lat.values
            kd_data = _interp_to_grid(kd_da.values.astype(float), k_lons, k_lats)
            # Score: 1.0 at KD490<0.04, 0 at KD490>0.15
            clarity_score = np.clip(1.0 - (kd_data - 0.04) / 0.11, 0, 1)
            _add_score("clarity", clarity_score)
        except Exception as e:
            print(f"[Hotspots] Clarity scoring failed: {e}")

    # 9. Current favorability — warm water advection into Perth Canyon
    #    Score = direction × speed × upstream_SST_suitability
    #    Eastward (onshore) flow of warm Leeuwin Current water into the canyon
    #    is what aggregates bait and marlin. Cold water pushed onshore is useless.
    cur_file = os.path.join(OUTPUT_DIR, "currents_raw.nc")
    if os.path.exists(cur_file):
        try:
            cds = xr.open_dataset(cur_file)
            # Select surface layer if multiple depths exist
            uo_raw = cds["uo"]
            vo_raw = cds["vo"]
            if "depth" in uo_raw.dims and uo_raw.sizes["depth"] > 1:
                uo_raw = uo_raw.isel(depth=0)
                vo_raw = vo_raw.isel(depth=0)
            uo_da = uo_raw.squeeze()
            vo_da = vo_raw.squeeze()
            c_lons = uo_da.longitude.values if "longitude" in uo_da.dims else uo_da.lon.values
            c_lats = uo_da.latitude.values if "latitude" in uo_da.dims else uo_da.lat.values
            uo_data = _interp_to_grid(uo_da.values.astype(float), c_lons, c_lats)
            vo_data = _interp_to_grid(vo_da.values.astype(float), c_lons, c_lats)
            cur_speed = np.sqrt(uo_data**2 + vo_data**2)

            # Score 1: eastward component — positive uo = flow toward coast/canyon
            # 0 at uo<=0, 1.0 at uo>=0.15 m/s (strong onshore)
            east_score = np.clip(uo_data / 0.15, 0, 1)

            # Score 2: current speed — stronger = more warm water transport
            # 0 at <0.03 m/s, 1.0 at >=0.20 m/s
            speed_score = np.clip((cur_speed - 0.03) / 0.17, 0, 1)

            # Score 3: upstream water temperature — sample SST ~1-2 pixels
            # UPSTREAM of each cell (opposite to current direction).
            # This tells us what temperature the current is bringing IN.
            # Use sst_smooth which is already computed above.
            upstream_sst = np.full_like(sst_smooth, np.nan)
            dlat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.05
            dlon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.05
            for yi in range(ny):
                for xi in range(nx):
                    if land[yi, xi] or np.isnan(uo_data[yi, xi]):
                        continue
                    # Look ~2 grid cells upstream (opposite flow direction)
                    u_val, v_val = float(uo_data[yi, xi]), float(vo_data[yi, xi])
                    spd = np.sqrt(u_val**2 + v_val**2)
                    if spd < 0.01:
                        upstream_sst[yi, xi] = sst_smooth[yi, xi]
                        continue
                    # Upstream offset in grid cells (2 cells in upstream direction)
                    src_xi = int(round(xi - 2 * u_val / spd))
                    src_yi = int(round(yi - 2 * v_val / spd))
                    src_xi = max(0, min(nx - 1, src_xi))
                    src_yi = max(0, min(ny - 1, src_yi))
                    t = sst_smooth[src_yi, src_xi]
                    upstream_sst[yi, xi] = t if not np.isnan(t) else sst_smooth[yi, xi]

            # Score upstream SST using same Gaussian as main SST score
            upstream_temp_score = np.exp(-0.5 * ((upstream_sst - optimal_temp) / sst_sigma) ** 2)
            upstream_temp_score[land] = np.nan

            # Combined: speed × upstream warmth, with eastward bonus
            # Validated: catches occur in all current directions, not just eastward.
            # Base score = speed × upstream_temp; eastward adds 30% bonus
            _east_bonus_factor = getattr(sys.modules[__name__], '_opt_east_bonus', 0.29)
            east_bonus = 1.0 + _east_bonus_factor * east_score
            current_score = speed_score * upstream_temp_score * east_bonus
            current_score = np.clip(current_score, 0, 1)
            current_score[land] = np.nan
            _add_score("current", current_score)

            ocean = ~land & ~np.isnan(current_score)
            fav_pct = np.sum(current_score[ocean] > 0.3) / np.sum(ocean) * 100
            mean_up_sst = np.nanmean(upstream_sst[ocean])
            print(f"[Hotspots] Current scoring: {fav_pct:.0f}% favorable, upstream SST mean {mean_up_sst:.1f}C")

            # 10. Current convergence — negative divergence = converging flow
            #     Convergence concentrates bait (pilchards, squid) at canyon head,
            #     which is the primary mechanism that aggregates marlin.
            #     Use the current grid (already on master grid via interpolation).
            dudx = np.gradient(uo_data, axis=1)
            dvdy = np.gradient(vo_data, axis=0)
            divergence = dudx + dvdy
            # Convergence = negative divergence. Stronger convergence = higher score.
            # Score: 0 at div>=0 (diverging), 1.0 at div<=-0.005 (strong convergence)
            conv_score = np.clip(-divergence / 0.005, 0, 1)
            # Smooth to reduce noise from grid-scale artefacts
            conv_filled = conv_score.copy()
            conv_filled[np.isnan(conv_filled)] = 0
            conv_score = gaussian_filter(conv_filled, sigma=1.0)
            conv_score[land] = np.nan

            # Bait trap synergy: convergence is more effective with strong current
            # (currents push bait into convergence zones = active aggregation)
            # Validated: both high = 69%, neither = 53% at catch locations
            _synergy = getattr(sys.modules[__name__], '_opt_synergy_factor', 0.45)
            synergy = 1.0 + _synergy * np.clip(current_score, 0, 1)
            conv_score_synergy = np.clip(conv_score * synergy, 0, 1)
            conv_score_synergy[land] = np.nan
            _add_score("convergence", conv_score_synergy)
            conv_pct = np.sum(conv_score_synergy[~np.isnan(conv_score_synergy) & ~land] > 0.3) / np.sum(~np.isnan(conv_score_synergy) & ~land) * 100
            print(f"[Hotspots] Convergence scoring: {conv_pct:.0f}% of cells have bait-concentrating flow")
        except Exception as e:
            print(f"[Hotspots] Current scoring failed: {e}")

    # --- Normalize by actual weights used (handles missing data gracefully) ---
    valid = weight_sum > 0
    final = np.full((ny, nx), np.nan)
    final[valid] = score[valid] / weight_sum[valid]
    final[land] = np.nan

    # Apply static multipliers: depth gates, shelf break boosts
    if "depth" in sub_scores:
        depth_mult = sub_scores["depth"]
        dmask = ~np.isnan(depth_mult) & valid
        final[dmask] *= depth_mult[dmask]  # zero out shallow water
    if "shelf_break" in sub_scores:
        _shelf_boost = getattr(sys.modules[__name__], '_opt_shelf_boost', 0.60)
        shelf_mult = 1.0 + _shelf_boost * sub_scores["shelf_break"]
        smask = ~np.isnan(shelf_mult) & valid
        final[smask] *= shelf_mult[smask]
        final = np.clip(final, 0, 1)  # cap at 1.0

    # Light spatial smoothing to reduce pixelation
    final_filled = final.copy()
    final_filled[np.isnan(final_filled)] = 0
    final_smooth = gaussian_filter(final_filled, sigma=0.8)
    final_smooth[land | ~valid] = np.nan

    fmin = float(np.nanmin(final_smooth[~land & valid]))
    fmax = float(np.nanmax(final_smooth[~land & valid]))
    fmean = float(np.nanmean(final_smooth[~land & valid]))
    print(f"[Hotspots] Score range: {fmin:.3f} - {fmax:.3f} (mean {fmean:.3f})")

    # --- Build depth mask for clipping to >100m ---
    clip_mask = None
    if tif_path and os.path.exists(tif_path):
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, mapping
            clip_mask = build_deep_water_mask(tif_path, depth_threshold=-50)
        except ImportError:
            pass

    # --- Helper to sample mean sub-scores within a polygon's bounding box ---
    def _sample_scores(coords_list):
        """Return (actual_intensity, sub_scores_dict) for the polygon area."""
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]
        lon_min, lon_max = min(xs), max(xs)
        lat_min, lat_max = min(ys), max(ys)
        col_mask = (lons >= lon_min) & (lons <= lon_max)
        row_mask = (lats >= lat_min) & (lats <= lat_max)
        # Actual composite score from the smoothed grid
        region_composite = final_smooth[np.ix_(row_mask, col_mask)]
        valid_composite = region_composite[~np.isnan(region_composite)]
        actual_intensity = round(float(np.mean(valid_composite)), 2) if len(valid_composite) > 0 else 0.0
        result = {}
        for name, arr in sub_scores.items():
            region = arr[np.ix_(row_mask, col_mask)]
            valid = region[~np.isnan(region)]
            if len(valid) > 0:
                w = BLUE_MARLIN_WEIGHTS.get(name, 0)
                mean_score = round(float(np.mean(valid)), 2)
                if name == "depth":
                    # Depth is a gate multiplier: show as ×N
                    result[name] = {"score": mean_score, "weight": -1}
                elif name == "shelf_break":
                    # Shelf break is a boost multiplier: show as ×N
                    _sb = getattr(sys.modules[__name__], '_opt_shelf_boost', 0.60)
                    result[name] = {"score": round(1.0 + _sb * mean_score, 2), "weight": -2}
                else:
                    result[name] = {"score": mean_score, "weight": w}
        return actual_intensity, result

    # --- Export as filled contour polygons with intensity bands ---
    # Fill NaN with 0 for contourf
    plot_data = final_smooth.copy()
    plot_data[np.isnan(plot_data)] = 0

    levels = [0] + HOTSPOT_BANDS + [1.0]
    fig, ax = plt.subplots()
    cf = ax.contourf(lons, lats, plot_data, levels=levels, extend="neither")
    plt.close(fig)

    features = []
    for band_idx, seg_list in enumerate(cf.allsegs):
        if band_idx == 0:
            continue  # skip the 0–0.15 band (background noise)
        intensity = round((levels[band_idx] + levels[band_idx + 1]) / 2, 2) if band_idx + 1 < len(levels) else 1.0
        band_label = f"{levels[band_idx]:.0%}–{levels[band_idx+1]:.0%}" if band_idx + 1 < len(levels) else f">{levels[band_idx]:.0%}"

        for seg in seg_list:
            if len(seg) < 4:
                continue
            coords = [[round(float(x), 4), round(float(y), 4)] for x, y in seg]
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            # Sample actual composite score and sub-scores for this polygon
            actual_intensity, breakdown = _sample_scores(coords)

            props = {
                "species": "blue",
                "type": "hotspot",
                "intensity": actual_intensity,
                "band": band_label,
            }
            for name, info in breakdown.items():
                props[f"s_{name}"] = info["score"]
                props[f"w_{name}"] = info["weight"]

            if clip_mask is not None:
                try:
                    poly = ShapelyPolygon([(c[0], c[1]) for c in coords]).buffer(0)
                    clipped = poly.intersection(clip_mask)
                    if clipped.is_empty:
                        continue
                    geom = mapping(clipped)
                    if geom["type"] == "Polygon":
                        features.append({"type": "Feature", "geometry": geom, "properties": props})
                    elif geom["type"] == "MultiPolygon":
                        for mc in geom["coordinates"]:
                            features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": mc}, "properties": props})
                    continue
                except Exception:
                    pass

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": props,
            })

    geojson = {"type": "FeatureCollection", "features": features}
    output_path = os.path.join(OUTPUT_DIR, "blue_marlin_hotspots.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Hotspots] {len(features)} polygons across {len(HOTSPOT_BANDS)} bands ->{output_path}")
    return {
        "path": output_path,
        "grid": final_smooth,
        "lats": lats,
        "lons": lons,
        "sub_scores": sub_scores,
        "weights": {k: v for k, v in BLUE_MARLIN_WEIGHTS.items()},
    }


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

    print(f"[Chlorophyll] {len(all_features)} contours at {[l[0] for l in CHL_LEVELS]} mg/m³ ->{output_path}")
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

    print(f"[Water Clarity] {len(all_features)} contours ->{output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3c. SSH / Eddy Detection (Sea Level Anomaly)
# ---------------------------------------------------------------------------
# SLA contour levels (m): eddy structure boundaries
# Positive SLA = warm-core anticyclonic eddy (clockwise in S. hemisphere)
#   ->traps warm water, marlin congregate on leading edge
# Negative SLA = cold-core cyclonic eddy ->upwelling, bait but cold
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

    print(f"[SSH/Eddies] {len(all_features)} contours ->{output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4a. Mixed Layer Depth Contours
# ---------------------------------------------------------------------------
MLD_LEVELS = [
    (30,  "shallow",  "#22d3ee"),   # cyan — very shallow MLD, strongly stratified
    (50,  "moderate", "#0ea5e9"),   # blue — key threshold, good fishing boundary
    (80,  "deep",     "#6366f1"),   # indigo — deep MLD, less favorable
]


def process_mld(mld_file):
    """Extract MLD contour lines. MLD < 50m = shallow mixed layer = good marlin fishing."""
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[MLD] Processing contours...")
    ds = xr.open_dataset(mld_file)

    mld = None
    for var in ["mlotst", "mld", "MLD", "mlp"]:
        if var in ds:
            mld = ds[var].squeeze()
            break
    if mld is None:
        raise ValueError(f"No MLD variable found. Available: {list(ds.data_vars)}")

    lons = mld.longitude.values if "longitude" in mld.dims else mld.lon.values
    lats = mld.latitude.values if "latitude" in mld.dims else mld.lat.values
    data = mld.values.copy().astype(float)

    mask = np.isnan(data)
    data[mask] = 0.0
    data_smooth = gaussian_filter(data, sigma=0.8)
    data_smooth[mask] = 0.0
    data_max = float(np.nanmax(data_smooth))

    all_features = []
    for level, label, color in MLD_LEVELS:
        if level > data_max:
            continue
        feats = _contours_to_geojson(
            data_smooth, lons, lats, level,
            properties={"type": "mld_contour", "mld": level, "label": label, "color": color}
        )
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 3]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "mld_contours.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"[MLD] {len(all_features)} contours ->{output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4b. Dissolved Oxygen at 100m Contours
# ---------------------------------------------------------------------------
O2_LEVELS = [
    (100, "hypoxic",  "#ef4444"),   # red — severely low O2, avoid
    (150, "low",      "#fb923c"),   # orange — borderline for billfish
    (200, "adequate", "#22c55e"),   # green — sufficient for marlin
]


def process_oxygen(oxygen_file):
    """Extract dissolved O2 contours at 100m depth. O2 < 150 mmol/m3 = hypoxic for marlin."""
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[Oxygen] Processing contours...")
    ds = xr.open_dataset(oxygen_file)

    o2 = None
    for var in ["o2", "O2", "doxy", "dissolved_oxygen"]:
        if var in ds:
            o2 = ds[var].squeeze()
            break
    if o2 is None:
        raise ValueError(f"No oxygen variable found. Available: {list(ds.data_vars)}")

    lons = o2.longitude.values if "longitude" in o2.dims else o2.lon.values
    lats = o2.latitude.values if "latitude" in o2.dims else o2.lat.values
    data = o2.values.copy().astype(float)

    mask = np.isnan(data) | (data <= 0)
    data[mask] = 0.0
    data_smooth = gaussian_filter(data, sigma=0.8)
    data_smooth[mask] = 0.0
    data_max = float(np.nanmax(data_smooth))

    all_features = []
    for level, label, color in O2_LEVELS:
        if level > data_max:
            continue
        feats = _contours_to_geojson(
            data_smooth, lons, lats, level,
            properties={"type": "o2_contour", "o2": level, "label": label, "color": color}
        )
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 3]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "oxygen_contours.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"[Oxygen] {len(all_features)} contours ->{output_path}")
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

    uo_raw = ds["uo"]
    vo_raw = ds["vo"]
    if "depth" in uo_raw.dims and uo_raw.sizes["depth"] > 1:
        uo_raw = uo_raw.isel(depth=0)
        vo_raw = vo_raw.isel(depth=0)
    uo = uo_raw.squeeze()
    vo = vo_raw.squeeze()

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

    print(f"[Currents] {len(features)} vectors ->{output_path}")
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

    print(f"[Bathymetry] {len(all_features)} contour segments ->{output_path}")
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

    print(f"[Bathymetry] Contours ->{output_path}")
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
    print(f"[Bathymetry] Downloaded {len(resp.content)//1024} KB ->{tif_path}")

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
# Fishing grounds — convex hull around Perth Canyon FADs and key marks.
# SST range is computed within this area (not the full bbox) so the report
# reflects conditions where people actually fish.
FISHING_GROUNDS_POLYGON = [
    # Ordered polygon enclosing the main FAD cluster
    (114.98, -31.92),   # Perth Canyon Head (west)
    (115.08, -31.85),   # north of canyon head
    (115.27, -31.87),   # north of FURUNO
    (115.33, -31.95),   # Club Marine (east)
    (115.33, -32.05),   # Club Marine (south-east)
    (115.17, -32.17),   # Fibrelite Boats (south)
    (115.00, -32.10),   # south of Rottnest Trench
    (114.92, -32.01),   # west of Rottnest Trench
    (114.98, -31.92),   # close polygon
]


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
                sst_da = ds[var].squeeze()
                sst = _kelvin_to_celsius(sst_da.values.copy().astype(float))
                lons = sst_da.longitude.values if "longitude" in sst_da.dims else sst_da.lon.values
                lats = sst_da.latitude.values if "latitude" in sst_da.dims else sst_da.lat.values

                # Mask to fishing grounds polygon (ray-casting point-in-polygon)
                poly = FISHING_GROUNDS_POLYGON
                fg_mask = np.zeros_like(sst, dtype=bool)
                for yi in range(len(lats)):
                    for xi in range(len(lons)):
                        px, py = float(lons[xi]), float(lats[yi])
                        inside = False
                        n = len(poly)
                        j = n - 1
                        for i in range(n):
                            pxi, pyi = poly[i]
                            pxj, pyj = poly[j]
                            if ((pyi > py) != (pyj > py)) and (px < (pxj - pxi) * (py - pyi) / (pyj - pyi) + pxi):
                                inside = not inside
                            j = i
                        fg_mask[yi, xi] = inside

                sst_fg = sst[fg_mask & ~np.isnan(sst)]
                sst_all = sst[~np.isnan(sst)]
                # Use fishing grounds SST if we have enough pixels, else fall back to full area
                sst_valid = sst_fg if len(sst_fg) > 3 else sst_all
                label = "fishing grounds" if len(sst_fg) > 3 else "full area"

                report["sst"] = {
                    "min": round(float(np.min(sst_valid)), 1),
                    "max": round(float(np.max(sst_valid)), 1),
                    "mean": round(float(np.mean(sst_valid)), 1),
                    "area": label,
                    "blue_marlin_prime": bool(
                        np.any((sst_valid >= 24) & (sst_valid <= 27))
                    ),
                    "blue_marlin_good": bool(
                        np.any((sst_valid >= 22) & (sst_valid <= 30))
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
    print(f"MARLIN ZONE REPORT - {date_str}")
    print(f"{'='*50}")
    if "sst" in report:
        s = report["sst"]
        print(f"SST Range ({s['area']}): {s['min']}C - {s['max']}C (mean {s['mean']}C)")
        print(f"Blue Marlin Prime (24-27C):   {'YES' if s['blue_marlin_prime'] else 'NO'}")
        print(f"Blue Marlin Good (22-30C):    {'YES' if s['blue_marlin_good'] else 'NO'}")
        print(f"Striped Marlin Zone (21-24C): {'YES' if s['striped_marlin_zone'] else 'NO'}")
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

    print(f"\nMarlin Zone Data Pipeline")
    print(f"   Date:   {date_str}")
    print(f"   Region: {bbox['lon_min']}-{bbox['lon_max']}E, {bbox['lat_min']}-{bbox['lat_max']}S")
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

        try:
            fetch_copernicus_mld(date_str, bbox)
        except Exception as e:
            print(f"[MLD] Fetch error: {e}")

        try:
            fetch_copernicus_oxygen(date_str, bbox)
        except Exception as e:
            print(f"[Oxygen] Fetch error: {e}")

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
            detect_sst_fronts(sst_file, deep_mask=deep_mask, tif_path=tif_path)
        except Exception as e:
            print(f"[SST Fronts] Error: {e}")

    # Generate blue marlin habitat hotspot heatmap
    try:
        generate_blue_marlin_hotspots(bbox, tif_path=tif_path)
    except Exception as e:
        print(f"[Hotspots] Error: {e}")

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

    # Process MLD contours
    mld_file = _nc("mld_raw.nc")
    if mld_file:
        try:
            process_mld(mld_file)
        except Exception as e:
            print(f"[MLD] Processing error: {e}")

    # Process dissolved oxygen contours
    o2_file = _nc("oxygen_raw.nc")
    if o2_file:
        try:
            process_oxygen(o2_file)
        except Exception as e:
            print(f"[Oxygen] Processing error: {e}")

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

    # Save deep water polygon as GeoJSON for the map fill layer
    if deep_mask is not None:
        save_deep_water_mask_geojson(deep_mask, os.path.join(OUTPUT_DIR, "deep_water.geojson"))

    # Generate report
    generate_report(date_str, bbox)

    # Copy GeoJSON files to base data/ dir so latest data is always at data/*.geojson
    import shutil
    geojson_files = [
        "sst_fronts.geojson", "chl_edges.geojson", "currents.geojson",
        "bathymetry_contours.geojson", "water_clarity.geojson", "ssh_eddies.geojson",
        "deep_water.geojson", "mld_contours.geojson", "oxygen_contours.geojson",
        "marlin_zones.geojson", "blue_marlin_hotspots.geojson",
    ]
    if not args.no_update_latest:
        for fname in geojson_files:
            src = os.path.join(OUTPUT_DIR, fname)
            dst = os.path.join(base_output, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"Done! GeoJSON files in {OUTPUT_DIR}/ and {base_output}/")
    else:
        print(f"Done! GeoJSON files in {OUTPUT_DIR}/ (base data/ not updated)")
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
        # matplotlib 3.8+ removed cf.collections; use cf.get_paths() directly
        all_paths = cf.get_paths() if hasattr(cf, 'get_paths') else [p for col in cf.collections for p in col.get_paths()]
        for path in all_paths:
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


def save_deep_water_mask_geojson(mask, output_path):
    """Save the deep water Shapely polygon as a GeoJSON FeatureCollection."""
    if mask is None:
        return
    try:
        from shapely.geometry import mapping
        gj = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(mask),
                "properties": {"label": "deep_water", "depth_threshold": -100}
            }]
        }
        with open(output_path, "w") as f:
            json.dump(gj, f)
        print(f"[Depth mask] Saved deep water polygon ->{output_path}")
    except Exception as e:
        print(f"[Depth mask] Could not save GeoJSON: {e}")


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
