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
    """Download SST data — tries IMOS 0.02deg first (matches GIBS MUR visual),
    then CMEMS L4 0.05deg, then reanalysis model 0.083deg."""
    import copernicusmarine
    import xarray as xr
    from datetime import datetime, timedelta

    output_file = os.path.join(OUTPUT_DIR, "sst_raw.nc")

    # 1. Try IMOS L3S 0.02deg (best match for GIBS MUR 0.01deg visual tiles)
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.strftime("%Y")
    ymd = dt.strftime("%Y%m%d")
    imos_base = "https://thredds.aodn.org.au/thredds/dodsC/IMOS/SRS/SST/ghrsst/L3S-1d/ngt/"
    for suffix in ["night", "day"]:
        for sensor in ["AVHRR_D", "MultiSensor", "AVHRR_N"]:
            try:
                url = (f"{imos_base}{year}/{ymd}092000-ABOM-L3S_GHRSST-SSTskin"
                       f"-{sensor}-1d_{suffix}.nc")
                print(f"[SST] Trying IMOS {sensor} {suffix} 0.02deg for {date_str}...")
                ds = xr.open_dataset(url, engine="netcdf4")
                sub = ds.sel(
                    lat=slice(bbox["lat_max"], bbox["lat_min"]),
                    lon=slice(bbox["lon_min"], bbox["lon_max"]),
                )
                sub.to_netcdf(output_file)
                sub.close()
                ds.close()
                print(f"[SST] IMOS {sensor} saved (0.02deg, {date_str})")
                return output_file
            except Exception:
                continue

    # 2. Try CMEMS L4 satellite observation (0.05deg NRT + reanalysis)
    for ds_id in [
        "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    ]:
        try:
            print(f"[SST] Trying CMEMS {ds_id.split('-')[0]} for {date_str}...")
            copernicusmarine.subset(
                dataset_id=ds_id,
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
            print(f"[SST] CMEMS L4 saved (0.05deg, {date_str})")
            return output_file
        except Exception:
            continue

    # 3. Last resort: physics reanalysis model (0.083deg)
    for ds_id, var in [
        ("cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao"),
        ("cmems_mod_glo_phy_anfc_0.083deg_P1D-m", "thetao"),
    ]:
        try:
            print(f"[SST] Trying model reanalysis for {date_str}...")
            copernicusmarine.subset(
                dataset_id=ds_id, variables=[var],
                minimum_longitude=bbox["lon_min"],
                maximum_longitude=bbox["lon_max"],
                minimum_latitude=bbox["lat_min"],
                maximum_latitude=bbox["lat_max"],
                start_datetime=f"{date_str}T00:00:00",
                end_datetime=f"{date_str}T23:59:59",
                minimum_depth=0, maximum_depth=1,
                output_filename=output_file,
                output_directory=".",
                overwrite=True,
            )
            print(f"[SST] Model reanalysis saved (0.083deg, {date_str})")
            return output_file
        except Exception:
            continue

    print(f"[SST] No SST data available for {date_str}")
    return None


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
    """Download chlorophyll-a — tries NRT L4 4km → MY L4 4km → model 0.25deg."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "chl_raw.nc")

    # 1. Try NRT L4 gapfree 4km (recent ~2 years)
    try:
        print(f"[Chlorophyll] Fetching NRT L4 4km for {date_str}...")
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
        print(f"[Chlorophyll] Saved NRT L4 4km to {output_file}")
        return output_file
    except Exception as e:
        print(f"[Chlorophyll] NRT L4 unavailable ({str(e)[:60]})")

    # 2. Try MY (reanalysis) L4 gapfree 4km (1997-present)
    try:
        print(f"[Chlorophyll] Fetching MY L4 4km for {date_str}...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
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
        print(f"[Chlorophyll] Saved MY L4 4km to {output_file}")
        return output_file
    except Exception as e:
        print(f"[Chlorophyll] MY L4 unavailable ({str(e)[:60]})")

    # 3. Last resort: biogeochemical model (0.25deg / 28km)
    print(f"[Chlorophyll] Falling back to model 0.25deg for {date_str}...")
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
    print(f"[Chlorophyll] Saved model 0.25deg to {output_file}")
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
    Tries NRT first (recent dates), then falls back to reanalysis (historical).
    """
    import copernicusmarine
    from datetime import datetime, timedelta

    output_file = os.path.join(OUTPUT_DIR, "ssh_raw.nc")
    base_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Try NRT product first (recent data, higher res 0.125deg)
    for delta in range(0, 8):
        try_date = (base_dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        print(f"[SSH/Eddies] Fetching SLA NRT for {try_date}...")
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
            print(f"[SSH/Eddies] Saved to {output_file} (NRT, data date: {try_date})")
            return output_file
        except Exception:
            if delta < 7:
                print(f"[SSH/Eddies] {try_date} NRT not available, trying earlier...")

    # Fall back to physics reanalysis model (zos, 0.083deg — covers 1993+)
    for delta in range(0, 5):
        try_date = (base_dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        print(f"[SSH/Eddies] Fetching zos reanalysis for {try_date}...")
        try:
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                variables=["zos"],
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
            print(f"[SSH/Eddies] Saved to {output_file} (reanalysis zos, data date: {try_date})")
            return output_file
        except Exception:
            if delta < 4:
                print(f"[SSH/Eddies] {try_date} reanalysis not available, trying earlier...")
            else:
                print(f"[SSH/Eddies] No SSH data available for {date_str}")
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
    O2 < 150 mmol/m3 at 100m = hypoxic, limits marlin vertical habitat.
    Tries ANFC (recent) then reanalysis (historical)."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "oxygen_raw.nc")

    # Try ANFC first (recent dates)
    for ds_id in [
        "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
        "cmems_mod_glo_bgc_my_0.25deg_P1D-m",
    ]:
        try:
            print(f"[Oxygen] Fetching {ds_id.split('_')[3]} for {date_str}...")
            copernicusmarine.subset(
                dataset_id=ds_id,
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
        except Exception as e:
            print(f"[Oxygen] {ds_id} failed: {str(e)[:60]}")
            continue

    print(f"[Oxygen] No oxygen data available for {date_str}")
    return None


def fetch_copernicus_salinity(date_str, bbox):
    """Download surface salinity from CMEMS physics model.
    The Leeuwin Current carries distinctively low-salinity tropical water (~34.5-35 PSU)
    vs surrounding Indian Ocean (~35.5-36 PSU). In summer, SST fronts can be masked by
    solar heating, but salinity (halocline) remains a reliable current boundary marker."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "salinity_raw.nc")

    for ds_id in [
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    ]:
        try:
            print(f"[Salinity] Fetching surface salinity for {date_str}...")
            copernicusmarine.subset(
                dataset_id=ds_id,
                variables=["so"],
                minimum_longitude=bbox["lon_min"],
                maximum_longitude=bbox["lon_max"],
                minimum_latitude=bbox["lat_min"],
                maximum_latitude=bbox["lat_max"],
                start_datetime=f"{date_str}T00:00:00",
                end_datetime=f"{date_str}T23:59:59",
                minimum_depth=0, maximum_depth=1,
                output_filename=output_file,
                output_directory=".",
                overwrite=True,
            )
            print(f"[Salinity] Saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"[Salinity] {ds_id} failed: {str(e)[:60]}")
            continue

    print(f"[Salinity] No salinity data available for {date_str}")
    return None


def fetch_copernicus_subsurface_temp(date_str, bbox):
    """Download subsurface temperature at 200-250m from CMEMS physics model.
    Doming isotherms at 250m reveal canyon-driven upwelling structure. The target
    signature is warm surface (LC cap) + cold lifted basement = productive biology."""
    import copernicusmarine

    output_file = os.path.join(OUTPUT_DIR, "subsurface_temp_raw.nc")

    for ds_id in [
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    ]:
        try:
            print(f"[SubsurfaceTemp] Fetching thetao at 200-250m for {date_str}...")
            copernicusmarine.subset(
                dataset_id=ds_id,
                variables=["thetao"],
                minimum_longitude=bbox["lon_min"],
                maximum_longitude=bbox["lon_max"],
                minimum_latitude=bbox["lat_min"],
                maximum_latitude=bbox["lat_max"],
                start_datetime=f"{date_str}T00:00:00",
                end_datetime=f"{date_str}T23:59:59",
                minimum_depth=200, maximum_depth=260,
                output_filename=output_file,
                output_directory=".",
                overwrite=True,
            )
            print(f"[SubsurfaceTemp] Saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"[SubsurfaceTemp] {ds_id} failed: {str(e)[:60]}")
            continue

    print(f"[SubsurfaceTemp] No subsurface temp data available for {date_str}")
    return None


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

    # 23°C isotherm — peak blue marlin catch temperature (median 22.8°C,
    # 41% of catches at 22-23°C).  Not a range boundary but the empirical
    # sweet spot from 46 validated Perth Canyon catches.
    if 23 not in seen_temps and data_min - 0.5 <= 23 <= data_max + 0.5:
        iso = _contours_to_geojson(
            np.where(mask, float(np.nanmean(iso_data[~mask])), iso_data),
            lons, lats, 23,
            properties={
                "type": "isotherm",
                "temperature": 23,
                "species": "blue",
                "tier": "peak",
                "color": "#f59e0b",  # amber — catch hotspot temperature
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
    # v22: Optuna 400-trial unique-only, obj=25.4
    "sst":           0.150,  # SST Gaussian
    "sst_front":     0.007,  # SST front gradient
    "front_corridor":0.064,  # SST front corridors
    "chl":           0.107,  # Chlorophyll
    "chl_curvature": 0.100,  # CHL curvature
    "ssh":           0.100,  # Sea level anomaly
    "current_shear": 0.129,  # Vorticity shear (edge-scored)
    "upwelling_edge":0.043,  # Canyon upwelling boundaries (edge-scored)
    "salinity_front":0.100,  # Halocline gradient
    "okubo_weiss":   0.021,  # Strain vs rotation (edge-scored)
    "shelf_break":   0.007,  # Shelf break proximity
    # Active minor features:
    "sst_intrusion": 0.00,
    "sst_chl_bivariate": 0.021,  # SST x CHL interaction
    "current":       0.000,
    "convergence":   0.00,
    "ftle":          0.093,  # Lagrangian coherent structures
    "vertical_velocity": 0.057,  # Derived vertical velocity
    "mld":           0.00,
    "o2":            0.00,
    "clarity":       0.00,
    "ssta":          0.00,
    "rugosity":      0.00,
    "sst_roc":       0.00,
    "thermocline_lift": 0.00,
    "stratification": 0.00,
    # Static factors applied as MULTIPLIERS, not additive:
    # depth:       0->1 gate (zero if <100m)
    # shelf_break: ALSO multiplicative boost (hybrid: additive + multiplicative)
}

# Intensity bands for contourf polygon export
# More bands at top end (80-100%) for fine spatial detail in the fishable zone
# Visual band breaks concentrated in the upper range where most ocean cells
# cluster during marlin season.  Finer spacing above 0.80 provides spatial
# contrast between feature-adjacent and background cells.
# Visual band breaks — floor at 0.65 so background ocean stays transparent
# and only elevated zones (feature intersections, shelf break) render.
# Finer spacing in 0.80–0.98 for contrast within hot zones.
# Display floor at 0.50 — shows broader context so "best water on the day"
# is visible even when conditions aren't perfect. 5% bands 50-70% for context,
# 2% bands 70-80%, 1% bands above 80% for fine spatial detail.
HOTSPOT_BANDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def compute_ssta(sst_grid, lats, lons, date_str, clim_path="data/sst_climatology.nc"):
    """Compute SST anomaly = observed SST - monthly climatology.
    Returns SSTA array (same shape as sst_grid) in degrees C."""
    import xarray as xr

    if not os.path.exists(clim_path):
        print(f"[SSTA] Climatology not found at {clim_path}")
        return np.full_like(sst_grid, np.nan)

    month = int(date_str[5:7])
    ds = xr.open_dataset(clim_path)
    clim = ds["sst_clim"].sel(month=month).values
    clim_lats = ds["latitude"].values
    clim_lons = ds["longitude"].values
    ds.close()

    # Interpolate climatology onto the observation grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (clim_lats, clim_lons), clim,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    clim_on_grid = interp((lat_grid, lon_grid))

    ssta = sst_grid - clim_on_grid
    return ssta


# SSTA contour levels for overlay generation
SSTA_LEVELS = [
    (-2.0, "very_cold", "#1d4ed8"),  # deep blue — strong negative anomaly
    (-1.0, "cold",      "#60a5fa"),  # light blue — moderate negative
    (-0.5, "cool",      "#bfdbfe"),  # pale blue — slight negative
    ( 0.5, "warm",      "#fdba74"),  # light orange — slight positive
    ( 1.0, "hot",       "#fb923c"),  # orange — moderate positive
    ( 2.0, "very_hot",  "#dc2626"),  # red — strong positive anomaly
]


def generate_ssta_contours(sst_file, date_str, clim_path="data/sst_climatology.nc"):
    """Generate SSTA contour lines GeoJSON from SST data and climatology."""
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    if not os.path.exists(clim_path):
        print(f"[SSTA] Climatology not found, skipping contours")
        return None

    print(f"[SSTA] Processing anomaly contours for {date_str}...")
    ds = xr.open_dataset(sst_file)
    sst = None
    for var in ["analysed_sst", "thetao", "sst", "SST"]:
        if var in ds:
            sst = ds[var].squeeze()
            break
    if sst is None:
        raise ValueError(f"No SST variable found. Available: {list(ds.data_vars)}")

    lons = sst.longitude.values if "longitude" in sst.dims else sst.lon.values
    lats = sst.latitude.values if "latitude" in sst.dims else sst.lat.values
    data = sst.values.copy().astype(float)
    ds.close()

    # Convert Kelvin to Celsius if needed
    if np.nanmean(data) > 100:
        data -= 273.15

    ssta = compute_ssta(data, lats, lons, date_str, clim_path)
    mask = np.isnan(ssta)
    ssta_filled = ssta.copy()
    ssta_filled[mask] = 0.0
    ssta_smooth = gaussian_filter(ssta_filled, sigma=0.8)
    ssta_smooth[mask] = 0.0

    all_features = []
    for level, label, color in SSTA_LEVELS:
        feats = _contours_to_geojson(
            ssta_smooth, lons, lats, level,
            properties={"type": "ssta_contour", "anomaly": level, "label": label, "color": color}
        )
        feats = [f for f in feats if len(f["geometry"]["coordinates"]) >= 3]
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    output_path = os.path.join(OUTPUT_DIR, "ssta_contours.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"[SSTA] {len(all_features)} contours -> {output_path}")
    return output_path


def compute_ftle(date_str, bbox, window_days=3):
    """Compute Finite-Time Lyapunov Exponents from multi-day velocity data.
    FTLE identifies Lagrangian Coherent Structures — transport barriers where
    baitfish passively accumulate. High FTLE ridges mark boundaries between
    water masses that separate or converge over the integration window.

    Requires current data for date_str and window_days-1 preceding days.
    Returns (ftle_field, lons, lats) or None if insufficient data.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    from datetime import datetime, timedelta

    target = datetime.strptime(date_str, "%Y-%m-%d")
    # OUTPUT_DIR is e.g. "data/2000-04-23" — go up to "data/" to find sibling dates
    data_dir = os.path.dirname(OUTPUT_DIR) or "data"

    # Collect velocity fields for the integration window
    u_fields = []
    v_fields = []
    dates = []
    for d in range(window_days):
        dt = target - timedelta(days=window_days - 1 - d)
        ds = dt.strftime("%Y-%m-%d")
        # Check multiple possible locations for current data
        for base in [
            os.path.join(data_dir, ds),
            os.path.join(data_dir, "lookback", ds),
            os.path.join(data_dir, "backtest", ds),
        ]:
            cur_file = os.path.join(base, "currents_raw.nc")
            if os.path.exists(cur_file):
                try:
                    cds = xr.open_dataset(cur_file)
                    uo = cds["uo"]
                    vo = cds["vo"]
                    if "depth" in uo.dims and uo.sizes["depth"] > 1:
                        uo = uo.isel(depth=0)
                        vo = vo.isel(depth=0)
                    uo = uo.squeeze()
                    vo = vo.squeeze()
                    c_lons = uo.longitude.values if "longitude" in uo.dims else uo.lon.values
                    c_lats = uo.latitude.values if "latitude" in uo.dims else uo.lat.values
                    u_fields.append((uo.values.astype(float), c_lons, c_lats))
                    v_fields.append((vo.values.astype(float), c_lons, c_lats))
                    dates.append(ds)
                    cds.close()
                except Exception:
                    pass
                break

    if len(u_fields) < 2:
        return None

    # Use the first day's grid as reference
    ref_lons = u_fields[0][1]
    ref_lats = u_fields[0][2]
    n_lat, n_lon = u_fields[0][0].shape

    # Build velocity interpolators for each day
    u_interps = []
    v_interps = []
    for (u_data, u_lons, u_lats), (v_data, v_lons, v_lats) in zip(u_fields, v_fields):
        u_filled = np.where(np.isnan(u_data), 0, u_data)
        v_filled = np.where(np.isnan(v_data), 0, v_data)
        u_interps.append(RegularGridInterpolator(
            (u_lats, u_lons), u_filled, method="linear",
            bounds_error=False, fill_value=0))
        v_interps.append(RegularGridInterpolator(
            (v_lats, v_lons), v_filled, method="linear",
            bounds_error=False, fill_value=0))

    # Seed particles on the reference grid
    lat_grid, lon_grid = np.meshgrid(ref_lats, ref_lons, indexing="ij")
    x0 = lon_grid.copy()
    y0 = lat_grid.copy()
    x = x0.copy()
    y = y0.copy()

    # Integration: forward advection using RK4
    # dt in seconds (1 day), n_steps = window_days
    dt_sec = 86400.0
    m_per_deg_lat = 111320.0
    cos_lat = np.cos(np.radians(np.mean(ref_lats)))
    m_per_deg_lon = 111320.0 * cos_lat

    n_steps = len(u_fields) - 1
    for step in range(n_steps):
        # Time fraction for interpolating between daily velocity snapshots
        u_interp = u_interps[step]
        v_interp = v_interps[step]
        u_next = u_interps[min(step + 1, len(u_interps) - 1)]
        v_next = v_interps[min(step + 1, len(v_interps) - 1)]

        def _vel(px, py, u_int, v_int):
            pts = np.column_stack([py.ravel(), px.ravel()])
            u = u_int(pts).reshape(px.shape) / m_per_deg_lon
            v = v_int(pts).reshape(px.shape) / m_per_deg_lat
            return u, v  # degrees/second

        # RK4 integration
        k1u, k1v = _vel(x, y, u_interp, v_interp)
        k2u, k2v = _vel(x + 0.5*dt_sec*k1u, y + 0.5*dt_sec*k1v, u_interp, v_interp)
        k3u, k3v = _vel(x + 0.5*dt_sec*k2u, y + 0.5*dt_sec*k2v, u_next, v_next)
        k4u, k4v = _vel(x + dt_sec*k3u, y + dt_sec*k3v, u_next, v_next)
        x += dt_sec / 6.0 * (k1u + 2*k2u + 2*k3u + k4u)
        y += dt_sec / 6.0 * (k1v + 2*k2v + 2*k3v + k4v)

    # Compute deformation gradient tensor and FTLE
    # dx_f/dx_0, dx_f/dy_0, dy_f/dx_0, dy_f/dy_0
    dx = x - x0
    dy = y - y0
    # Finite differences for the deformation gradient
    dxdx = np.gradient(dx, axis=1) + 1.0  # +1 for identity
    dxdy = np.gradient(dx, axis=0)
    dydx = np.gradient(dy, axis=1)
    dydy = np.gradient(dy, axis=0) + 1.0

    # Cauchy-Green strain tensor eigenvalues
    # C = F^T * F, lambda_max = max eigenvalue
    a = dxdx**2 + dydx**2
    b = dxdx * dxdy + dydx * dydy
    c = dxdy**2 + dydy**2
    # Max eigenvalue of 2x2 symmetric matrix
    trace = a + c
    det = a * c - b**2
    discriminant = np.clip(trace**2 - 4 * det, 0, None)
    lambda_max = 0.5 * (trace + np.sqrt(discriminant))
    lambda_max = np.clip(lambda_max, 1.0, None)

    # FTLE = (1/T) * ln(sqrt(lambda_max))
    T = n_steps * dt_sec
    ftle = np.log(np.sqrt(lambda_max)) / T

    print(f"[FTLE] Computed from {len(dates)} days of velocity data "
          f"(max FTLE: {np.nanmax(ftle):.2e})")

    return ftle, ref_lons, ref_lats


def generate_blue_marlin_hotspots(bbox, tif_path=None, date_str=None):
    """
    Build a composite habitat suitability grid for blue marlin by scoring
    each pixel across all available ocean variables, then export as filled
    GeoJSON polygons with intensity bands.
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter, sobel, laplace, distance_transform_edt, convolve
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

    # Preserve native-resolution data for gradient computation BEFORE upsampling.
    # Computing Sobel on interpolated data creates artifact edges at the source resolution.
    # Gradients should be computed at native resolution, then interpolated to master grid.
    sst_native = sst.copy()
    lons_native = lons.copy()
    lats_native = lats.copy()
    was_upsampled = False

    # Upsample coarse grids (>0.05 deg) to ~0.02 deg for finer spatial detail
    # ANFC forecast data is 0.083 deg (~5nm cells) which is too coarse for sub-zone
    # discrimination. Upsampling to 0.02 deg (~1.2nm) matches IMOS observation resolution.
    grid_step = abs(np.diff(lons).mean()) if len(lons) > 1 else 1
    if grid_step > 0.05:
        was_upsampled = True
        from scipy.interpolate import RegularGridInterpolator as _RGI
        target_step = 0.02
        fine_lons = np.arange(lons.min(), lons.max() + target_step * 0.5, target_step)
        fine_lats = np.arange(lats.min(), lats.max() + target_step * 0.5, target_step)
        # Ensure monotonic direction matches original
        if lats[0] > lats[-1]:
            fine_lats = fine_lats[::-1]
        interp_sst = _RGI((lats, lons), sst, method="linear",
                          bounds_error=False, fill_value=np.nan)
        fg_lat, fg_lon = np.meshgrid(fine_lats, fine_lons, indexing="ij")
        sst = interp_sst((fg_lat, fg_lon))
        lons = fine_lons
        lats = fine_lats
        print(f"[Hotspots] Upsampled {grid_step:.3f} -> {target_step} deg "
              f"({sst.shape[0]}x{sst.shape[1]} grid)")

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

    _pool_pct = getattr(sys.modules[__name__], '_opt_pool_percentile', 75)

    def _maxpool_to_grid(data_hr, hr_lons, hr_lats):
        """Percentile-pool a high-res array to the master grid.
        For each coarse cell, take the Nth percentile from all high-res pixels
        that fall within it. Default 85th — reduces offshore bias from extreme
        deep-edge pixels while still preserving shelf-edge signal."""
        result = np.zeros((ny, nx))
        dlat = abs(lats[1] - lats[0]) / 2 if ny > 1 else 0.04
        dlon = abs(lons[1] - lons[0]) / 2 if nx > 1 else 0.04
        for yi in range(ny):
            for xi in range(nx):
                b_row = (hr_lats >= lats[yi] - dlat) & (hr_lats <= lats[yi] + dlat)
                b_col = (hr_lons >= lons[xi] - dlon) & (hr_lons <= lons[xi] + dlon)
                if np.any(b_row) and np.any(b_col):
                    patch = data_hr[np.ix_(b_row, b_col)]
                    result[yi, xi] = np.percentile(patch, _pool_pct) if _pool_pct < 100 else np.max(patch)
                else:
                    byi = np.argmin(np.abs(hr_lats - lats[yi]))
                    bxi = np.argmin(np.abs(hr_lons - lons[xi]))
                    result[yi, xi] = data_hr[byi, bxi]
        return result

    def _native_gradient(data_native, src_lons, src_lats, sigma=1.5):
        """Compute Sobel gradient at native resolution, then interpolate magnitude
        to master grid. Avoids artifact edges from computing gradients on
        linearly interpolated data."""
        filled = data_native.copy()
        native_land = np.isnan(filled)
        filled[native_land] = np.nanmean(filled) if np.any(~native_land) else 0
        smoothed = gaussian_filter(filled, sigma=sigma)
        gx_n = sobel(smoothed, axis=1)
        gy_n = sobel(smoothed, axis=0)
        gmag = np.sqrt(gx_n**2 + gy_n**2)
        gmag[native_land] = 0
        # Interpolate gradient magnitude to master grid
        gmag_master = _interp_to_grid(gmag, src_lons, src_lats)
        # Also interpolate component vectors for alignment calculations
        gx_master = _interp_to_grid(gx_n, src_lons, src_lats)
        gy_master = _interp_to_grid(gy_n, src_lons, src_lats)
        return gmag_master, gx_master, gy_master

    # Value-space edge scoring: Gaussian centered on a "sweet spot" value.
    # Catches cluster at feature EDGES (not peaks), so we score proximity to
    # a calibrated center value with configurable width.
    EDGE_FEATURES = {"okubo_weiss", "upwelling_edge",
                     "current_shear", "chl_curvature"}
    EDGE_DEFAULTS = {
        "okubo_weiss":         (0.30, 0.65),
        "upwelling_edge":      (0.75, 0.10),
        "current_shear":       (0.80, 0.60),
        "chl_curvature":       (0.50, 0.80),
    }

    def _add_score(name, values, mask=None):
        """Add a weighted sub-score. Values should be 0–1.
        Edge features get value-space Gaussian transform before weighting."""
        w = BLUE_MARLIN_WEIGHTS.get(name, 0)
        if w == 0:
            # Store for diagnostics even at zero weight
            sub_scores[name] = np.where(np.isnan(values), np.nan, np.clip(values, 0, 1))
            return
        v = np.clip(values, 0, 1)
        if name in EDGE_FEATURES:
            # Store raw values for diagnostics
            sub_scores[f"{name}_raw"] = v.copy()
            # Value-space Gaussian: score peaks at center, decays with width
            c_default, w_default = EDGE_DEFAULTS[name]
            center = getattr(sys.modules[__name__], f'_opt_{name}_edge_center', c_default)
            width = getattr(sys.modules[__name__], f'_opt_{name}_edge_width', w_default)
            v = np.exp(-0.5 * ((v - center) / max(width, 0.01)) ** 2)
            v = np.clip(v, 0, 1)
        valid_cells = ~np.isnan(v) & ~land
        if mask is not None:
            valid_cells &= ~mask
        score[valid_cells] += w * v[valid_cells]
        weight_sum[valid_cells] += w
        sub_scores[name] = v.copy()

    # 1. SST score — Gaussian centered at optimal temp
    #    Validated: 81% of 71 catches at 22-24°C, median 22.9°C
    sst_filled = sst.copy()
    sst_filled[land] = np.nanmean(sst)
    sst_smooth = gaussian_filter(sst_filled, sigma=0.5)
    sst_smooth[land] = np.nan
    optimal_temp = getattr(sys.modules[__name__], '_opt_sst_optimal', 23.75)
    sst_sigma = getattr(sys.modules[__name__], '_opt_sst_sigma', 2.50)
    sst_sigma_above = getattr(sys.modules[__name__], '_opt_sst_sigma_above', 4.0)
    if sst_sigma_above is not None:
        # Asymmetric Gaussian: tighter below optimal (cooling penalty), wider above
        sigma_map = np.where(sst_smooth < optimal_temp, sst_sigma, sst_sigma_above)
        sst_score = np.exp(-0.5 * ((sst_smooth - optimal_temp) / sigma_map) ** 2)
    else:
        sst_score = np.exp(-0.5 * ((sst_smooth - optimal_temp) / sst_sigma) ** 2)
    _add_score("sst", sst_score)

    # 1b. SST Anomaly — REMOVED from scoring.
    # Analysis: 69% of catches in cool anomaly water (SSTA < -0.5C),
    # only 10% in warm anomaly. Delta catch vs background = -0.11C.
    # SSTA has no discriminative value for blue marlin in Perth Canyon.

    # 2. SST front score — Sobel gradient magnitude, modulated by SST suitability
    #    A front at 20°C is useless for marlin — only score fronts in warm water
    #    Compute gradient at NATIVE resolution to avoid interpolation artifacts,
    #    then interpolate gradient magnitude to master grid.
    from scipy.ndimage import binary_dilation
    coast_buf = binary_dilation(land, iterations=2)
    if was_upsampled:
        grad_mag, gx, gy = _native_gradient(sst_native, lons_native, lats_native, sigma=1.5)
    else:
        sst_for_grad = sst_filled.copy()
        sst_grad = gaussian_filter(sst_for_grad, sigma=1.5)
        gx = sobel(sst_grad, axis=1)
        gy = sobel(sst_grad, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[coast_buf] = 0
    # Normalize by 90th percentile (robust to outlier pixels, more
    # temporally consistent than max-based normalization)
    ocean_grad = grad_mag[~coast_buf & ~land]
    g90 = np.nanpercentile(ocean_grad, 90) if len(ocean_grad) > 0 else 0
    if g90 > 0:
        front_raw = np.clip(grad_mag / g90, 0, 1)
    else:
        front_raw = np.zeros_like(grad_mag)
    # Edge-hunting: catches sit ADJACENT to SST fronts, not on the sharpest gradient.
    # Widen the front influence so score bleeds to both sides of the front line.
    _front_sig = getattr(sys.modules[__name__], '_edge_front_sigma', 3.5)
    front_widened = gaussian_filter(np.where(np.isnan(front_raw), 0, front_raw), sigma=max(_front_sig, 0.1))
    fw_ocean = front_widened[~land & ~coast_buf]
    fw_90 = np.nanpercentile(fw_ocean, 90) if len(fw_ocean) > 0 else 1
    front_score = np.clip(front_widened / fw_90, 0, 1) if fw_90 > 0 else front_raw
    # Modulate: fronts only count where SST is suitable for marlin
    front_score = front_score * sst_score
    # Floor: warm water (SST score > 0.6) gets minimum front score
    # Validated: catches in warm water without fronts still productive (54%)
    _front_floor = getattr(sys.modules[__name__], '_opt_front_floor', 0.07)
    warm_mask = sst_score > 0.6
    front_score = np.where(warm_mask, np.maximum(front_score, _front_floor), front_score)
    _add_score("sst_front", front_score)

    # 2a. SST Front Corridors — narrow front corridors/pinch points funnel prey
    #     Catches often sit inside tight front pockets where gradients converge
    #     from multiple directions. Score cells near fronts from 2+ quadrants.
    try:
        _corr_pct = getattr(sys.modules[__name__], '_opt_corridor_pct', 85)
        _corr_thresh = np.nanpercentile(front_score[~land & ~coast_buf], _corr_pct) if np.any(~land & ~coast_buf) else 0.26
        front_mask = (front_score > _corr_thresh).astype(float)
        front_mask[land | coast_buf] = 0

        # Distance to nearest front (in grid cells)
        dist_to_front = distance_transform_edt(1 - front_mask)
        proximity = np.clip(1.0 - dist_to_front / 4.0, 0, 1)

        # Multi-direction check: convolve front_mask with 4 quadrant kernels
        _gs = abs(lons[1] - lons[0]) if nx > 1 else 0.083
        k_size = max(5, int(round(0.15 / _gs)))
        if k_size % 2 == 0:
            k_size += 1
        half = k_size // 2
        quadrants = [
            (slice(0, half), slice(None)),       # top
            (slice(half + 1, None), slice(None)), # bottom
            (slice(None), slice(0, half)),        # left
            (slice(None), slice(half + 1, None)), # right
        ]
        dir_count = np.zeros((ny, nx))
        for qs in quadrants:
            k = np.zeros((k_size, k_size))
            k[qs] = 1.0
            k /= max(k.sum(), 1)
            d = convolve(front_mask, k, mode='constant', cval=0)
            dir_count += (d > 0.08).astype(float)

        # Corridor score: near fronts AND fronts from 2+ directions
        corridor_score = proximity * np.clip((dir_count - 1) / 2.0, 0, 1)
        corridor_score *= sst_score  # only in warm water
        corridor_score[land] = np.nan
        _add_score("front_corridor", corridor_score)
    except Exception as e:
        print(f"[Hotspots] Front corridor scoring failed: {e}")

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
    _intrusion_thresh = getattr(sys.modules[__name__], '_opt_intrusion_threshold', 0.45)
    _intrusion_baseline = getattr(sys.modules[__name__], '_opt_intrusion_baseline', 0.29)
    intrusion_score = np.clip(cross_grad / _intrusion_thresh, 0, 1)
    # Warm-water baseline: if SST is suitable, give minimum score even without
    # a clear cross-shelf gradient (LC may be present without E-W gradient)
    warm_baseline = sst_score > 0.5
    intrusion_score = np.where(warm_baseline, np.maximum(intrusion_score, _intrusion_baseline), intrusion_score)
    intrusion_score[land] = np.nan
    _add_score("sst_intrusion", intrusion_score)

    # 3. Chlorophyll score — peaks at 0.15–0.30 mg/m³ (shelf edge bait zone)
    chl_grid = None  # saved for boundary convergence
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
            chl_native = chl_da.values.astype(float)  # preserve for native-res gradients
            chl_data = _interp_to_grid(chl_native, chl_lons, chl_lats)
            chl_grid = chl_data  # save for boundary convergence
            # Peak score at optimal CHL, Gaussian falloff in log space
            _chl_opt = getattr(sys.modules[__name__], '_opt_chl_threshold', 0.20)
            _chl_sig = getattr(sys.modules[__name__], '_opt_chl_sigma', 0.45)
            chl_log = np.log10(np.clip(chl_data, 0.01, 10))
            optimal_chl = np.log10(_chl_opt)
            chl_score = np.exp(-0.5 * ((chl_log - optimal_chl) / _chl_sig) ** 2)
            _add_score("chl", chl_score)

            # 3a2. Bivariate SST-CHL kernel — captures interaction that linear
            #      sum misses. Warm+barren and cold+productive both score LOW
            #      in a 2D Gaussian, whereas linear sum scores them MEDIUM.
            #      Sweet spot: SST 22.2-23.4C + CHL 0.12-0.16 mg/m3 = "clean
            #      blue water adjacent to upwelling productivity"
            try:
                _bv_rho = getattr(sys.modules[__name__], '_opt_bivariate_rho', 0.0)
                sst_dev = (sst_smooth - optimal_temp) / sst_sigma
                chl_dev = (chl_log - optimal_chl) / _chl_sig
                rho = np.clip(_bv_rho, -0.9, 0.9)
                rho2 = rho ** 2
                # 2D Gaussian: exp(-0.5/(1-rho^2) * (x^2 + y^2 - 2*rho*x*y))
                z = (sst_dev**2 + chl_dev**2 - 2 * rho * sst_dev * chl_dev) / (1 - rho2)
                bivariate_score = np.exp(-0.5 * z)
                bivariate_score[land] = np.nan
                _add_score("sst_chl_bivariate", bivariate_score)
            except Exception as e:
                print(f"[Hotspots] Bivariate SST-CHL scoring failed: {e}")

            # 3b. CHL Curvature — pockets & peninsulas indicate dynamic mixing
            #     High absolute Laplacian = concave/convex CHL features where
            #     nutrients pool and bait congregates. Catches cluster near these.
            try:
                chl_for_lap = chl_log.copy()
                chl_for_lap[np.isnan(chl_for_lap) | land] = np.nanmean(chl_log[~land])
                chl_lap_smooth = gaussian_filter(chl_for_lap, sigma=2.0)
                chl_laplacian = laplace(chl_lap_smooth)
                chl_laplacian[coast_buf] = 0
                chl_curv = np.abs(chl_laplacian)
                cc90 = np.nanpercentile(chl_curv[~coast_buf & ~land], 90)
                if cc90 > 0:
                    chl_curv_score = np.clip(chl_curv / cc90, 0, 1)
                    chl_curv_score[land] = np.nan
                    _add_score("chl_curvature", chl_curv_score)
            except Exception as e:
                print(f"[Hotspots] CHL curvature scoring failed: {e}")
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
            # Already gradient-based, but catches sit 1.70x gradient lift from peaks —
            # marlin patrol the EDGE of steep zones, not the steepest point.
            # Widen influence with gaussian blur so score bleeds outward.
            bathy_filled = bathy.copy()
            bathy_filled[np.isnan(bathy_filled)] = 0
            dgx = sobel(bathy_filled, axis=1)
            dgy = sobel(bathy_filled, axis=0)
            depth_gradient = np.sqrt(dgx**2 + dgy**2)
            depth_gradient[np.isnan(bathy)] = 0
            # Interpolate gradient to master grid
            shelf_break = _interp_to_grid(depth_gradient, b_lons, b_lats)
            shelf_raw = np.clip(shelf_break / 100, 0, 1)
            # Widen: blur the gradient score so high values bleed to adjacent water
            _shelf_sig = getattr(sys.modules[__name__], '_edge_shelf_sigma', 2.0)
            shelf_widened = gaussian_filter(np.where(np.isnan(shelf_raw), 0, shelf_raw), sigma=max(_shelf_sig, 0.1))
            # Renormalize
            sw_ocean = shelf_widened[~land]
            sw_90 = np.nanpercentile(sw_ocean, 90) if len(sw_ocean) > 0 else 1
            shelf_score = np.clip(shelf_widened / sw_90, 0, 1) if sw_90 > 0 else shelf_raw
            shelf_score[land] = np.nan

            # Proximity-to-shelf-edge complement: catches cluster on the FLAT
            # shelf lip (median depth 229m, 0.69x Sobel gradient of 5nm offshore).
            # Blend Sobel gradient with a Gaussian proximity score centered at
            # the shelf break depth to pull peaks shoreward toward catch locations.
            depth_master_early = _interp_to_grid(
                np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats)
            _shelf_prox_depth = getattr(sys.modules[__name__], '_shelf_prox_depth', 270)
            _shelf_prox_sigma = getattr(sys.modules[__name__], '_shelf_prox_sigma', 50)
            _shelf_prox_blend = getattr(sys.modules[__name__], '_shelf_prox_blend', 0.80)
            if _shelf_prox_blend > 0:
                shelf_prox = np.exp(-0.5 * ((depth_master_early - _shelf_prox_depth) / max(_shelf_prox_sigma, 1))**2)
                shelf_prox[land] = np.nan
                shelf_score = (1.0 - _shelf_prox_blend) * shelf_score + _shelf_prox_blend * shelf_prox

            # Hybrid: additive component via weighted sum + multiplicative boost later
            _add_score("shelf_break", shelf_score)
            sb_pct = np.sum(shelf_score[~np.isnan(shelf_score)] > 0.5) / np.sum(~np.isnan(shelf_score)) * 100
            print(f"[Hotspots] Shelf break: {sb_pct:.0f}% of cells >50% (applied as x1.0-1.5 multiplier)")

            # 4b2. Bathymetric Rugosity (VRM) — Vector Ruggedness Measure
            #      Distinguishes complex rocky holding zones from smooth transit slopes.
            #      High VRM = rugged terrain where baitfish shelter and predators ambush.
            #      Computed at native bathy resolution, then interpolated to master grid.
            try:
                # Surface normal vectors from slopes (dgx, dgy already computed)
                # Normal = (-dz/dx, -dz/dy, 1) — normalize to unit vectors
                norm_mag = np.sqrt(dgx**2 + dgy**2 + 1.0)
                nx_comp = -dgx / norm_mag
                ny_comp = -dgy / norm_mag
                nz_comp = 1.0 / norm_mag

                # VRM: in a moving window, compute resultant vector length
                # VRM = 1 - (|resultant| / n_cells)
                _vrm_window = 5  # 5x5 cells at native bathy res
                from scipy.ndimage import uniform_filter
                sum_nx = uniform_filter(nx_comp, size=_vrm_window, mode='constant', cval=0)
                sum_ny = uniform_filter(ny_comp, size=_vrm_window, mode='constant', cval=0)
                sum_nz = uniform_filter(nz_comp, size=_vrm_window, mode='constant', cval=0)
                # uniform_filter returns mean, multiply by n_cells to get sum
                n_cells = _vrm_window ** 2
                resultant_len = np.sqrt((sum_nx * n_cells)**2 + (sum_ny * n_cells)**2 + (sum_nz * n_cells)**2)
                vrm_raw = 1.0 - (resultant_len / n_cells)
                vrm_raw[np.isnan(bathy)] = 0

                # Interpolate to master grid
                vrm_master = _interp_to_grid(vrm_raw, b_lons, b_lats)
                # Normalize: 0-1 by 95th percentile (rugosity is rare)
                vrm_ocean = vrm_master[~land & ~np.isnan(vrm_master)]
                vrm_95 = np.nanpercentile(vrm_ocean, 95) if len(vrm_ocean) > 0 else 1
                if vrm_95 > 0:
                    rugosity_score = np.clip(vrm_master / vrm_95, 0, 1)
                else:
                    rugosity_score = np.zeros((ny, nx))
                # Only meaningful in the 100-500m zone where marlin operate
                if depth_master_early is not None:
                    rug_depth_mod = np.clip((depth_master_early - 80) / 70, 0, 1) * np.clip((600 - depth_master_early) / 200, 0, 1)
                    rugosity_score *= rug_depth_mod
                rugosity_score[land] = np.nan
                _add_score("rugosity", rugosity_score)
                rug_pct = np.sum(rugosity_score[~np.isnan(rugosity_score)] > 0.3) / max(np.sum(~np.isnan(rugosity_score)), 1) * 100
                print(f"[Hotspots] Rugosity (VRM): {rug_pct:.0f}% of cells in rugged zones")
            except Exception as e:
                print(f"[Hotspots] Rugosity scoring failed: {e}")

            # 4c. Bathymetry-Feature Alignment — features following canyon walls
            #     When SST fronts or CHL edges align with the bathymetry gradient,
            #     prey aggregates along the shelf break. Score |cos(angle)| between
            #     bathy gradient and ocean feature gradients.
            try:
                bathy_gx_m = _interp_to_grid(dgx, b_lons, b_lats)
                bathy_gy_m = _interp_to_grid(dgy, b_lons, b_lats)
                bathy_gmag = np.sqrt(bathy_gx_m**2 + bathy_gy_m**2)
                bathy_gmag = np.where(bathy_gmag < 1e-10, 1e-10, bathy_gmag)

                # SST gradient alignment (gx, gy already computed)
                sst_gmag = np.where(grad_mag < 1e-10, 1e-10, grad_mag)
                dot_sst = np.abs(bathy_gx_m * gx + bathy_gy_m * gy) / (bathy_gmag * sst_gmag)
                align_score = np.clip(dot_sst, 0, 1)

                # CHL gradient alignment (if available)
                if chl_grid is not None:
                    chl_a = np.log10(np.clip(chl_grid, 0.01, 10))
                    chl_a[np.isnan(chl_a) | land] = np.nanmean(chl_a[~land])
                    chl_as = gaussian_filter(chl_a, sigma=1.5)
                    chl_agx = sobel(chl_as, axis=1)
                    chl_agy = sobel(chl_as, axis=0)
                    chl_amag = np.sqrt(chl_agx**2 + chl_agy**2)
                    chl_amag = np.where(chl_amag < 1e-10, 1e-10, chl_amag)
                    dot_chl = np.abs(bathy_gx_m * chl_agx + bathy_gy_m * chl_agy) / (bathy_gmag * chl_amag)
                    align_score = np.maximum(align_score, np.clip(dot_chl, 0, 1))

                # Weight by shelf break presence (alignment only matters at the shelf)
                align_score *= shelf_score
                align_score[land] = np.nan
                _add_score("bathy_align", align_score)
            except Exception as e:
                print(f"[Hotspots] Bathy alignment scoring failed: {e}")

            # Compute depth score at native bathy resolution FIRST, then
            # max-pool to coarse grid. This prevents shelf-edge catches from
            # being scored as shallow due to linear interpolation averaging
            # deep water with nearby land/shallow cells.
            abs_depth_hr = np.where(np.isnan(bathy), 0, -bathy)
            # Depth score: based on catch depth distribution (n=49)
            # Min=93m, 5th%=108m, 25th%=166m, median=229m, mean=365m
            # 0 catches <50m, 2 in 50-100m, 5 in 100-150m, 65% in 150-500m
            # Gate must be gentle since it's multiplicative (harsh penalties cascade).
            # Key: suppress <80m strongly, ramp 80-200m, full 200m+
            # 0m-80m:    0.0 (zero catches anywhere near this shallow)
            # 80-150m:   0.7-0.9 (7 catches in this range, keep viable)
            # 150-200m:  0.9-1.0 (15 catches, ramp to full)
            # 200-500m:  1.0 (primary catch zone - 65% of catches)
            # 500-800m:  1.0-0.80 (catches drop off sharply beyond 500m)
            # 800-1500m: 0.80-0.55 (very few catches, suppress offshore)
            # 1500m+:    0.50 (screenshot analysis: red zones extend into
            #            deep offshore where zero catches occur)
            _dt_start = getattr(sys.modules[__name__], '_depth_taper_start', 500)
            _dt_mid = getattr(sys.modules[__name__], '_depth_taper_mid', 1500)
            _dt_floor = getattr(sys.modules[__name__], '_depth_floor', 0.95)
            # Shallow side: 40% of catches at 100-200m — must not suppress this zone
            _dt_shallow_full = getattr(sys.modules[__name__], '_depth_shallow_full', 180)
            _dt_shallow_floor = getattr(sys.modules[__name__], '_depth_shallow_floor', 0.50)
            # Taper: full at _dt_start, 0.80 at _dt_mid, _dt_floor beyond 2x _dt_mid
            _dt_knee = 0.80  # score at _dt_mid
            _dt_zero = getattr(sys.modules[__name__], '_depth_zero_cut', 80)
            depth_score_hr = np.where(abs_depth_hr < _dt_zero, 0,
                             np.where(abs_depth_hr < _dt_shallow_full,
                                      _dt_shallow_floor + (1.0 - _dt_shallow_floor) * (abs_depth_hr - _dt_zero) / max(_dt_shallow_full - _dt_zero, 1),
                             np.where(abs_depth_hr < _dt_start, 1.0,
                             np.where(abs_depth_hr < _dt_mid, 1.0 - (1.0 - _dt_knee) * (abs_depth_hr - _dt_start) / max(_dt_mid - _dt_start, 1),
                             np.where(abs_depth_hr < _dt_mid * 2, _dt_knee - (_dt_knee - _dt_floor) * (abs_depth_hr - _dt_mid) / max(_dt_mid, 1),
                             _dt_floor)))))
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
    ssh_grid = None  # saved for boundary convergence
    ssh_file = os.path.join(OUTPUT_DIR, "ssh_raw.nc")
    if os.path.exists(ssh_file):
        try:
            sds = xr.open_dataset(ssh_file)
            sv = "sla" if "sla" in sds else ("zos" if "zos" in sds else "adt")
            sla_da = sds[sv].squeeze()
            s_lons = sla_da.longitude.values if "longitude" in sla_da.dims else sla_da.lon.values
            s_lats = sla_da.latitude.values if "latitude" in sla_da.dims else sla_da.lat.values
            sla_data = _interp_to_grid(sla_da.values.astype(float), s_lons, s_lats)
            ssh_grid = sla_data  # save for boundary convergence
            # Absolute SLA: warm water mass indicator
            # Higher SLA = stronger warm eddy presence
            abs_score = np.clip(sla_data / 0.12, 0, 1)
            # Relative SLA: eddy structure (local highs above background)
            sla_filled = sla_data.copy()
            sla_filled[np.isnan(sla_filled)] = np.nanmean(sla_data)
            sla_bg = gaussian_filter(sla_filled, sigma=4)
            sla_relative = sla_data - sla_bg
            rel_score = np.clip(sla_relative / 0.04, 0, 1)
            # Blend: 40% absolute + 60% relative (eddy edges)
            # Screenshot analysis: absolute SLA biases scoring offshore into deep
            # water where catches never occur. Eddy EDGES (relative) are where
            # marlin hunt — the boundary between warm/cool water masses.
            _ssh_ab = getattr(sys.modules[__name__], '_ssh_abs_blend', 0.2)
            ssh_score = _ssh_ab * abs_score + (1.0 - _ssh_ab) * rel_score
            ssh_score[land] = np.nan
            _add_score("ssh", ssh_score)
            mean_abs = np.nanmean(sla_data[~land])
            pct_high = np.sum(ssh_score[~np.isnan(ssh_score)] > 0.5) / np.sum(~np.isnan(ssh_score)) * 100
            print(f"[Hotspots] SSH scoring: {pct_high:.0f}% >50%, mean abs SLA={mean_abs:.3f}m")
        except Exception as e:
            print(f"[Hotspots] SSH scoring failed: {e}")

    # 6. MLD score — edge-hunting: marlin hunt MLD TRANSITIONS, not shallowest MLD
    #    Analysis shows 1.83x gradient lift at catch locations — catches sit where
    #    MLD changes rapidly (thermocline depth fronts), not where it's shallowest.
    #    Blend: 60% value (shallow is still good) + 40% edge (transitions score high)
    mld_grid = None  # saved for boundary convergence
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
            mld_grid = mld_data  # save for boundary convergence
            # Value component: 1.0 at MLD<20m, 0.5 at 50m, 0 at 100m
            mld_value = np.clip(1.0 - (mld_data - 20) / 80, 0, 1)
            # Edge component: gradient of MLD field (where thermocline depth changes rapidly)
            mld_for_grad = mld_data.copy()
            mld_for_grad[np.isnan(mld_for_grad)] = 0
            mld_smooth = gaussian_filter(mld_for_grad, sigma=1.5)
            mld_gy, mld_gx = np.gradient(mld_smooth)
            mld_edge = np.sqrt(mld_gx**2 + mld_gy**2)
            mld_edge[land | coast_buf] = 0
            mld_e90 = np.nanpercentile(mld_edge[~land & ~coast_buf], 90) if np.any(~land & ~coast_buf) else 0
            mld_edge_score = np.clip(mld_edge / mld_e90, 0, 1) if mld_e90 > 0 else np.zeros((ny, nx))
            # Widen edge influence so it reaches where marlin actually sit
            mld_edge_score = gaussian_filter(mld_edge_score, sigma=1.5)
            mld_edge_score = np.clip(mld_edge_score / (np.nanpercentile(mld_edge_score[~land & ~coast_buf], 90) + 1e-6), 0, 1)
            # Blend: 60% value + 40% edge
            _mld_eb = getattr(sys.modules[__name__], '_edge_mld_blend', 0.0)
            mld_score = (1.0 - _mld_eb) * mld_value + _mld_eb * mld_edge_score
            mld_score[land] = np.nan
            _add_score("mld", mld_score)
        except Exception as e:
            print(f"[Hotspots] MLD scoring failed: {e}")

    # 7. Oxygen score — REMOVED from scoring.
    # 0.25° resolution too coarse for canyon-scale features (13x13 grid).
    # Always 220-250 mmol/m³ off Perth — never limiting, zero discrimination.

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
            uo_native = uo_da.values.astype(float)  # preserve for native-res derivatives
            vo_native = vo_da.values.astype(float)
            uo_data = _interp_to_grid(uo_native, c_lons, c_lats)
            vo_data = _interp_to_grid(vo_native, c_lons, c_lats)
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
                    # Upstream offset: look back ~20km (approx 0.18 degrees)
                    # Normalize vector, multiply by distance in degrees, divide by grid resolution
                    dist_deg = 0.18
                    src_xi = int(round(xi - (u_val / spd) * (dist_deg / dlon)))
                    src_yi = int(round(yi - (v_val / spd) * (dist_deg / dlat)))
                    src_xi = max(0, min(nx - 1, src_xi))
                    src_yi = max(0, min(ny - 1, src_yi))
                    t = sst_smooth[src_yi, src_xi]
                    upstream_sst[yi, xi] = t if not np.isnan(t) else sst_smooth[yi, xi]

            # Score upstream SST using same Gaussian as main SST score
            if sst_sigma_above is not None:
                sigma_map_up = np.where(upstream_sst < optimal_temp, sst_sigma, sst_sigma_above)
                upstream_temp_score = np.exp(-0.5 * ((upstream_sst - optimal_temp) / sigma_map_up) ** 2)
            else:
                upstream_temp_score = np.exp(-0.5 * ((upstream_sst - optimal_temp) / sst_sigma) ** 2)
            upstream_temp_score[land] = np.nan

            # Combined: speed × upstream warmth, with eastward bonus
            # Validated: catches occur in all current directions, not just eastward.
            # Base score = speed × upstream_temp; eastward adds 30% bonus
            _east_bonus_factor = getattr(sys.modules[__name__], '_opt_east_bonus', 0.03)
            east_bonus = 1.0 + _east_bonus_factor * east_score
            current_value = speed_score * upstream_temp_score * east_bonus
            current_value = np.clip(current_value, 0, 1)

            # Edge-hunting: marlin hunt current BOUNDARIES (1.64x gradient lift).
            # Score the gradient of current speed — where speed changes rapidly
            # = shear lines between fast/slow water = bait aggregation zones.
            # Compute speed gradient at NATIVE current resolution.
            spd_native = np.sqrt(np.where(np.isnan(uo_native), 0, uo_native)**2 +
                                 np.where(np.isnan(vo_native), 0, vo_native)**2)
            spd_native_smooth = gaussian_filter(spd_native, sigma=1.5)
            ceg_y, ceg_x = np.gradient(spd_native_smooth)
            cur_edge_native = np.sqrt(ceg_x**2 + ceg_y**2)
            cur_edge = _interp_to_grid(cur_edge_native, c_lons, c_lats)
            cur_edge[land | coast_buf] = 0
            cur_e90 = np.nanpercentile(cur_edge[~land & ~coast_buf], 90) if np.any(~land & ~coast_buf) else 0
            cur_edge_score = np.clip(cur_edge / cur_e90, 0, 1) if cur_e90 > 0 else np.zeros((ny, nx))
            # Widen edge influence
            cur_edge_score = gaussian_filter(cur_edge_score, sigma=1.5)
            cur_edge_score = np.clip(cur_edge_score / (np.nanpercentile(cur_edge_score[~land & ~coast_buf], 90) + 1e-6), 0, 1)
            # Blend: 60% value + 40% edge
            _cur_eb = getattr(sys.modules[__name__], '_edge_current_blend', 0.4)
            current_score = (1.0 - _cur_eb) * current_value + _cur_eb * cur_edge_score
            current_score[land] = np.nan
            _add_score("current", current_score)

            ocean = ~land & ~np.isnan(current_score)
            fav_pct = np.sum(current_score[ocean] > 0.3) / np.sum(ocean) * 100
            mean_up_sst = np.nanmean(upstream_sst[ocean])
            print(f"[Hotspots] Current scoring: {fav_pct:.0f}% favorable, upstream SST mean {mean_up_sst:.1f}C")

            # 10. Current convergence — edge-hunting scoring
            #     Analysis shows 88% of catches sit on the EDGE of convergence
            #     zones (ratio 0.29), not inside them. Marlin patrol the boundary
            #     where bait is being pushed together, not the centre.
            #     Compute divergence at NATIVE current resolution to avoid artifacts.
            dudx_n = np.gradient(np.where(np.isnan(uo_native), 0, uo_native), axis=1)
            dvdy_n = np.gradient(np.where(np.isnan(vo_native), 0, vo_native), axis=0)
            divergence_n = dudx_n + dvdy_n
            divergence = _interp_to_grid(divergence_n, c_lons, c_lats)
            # Raw convergence field (negative divergence)
            conv_raw = np.clip(-divergence / 0.005, 0, 1)
            conv_filled = conv_raw.copy()
            conv_filled[np.isnan(conv_filled)] = 0
            conv_smooth = gaussian_filter(conv_filled, sigma=1.0)
            conv_smooth[land] = np.nan

            # Edge score: gradient of the convergence field
            # High gradient = boundary between converging and non-converging water
            conv_for_grad = conv_smooth.copy()
            conv_for_grad[np.isnan(conv_for_grad)] = 0
            conv_gy, conv_gx = np.gradient(conv_for_grad)
            conv_edge = np.sqrt(conv_gx**2 + conv_gy**2)
            conv_edge[land | coast_buf] = 0
            # Normalize to 90th percentile
            ce_ocean = conv_edge[~land & ~coast_buf]
            ce90 = np.nanpercentile(ce_ocean, 90) if len(ce_ocean) > 0 else 0
            conv_edge_score = np.clip(conv_edge / ce90, 0, 1) if ce90 > 0 else np.zeros((ny, nx))

            # Blend: 70% edge score + 30% raw proximity (must be near convergence)
            # The raw component ensures we don't score edges of divergence
            conv_proximity = np.clip(conv_smooth / 0.5, 0, 1)  # 1.0 if conv_raw >= 0.5
            # Expand proximity: also score cells adjacent to convergence
            prox_filled = conv_proximity.copy()
            prox_filled[np.isnan(prox_filled)] = 0
            conv_nearby = gaussian_filter(prox_filled, sigma=1.5)
            conv_nearby = np.clip(conv_nearby * 3, 0, 1)  # boost so 0.33 nearby -> 1.0
            conv_nearby[land] = np.nan

            _conv_eb = getattr(sys.modules[__name__], '_edge_conv_blend', 0.0)
            conv_score = _conv_eb * conv_edge_score + (1.0 - _conv_eb) * conv_nearby
            conv_score[land] = np.nan

            # Bait trap synergy: convergence edges more effective with strong current
            _synergy = getattr(sys.modules[__name__], '_opt_synergy_factor', 0.25)
            synergy = 1.0 + _synergy * np.clip(current_score, 0, 1)
            conv_score_synergy = np.clip(conv_score * synergy, 0, 1)
            conv_score_synergy[land] = np.nan
            _add_score("convergence", conv_score_synergy)
            conv_pct = np.sum(conv_score_synergy[~np.isnan(conv_score_synergy) & ~land] > 0.3) / np.sum(~np.isnan(conv_score_synergy) & ~land) * 100
            print(f"[Hotspots] Convergence scoring: {conv_pct:.0f}% of cells near convergence boundaries")

            # 10a. Vertical velocity (derived from continuity equation)
            #      Continuity: ∇·u + ∂w/∂z = 0 → positive surface divergence = upwelling
            #      Complements convergence (which uses negative divergence).
            #      No new data needed — reuses divergence computed above.
            if BLUE_MARLIN_WEIGHTS.get("vertical_velocity", 0) > 0:
                w_proxy = np.clip(divergence / 0.005, 0, 1)
                w_proxy[land] = np.nan
                _add_score("vertical_velocity", w_proxy)

            # 10b. Current shear (vorticity) — Leeuwin/Undercurrent boundary
            #      Vorticity = dv/dx - du/dy. High |vorticity| at canyon edges
            #      marks the shear boundary where baitfish get "stacked".
            #      The Leeuwin Current flows south along the shelf while the
            #      Capes Undercurrent flows north at depth — their interface
            #      creates strong lateral shear that concentrates prey.
            # Compute vorticity at NATIVE current resolution
            dvdx_n = np.gradient(np.where(np.isnan(vo_native), 0, vo_native), axis=1)
            dudy_n = np.gradient(np.where(np.isnan(uo_native), 0, uo_native), axis=0)
            vorticity_n = dvdx_n - dudy_n
            vorticity = _interp_to_grid(vorticity_n, c_lons, c_lats)
            abs_vort = np.abs(vorticity)
            abs_vort[land] = np.nan
            # Smooth to reduce grid noise
            vort_filled = abs_vort.copy()
            vort_filled[np.isnan(vort_filled)] = 0
            abs_vort = gaussian_filter(vort_filled, sigma=1.0)
            abs_vort[land] = np.nan
            # Normalize to 90th percentile (same approach as SST gradient)
            vort_ocean = abs_vort[~land & ~np.isnan(abs_vort)]
            v90 = np.nanpercentile(vort_ocean, 90) if len(vort_ocean) > 0 else 0
            if v90 > 0:
                shear_score = np.clip(abs_vort / v90, 0, 1)
            else:
                shear_score = np.zeros((ny, nx))
            shear_score[land] = np.nan
            # Modulate by depth: shear at the shelf edge (80-150m) is surface
            # friction, not Leeuwin/Undercurrent interaction. Only score shear
            # where depth >150m (where the undercurrent actually operates).
            # Dedicated shear depth ramp — Undercurrent operates at 100-150m shelf slope
            # in Perth Canyon context, so use a lower threshold than the main depth gate
            _shear_depth_thresh = getattr(sys.modules[__name__], '_opt_shear_depth_thresh', 60)
            _shear_depth_full = getattr(sys.modules[__name__], '_opt_shear_depth_full', 300)
            if depth_master_early is not None:
                shear_depth_mod = np.clip(
                    (depth_master_early - _shear_depth_thresh) / max(_shear_depth_full - _shear_depth_thresh, 1),
                    0, 1)
                shear_score = shear_score * shear_depth_mod
            elif "depth" in sub_scores:
                depth_mod = np.where(np.isnan(sub_scores["depth"]), 0, sub_scores["depth"])
                shear_score = shear_score * depth_mod
            _add_score("current_shear", shear_score)
            shear_pct = np.sum(shear_score[~np.isnan(shear_score) & ~land] > 0.3) / np.sum(~np.isnan(shear_score) & ~land) * 100
            print(f"[Hotspots] Current shear: {shear_pct:.0f}% of cells in shear zones (Leeuwin/Undercurrent boundary)")

        except Exception as e:
            print(f"[Hotspots] Current scoring failed: {e}")

    # 10c. Upwelling proxy — canyon-forced upwelling edges
    #      Upwelling = cooler SST + elevated CHL + shallow MLD near shelf break.
    #      Blue marlin hunt the WARM SIDE of upwelling boundaries where bait
    #      is pushed up by topographic forcing. Score the edge, not the core.
    #      Research: "Filter out cold-water upwellings unsuitable for Blue Marlin"
    #      — we want the warm-water boundary adjacent to upwelling.
    try:
        # Need SST, CHL, and either MLD or shelf break data
        if sst_smooth is not None and chl_grid is not None:
            # Upwelling indicator: cooler-than-mean SST + above-median CHL
            # (upwelling brings cold, nutrient-rich water to surface)
            sst_mean_ocean = np.nanmean(sst_smooth[~land])
            sst_cool = np.clip((sst_mean_ocean - sst_smooth) / 1.5, 0, 1)  # 1.0 if 1.5C cooler than mean
            sst_cool[land] = 0

            chl_med_ocean = np.nanmedian(chl_grid[~land & ~np.isnan(chl_grid)])
            chl_elevated = np.clip((chl_grid - chl_med_ocean) / (chl_med_ocean * 2), 0, 1) if chl_med_ocean > 0 else np.zeros((ny, nx))
            chl_elevated[land | np.isnan(chl_grid)] = 0

            # Upwelling core: where SST is cool AND CHL is elevated
            upwelling_core = sst_cool * chl_elevated
            if mld_grid is not None:
                # Shallow MLD reinforces upwelling signal
                mld_shallow = np.clip(1.0 - (mld_grid - 15) / 35, 0, 1)  # 1.0 at <15m, 0 at 50m
                mld_shallow[land | np.isnan(mld_grid)] = 0.5  # neutral where missing
                upwelling_core = upwelling_core * (0.5 + 0.5 * mld_shallow)

            # Smooth the core to get stable edges
            up_filled = upwelling_core.copy()
            up_filled[np.isnan(up_filled)] = 0
            upwelling_smooth = gaussian_filter(up_filled, sigma=1.5)

            # Upwelling EDGE: gradient of the upwelling field
            # This marks the boundary where warm Leeuwin water meets cold upwelled water
            up_gx = sobel(upwelling_smooth, axis=1)
            up_gy = sobel(upwelling_smooth, axis=0)
            up_edge_raw = np.sqrt(up_gx**2 + up_gy**2)
            up_edge_raw[land | coast_buf] = 0

            # Edge-hunting: marlin patrol the WARM SIDE of upwelling edges,
            # sitting further out than the sharp gradient peak (ratio=0.21).
            # Widen the edge influence with gaussian blur so score bleeds
            # outward to where marlin actually hunt.
            _up_sig = getattr(sys.modules[__name__], '_edge_upwell_sigma', 4.0)
            up_edge = gaussian_filter(up_edge_raw, sigma=max(_up_sig, 0.1))
            up_edge[land | coast_buf] = 0

            # Score the warm side of the edge: where SST is suitable
            # "Filter out cold-water upwellings unsuitable for Blue Marlin"
            up_e90 = np.nanpercentile(up_edge[~land & ~coast_buf], 90) if np.any(~land & ~coast_buf) else 0
            if up_e90 > 0:
                upwelling_edge_score = np.clip(up_edge / up_e90, 0, 1)
                # Only score where SST is still suitable for marlin (warm side)
                upwelling_edge_score *= sst_score
                upwelling_edge_score[land] = np.nan
                _add_score("upwelling_edge", upwelling_edge_score)
                up_pct = np.sum(upwelling_edge_score[~np.isnan(upwelling_edge_score) & ~land] > 0.3) / np.sum(~np.isnan(upwelling_edge_score) & ~land) * 100
                print(f"[Hotspots] Upwelling edge: {up_pct:.0f}% of cells near canyon upwelling boundaries")
            else:
                _add_score("upwelling_edge", np.where(land, np.nan, 0.0))
        else:
            _add_score("upwelling_edge", np.where(land, np.nan, 0.0))
    except Exception as e:
        print(f"[Hotspots] Upwelling edge scoring failed: {e}")
        _add_score("upwelling_edge", np.where(land, np.nan, 0.0))

    # 10d. Vertical velocity (wo) — REMOVED
    #      CMEMS physics models (ANFC + reanalysis) do not provide the 'wo' variable.
    #      Upwelling detection relies on SST/CHL proxy via upwelling_edge scoring.

    # 10e. FTLE / Lagrangian Coherent Structures — transport barriers
    #      High FTLE ridges mark boundaries where water parcels separate,
    #      creating accumulation zones for passively drifting baitfish.
    #      Uses multi-day velocity fields (requires adjacent-day current data).
    if date_str and BLUE_MARLIN_WEIGHTS.get("ftle", 0) > 0:
        try:
            ftle_result = compute_ftle(date_str, bbox, window_days=3)
            if ftle_result is not None:
                ftle_field, ftle_lons, ftle_lats = ftle_result
                ftle_master = _interp_to_grid(ftle_field, ftle_lons, ftle_lats)
                # Normalize by 95th percentile (FTLE ridges are sparse)
                ftle_ocean = ftle_master[~land & ~np.isnan(ftle_master)]
                ftle_95 = np.nanpercentile(ftle_ocean, 95) if len(ftle_ocean) > 0 else 1
                if ftle_95 > 0:
                    ftle_score = np.clip(ftle_master / ftle_95, 0, 1)
                else:
                    ftle_score = np.zeros((ny, nx))
                ftle_score[land] = np.nan
                _add_score("ftle", ftle_score)
                ftle_pct = np.sum(ftle_score[~np.isnan(ftle_score)] > 0.3) / max(np.sum(~np.isnan(ftle_score)), 1) * 100
                print(f"[Hotspots] FTLE: {ftle_pct:.0f}% of cells on transport barriers")
        except Exception as e:
            print(f"[Hotspots] FTLE scoring failed: {e}")

    # 11. Salinity front — halocline marks LC edge when SST front is masked
    #     The Leeuwin Current carries low-salinity tropical water (~34.5-35 PSU)
    #     vs surrounding Indian Ocean (~35.5-36 PSU). In summer when SST fronts
    #     disappear due to solar heating, salinity gradients remain sharp.
    sal_file = os.path.join(OUTPUT_DIR, "salinity_raw.nc")
    sal_grid = None
    if os.path.exists(sal_file) and BLUE_MARLIN_WEIGHTS.get("salinity_front", 0) > 0:
        try:
            sal_ds = xr.open_dataset(sal_file)
            sal_da = sal_ds["so"].squeeze()
            if "depth" in sal_da.dims:
                sal_da = sal_da.isel(depth=0)
            s_lons = sal_da.longitude.values if "longitude" in sal_da.dims else sal_da.lon.values
            s_lats = sal_da.latitude.values if "latitude" in sal_da.dims else sal_da.lat.values
            sal_grid = _interp_to_grid(sal_da.values.astype(float), s_lons, s_lats)

            # Compute Sobel gradient of salinity (same pattern as SST front)
            sal_filled = sal_grid.copy()
            sal_filled[np.isnan(sal_filled)] = np.nanmean(sal_filled)
            sal_smooth = gaussian_filter(sal_filled, sigma=1.5)
            sal_gx = sobel(sal_smooth, axis=1)
            sal_gy = sobel(sal_smooth, axis=0)
            sal_grad = np.sqrt(sal_gx**2 + sal_gy**2)
            sal_grad[coast_buf] = 0

            # Normalize by 90th percentile
            sal_ocean = sal_grad[~coast_buf & ~land]
            sg90 = np.nanpercentile(sal_ocean, 90) if len(sal_ocean) > 0 else 0
            if sg90 > 0:
                sal_front_raw = np.clip(sal_grad / sg90, 0, 1)
            else:
                sal_front_raw = np.zeros_like(sal_grad)

            # Widen front influence (edge-hunting pattern)
            sal_front_widened = gaussian_filter(np.where(np.isnan(sal_front_raw), 0, sal_front_raw), sigma=4.0)
            sf_ocean = sal_front_widened[~land & ~coast_buf]
            sf_90 = np.nanpercentile(sf_ocean, 90) if len(sf_ocean) > 0 else 1
            sal_front_score = np.clip(sal_front_widened / sf_90, 0, 1) if sf_90 > 0 else sal_front_raw
            sal_front_score[land] = np.nan
            _add_score("salinity_front", sal_front_score)
            sf_pct = np.sum(sal_front_score[~np.isnan(sal_front_score) & ~land] > 0.3) / np.sum(~np.isnan(sal_front_score) & ~land) * 100
            print(f"[Hotspots] Salinity front: {sf_pct:.0f}% of cells near halocline boundaries")
        except Exception as e:
            print(f"[Hotspots] Salinity front scoring failed: {e}")

    # 12. Thermocline lift — cold 250m water = canyon upwelling structure
    #     Warm Surface + Cold Lifted Basement is the most productive vertical profile.
    #     Score inversely proportional to 250m temperature: colder = more upwelling.
    sub_file = os.path.join(OUTPUT_DIR, "subsurface_temp_raw.nc")
    if os.path.exists(sub_file) and BLUE_MARLIN_WEIGHTS.get("thermocline_lift", 0) > 0:
        try:
            sub_ds = xr.open_dataset(sub_file)
            sub_da = sub_ds["thetao"].squeeze()
            if "depth" in sub_da.dims:
                sub_da = sub_da.isel(depth=0)
            st_lons = sub_da.longitude.values if "longitude" in sub_da.dims else sub_da.lon.values
            st_lats = sub_da.latitude.values if "latitude" in sub_da.dims else sub_da.lat.values
            sub_grid = _interp_to_grid(sub_da.values.astype(float), st_lons, st_lats)

            # Score: inverse temperature. Colder at 250m = stronger upwelling structure.
            # Typical range off Perth: 8-16C at 250m. Upwelling domes might be 8-10C.
            # Background ~14-16C. Score: 1.0 at <=8C, 0.0 at >=16C.
            _thermo_cold = getattr(sys.modules[__name__], '_opt_thermo_cold', 8.0)
            _thermo_warm = getattr(sys.modules[__name__], '_opt_thermo_warm', 16.0)
            thermo_score = np.clip((_thermo_warm - sub_grid) / max(_thermo_warm - _thermo_cold, 1), 0, 1)
            thermo_score[land] = np.nan

            # Bonus: warm surface + cold basement signature
            # Where SST is suitable (high sst_score) AND basement is cold,
            # we have the ideal Warm Cap + Cold Upwelling profile
            if sst_score is not None:
                warm_cap = np.clip(sst_score, 0, 1)
                # Multiply: high score only where BOTH conditions met
                thermo_combined = thermo_score * warm_cap
                # Blend: 60% raw thermocline + 40% combined signature
                thermo_final = 0.6 * thermo_score + 0.4 * thermo_combined
                thermo_final[land] = np.nan
            else:
                thermo_final = thermo_score

            _add_score("thermocline_lift", thermo_final)
            th_pct = np.sum(thermo_final[~np.isnan(thermo_final) & ~land] > 0.3) / np.sum(~np.isnan(thermo_final) & ~land) * 100
            print(f"[Hotspots] Thermocline lift: {th_pct:.0f}% of cells showing upwelling structure")
        except Exception as e:
            print(f"[Hotspots] Thermocline lift scoring failed: {e}")

    # 12b. Stratification Index — thermal barrier strength (bait entrapment)
    #      ΔT = SST - T_250m. High ΔT = strong thermocline = baitfish compressed
    #      into warm surface layer where marlin hunt. Perth Canyon: warm Leeuwin
    #      Current capping cold Leeuwin Undercurrent creates ideal stratification.
    #      Score: Gaussian centered on optimal ΔT (~6-8°C), with configurable params.
    if os.path.exists(sub_file) and BLUE_MARLIN_WEIGHTS.get("stratification", 0) > 0:
        try:
            sub_ds2 = xr.open_dataset(sub_file)
            sub_da2 = sub_ds2["thetao"].squeeze()
            if "depth" in sub_da2.dims:
                sub_da2 = sub_da2.isel(depth=0)
            st2_lons = sub_da2.longitude.values if "longitude" in sub_da2.dims else sub_da2.lon.values
            st2_lats = sub_da2.latitude.values if "latitude" in sub_da2.dims else sub_da2.lat.values
            t_deep = _interp_to_grid(sub_da2.values.astype(float), st2_lons, st2_lats)

            # ΔT = surface SST - deep temp
            delta_t = sst - t_deep

            # Score: 1.0 when ΔT >= threshold (strong barrier), taper below
            # Typical Perth range: 4-12°C. Strong stratification > 6°C.
            _strat_strong = getattr(sys.modules[__name__], '_opt_strat_strong', 6.0)
            _strat_weak = getattr(sys.modules[__name__], '_opt_strat_weak', 2.0)
            strat_score = np.clip((delta_t - _strat_weak) / max(_strat_strong - _strat_weak, 1), 0, 1)
            strat_score[land] = np.nan
            _add_score("stratification", strat_score)
            mean_dt = np.nanmean(delta_t[~land]) if np.any(~land) else 0
            print(f"[Hotspots] Stratification: mean ΔT={mean_dt:.1f}°C (strong>{_strat_strong}°C)")
        except Exception as e:
            print(f"[Hotspots] Stratification scoring failed: {e}")

    _depth_grid = None  # populated if bathy data available; used in hover info

    # 13. SST Rate of Change (Heating/Cooling)
    #      Reward areas that are actively warming (new water arriving).
    #      Penalize areas that are cooling (upwelling/mixing).
    #      This is highly specific to "today" vs general location.
    if BLUE_MARLIN_WEIGHTS.get("sst_roc", 0) > 0 and date_str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Look for yesterday's data in data/YYYY-MM-DD or main dir
            prev_file = os.path.join(os.path.dirname(OUTPUT_DIR), prev_date, "sst_raw.nc")
            if not os.path.exists(prev_file):
                prev_file = os.path.join("data", prev_date, "sst_raw.nc")
                
            if os.path.exists(prev_file):
                pds = xr.open_dataset(prev_file)
                prev_da = None
                for var in ["thetao", "analysed_sst", "sst"]:
                    if var in pds: prev_da = pds[var].squeeze(); break
                
                if prev_da is not None:
                    prev_val = prev_da.values.astype(float)
                    if np.nanmean(prev_val) > 100: prev_val -= 273.15
                    
                    prev_grid = _interp_to_grid(prev_val, 
                                              prev_da.longitude.values if "longitude" in prev_da.dims else prev_da.lon.values, 
                                              prev_da.latitude.values if "latitude" in prev_da.dims else prev_da.lat.values)
                    
                    # Calculate Delta
                    sst_delta = sst_smooth - prev_grid
                    
                    # Score: sigmoid centered at +0.2C (slight warming is good)
                    # < -0.5C (cooling) -> 0.0
                    # > +0.5C (warming) -> 1.0
                    roc_score = np.clip((sst_delta + 0.5) / 1.0, 0, 1)
                    roc_score[land] = np.nan
                    _add_score("sst_roc", roc_score)
                pds.close()
        except Exception as e:
            print(f"[Hotspots] SST RoC failed: {e}")

    # 14. Okubo-Weiss Parameter (Strain vs Vorticity)
    #      W = Sn^2 + Ss^2 - w^2
    #      Positive W = Strain dominated (deformation, fronts, filaments) -> Bait aggregation
    #      Negative W = Vorticity dominated (eddy cores) -> Trapping/Stable
    #      We want positive W (edges/filaments).
    if BLUE_MARLIN_WEIGHTS.get("okubo_weiss", 0) > 0 and 'uo_native' in locals():
        try:
            # Use native resolution currents to capture gradients
            u_n = np.where(np.isnan(uo_native), 0, uo_native)
            v_n = np.where(np.isnan(vo_native), 0, vo_native)
            
            # Gradients (indices)
            ux = np.gradient(u_n, axis=1); uy = np.gradient(u_n, axis=0)
            vx = np.gradient(v_n, axis=1); vy = np.gradient(v_n, axis=0)
            
            # Strain and Vorticity components
            normal_strain = ux - vy
            shear_strain = vx + uy
            vorticity = vx - uy
            
            # Okubo-Weiss W
            W = normal_strain**2 + shear_strain**2 - vorticity**2
            
            # Interpolate W to master grid
            W_grid = _interp_to_grid(W, c_lons, c_lats)
            
            # Score: We want Strain (>0). Vorticity (<0) is the core.
            # Normalize positive values.
            W_pos = np.clip(W_grid, 0, None)
            w90 = np.nanpercentile(W_pos[~land & (W_pos > 0)], 90) if np.any(W_pos > 0) else 1
            
            ow_score = np.clip(W_pos / w90, 0, 1) if w90 > 0 else np.zeros_like(W_grid)
            ow_score[land] = np.nan
            _add_score("okubo_weiss", ow_score)
            
            ow_pct = np.sum(ow_score[~np.isnan(ow_score)] > 0.3) / max(np.sum(~np.isnan(ow_score)), 1) * 100
            print(f"[Hotspots] Okubo-Weiss: {ow_pct:.0f}% of cells in strain/filament zones")
        except Exception as e:
            print(f"[Hotspots] Okubo-Weiss failed: {e}")

    # 11. Feature banding — distance-decay bands around oceanographic feature lines
    #     Instead of scoring only at exact contour cells, create ~2nm bands around
    #     SST fronts, CHL edges, SSH eddies, and bathymetry contours (100/200/500/1000m).
    #     Score decays smoothly from 1.0 at the feature to 0 at band_width distance.
    #     Where multiple bands overlap, multiplicative boost rewards convergence zones.
    try:
        _band_width_nm = getattr(sys.modules[__name__], '_opt_band_width_nm', 4.5)
        _band_boost = getattr(sys.modules[__name__], '_opt_band_boost', 0.40)
        _band_decay = getattr(sys.modules[__name__], '_opt_band_decay', 0.60)
        _band_front_thresh = getattr(sys.modules[__name__], '_opt_band_front_thresh', 0.25)
        _band_chl_thresh = getattr(sys.modules[__name__], '_opt_band_chl_thresh', 0.45)

        _grid_step_b = abs(lons[1] - lons[0]) if nx > 1 else 0.083
        # 1 nm ≈ 0.0167 degrees latitude
        band_width_deg = _band_width_nm * 0.0167
        band_width_cells = band_width_deg / _grid_step_b

        # Lat/lon aspect ratio for EDT: at ~32S, 1° lon ≈ cos(32°) × 1° lat
        _mean_lat = abs(np.mean(lats))
        _cos_lat = np.cos(np.radians(_mean_lat))
        _edt_sampling = [1.0, _cos_lat]

        # Upsample factor: ensure bands span at least 3 cells for smooth shapes
        _upsample = max(1, int(np.ceil(3.0 / max(band_width_cells, 0.1))))

        def _band_score(binary_mask, weight=1.0, width_nm=None):
            """Compute distance-decay band score from a binary feature mask.
            Upsamples internally if the grid is too coarse for the band width.
            weight: scale factor for this band type (e.g. 1.5 for 200-500m bathy).
            width_nm: override band width in nautical miles (default: global _band_width_nm)."""
            if not np.any(binary_mask):
                return np.zeros((ny, nx))

            bwc = (width_nm * 0.0167 / _grid_step_b) if width_nm else band_width_cells

            if _upsample > 1:
                from scipy.ndimage import zoom
                fine_mask = zoom(binary_mask.astype(float), _upsample, order=0) > 0.5
                fine_bwc = bwc * _upsample
                fine_dist = distance_transform_edt(~fine_mask, sampling=_edt_sampling)
                fine_norm = np.clip(fine_dist / fine_bwc, 0, 1)
                fine_band = np.clip(1.0 - fine_norm ** _band_decay, 0, 1)
                band = zoom(fine_band, 1.0 / _upsample, order=1)[:ny, :nx]
            else:
                dist = distance_transform_edt(~binary_mask, sampling=_edt_sampling)
                normalised = np.clip(dist / bwc, 0, 1)
                band = np.clip(1.0 - normalised ** _band_decay, 0, 1)

            band[land] = 0
            return band * weight

        band_layers = {}  # name -> band_score array

        # SST front band — contour-crossing at the gradient threshold.
        # Uses the same interpolation logic as the visible front lines:
        # cells where the gradient crosses 0.5 between neighbors, so the
        # band aligns exactly with the visible contour on the map.
        _thresh = SST_GRADIENT_THRESHOLD
        cross_h = (grad_mag[:, :-1] - _thresh) * (grad_mag[:, 1:] - _thresh) <= 0
        cross_v = (grad_mag[:-1, :] - _thresh) * (grad_mag[1:, :] - _thresh) <= 0
        sst_front_mask = np.zeros((ny, nx), dtype=bool)
        sst_front_mask[:, :-1] |= cross_h
        sst_front_mask[:, 1:] |= cross_h
        sst_front_mask[:-1, :] |= cross_v
        sst_front_mask[1:, :] |= cross_v
        # Also include cells above threshold (the core of the front)
        sst_front_mask |= (grad_mag > _thresh)
        sst_front_mask &= ~coast_buf & ~land
        if np.any(sst_front_mask):
            band_layers["sst_front"] = _band_score(sst_front_mask)

        # Isotherm bands — contour-crossing detection for a true 1-pixel line
        # matching the visible isotherm on the map.  The old temp-slab approach
        # (|SST - temp| < 0.3) covered 9-34% of ocean in gentle gradients.
        try:
            iso_smooth = gaussian_filter(sst_filled, sigma=0.5)
            for iso_temp in [22, 23, 24]:
                # Cell is on the contour where SST crosses the target between neighbors
                cross_h = (iso_smooth[:, :-1] - iso_temp) * (iso_smooth[:, 1:] - iso_temp) <= 0
                cross_v = (iso_smooth[:-1, :] - iso_temp) * (iso_smooth[1:, :] - iso_temp) <= 0
                cross_mask = np.zeros((ny, nx), dtype=bool)
                cross_mask[:, :-1] |= cross_h
                cross_mask[:, 1:] |= cross_h
                cross_mask[:-1, :] |= cross_v
                cross_mask[1:, :] |= cross_v
                cross_mask &= ~land & ~coast_buf
                if np.any(cross_mask):
                    # Isotherms have weak catch proximity (median 16-49nm)
                    # vs SST fronts (2.92nm). Downweight to 0.3.
                    band_layers[f"isotherm_{iso_temp}C"] = _band_score(cross_mask, weight=0.3)
        except Exception:
            pass

        # CHL edge band — gradient-only.  The contour tolerance approach
        # (4 levels x 30% tolerance) covered 67-78% of ocean — not selective.
        # Gradient-only fires where CHL changes sharply (~10-15% coverage).
        if chl_grid is not None:
            try:
                chl_for_grad = np.log10(np.clip(chl_grid, 0.01, 10))
                chl_for_grad[np.isnan(chl_for_grad) | land] = np.nanmean(chl_for_grad[~land])
                chl_smooth = gaussian_filter(chl_for_grad, sigma=1.5)
                cgx = sobel(chl_smooth, axis=1)
                cgy = sobel(chl_smooth, axis=0)
                chl_grad = np.sqrt(cgx**2 + cgy**2)
                chl_grad[coast_buf] = 0
                cg90 = np.nanpercentile(chl_grad[~coast_buf & ~land], 90)
                chl_edge_mask = (chl_grad / cg90 > _band_chl_thresh) & ~coast_buf & ~land if cg90 > 0 else np.zeros_like(land)
                if np.any(chl_edge_mask):
                    band_layers["chl_edge"] = _band_score(chl_edge_mask)
            except Exception:
                pass

        # 0.15 mg/m3 CHL contour — the catch clustering boundary.
        # 50% of blue marlin catches at 0.10-0.15 mg/m3 (median 0.142).
        # This marks the oligotrophic/mesotrophic shelf-edge transition
        # where bait concentrates. Only 4-10% coverage — genuinely selective.
        if chl_grid is not None:
            try:
                chl_for_contour = gaussian_filter(
                    np.where(np.isnan(chl_grid) | land, 0, chl_grid), sigma=0.8)
                cross_h = (chl_for_contour[:, :-1] - 0.15) * (chl_for_contour[:, 1:] - 0.15) <= 0
                cross_v = (chl_for_contour[:-1, :] - 0.15) * (chl_for_contour[1:, :] - 0.15) <= 0
                chl_cross = np.zeros((ny, nx), dtype=bool)
                chl_cross[:, :-1] |= cross_h
                chl_cross[:, 1:] |= cross_h
                chl_cross[:-1, :] |= cross_v
                chl_cross[1:, :] |= cross_v
                chl_cross &= ~land & ~coast_buf
                if np.any(chl_cross):
                    band_layers["chl_015"] = _band_score(chl_cross)
            except Exception:
                pass

        # MLD edge band — REMOVED.  At 0.083 deg resolution the MLD field is
        # already smooth (model output, not observed), so contours represent
        # broad transitions rather than sharp thermocline edges.  The band
        # covered 66-73% of ocean cells — not selective.

        # SLA contour bands — contour-crossing only (no erosion edges).
        # Cold eddy REMOVED: SLA is always positive in Perth Canyon (Leeuwin
        # Current), so "cold_eddy" at -0.03m never exists — the band was
        # detecting coastline artifacts and covering 45-47% of ocean.
        # Warm eddy kept at positive SLA levels only (11-15% coverage).
        if ssh_grid is not None:
            try:
                ssh_for_grad = ssh_grid.copy()
                ssh_for_grad[np.isnan(ssh_for_grad) | land] = np.nanmean(ssh_grid[~land])
                ssh_smooth_we = gaussian_filter(ssh_for_grad, sigma=1.0)

                # Contour-crossing masks at positive SLA levels only
                warm_contour = np.zeros((ny, nx), dtype=bool)
                for level, _, _ in SLA_LEVELS:
                    if level < 0:
                        continue  # skip negative levels — no cold eddies here
                    cross_h = (ssh_smooth_we[:, :-1] - level) * (ssh_smooth_we[:, 1:] - level) <= 0
                    cross_v = (ssh_smooth_we[:-1, :] - level) * (ssh_smooth_we[1:, :] - level) <= 0
                    cross = np.zeros((ny, nx), dtype=bool)
                    cross[:, :-1] |= cross_h
                    cross[:, 1:] |= cross_h
                    cross[:-1, :] |= cross_v
                    cross[1:, :] |= cross_v
                    warm_contour |= cross & ~land & ~coast_buf
                if np.any(warm_contour):
                    band_layers["warm_eddy"] = _band_score(warm_contour)

            except Exception:
                pass

        # SSTA edge band — REMOVED. No discriminative value for blue marlin.
        # 69% of catches in cool anomaly, warm intrusion edge is not a fish feature.

        # Bathymetry contour bands — weighted by catch proximity analysis.
        # Proximity ranking (median nm to nearest contour, % within 3nm):
        #   200m: 1.51nm, 67% | 400m: 2.33nm, 59% | 300m: 2.45nm, 53%
        #   500m: 2.57nm, 65% | 600m: 2.95nm, 57% | 700m: 3.27nm, 41%
        #   800m: 3.69nm, 33% | 900m: 4.22nm, 31% | 1000m: 4.55nm, 29%
        #   100m: 6.95nm, 10% (too shallow — catches are offshore of shelf)
        # Only 200m and 500m exceed 0.3 threshold for band-overlap counting.
        # Intermediate contours add gentle score without inflating band count
        # in deep water (which pulls peaks seaward of catches).
        _bw200 = getattr(sys.modules[__name__], '_opt_bathy_w_200', 1.0)
        _bw500 = getattr(sys.modules[__name__], '_opt_bathy_w_500', 0.7)
        _bathy_band_weights = {
            100: 0.1,
            200: _bw200,
            300: 0.25,
            400: 0.25,
            500: _bw500,
            600: 0.2,
            700: 0.15,
            800: 0.1,
            900: 0.1,
            1000: 0.1,
        }
        if tif_path and 'bathy' in dir():
            try:
                depth_master = _interp_to_grid(
                    np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats
                )
                _depth_grid = depth_master  # make available for hover info
                _band_shore_ratio = getattr(sys.modules[__name__], '_opt_band_shore_ratio', 0.30)
                _band_deep_ratio = getattr(sys.modules[__name__], '_opt_band_deep_ratio', 0.30)
                _shallow_cut = getattr(sys.modules[__name__], '_opt_shallow_cut', 0.65)
                for depth_m, bw in _bathy_band_weights.items():
                    # Symmetric tolerances — catches sit on shoreward side of contours
                    tol_shore = max(20, depth_m * _band_shore_ratio)
                    tol_deep = max(30, depth_m * _band_deep_ratio)
                    contour_mask = ((depth_master >= depth_m - tol_shore) &
                                    (depth_master <= depth_m + tol_deep) & ~land)
                    if np.any(contour_mask):
                        bathy_band = _band_score(contour_mask, weight=bw)
                        # Soft taper instead of hard zero for shoreward suppression
                        shallow_taper_depth = depth_m * _shallow_cut
                        shallow_taper = np.clip(
                            (depth_master - shallow_taper_depth) / max(depth_m * 0.15, 10), 0, 1)
                        bathy_band *= shallow_taper
                        band_layers[f"bathy_{depth_m}m"] = bathy_band
            except Exception as e:
                print(f"[Hotspots] Bathy contour banding failed: {e}")

        # FAD proximity band — small boost near known FAD buoy positions
        # Catches cluster near FADs; this gives a mild nudge to nearby cells.
        _fad_positions = [
            # Perth Canyon group
            (115.3333, -32.05),    # Club Marine (PC48)
            (115.2333, -32.0),     # PGFC (PC47)
            (115.2667, -31.9667),  # FURUNO (PC46)
            (115.2, -31.9167),     # Compleat Angler (PC45)
            # Fremantle
            (115.1833, -32.0833),  # Fremantle Sailing Club
            # Woodman Pt group
            (115.1600, -32.1130),  # Fibrelite Boats (WP06)
            (115.1210, -32.1130),  # Woodman Pt 05
            # Rockingham / Mandurah
            (115.1217, -32.2479),  # Rockingham 08
            (115.0736, -32.3585),  # Rockingham 07
            (115.0700, -32.5260),  # Mandurah 09
            (115.0287, -32.6274),  # Mandurah 10
            # North Metro group
            (115.1754, -31.7287),  # North Metro 04
            (115.1752, -31.7003),  # North Metro 03
            (115.1333, -31.6999),  # North Metro 02
            (115.1332, -31.6171),  # North Metro 01
        ]
        try:
            fad_mask = np.zeros((ny, nx), dtype=bool)
            for flon, flat in _fad_positions:
                ci = np.argmin(np.abs(lons - flon))
                ri = np.argmin(np.abs(lats - flat))
                if 0 <= ri < ny and 0 <= ci < nx:
                    fad_mask[ri, ci] = True
            if np.any(fad_mask):
                band_layers["fad"] = _band_score(fad_mask, weight=0.4, width_nm=1.0)
        except Exception:
            pass

        # Band overlap — purely multiplicative (no additive sub-score to avoid double-counting)
        # Bands only boost cells that already score well from oceanographic fundamentals.
        if band_layers:
            band_stack = np.array(list(band_layers.values()))
            band_sum = np.sum(band_stack, axis=0)
            # Only count cells with substantial band presence (>0.3 intensity)
            # to avoid the smooth band edges inflating overlap counts everywhere
            band_count = np.sum(band_stack > 0.3, axis=0).astype(float)
            counted_sum = np.sum(np.where(band_stack > 0.3, band_stack, 0), axis=0)
            mean_band = np.where(band_count > 0, counted_sum / np.maximum(band_count, 1), 0)

            # Store for multiplicative boost applied post-normalization
            _feature_band_count = band_count
            _feature_band_mean = mean_band

            # Floor boost for key feature lines — cells on these lines get
            # lifted to at least 0.62 so they always create visible zones.
            # SST front: broader mask (abs OR rel gradient) for floor only.
            # CHL 0.15: the catch-clustering boundary (4-10% coverage).
            _key_feature_floor = getattr(sys.modules[__name__], '_opt_key_feature_floor', 0.40)
            key_floor = np.zeros((ny, nx))
            # SST front floor (broad mask)
            _rel_floor_mask = (grad_mag / g90 > _band_front_thresh) & ~coast_buf & ~land if g90 > 0 else sst_front_mask
            broad_front_mask = sst_front_mask | _rel_floor_mask
            if np.any(broad_front_mask):
                broad_front_band = _band_score(broad_front_mask)
                key_floor = np.maximum(key_floor, np.where(broad_front_band > 0.5, _key_feature_floor, 0))
            # CHL 0.15 contour floor
            if "chl_015" in band_layers:
                key_floor = np.maximum(key_floor, np.where(band_layers["chl_015"] > 0.5, _key_feature_floor, 0))
            _feature_key_floor = key_floor

            n_bands = len(band_layers)
            multi2 = np.sum(band_count[~land] >= 2) / max(np.sum(~land), 1) * 100
            multi3 = np.sum(band_count[~land] >= 3) / max(np.sum(~land), 1) * 100
            print(f"[Hotspots] Feature bands: {n_bands} types, {multi2:.0f}% cells in 2+ bands, {multi3:.0f}% in 3+ bands")
        else:
            _feature_band_count = np.zeros((ny, nx))
            _feature_band_mean = np.zeros((ny, nx))
            _feature_key_floor = np.zeros((ny, nx))
    except Exception as e:
        _feature_band_count = np.zeros((ny, nx))
        _feature_band_mean = np.zeros((ny, nx))
        _feature_key_floor = np.zeros((ny, nx))
        print(f"[Hotspots] Feature banding failed: {e}")

    # --- Normalize by actual weights used (handles missing data gracefully) ---
    valid = weight_sum > 0
    final = np.full((ny, nx), np.nan)
    final[valid] = score[valid] / weight_sum[valid]
    final[land] = np.nan

    # --- Profile / Hybrid scoring modes ---
    _scoring_mode = getattr(sys.modules[__name__], '_scoring_mode', 'weighted_sum')
    _default_profile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'catch_profile.npz')
    _profile_path = getattr(sys.modules[__name__], '_profile_path', _default_profile)
    if _scoring_mode in ('profile', 'hybrid') and _profile_path:
        try:
            profile = np.load(_profile_path, allow_pickle=True)
            prof_mu = profile['mean']
            prof_inv_cov = profile['inv_cov']
            prof_features = list(profile['feature_names'])
            n_feat = len(prof_features)

            feat_matrix = np.zeros((ny * nx, n_feat))
            feat_available = np.ones(ny * nx, dtype=bool)
            for fi, fname in enumerate(prof_features):
                if fname in sub_scores and isinstance(sub_scores[fname], np.ndarray):
                    vals = sub_scores[fname].ravel().copy()
                    nan_mask = np.isnan(vals)
                    vals[nan_mask] = 0.0
                    feat_available &= ~nan_mask
                    feat_matrix[:, fi] = vals
                else:
                    feat_matrix[:, fi] = prof_mu[fi]

            diff = feat_matrix - prof_mu
            mahal_sq = np.sum(diff @ prof_inv_cov * diff, axis=1)
            mahal_score = np.exp(-0.5 * mahal_sq / n_feat).reshape((ny, nx))

            ocean_valid = valid & ~land & feat_available.reshape((ny, nx))
            ocean_mahal = mahal_score[ocean_valid]
            p90_m = np.nanpercentile(ocean_mahal, 90)

            print(f"[Hotspots] Profile scoring: {n_feat} features, "
                  f"Mahalanobis d^2 range: {np.min(mahal_sq[feat_available]):.1f} - "
                  f"{np.percentile(mahal_sq[feat_available], 99):.1f}")

            if _scoring_mode == 'profile':
                # Pure profile: replace weighted sum entirely
                final[:] = np.nan
                if p90_m > 0:
                    final[ocean_valid] = np.clip(mahal_score[ocean_valid] / p90_m, 0, 1)
                else:
                    final[ocean_valid] = mahal_score[ocean_valid]
                _profile_pure = getattr(sys.modules[__name__], '_profile_pure', False)
                if _profile_pure:
                    if "depth" in sub_scores:
                        depth_mult = sub_scores["depth"]
                        dmask = ~np.isnan(depth_mult) & valid
                        final[dmask] *= depth_mult[dmask]
                    print("[Hotspots] PURE PROFILE: skipping all post-processing")
                    # Skip to output
                    final[land] = np.nan
            elif _scoring_mode == 'hybrid':
                # Hybrid: geometric mean of weighted sum and profile similarity
                ws_score = final.copy()
                prof_rescaled = np.full((ny, nx), np.nan)
                if p90_m > 0:
                    prof_rescaled[ocean_valid] = np.clip(mahal_score[ocean_valid] / p90_m, 0, 1)
                else:
                    prof_rescaled[ocean_valid] = mahal_score[ocean_valid]
                # Geometric mean: sqrt(weighted_sum * profile)
                hmask = ocean_valid & ~np.isnan(ws_score)
                final[hmask] = np.sqrt(np.clip(ws_score[hmask], 0, 1) * np.clip(prof_rescaled[hmask], 0, 1))
                print(f"[Hotspots] HYBRID: geometric mean of weighted sum and profile")
        except Exception as e:
            print(f"[Hotspots] Profile/hybrid scoring failed ({e}), using weighted sum")

    _skip_post_processing = (_scoring_mode == 'profile' and
                             getattr(sys.modules[__name__], '_profile_pure', False))

    # --- Gamma contrast: compress the bloated 0.8-1.0 range so feature band
    # multipliers have headroom to selectively boost cells near oceanographic
    # feature lines.  gamma > 1 compresses high scores; 1.0 = no change.
    # Apply static multipliers: depth gates, shelf break boosts
    if "depth" in sub_scores and not _skip_post_processing:
        depth_mult = sub_scores["depth"]
        dmask = ~np.isnan(depth_mult) & valid
        final[dmask] *= depth_mult[dmask]  # zero out shallow water
    if "shelf_break" in sub_scores and not _skip_post_processing:
        # Multiplicative component of hybrid shelf scoring (additive part already in weighted sum)
        _shelf_boost = getattr(sys.modules[__name__], '_opt_shelf_boost', 0.12)
        shelf_mult = 1.0 + _shelf_boost * sub_scores["shelf_break"]
        smask = ~np.isnan(shelf_mult) & valid
        final[smask] *= shelf_mult[smask]
        final = np.clip(final, 0, 1)  # cap at 1.0

    # SST x shelf_break interaction — catches happen where warm water meets the
    # shelf edge. Neither feature alone is sufficient: warm water offshore scores
    # high on SST but has no shelf; shelf inshore scores high on shelf but may be
    # cool. The interaction rewards cells where BOTH are strong.
    _sst_shelf_interact = getattr(sys.modules[__name__], '_opt_sst_shelf_interact', 0.0)
    if _sst_shelf_interact > 0 and "sst" in sub_scores and "shelf_break" in sub_scores and not _skip_post_processing:
        sst_s = sub_scores["sst"]
        shelf_s = sub_scores["shelf_break"]
        # Geometric mean: high only when both are high
        interact = np.sqrt(np.clip(sst_s, 0, 1) * np.clip(shelf_s, 0, 1))
        interact_mult = 1.0 + _sst_shelf_interact * interact
        imask = ~np.isnan(interact) & valid
        final[imask] *= interact_mult[imask]
        final = np.clip(final, 0, 1)

    # Multi-feature edge overlap — catches have 24% more features simultaneously
    # transitioning than score peaks (3.81 vs 3.07). Marlin hunt where multiple
    # oceanographic features are all changing at once (convergence of gradients),
    # not where any single feature peaks. Boost pixels where 3+ sub_scores have
    # above-average gradient magnitude.
    if not _skip_post_processing:
      try:
        edge_features = ["sst_front", "convergence", "current_shear", "upwelling_edge",
                         "chl_curvature", "current", "mld", "shelf_break", "front_corridor"]
        edge_count = np.zeros((ny, nx))
        for feat in edge_features:
            if feat not in sub_scores or not isinstance(sub_scores[feat], np.ndarray):
                continue
            fg = sub_scores[feat].copy()
            fg[np.isnan(fg)] = 0
            fg_smooth = gaussian_filter(fg, sigma=1.0)
            gy, gx = np.gradient(fg_smooth)
            gmag = np.sqrt(gx**2 + gy**2)
            gmag[land | coast_buf] = 0
            g75 = np.nanpercentile(gmag[~land & ~coast_buf], 75) if np.any(~land & ~coast_buf) else 0
            if g75 > 0:
                edge_count += (gmag > g75).astype(float)
        _edge_bs = getattr(sys.modules[__name__], '_edge_boost_strength', 0.05)
        edge_mult = np.clip(1.0 + _edge_bs * (edge_count - 2), 1.0, 1.0 + _edge_bs * 3)
        final[valid] *= edge_mult[valid]
        final = np.clip(final, 0, 1)
        ec3 = np.sum(edge_count[~land] >= 3) / max(np.sum(~land), 1) * 100
        ec4 = np.sum(edge_count[~land] >= 4) / max(np.sum(~land), 1) * 100
        print(f"[Hotspots] Multi-feature edges: {ec3:.0f}% with 3+ transitions, {ec4:.0f}% with 4+")
      except Exception as e:
        print(f"[Hotspots] Multi-feature edge scoring failed: {e}")

    # Lunar phase modifier — habitat compression via Diel Vertical Migration
    # New moon: DSL rises higher, MLD shallower (17m), bait compressed at surface
    # Full moon: DSL stays deep, MLD deeper (23m), bait dispersed vertically
    # Effect: new moon = ~5% boost, full moon = neutral (no penalty)
    # Lunar phase 0 = new moon, 0.5 = full moon
    if date_str and not _skip_post_processing:
        try:
            from math import sin, pi
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Approximate lunar phase using synodic month (29.53 days)
            # Reference: 2000-01-06 was a new moon
            ref_new = datetime(2000, 1, 6)
            days_since = (dt - ref_new).days
            lunar_cycle = (days_since % 29.53) / 29.53  # 0=new, 0.5=full
            # Moon illumination: 0 at new, 1 at full
            moon_illum = 0.5 * (1 - np.cos(2 * pi * lunar_cycle))
            # Habitat compression bonus: strongest at new moon (illum=0)
            # Range: 1.0 (full moon) to 1.05 (new moon)
            _lunar_mod = getattr(sys.modules[__name__], '_opt_lunar_boost', 0.10)
            lunar_boost = 1.0 + _lunar_mod * (1.0 - moon_illum)
            final[valid] *= lunar_boost
            final = np.clip(final, 0, 1)
            phase_name = "New" if moon_illum < 0.25 else "Full" if moon_illum > 0.75 else "Quarter"
            print(f"[Hotspots] Lunar phase: {phase_name} ({moon_illum:.0%} illum) -> x{lunar_boost:.3f} habitat compression boost")
        except Exception:
            pass

    # Feature band boost — graduated multiplier based on band count.
    # Cells with more overlapping feature bands score progressively higher.
    # Cells with 0 bands get suppressed; 1 band gets mild suppression;
    # 2+ bands get a boost. This creates a gradient instead of a cliff.
    if np.any(_feature_band_count > 0) and not _skip_post_processing:
        _band_single = getattr(sys.modules[__name__], '_opt_band_single', 0.06)
        _band_overlap = getattr(sys.modules[__name__], '_opt_band_overlap', 0.20)
        # Graduated band multiplier — smooth ramp from 0 bands to 3+
        # Softened: catches average 2.19 bands vs peaks at 2.85. Old penalties
        # (0-band x0.65, 1-band x0.85) pushed zones offshore where bands converge.
        # 0 bands: ×0.75 (25% penalty — still penalize featureless water)
        # 1 band:  ×0.90 (10% penalty — legitimate edge, don't penalize hard)
        # 2 bands: ×1.0 + boost (baseline)
        # 3+ bands: ×1.0 + larger boost (convergence zone)
        _zero_bm = getattr(sys.modules[__name__], '_opt_zero_band_mult', 0.55)
        _one_bm = getattr(sys.modules[__name__], '_opt_one_band_mult', 0.80)
        band_base_mult = np.where(
            _feature_band_count < 0.1, _zero_bm,
            np.where(_feature_band_count < 1.5, _one_bm,
                     1.0))
        # Additive boost for 2+ bands (convergence)
        extra = np.clip(_feature_band_count - 2, 0, None)
        band_mult = (band_base_mult
                     + _band_single * np.clip(_feature_band_count - 1, 0, None) * _feature_band_mean
                     + _band_overlap * extra * _feature_band_mean)
        final[valid] *= band_mult[valid]
        # Don't clip to 1.0 here — allow scores >1.0 so band-boosted cells
        # stand out visually. Will be rescaled to [0,1] before contouring.
        ov1 = np.sum(_feature_band_count[valid & ~land] >= 1) / max(np.sum(valid & ~land), 1) * 100
        ov2 = np.sum(_feature_band_count[valid & ~land] >= 2) / max(np.sum(valid & ~land), 1) * 100
        ov3 = np.sum(_feature_band_count[valid & ~land] >= 3) / max(np.sum(valid & ~land), 1) * 100
        print(f"[Hotspots] Band boost: {ov1:.0f}% 1+ bands, {ov2:.0f}% 2+ bands, {ov3:.0f}% 3+ bands")

    # Rescale: band boost can push scores above 1.0. Compress cells
    # above 1.0 back into [0.75, 1.0] range — wide enough to preserve
    # discrimination between 2-band and 3+-band cells.
    # Cells at or below 1.0 keep their original scores.
    raw_max = float(np.nanmax(final[valid & ~land])) if np.any(valid & ~land) else 1.0
    if raw_max > 1.0 and not _skip_post_processing:
        above = final > 1.0
        if np.any(above):
            # Linear map [1.0, raw_max] -> [0.75, 1.0]
            final[above] = 0.75 + 0.25 * (final[above] - 1.0) / (raw_max - 1.0)
        final = np.clip(final, 0, 1)
        print(f"[Hotspots] Band boost pushed max to {raw_max:.2f}, top-only rescale [1.0-{raw_max:.2f}] -> [0.75-1.0]")

    # Score-gradient reward — catches happen at transition zones (~87% of
    # local peak, steep E-W gradient at shelf edge).  Boost cells where the
    # combined score landscape is changing rapidly AND the score is already
    # moderate-to-high (>0.5).  This lifts the edges of good zones without
    # inflating featureless water.
    _score_grad_blend = getattr(sys.modules[__name__], '_opt_score_grad_blend', 0.20)
    if _score_grad_blend > 0 and not _skip_post_processing:
        try:
            score_for_grad = np.where(np.isnan(final), 0, final)
            score_smooth = gaussian_filter(score_for_grad, sigma=1.5)
            gy, gx = np.gradient(score_smooth)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mag[land] = 0
            # Normalize to [0, 1] using 99th percentile
            ocean_grad = grad_mag[~land & valid]
            g99 = np.nanpercentile(ocean_grad, 99) if len(ocean_grad) > 0 else 1
            grad_norm = np.clip(grad_mag / max(g99, 1e-6), 0, 1)
            # Only reward transitions within decent-scoring zones (>0.5)
            zone_mask = score_for_grad > 0.5
            grad_reward = np.where(zone_mask, grad_norm, 0)
            # Multiplicative boost: transition zones get up to +blend boost
            grad_mult = 1.0 + _score_grad_blend * grad_reward
            final[valid] *= grad_mult[valid]
            final = np.clip(final, 0, 1)
            pct_boosted = np.sum(grad_reward[~land & valid] > 0.3) / max(np.sum(~land & valid), 1) * 100
            print(f"[Hotspots] Score-gradient reward: {pct_boosted:.0f}% of ocean cells boosted (blend={_score_grad_blend})")
        except Exception as e:
            print(f"[Hotspots] Score-gradient reward failed: {e}")

    # Floor boost for SST front lines — cells on the front are lifted
    # to at least _key_feature_floor (0.55).  Uses np.maximum so cells
    # already above the floor are untouched — no clipping, no inflation.
    if np.any(_feature_key_floor > 0) and not _skip_post_processing:
        before = final.copy()
        final[valid] = np.maximum(final[valid], _feature_key_floor[valid])
        n_lifted = np.sum((final[valid & ~land] > before[valid & ~land]))
        n_front = np.sum(_feature_key_floor[valid & ~land] > 0)
        n_ocean = max(np.sum(valid & ~land), 1)
        print(f"[Hotspots] Feature line floor: {n_lifted}/{n_front} cells lifted to >=0.62 ({n_front}/{n_ocean} = {n_front/n_ocean*100:.0f}% on feature lines)")

    # Spatial smoothing — sigma scales with grid resolution
    # Target ~1nm physical smoothing: just enough to connect pixels into contours
    # without flattening the spatial gradients we need for sub-zone detail
    _grid_step = abs(lons[1] - lons[0]) if nx > 1 else 0.083
    _smooth_sigma = max(0.6, 0.015 / _grid_step)  # ~1nm — softens polygon edges for gradient look
    final_filled = final.copy()
    final_filled[np.isnan(final_filled)] = 0
    final_smooth = gaussian_filter(final_filled, sigma=_smooth_sigma)
    final_smooth[land | ~valid] = np.nan

    fmin = float(np.nanmin(final_smooth[~land & valid]))
    fmax = float(np.nanmax(final_smooth[~land & valid]))
    fmean = float(np.nanmean(final_smooth[~land & valid]))
    print(f"[Hotspots] Score range: {fmin:.3f} - {fmax:.3f} (mean {fmean:.3f})")

    # --- Build depth mask for clipping to >100m ---
    from shapely.geometry import Polygon as ShapelyPolygon, mapping
    clip_mask = None
    if tif_path and os.path.exists(tif_path):
        try:
            clip_mask = build_deep_water_mask(tif_path, depth_threshold=-50)
        except Exception:
            pass

    # --- Helper to sample mean sub-scores within a polygon's bounding box ---
    def _sample_scores(coords_list):
        """Return (actual_intensity, sub_scores_dict) for the polygon area.
        Uses point-in-polygon masking instead of bounding box to avoid
        diluting scores with cells outside the polygon."""
        from matplotlib.path import Path
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]
        lon_min, lon_max = min(xs), max(xs)
        lat_min, lat_max = min(ys), max(ys)
        col_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        row_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        if len(col_idx) == 0 or len(row_idx) == 0:
            return 0.0, {}

        # Build point-in-polygon mask for the bbox subset
        sub_lons = lons[col_idx]
        sub_lats = lats[row_idx]
        mesh_lon, mesh_lat = np.meshgrid(sub_lons, sub_lats)
        points = np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()])
        poly_path = Path([(c[0], c[1]) for c in coords_list])
        inside = poly_path.contains_points(points).reshape(mesh_lon.shape)

        # Actual composite score from the smoothed grid
        region_composite = final_smooth[np.ix_(row_idx, col_idx)]
        valid_composite = region_composite[inside & ~np.isnan(region_composite)]
        actual_intensity = round(float(np.mean(valid_composite)), 2) if len(valid_composite) > 0 else 0.0
        result = {}
        for name, arr in sub_scores.items():
            region = arr[np.ix_(row_idx, col_idx)]
            valid = region[inside & ~np.isnan(region)]
            if len(valid) > 0:
                w = BLUE_MARLIN_WEIGHTS.get(name, 0)
                mean_score = round(float(np.mean(valid)), 2)
                if name == "depth":
                    result[name] = {"score": mean_score, "weight": -1}
                elif name == "shelf_break":
                    _sb = getattr(sys.modules[__name__], '_opt_shelf_boost', 0.12)
                    result[name] = {"score": round(1.0 + _sb * mean_score, 2), "weight": w}
                else:
                    result[name] = {"score": mean_score, "weight": w}
        if _depth_grid is not None:
            depth_region = _depth_grid[np.ix_(row_idx, col_idx)]
            depth_valid = depth_region[inside & ~np.isnan(depth_region) & (depth_region > 0)]
            if len(depth_valid) > 0:
                result["depth_m"] = {"score": round(float(np.mean(depth_valid))), "weight": -3}

        bc_region = _feature_band_count[np.ix_(row_idx, col_idx)]
        bc_valid = bc_region[inside & ~np.isnan(bc_region)]
        if len(bc_valid) > 0:
            result["bands"] = {"score": round(float(np.mean(bc_valid)), 1), "weight": -4}

        return actual_intensity, result

    # --- Export as filled contour polygons with intensity bands ---
    # Fill NaN with 0 for contourf
    plot_data = final_smooth.copy()
    plot_data[np.isnan(plot_data)] = 0

    levels = [0] + HOTSPOT_BANDS + [1.0]
    fig, ax = plt.subplots()
    cf = ax.contourf(lons, lats, plot_data, levels=levels, extend="neither")
    plt.close(fig)

    # Build filled polygons per band, then subtract higher bands for non-overlapping rings
    from shapely.ops import unary_union
    band_polys = {}
    for band_idx, seg_list in enumerate(cf.allsegs):
        if band_idx == 0:
            continue
        parts = []
        for seg in seg_list:
            if len(seg) < 4:
                continue
            coords = [(round(float(x), 4), round(float(y), 4)) for x, y in seg]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                p = ShapelyPolygon(coords).buffer(0)
                if not p.is_empty and p.area > 0:
                    parts.append(p)
            except Exception:
                pass
        if parts:
            band_polys[band_idx] = unary_union(parts)

    features = []
    for band_idx in sorted(band_polys.keys()):
        intensity = round(levels[band_idx], 2)
        band_label = f"{levels[band_idx]:.0%}–{levels[band_idx+1]:.0%}" if band_idx + 1 < len(levels) else f">{levels[band_idx]:.0%}"

        ring = band_polys[band_idx]
        for higher_idx in sorted(band_polys.keys()):
            if higher_idx > band_idx:
                try:
                    ring = ring.difference(band_polys[higher_idx])
                except Exception:
                    pass
        if ring.is_empty:
            continue
        if clip_mask is not None:
            try:
                ring = ring.intersection(clip_mask)
            except Exception:
                pass
        if ring.is_empty:
            continue

        # Sample sub-scores from ring's representative point
        rep = ring.representative_point()
        sample_coords = [[rep.x - 0.01, rep.y - 0.01], [rep.x + 0.01, rep.y - 0.01],
                         [rep.x + 0.01, rep.y + 0.01], [rep.x - 0.01, rep.y + 0.01],
                         [rep.x - 0.01, rep.y - 0.01]]
        _, breakdown = _sample_scores(sample_coords)

        props = {
            "species": "blue",
            "type": "hotspot",
            "intensity": intensity,
            "band": band_label,
        }
        for name, info in breakdown.items():
            props[f"s_{name}"] = info["score"]
            props[f"w_{name}"] = info["weight"]

        geom = mapping(ring)
        if geom["type"] == "Polygon":
            features.append({"type": "Feature", "geometry": geom, "properties": props})
        elif geom["type"] == "MultiPolygon":
            for mc in geom["coordinates"]:
                features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": mc}, "properties": props})

    geojson = {"type": "FeatureCollection", "features": features}
    output_path = os.path.join(OUTPUT_DIR, "blue_marlin_hotspots.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[Hotspots] {len(features)} polygons across {len(HOTSPOT_BANDS)} bands ->{output_path}")

    # Export depth grid as point GeoJSON for per-pixel hover queries.
    # Each grid cell becomes a point at its centre carrying the depth value.
    if _depth_grid is not None:
        depth_pts = []
        for ri in range(ny):
            for ci in range(nx):
                d = _depth_grid[ri, ci]
                if np.isnan(d) or d <= 0:
                    continue
                depth_pts.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(float(lons[ci]), 4), round(float(lats[ri]), 4)]},
                    "properties": {"d": round(float(d))},
                })
        if depth_pts:
            depth_geojson = {"type": "FeatureCollection", "features": depth_pts}
            depth_path = os.path.join(OUTPUT_DIR, "depth_grid.geojson")
            with open(depth_path, "w") as f:
                json.dump(depth_geojson, f)
            print(f"[Hotspots] Depth grid: {len(depth_pts)} points ->{depth_path}")

    # ---- Export invisible scoring features as GeoJSON overlays ----
    # These features contribute to the score but have no visible map overlay.
    # Generating contour lines lets the user see what the model "sees".

    _overlay_configs = [
        # (score_key, filename, contour_levels, type_name, labels, colors)
        ("front_corridor", "front_corridors.geojson",
         [0.3, 0.5, 0.7],
         "front_corridor",
         ["moderate", "strong", "intense"],
         ["#fb923c", "#f97316", "#ea580c"]),   # orange tones
        ("current_shear", "shear_zones.geojson",
         [0.3, 0.5, 0.7],
         "current_shear",
         ["moderate", "strong", "intense"],
         ["#38bdf8", "#0ea5e9", "#0284c7"]),   # sky blue tones
        ("upwelling_edge", "upwelling_edges.geojson",
         [0.3, 0.5],
         "upwelling_edge",
         ["moderate", "strong"],
         ["#2dd4bf", "#0d9488"]),               # teal tones
        ("chl_curvature", "chl_curvature.geojson",
         [0.4, 0.6],
         "chl_curvature",
         ["moderate", "strong"],
         ["#86efac", "#22c55e"]),               # green tones
        ("salinity_front", "salinity_fronts.geojson",
         [0.3, 0.5, 0.7],
         "salinity_front",
         ["moderate", "strong", "intense"],
         ["#c084fc", "#a855f7", "#7c3aed"]),   # purple tones
        ("ftle", "ftle_ridges.geojson",
         [0.3, 0.5],
         "ftle",
         ["moderate", "strong"],
         ["#fca5a5", "#ef4444"]),               # red tones
        ("vertical_velocity", "vertical_velocity.geojson",
         [0.3, 0.5],
         "vertical_velocity",
         ["moderate", "strong"],
         ["#93c5fd", "#3b82f6"]),               # blue tones
        ("okubo_weiss", "okubo_weiss.geojson",
         [0.3, 0.5],
         "okubo_weiss",
         ["moderate", "strong"],
         ["#fde68a", "#f59e0b"]),               # amber tones
        ("sst_chl_bivariate", "bivariate.geojson",
         [0.4, 0.6],
         "sst_chl_bivariate",
         ["moderate", "strong"],
         ["#67e8f9", "#06b6d4"]),               # cyan tones
    ]

    for score_key, fname, levels, type_name, labels, colors in _overlay_configs:
        if score_key not in sub_scores:
            continue
        sgrid = sub_scores[score_key]
        if not isinstance(sgrid, np.ndarray) or np.all(np.isnan(sgrid)):
            continue
        # Fill NaN for contour extraction
        sgrid_filled = sgrid.copy()
        sgrid_filled[np.isnan(sgrid_filled)] = 0
        all_feats = []
        for lvl, lbl, clr in zip(levels, labels, colors):
            feats = _contours_to_geojson(
                sgrid_filled, lons, lats, lvl,
                properties={"type": type_name, "level": lvl, "label": lbl, "color": clr},
                min_pts=4,
            )
            all_feats.extend(feats)
        if all_feats:
            ov_geojson = {"type": "FeatureCollection", "features": all_feats}
            ov_path = os.path.join(OUTPUT_DIR, fname)
            with open(ov_path, "w") as f:
                json.dump(ov_geojson, f)
            print(f"[Hotspots] {type_name}: {len(all_feats)} contours ->{ov_path}")

    # ---- Generate feature score heatmap PNGs for map overlay ----
    _heatmap_features = [
        "sst", "chl", "ssh", "salinity_front", "current_shear",
        "chl_curvature", "okubo_weiss", "upwelling_edge", "ftle",
        "vertical_velocity", "front_corridor", "shelf_break",
        "sst_front", "sst_chl_bivariate",
    ]
    try:
        from PIL import Image
        hm_bounds = [float(lons[0]), float(lats[-1]), float(lons[-1]), float(lats[0])]
        for feat_name in _heatmap_features:
            if feat_name not in sub_scores:
                continue
            grid = sub_scores[feat_name]
            if not isinstance(grid, np.ndarray) or np.all(np.isnan(grid)):
                continue
            h, w = grid.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            valid = ~np.isnan(grid)
            vals = np.clip(grid[valid], 0, 1)
            # Colormap: blue(0) -> cyan(0.25) -> green(0.5) -> yellow(0.75) -> red(1.0)
            r = np.clip(np.where(vals < 0.5, 0, (vals - 0.5) * 2) * 255, 0, 255).astype(np.uint8)
            g = np.clip(np.where(vals < 0.5, vals * 2, 2 - vals * 2) * 255, 0, 255).astype(np.uint8)
            b = np.clip(np.where(vals < 0.5, (0.5 - vals) * 2, 0) * 255, 0, 255).astype(np.uint8)
            alpha = np.full_like(vals, 160, dtype=np.uint8)  # ~63% opacity
            rgba[valid, 0] = r
            rgba[valid, 1] = g
            rgba[valid, 2] = b
            rgba[valid, 3] = alpha
            img = Image.fromarray(rgba, 'RGBA')
            # Upscale 4x with nearest-neighbor for crisp pixels
            img = img.resize((w * 4, h * 4), Image.NEAREST)
            hm_path = os.path.join(OUTPUT_DIR, f"{feat_name}_heatmap.png")
            img.save(hm_path)
        # Save bounds metadata for the JS to use
        hm_meta = {"bounds": hm_bounds, "features": [f for f in _heatmap_features if f in sub_scores]}
        hm_meta_path = os.path.join(OUTPUT_DIR, "heatmap_meta.json")
        with open(hm_meta_path, "w") as f:
            json.dump(hm_meta, f)
    except ImportError:
        pass  # PIL not available, skip heatmaps
    except Exception as e:
        print(f"[Hotspots] Heatmap generation failed: {e}")

    return {
        "path": output_path,
        "grid": final_smooth,
        "lats": lats,
        "lons": lons,
        "sub_scores": sub_scores,
        "weights": {k: v for k, v in BLUE_MARLIN_WEIGHTS.items()},
        "band_count": _feature_band_count,
        "band_mean": _feature_band_mean,
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
    Handles both SLA (satellite altimetry) and zos (ANFC model) variables.
    For zos: subtract spatial mean to convert absolute height to pseudo-anomaly.
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    print("[SSH/Eddies] Processing SLA contours...")
    ds = xr.open_dataset(ssh_file)
    # variable may be 'sla', 'adt', or 'zos' (ANFC forecast)
    varname = None
    for v in ["sla", "adt", "zos"]:
        if v in ds:
            varname = v
            break
    if varname is None:
        print(f"[SSH/Eddies] No SSH variable found. Available: {list(ds.data_vars)}")
        return None
    sla = ds[varname].squeeze()

    lons = sla.longitude.values if "longitude" in sla.dims else sla.lon.values
    lats = sla.latitude.values if "latitude" in sla.dims else sla.lat.values
    data = sla.values.copy().astype(float)

    # Convert zos (absolute sea surface height) to pseudo-anomaly
    if varname == "zos":
        spatial_mean = float(np.nanmean(data))
        data = data - spatial_mean
        print(f"[SSH/Eddies] zos -> pseudo-anomaly (subtracted mean {spatial_mean:.3f}m)")

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
    (10,  "very_shallow", "#a78bfa"),  # violet — extreme stratification, marlin pinned to surface
    (20,  "shallow",      "#22d3ee"),  # cyan — typical summer, strong stratification
    (30,  "moderate",     "#0ea5e9"),  # blue — transitional mixing
    (50,  "deep",         "#6366f1"),  # indigo — deep MLD, weaker stratification
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
def extract_bathymetry_contours(gebco_file, depths=[-50, -100, -200, -300, -400, -500, -600, -700, -800, -900, -1000]):
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
        -200:  {"label": "200m shelf break","color": "#f59e0b"},
        -300:  {"label": "300m contour",    "color": "#e17b24"},
        -400:  {"label": "400m contour",    "color": "#d97706"},
        -500:  {"label": "500m contour",    "color": "#06b6d4"},
        -600:  {"label": "600m contour",    "color": "#0891b2"},
        -700:  {"label": "700m contour",    "color": "#0e7490"},
        -800:  {"label": "800m contour",    "color": "#155e75"},
        -900:  {"label": "900m contour",    "color": "#1e40af"},
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
def fetch_bathymetry_gmrt(bbox, depths=[-100, -150, -200, -250, -500, -1000]):
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

        # Salinity — halocline marks LC boundary when SST front is masked in summer
        if BLUE_MARLIN_WEIGHTS.get("salinity_front", 0) > 0:
            try:
                fetch_copernicus_salinity(date_str, bbox)
            except Exception as e:
                print(f"[Salinity] Fetch error: {e}")

        # Subsurface temperature at 250m — canyon upwelling + stratification index
        if BLUE_MARLIN_WEIGHTS.get("thermocline_lift", 0) > 0 or BLUE_MARLIN_WEIGHTS.get("stratification", 0) > 0:
            try:
                fetch_copernicus_subsurface_temp(date_str, bbox)
            except Exception as e:
                print(f"[SubsurfaceTemp] Fetch error: {e}")

        # Oxygen fetch removed — 0.25° too coarse, never limiting off Perth

    def _nc(name):
        """Find a NetCDF file: prefer dated dir, fall back to base dir."""
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            return p
        p2 = os.path.join(base_output, name)
        return p2 if os.path.exists(p2) else None

    # Download bathymetry FIRST so we can build the depth mask for clipping
    # Prefer high-res merged bathymetry if available, fall back to GMRT
    hires_path = os.path.join(base_output, "perth_canyon_bathy_hires.tif")
    tif_path = os.path.join(OUTPUT_DIR, "bathy_gmrt.tif")
    if os.path.exists(hires_path):
        tif_path = hires_path
        print(f"[Bathymetry] Using high-res bathymetry: {hires_path}")
    else:
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

    # Generate SSTA contours (visual only — not used in scoring)
    if sst_file:
        try:
            generate_ssta_contours(sst_file, date_str)
        except Exception as e:
            print(f"[SSTA] Error: {e}")

    # Generate blue marlin habitat hotspot heatmap
    try:
        generate_blue_marlin_hotspots(bbox, tif_path=tif_path, date_str=date_str)
    except Exception as e:
        print(f"[Hotspots] Error: {e}")

    # Generate Spanish Mackerel habitat hotspot heatmap
    try:
        from species.spanish_mackerel import generate_spanish_mackerel_hotspots
        generate_spanish_mackerel_hotspots(bbox, tif_path=tif_path, date_str=date_str)
    except Exception as e:
        print(f"[SM-Hotspots] Error: {e}")

    # Generate SBT habitat hotspot heatmap
    try:
        from species.southern_bluefin_tuna import generate_sbt_hotspots
        generate_sbt_hotspots(bbox, tif_path=tif_path, date_str=date_str)
    except Exception as e:
        print(f"[SBT-Hotspots] Error: {e}")

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

    # Oxygen contours removed — 0.25° too coarse, never limiting off Perth

    # Process currents
    cur_file = _nc("currents_raw.nc")
    if cur_file:
        try:
            process_currents(cur_file)
        except Exception as e:
            print(f"[Currents] Error: {e}")

    # Bathymetry contours — use GMRT (smooth, consistent) even when hires is
    # available for scoring. The hires merged TIF has seam artifacts that make
    # contour lines jagged. GEBCO file still takes priority if provided.
    gmrt_contour_path = os.path.join(OUTPUT_DIR, "bathy_gmrt.tif")
    if not os.path.exists(gmrt_contour_path):
        gmrt_contour_path = os.path.join(base_output, "bathy_gmrt.tif")
    if args.gebco:
        try:
            extract_contours_gdal(args.gebco)
        except Exception as e:
            print(f"[Bathymetry] GDAL failed: {e}, trying rasterio...")
            try:
                extract_bathymetry_contours(args.gebco)
            except Exception as e2:
                print(f"[Bathymetry] Error: {e2}")
    elif gmrt_contour_path and os.path.exists(gmrt_contour_path):
        try:
            extract_bathymetry_contours(gmrt_contour_path)
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
