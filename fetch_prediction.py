#!/usr/bin/env python3
"""
fetch_prediction.py — Fetch ocean data for marlin prediction.

Downloads 8 days of recent observations + 7 days of forecast data,
computes trend features, and scores each upcoming day using the
prediction model from analyze_trends_v2.py.

Data sources:
  OBSERVATIONS (past 8 days):
    SST:      IMOS AODN L3S 0.02deg gridded (Australian, highest res)
              Fallback: Copernicus NRT observation
    Currents: Copernicus ANFC model 0.083deg
    CHL:      Copernicus NRT observation
    MLD:      Copernicus ANFC model 0.083deg

  FORECASTS (next 7 days):
    SST:      Copernicus ANFC model (thetao) 0.083deg
    Currents: Copernicus ANFC model 0.083deg
    CHL:      Copernicus ANFC model 0.25deg
    MLD:      Copernicus ANFC model 0.083deg

Usage:
    python fetch_prediction.py                    # fetch + predict
    python fetch_prediction.py --days-back 8      # lookback window
    python fetch_prediction.py --days-ahead 7     # forecast window
    python fetch_prediction.py --skip-fetch       # use cached data only
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = "data"
PREDICT_DIR = os.path.join(BASE_DIR, "prediction")
BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}
CANYON_BBOX = {
    "lon_min": 114.8, "lon_max": 115.6,
    "lat_min": -32.3, "lat_max": -31.5,
}

# IMOS AODN THREDDS server for Australian SST
IMOS_SST_THREDDS = (
    "https://thredds.aodn.org.au/thredds/dodsC/IMOS/SRS/SST/ghrsst/L3S-1d/ngt/"
)


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------
def fetch_imos_sst(date_str, out_dir):
    """Fetch SST from IMOS AODN L3S 0.02deg gridded (Australian satellite).

    Much higher resolution than Copernicus for Perth Canyon.
    Access via OPeNDAP — no API key needed.
    """
    out_path = os.path.join(out_dir, "sst_raw.nc")
    if os.path.exists(out_path):
        return True

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # IMOS L3S filename pattern: YYYYMMDD...
    year = dt.strftime("%Y")
    ymd = dt.strftime("%Y%m%d")

    # Try OPeNDAP access
    try:
        url = (f"{IMOS_SST_THREDDS}{year}/{ymd}092000-ABOM-L3S_GHRSST-SSTskin"
               f"-AVHRR_D-1d_night.nc")
        print(f"    [SST] Trying IMOS AODN L3S: {date_str}...")
        ds = xr.open_dataset(url, engine="netcdf4")
        # Subset to our bbox
        sub = ds.sel(
            lat=slice(BBOX["lat_max"], BBOX["lat_min"]),  # lat is descending
            lon=slice(BBOX["lon_min"], BBOX["lon_max"]),
        )
        sub.to_netcdf(out_path)
        sub.close()
        ds.close()
        print(f"    [SST] IMOS L3S saved ({date_str})")
        return True
    except Exception as e:
        print(f"    [SST] IMOS unavailable: {str(e)[:60]}")

    # Fallback: try alternate IMOS filename patterns
    for suffix in ["night", "day"]:
        for sensor in ["AVHRR_D", "AVHRR_N", "MultiSensor"]:
            try:
                url = (f"{IMOS_SST_THREDDS}{year}/{ymd}092000-ABOM-L3S_GHRSST-SSTskin"
                       f"-{sensor}-1d_{suffix}.nc")
                ds = xr.open_dataset(url, engine="netcdf4")
                sub = ds.sel(
                    lat=slice(BBOX["lat_max"], BBOX["lat_min"]),
                    lon=slice(BBOX["lon_min"], BBOX["lon_max"]),
                )
                sub.to_netcdf(out_path)
                sub.close()
                ds.close()
                print(f"    [SST] IMOS {sensor} {suffix} saved ({date_str})")
                return True
            except Exception:
                continue

    return False


def fetch_copernicus_observation(date_str, out_dir):
    """Fetch SST from Copernicus NRT observation as fallback."""
    import copernicusmarine

    out_path = os.path.join(out_dir, "sst_raw.nc")
    if os.path.exists(out_path):
        return True

    try:
        print(f"    [SST] Copernicus NRT observation: {date_str}...")
        copernicusmarine.subset(
            dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            variables=["analysed_sst"],
            minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
            minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T23:59:59",
            output_filename=out_path, output_directory=".", overwrite=True,
        )
        return True
    except Exception as e:
        print(f"    [SST] Copernicus NRT failed: {str(e)[:60]}")
        return False


def fetch_copernicus_forecast(date_str, out_dir, variable_set="all"):
    """Fetch forecast data from Copernicus ANFC (analysis+forecast) products.

    These products support future dates (10+ days ahead).
    """
    import copernicusmarine

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # --- SST (from ANFC physics model) ---
    sst_path = os.path.join(out_dir, "sst_raw.nc")
    if not os.path.exists(sst_path):
        for ds_id in [
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        ]:
            try:
                print(f"    [SST] ANFC forecast: {date_str}...")
                copernicusmarine.subset(
                    dataset_id=ds_id, variables=["thetao"],
                    minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
                    minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
                    start_datetime=f"{date_str}T00:00:00",
                    end_datetime=f"{date_str}T23:59:59",
                    minimum_depth=0, maximum_depth=1,
                    output_filename=sst_path, output_directory=".", overwrite=True,
                )
                results["sst"] = True
                break
            except Exception as e:
                print(f"    [SST] {ds_id} failed: {str(e)[:60]}")
                continue
    else:
        results["sst"] = True

    # --- Currents ---
    cur_path = os.path.join(out_dir, "currents_raw.nc")
    if not os.path.exists(cur_path):
        try:
            print(f"    [Currents] ANFC: {date_str}...")
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                variables=["uo", "vo"],
                minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
                minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
                start_datetime=f"{date_str}T00:00:00",
                end_datetime=f"{date_str}T23:59:59",
                minimum_depth=0, maximum_depth=1,
                output_filename=cur_path, output_directory=".", overwrite=True,
            )
            results["currents"] = True
        except Exception as e:
            print(f"    [Currents] failed: {str(e)[:60]}")
    else:
        results["currents"] = True

    # --- CHL ---
    chl_path = os.path.join(out_dir, "chl_raw.nc")
    if not os.path.exists(chl_path):
        for ds_id, var in [
            ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
            ("cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m", "chl"),
        ]:
            try:
                print(f"    [CHL] {date_str}...")
                kwargs = dict(
                    dataset_id=ds_id, variables=[var],
                    minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
                    minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
                    start_datetime=f"{date_str}T00:00:00",
                    end_datetime=f"{date_str}T23:59:59",
                    output_filename=chl_path, output_directory=".", overwrite=True,
                )
                if var == "chl":
                    kwargs["minimum_depth"] = 0
                    kwargs["maximum_depth"] = 1
                copernicusmarine.subset(**kwargs)
                results["chl"] = True
                break
            except Exception:
                continue
    else:
        results["chl"] = True

    # --- MLD ---
    mld_path = os.path.join(out_dir, "mld_raw.nc")
    if not os.path.exists(mld_path):
        try:
            print(f"    [MLD] ANFC: {date_str}...")
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                variables=["mlotst"],
                minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
                minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
                start_datetime=f"{date_str}T00:00:00",
                end_datetime=f"{date_str}T23:59:59",
                output_filename=mld_path, output_directory=".", overwrite=True,
            )
            results["mld"] = True
        except Exception as e:
            print(f"    [MLD] failed: {str(e)[:60]}")
    else:
        results["mld"] = True

    return results


def fetch_day(date_str, out_dir, is_forecast=False):
    """Fetch all data for a single day.

    For past dates: try IMOS SST first (higher res), then Copernicus.
    For future dates: use Copernicus ANFC forecast products.
    """
    os.makedirs(out_dir, exist_ok=True)

    if is_forecast:
        # All forecast data comes from Copernicus ANFC
        results = fetch_copernicus_forecast(date_str, out_dir)
        return results

    # --- Observation data for past dates ---
    results = {}

    # SST: try IMOS first (0.02deg Australian), then Copernicus (0.05deg global)
    sst_path = os.path.join(out_dir, "sst_raw.nc")
    if not os.path.exists(sst_path):
        if fetch_imos_sst(date_str, out_dir):
            results["sst"] = "imos"
        elif fetch_copernicus_observation(date_str, out_dir):
            results["sst"] = "copernicus_nrt"
        else:
            # Last resort: ANFC model
            r = fetch_copernicus_forecast(date_str, out_dir)
            if r.get("sst"):
                results["sst"] = "copernicus_anfc"
    else:
        results["sst"] = "cached"

    # Currents, CHL, MLD from Copernicus
    r = fetch_copernicus_forecast(date_str, out_dir)
    results.update({k: v for k, v in r.items() if k != "sst"})

    return results


# ---------------------------------------------------------------------------
# Extract metrics + score (reuse from analyze_trends_v2)
# ---------------------------------------------------------------------------
def _extract_var(ds, var_names):
    for v in var_names:
        if v in ds.data_vars:
            data = ds[v]
            if "time" in data.dims:
                data = data.isel(time=0)
            if "depth" in data.dims:
                data = data.isel(depth=0)
            return data
    return None


def _clip(data, bbox):
    if data is None or bbox is None:
        return data
    # Handle both lat/latitude naming
    for ln in ["latitude", "lat"]:
        if ln in data.dims:
            break
    for lo in ["longitude", "lon"]:
        if lo in data.dims:
            break
    lats, lons = data[ln].values, data[lo].values
    return data.isel(**{
        ln: (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"]),
        lo: (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"]),
    })


def extract_metrics(directory, bbox=None):
    """Extract ocean metrics from a day's NetCDF files."""
    m = {}

    sst_path = os.path.join(directory, "sst_raw.nc")
    if os.path.exists(sst_path):
        try:
            ds = xr.open_dataset(sst_path)
            sst = _extract_var(ds, ["sea_surface_temperature", "analysed_sst",
                                     "thetao", "sst"])
            if sst is not None:
                if float(np.nanmean(sst.values)) > 100:
                    sst = sst - 273.15
                sub = _clip(sst, bbox)
                v = sub.values.flatten()
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    m["sst_mean"] = float(np.mean(v))
                    m["sst_min"] = float(np.min(v))
                    m["sst_max"] = float(np.max(v))
                    m["sst_range"] = m["sst_max"] - m["sst_min"]
                    m["sst_std"] = float(np.std(v))
                    m["sst_p25"] = float(np.percentile(v, 25))
                    m["sst_p75"] = float(np.percentile(v, 75))
                    m["sst_iqr"] = m["sst_p75"] - m["sst_p25"]
                    if len(v) > 2:
                        m["sst_skew"] = float(stats.skew(v))
                    m["sst_warm_frac"] = float(np.sum(v > 23) / len(v))
                    m["sst_hot_frac"] = float(np.sum(v > 24) / len(v))
                    arr = sub.values
                    if arr.ndim == 2 and min(arr.shape) >= 3:
                        gy, gx = np.gradient(arr)
                        mag = np.sqrt(gx**2 + gy**2)
                        mc = mag[~np.isnan(mag)]
                        if len(mc) > 0:
                            m["sst_gradient"] = float(np.mean(mc))
                            m["sst_gradient_max"] = float(np.max(mc))
                            m["sst_front_frac"] = float(np.sum(mc > 0.15) / len(mc))
            ds.close()
        except Exception:
            pass

    cur_path = os.path.join(directory, "currents_raw.nc")
    if os.path.exists(cur_path):
        try:
            ds = xr.open_dataset(cur_path)
            uo = _extract_var(ds, ["uo"])
            vo = _extract_var(ds, ["vo"])
            if uo is not None and vo is not None:
                us, vs = _clip(uo, bbox), _clip(vo, bbox)
                uv, vv = us.values.flatten(), vs.values.flatten()
                mask = ~np.isnan(uv) & ~np.isnan(vv)
                uv, vv = uv[mask], vv[mask]
                if len(uv) > 0:
                    spd = np.sqrt(uv**2 + vv**2)
                    m["cur_speed"] = float(np.mean(spd))
                    m["cur_speed_max"] = float(np.max(spd))
                    m["leeuwin"] = float(-np.mean(vv))
                    m["onshore"] = float(np.mean(uv))
                    up, vp = uv - np.mean(uv), vv - np.mean(vv)
                    m["eke"] = float(0.5 * np.mean(up**2 + vp**2))
                    ua, va = us.values, vs.values
                    if ua.ndim >= 2 and min(ua.shape[-2:]) >= 3:
                        dvdx = np.gradient(va, axis=-1)
                        dudy = np.gradient(ua, axis=-2)
                        vort = (dvdx - dudy).flatten()
                        vort = vort[~np.isnan(vort)]
                        if len(vort) > 0:
                            m["vorticity"] = float(np.mean(vort))
                            m["vort_std"] = float(np.std(vort))
                            m["anticyc_frac"] = float(np.sum(vort < 0) / len(vort))
                        dudx = np.gradient(ua, axis=-1)
                        dvdy = np.gradient(va, axis=-2)
                        div = (dudx + dvdy).flatten()
                        div = div[~np.isnan(div)]
                        if len(div) > 0:
                            m["convergence"] = float(-np.mean(div))
            ds.close()
        except Exception:
            pass

    chl_path = os.path.join(directory, "chl_raw.nc")
    if os.path.exists(chl_path):
        try:
            ds = xr.open_dataset(chl_path)
            chl = _extract_var(ds, ["CHL", "chl", "chlor_a"])
            if chl is not None:
                sub = _clip(chl, bbox)
                v = sub.values.flatten()
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    m["chl_mean"] = float(np.mean(v))
                    m["chl_std"] = float(np.std(v))
                    m["chl_log"] = float(np.mean(np.log10(np.clip(v, 1e-4, None))))
                    m["blue_water_frac"] = float(np.sum(v < 0.1) / len(v))
                    arr = sub.values
                    if arr.ndim == 2 and min(arr.shape) >= 3:
                        gy, gx = np.gradient(np.log10(np.clip(arr, 1e-4, None)))
                        mag = np.sqrt(gx**2 + gy**2)
                        mc = mag[~np.isnan(mag)]
                        if len(mc) > 0:
                            m["chl_gradient"] = float(np.mean(mc))
            ds.close()
        except Exception:
            pass

    mld_path = os.path.join(directory, "mld_raw.nc")
    if os.path.exists(mld_path):
        try:
            ds = xr.open_dataset(mld_path)
            mld = _extract_var(ds, ["mlotst"])
            if mld is not None:
                sub = _clip(mld, bbox)
                v = sub.values.flatten()
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    m["mld_mean"] = float(np.mean(v))
                    m["mld_std"] = float(np.std(v))
                    m["mld_min"] = float(np.min(v))
                    m["mld_shallow_frac"] = float(np.sum(v < 20) / len(v))
            ds.close()
        except Exception:
            pass

    return m if m else None


# ---------------------------------------------------------------------------
# Prediction scoring
# ---------------------------------------------------------------------------
def score_day(model, lookback_metrics):
    """Score a single day using the prediction model.

    lookback_metrics: list of metric dicts for the 7 days prior to this day.
    """
    if not model or "parameters" not in model:
        return {"score": 0, "confidence": 0, "error": "No model"}

    # Build offsets
    series = []
    for i, m in enumerate(lookback_metrics):
        series.append({"offset": i - len(lookback_metrics), "metrics": m})

    # Compute trends from the lookback window
    param_scores = {}
    total_weight = 0

    for key, param in model["parameters"].items():
        offs = []
        vals = []
        for d in series:
            if d["metrics"] and key in d["metrics"]:
                v = d["metrics"][key]
                if not np.isnan(v):
                    offs.append(d["offset"])
                    vals.append(v)

        if len(vals) < 3:
            continue

        arr = np.array(vals)
        off = np.array(offs)
        weight = param.get("weight", 0)
        sub_scores = []

        # Score 1: Absolute value match (40%)
        catch_mean = param.get("catch_day_mean")
        catch_std = param.get("catch_day_std")
        if catch_mean is not None and catch_std is not None:
            latest = np.mean(arr[-3:]) if len(arr) >= 3 else arr[-1]
            z = abs(latest - catch_mean) / max(catch_std, 1e-6)
            sub_scores.append(("value", 0.40, max(0, 1 - z / 3)))

        # Score 2: Trend direction match (35%)
        expected_slope = param.get("expected_slope")
        if expected_slope is not None:
            actual_slope = stats.linregress(off, arr).slope
            if expected_slope != 0:
                dir_match = 1.0 if np.sign(actual_slope) == np.sign(expected_slope) else 0.0
                slope_std = param.get("slope_std", abs(expected_slope))
                mag_z = abs(actual_slope - expected_slope) / max(slope_std, 1e-8)
                mag_score = max(0, 1 - mag_z / 3)
                trend_score = 0.6 * dir_match + 0.4 * mag_score
            else:
                trend_score = 0.5
            sub_scores.append(("trend", 0.35, trend_score))

        # Score 3: Early->late shift (25%)
        expected_change = param.get("expected_change")
        if expected_change is not None:
            early = arr[off <= -4] if np.any(off <= -4) else arr[:2]
            late = arr[off >= -2] if np.any(off >= -2) else arr[-2:]
            if len(early) > 0 and len(late) > 0:
                actual_change = np.mean(late) - np.mean(early)
                if expected_change != 0:
                    ratio = actual_change / expected_change
                    sub_scores.append(("shift", 0.25, max(0, min(1, ratio))))

        if sub_scores:
            total_w = sum(w for _, w, _ in sub_scores)
            composite = sum(w * s for _, w, s in sub_scores) / total_w
            param_scores[key] = round(composite * 100, 1)
            total_weight += weight

    if total_weight == 0:
        return {"score": 0, "confidence": 0, "n_params": 0}

    overall = sum(
        param_scores[k] * model["parameters"][k]["weight"]
        for k in param_scores
    ) / total_weight

    return {
        "score": round(overall, 1),
        "confidence": round(len(param_scores) / max(1, len(model["parameters"])) * 100, 1),
        "n_params": len(param_scores),
        "top_params": dict(sorted(param_scores.items(), key=lambda x: -x[1])[:5]),
        "bottom_params": dict(sorted(param_scores.items(), key=lambda x: x[1])[:3]),
    }


# ---------------------------------------------------------------------------
# Generate prediction hotspot map
# ---------------------------------------------------------------------------
def generate_prediction_geojson(date_str, score, metrics, out_dir):
    """Generate a GeoJSON hotspot overlay for a predicted day.

    Uses the prediction score to modulate the existing scoring pipeline.
    """
    import marlin_data
    from marlin_data import generate_blue_marlin_hotspots

    os.makedirs(out_dir, exist_ok=True)
    marlin_data.OUTPUT_DIR = out_dir

    tif_path = os.path.join(out_dir, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        src_tif = os.path.join(BASE_DIR, "bathy_gmrt.tif")
        if os.path.exists(src_tif):
            import shutil
            shutil.copy2(src_tif, tif_path)
        else:
            # Search in dated dirs
            for d in os.listdir(BASE_DIR):
                candidate = os.path.join(BASE_DIR, d, "bathy_gmrt.tif")
                if os.path.exists(candidate):
                    import shutil
                    shutil.copy2(candidate, tif_path)
                    break

    try:
        result = generate_blue_marlin_hotspots(BBOX, tif_path=tif_path)
        if result:
            return result["path"]
    except Exception as e:
        print(f"    [Map] Error generating hotspot map: {str(e)[:80]}")

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch data and predict marlin activity")
    parser.add_argument("--days-back", type=int, default=8,
                        help="Days of historical data to fetch (default: 8)")
    parser.add_argument("--days-ahead", type=int, default=7,
                        help="Days of forecast to fetch (default: 7)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data download, use cached files only")
    parser.add_argument("--skip-maps", action="store_true",
                        help="Skip hotspot map generation")
    args = parser.parse_args()

    today = datetime.now()
    print("=" * 60)
    print("BLUE MARLIN PREDICTION — Perth Canyon")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Lookback: {args.days_back} days | Forecast: {args.days_ahead} days")
    print("=" * 60)

    # Load prediction model
    model_path = os.path.join(BASE_DIR, "prediction_model_v2.json")
    if not os.path.exists(model_path):
        print("ERROR: No prediction model found. Run analyze_trends_v2.py first.")
        return
    with open(model_path) as f:
        model = json.load(f)
    print(f"\nModel loaded: {len(model['parameters'])} parameters")

    # Build date range: past days + future days
    dates = []
    for offset in range(-args.days_back, args.days_ahead + 1):
        dt = today + timedelta(days=offset)
        dates.append({
            "date": dt.strftime("%Y-%m-%d"),
            "offset": offset,
            "is_forecast": offset > 0,
            "is_today": offset == 0,
            "label": "FORECAST" if offset > 0 else ("TODAY" if offset == 0 else "PAST"),
        })

    # --- Fetch data ---
    if not args.skip_fetch:
        print(f"\nFetching data for {len(dates)} dates...")
        for d in dates:
            out_dir = os.path.join(PREDICT_DIR, d["date"])
            tag = f"[{d['label']}]"
            print(f"\n  {tag} {d['date']} (day {d['offset']:+d})")
            try:
                results = fetch_day(d["date"], out_dir, is_forecast=d["is_forecast"])
                n_ok = sum(1 for v in results.values() if v)
                print(f"    -> {n_ok}/4 variables OK")
            except Exception as e:
                print(f"    -> ERROR: {str(e)[:80]}")

    # --- Extract metrics for all dates ---
    print(f"\nExtracting ocean metrics...")
    all_metrics = []
    for d in dates:
        out_dir = os.path.join(PREDICT_DIR, d["date"])
        if os.path.exists(out_dir):
            metrics = extract_metrics(out_dir, CANYON_BBOX)
        else:
            metrics = None
        d["metrics"] = metrics
        all_metrics.append(metrics)
        status = f"{len(metrics)} params" if metrics else "NO DATA"
        print(f"  {d['date']} ({d['label']}): {status}")

    # --- Score each day from today onwards ---
    print(f"\n{'=' * 60}")
    print("PREDICTIONS")
    print(f"{'=' * 60}")

    predictions = []
    for i, d in enumerate(dates):
        if d["offset"] < 0:
            continue  # Only score today + future

        # Build 7-day lookback for this day
        lookback_start = i - 7
        lookback_end = i
        if lookback_start < 0:
            lookback_start = 0

        lookback = [dates[j]["metrics"] for j in range(lookback_start, lookback_end)]

        # Need at least 4 days of data
        n_with_data = sum(1 for m in lookback if m)
        if n_with_data < 4:
            print(f"  {d['date']} ({d['label']}): INSUFFICIENT DATA ({n_with_data}/7 days)")
            continue

        result = score_day(model, lookback)
        result["date"] = d["date"]
        result["offset"] = d["offset"]
        result["label"] = d["label"]
        result["metrics"] = d["metrics"]
        predictions.append(result)

        # Display
        score = result["score"]
        conf = result["confidence"]
        bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
        if score >= 70:
            level = "HIGH ACTIVITY"
        elif score >= 50:
            level = "MODERATE"
        elif score >= 30:
            level = "LOW"
        else:
            level = "UNLIKELY"

        print(f"\n  {d['date']} ({d['label']}, day {d['offset']:+d}):")
        print(f"    Score: {score:.1f}% [{bar}]  {level}")
        print(f"    Confidence: {conf:.0f}% ({result['n_params']} parameters)")
        if result.get("top_params"):
            top = list(result["top_params"].items())[:3]
            print(f"    Best:  {', '.join(f'{k}={v:.0f}%' for k, v in top)}")
        if result.get("bottom_params"):
            bot = list(result["bottom_params"].items())[:3]
            print(f"    Weak:  {', '.join(f'{k}={v:.0f}%' for k, v in bot)}")

    # --- Generate hotspot maps for forecast days ---
    if not args.skip_maps and predictions:
        print(f"\n{'=' * 60}")
        print("GENERATING HOTSPOT MAPS")
        print(f"{'=' * 60}")
        for pred in predictions:
            out_dir = os.path.join(PREDICT_DIR, pred["date"])
            if not os.path.exists(os.path.join(out_dir, "sst_raw.nc")):
                print(f"  {pred['date']}: Skipped (no SST data)")
                continue
            print(f"  {pred['date']} (score: {pred['score']:.1f}%)...")
            path = generate_prediction_geojson(
                pred["date"], pred["score"], pred["metrics"], out_dir
            )
            if path:
                print(f"    -> {path}")
                pred["map_path"] = path

    # --- Save prediction results ---
    output = {
        "generated": today.strftime("%Y-%m-%d %H:%M"),
        "model_version": model.get("version", 1),
        "lookback_days": args.days_back,
        "forecast_days": args.days_ahead,
        "predictions": [
            {k: v for k, v in p.items() if k != "metrics"}
            for p in predictions
        ],
    }
    output_path = os.path.join(PREDICT_DIR, "prediction_results.json")
    os.makedirs(PREDICT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for pred in predictions:
        score = pred["score"]
        label = pred["label"]
        date = pred["date"]
        if score >= 70:
            marker = ">>> HIGH <<<"
        elif score >= 50:
            marker = "    MODERATE"
        else:
            marker = "    low"
        print(f"  {date} ({label:>8}): {score:5.1f}%  {marker}")

    best = max(predictions, key=lambda x: x["score"]) if predictions else None
    if best and best["score"] >= 50:
        print(f"\n  Best day: {best['date']} ({best['score']:.1f}%)")
    print()


if __name__ == "__main__":
    main()
