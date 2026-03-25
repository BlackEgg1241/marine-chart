#!/usr/bin/env python3
"""
analyze_trends.py — Deep analysis of 7-day lookback ocean data for blue marlin
prediction.

Extracts temporal trends from SST, currents, CHL, MLD in the 7 days prior to
each historical blue marlin catch. Identifies which parameters and trend
patterns are most predictive, then builds a prediction algorithm.

Usage:
    python analyze_trends.py                # full analysis + report
    python analyze_trends.py --predict      # run prediction for current data
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BASE_DIR = "data"
LOOKBACK_DIR = os.path.join(BASE_DIR, "lookback")
BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}

# Canyon/shelf-break region where most marlin are caught
# (narrower focus for trend extraction — the productive zone)
CANYON_BBOX = {
    "lon_min": 114.8, "lon_max": 115.6,
    "lat_min": -32.3, "lat_max": -31.5,
}


def ddm_to_dd(raw_str, negative=False):
    """Convert degrees.minutes string (e.g. '31.49') to decimal degrees."""
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_blue_marlin_catches():
    """Load blue marlin catches with corrected DDM coordinates."""
    catches = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Species_Name"] != "BLUE MARLIN":
                continue
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            catches.append({
                "date": r["Release_Date"][:10],
                "lat": lat, "lon": lon,
                "tag": r["Tag_Number"],
            })
    return catches


# ---------------------------------------------------------------------------
# Data extraction from NetCDF files
# ---------------------------------------------------------------------------
def _extract_var(ds, var_names):
    """Extract first available variable from dataset."""
    for v in var_names:
        if v in ds.data_vars:
            data = ds[v]
            # Squeeze out time and depth dimensions
            if "time" in data.dims:
                data = data.isel(time=0)
            if "depth" in data.dims:
                data = data.isel(depth=0)
            return data
    return None


def _spatial_mean(data, bbox=None):
    """Compute spatial mean, optionally within a sub-bbox."""
    if data is None:
        return np.nan
    if bbox:
        lat_name = "latitude" if "latitude" in data.dims else "lat"
        lon_name = "longitude" if "longitude" in data.dims else "lon"
        lats = data[lat_name].values
        lons = data[lon_name].values
        lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
        lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
        data = data.isel(**{lat_name: lat_mask, lon_name: lon_mask})
    vals = data.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.nanmean(vals))


def _spatial_std(data, bbox=None):
    """Compute spatial standard deviation (measure of spatial variability)."""
    if data is None:
        return np.nan
    if bbox:
        lat_name = "latitude" if "latitude" in data.dims else "lat"
        lon_name = "longitude" if "longitude" in data.dims else "lon"
        lats = data[lat_name].values
        lons = data[lon_name].values
        lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
        lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
        data = data.isel(**{lat_name: lat_mask, lon_name: lon_mask})
    vals = data.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    return float(np.nanstd(vals))


def _gradient_magnitude(data, bbox=None):
    """Compute mean gradient magnitude (front strength indicator)."""
    if data is None:
        return np.nan
    if bbox:
        lat_name = "latitude" if "latitude" in data.dims else "lat"
        lon_name = "longitude" if "longitude" in data.dims else "lon"
        lats = data[lat_name].values
        lons = data[lon_name].values
        lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
        lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
        data = data.isel(**{lat_name: lat_mask, lon_name: lon_mask})
    arr = data.values
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return np.nan
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    vals = mag.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.nanmean(vals))


def extract_day_metrics(directory, bbox=None):
    """Extract all ocean metrics from a single day's NetCDF files.

    Returns dict of metrics or None if data missing.
    """
    metrics = {}

    # --- SST ---
    sst_path = os.path.join(directory, "sst_raw.nc")
    if os.path.exists(sst_path):
        try:
            ds = xr.open_dataset(sst_path)
            sst_data = _extract_var(ds, ["analysed_sst", "thetao"])
            if sst_data is not None:
                # Convert Kelvin to Celsius if needed
                mean_val = float(np.nanmean(sst_data.values))
                if mean_val > 100:
                    sst_data = sst_data - 273.15
                metrics["sst_mean"] = _spatial_mean(sst_data, bbox)
                metrics["sst_std"] = _spatial_std(sst_data, bbox)
                metrics["sst_gradient"] = _gradient_magnitude(sst_data, bbox)
                # Max SST (warm intrusion indicator)
                if bbox:
                    lat_name = "latitude" if "latitude" in sst_data.dims else "lat"
                    lon_name = "longitude" if "longitude" in sst_data.dims else "lon"
                    lats = sst_data[lat_name].values
                    lons = sst_data[lon_name].values
                    lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
                    lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
                    sub = sst_data.isel(**{lat_name: lat_mask, lon_name: lon_mask})
                else:
                    sub = sst_data
                vals = sub.values.flatten()
                vals = vals[~np.isnan(vals)]
                metrics["sst_max"] = float(np.max(vals)) if len(vals) > 0 else np.nan
                metrics["sst_min"] = float(np.min(vals)) if len(vals) > 0 else np.nan
                metrics["sst_range"] = metrics["sst_max"] - metrics["sst_min"]
            ds.close()
        except Exception:
            pass

    # --- Currents ---
    cur_path = os.path.join(directory, "currents_raw.nc")
    if os.path.exists(cur_path):
        try:
            ds = xr.open_dataset(cur_path)
            uo = _extract_var(ds, ["uo"])
            vo = _extract_var(ds, ["vo"])
            if uo is not None and vo is not None:
                # Current speed
                speed = np.sqrt(uo**2 + vo**2)
                metrics["current_speed_mean"] = _spatial_mean(speed, bbox)
                metrics["current_speed_max"] = float(np.nanmax(speed.values))

                # Convergence/divergence (du/dx + dv/dy)
                uo_arr = uo.values
                vo_arr = vo.values
                if uo_arr.ndim >= 2 and uo_arr.shape[-2] >= 3 and uo_arr.shape[-1] >= 3:
                    dudx = np.gradient(uo_arr, axis=-1)
                    dvdy = np.gradient(vo_arr, axis=-2)
                    div = dudx + dvdy
                    # Negative divergence = convergence
                    metrics["convergence_mean"] = float(-np.nanmean(div))
                    conv_vals = -div.flatten()
                    conv_vals = conv_vals[~np.isnan(conv_vals)]
                    if len(conv_vals) > 0:
                        metrics["convergence_max"] = float(np.max(conv_vals))

                # Current direction (dominant flow) in the canyon area
                if bbox:
                    lat_name = "latitude" if "latitude" in uo.dims else "lat"
                    lon_name = "longitude" if "longitude" in uo.dims else "lon"
                    lats = uo[lat_name].values
                    lons = uo[lon_name].values
                    lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
                    lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
                    uo_sub = uo.isel(**{lat_name: lat_mask, lon_name: lon_mask})
                    vo_sub = vo.isel(**{lat_name: lat_mask, lon_name: lon_mask})
                else:
                    uo_sub = uo
                    vo_sub = vo
                u_mean = float(np.nanmean(uo_sub.values))
                v_mean = float(np.nanmean(vo_sub.values))
                metrics["current_dir_degrees"] = float(np.degrees(np.arctan2(u_mean, v_mean)) % 360)
                # Southward component (Leeuwin Current strength)
                metrics["leeuwin_strength"] = float(-np.nanmean(vo_sub.values))
            ds.close()
        except Exception:
            pass

    # --- Chlorophyll ---
    chl_path = os.path.join(directory, "chl_raw.nc")
    if os.path.exists(chl_path):
        try:
            ds = xr.open_dataset(chl_path)
            chl_data = _extract_var(ds, ["CHL", "chl", "chlor_a"])
            if chl_data is not None:
                metrics["chl_mean"] = _spatial_mean(chl_data, bbox)
                metrics["chl_std"] = _spatial_std(chl_data, bbox)
                metrics["chl_gradient"] = _gradient_magnitude(chl_data, bbox)
            ds.close()
        except Exception:
            pass

    # --- MLD ---
    mld_path = os.path.join(directory, "mld_raw.nc")
    if os.path.exists(mld_path):
        try:
            ds = xr.open_dataset(mld_path)
            mld_data = _extract_var(ds, ["mlotst"])
            if mld_data is not None:
                metrics["mld_mean"] = _spatial_mean(mld_data, bbox)
                metrics["mld_std"] = _spatial_std(mld_data, bbox)
            ds.close()
        except Exception:
            pass

    return metrics if len(metrics) > 0 else None


# ---------------------------------------------------------------------------
# Build 7-day time series for each catch event
# ---------------------------------------------------------------------------
def build_catch_timeseries(catch_dates, lookback_days=7):
    """Build time series of ocean metrics for each catch date.

    Returns list of dicts: {
        'catch_date': str,
        'days': [{day_offset, date, metrics}, ...],  # -7 to 0
    }
    """
    events = []
    for cd in sorted(set(catch_dates)):
        dt = datetime.strptime(cd, "%Y-%m-%d")
        series = []

        for offset in range(-lookback_days, 1):  # -7 to 0 (catch day)
            day_date = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")

            # Check lookback dir, then catch data dir
            if offset == 0:
                directory = os.path.join(BASE_DIR, day_date)
            else:
                directory = os.path.join(LOOKBACK_DIR, day_date)
                if not os.path.exists(directory):
                    directory = os.path.join(BASE_DIR, day_date)

            if os.path.exists(directory):
                metrics = extract_day_metrics(directory, CANYON_BBOX)
            else:
                metrics = None

            series.append({
                "day_offset": offset,
                "date": day_date,
                "metrics": metrics,
            })

        events.append({
            "catch_date": cd,
            "days": series,
        })

    return events


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------
def compute_trends(events):
    """Compute temporal trends for each metric across the 7-day lookback window.

    For each event and metric, calculates:
    - slope: linear regression slope (rate of change per day)
    - r_squared: how linear the trend is
    - day0_value: value on catch day
    - day7_mean: mean of days -7 to -4 (early window)
    - day3_mean: mean of days -3 to -1 (recent window)
    - momentum: day3_mean - day7_mean (recent acceleration)
    - volatility: std of daily values (stability measure)
    """
    all_metrics_keys = set()
    for ev in events:
        for d in ev["days"]:
            if d["metrics"]:
                all_metrics_keys.update(d["metrics"].keys())
    all_metrics_keys = sorted(all_metrics_keys)

    for ev in events:
        ev["trends"] = {}
        for key in all_metrics_keys:
            offsets = []
            values = []
            for d in ev["days"]:
                if d["metrics"] and key in d["metrics"]:
                    v = d["metrics"][key]
                    if not np.isnan(v):
                        offsets.append(d["day_offset"])
                        values.append(v)

            if len(values) < 3:
                continue

            offsets_arr = np.array(offsets)
            values_arr = np.array(values)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(offsets_arr, values_arr)

            # Day 0 value (catch day)
            day0_vals = [v for o, v in zip(offsets, values) if o == 0]
            day0_value = day0_vals[0] if day0_vals else np.nan

            # Early window (days -7 to -4)
            early_vals = [v for o, v in zip(offsets, values) if -7 <= o <= -4]
            early_mean = np.mean(early_vals) if early_vals else np.nan

            # Recent window (days -3 to -1)
            recent_vals = [v for o, v in zip(offsets, values) if -3 <= o <= -1]
            recent_mean = np.mean(recent_vals) if recent_vals else np.nan

            # Momentum (recent vs early)
            if not np.isnan(early_mean) and not np.isnan(recent_mean):
                momentum = recent_mean - early_mean
            else:
                momentum = np.nan

            # Day-to-day changes
            daily_changes = np.diff(values_arr)
            volatility = float(np.std(daily_changes)) if len(daily_changes) > 1 else np.nan

            # Acceleration (change in slope: early 3 days vs last 3 days)
            if len(values) >= 6:
                early_slope = np.polyfit(offsets_arr[:3], values_arr[:3], 1)[0] if len(offsets_arr[:3]) >= 2 else 0
                late_slope = np.polyfit(offsets_arr[-3:], values_arr[-3:], 1)[0] if len(offsets_arr[-3:]) >= 2 else 0
                acceleration = late_slope - early_slope
            else:
                acceleration = np.nan

            ev["trends"][key] = {
                "slope": float(slope),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "day0_value": float(day0_value) if not np.isnan(day0_value) else None,
                "early_mean": float(early_mean) if not np.isnan(early_mean) else None,
                "recent_mean": float(recent_mean) if not np.isnan(recent_mean) else None,
                "momentum": float(momentum) if not np.isnan(momentum) else None,
                "volatility": float(volatility) if not np.isnan(volatility) else None,
                "acceleration": float(acceleration) if not np.isnan(acceleration) else None,
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr)),
                "n_days": len(values),
            }

    return all_metrics_keys


# ---------------------------------------------------------------------------
# Statistical analysis — parameter importance
# ---------------------------------------------------------------------------
def analyze_parameter_importance(events, metric_keys):
    """Analyze which parameters and trends are most significant across all catches.

    Returns dict with analysis results for each metric.
    """
    results = {}

    for key in metric_keys:
        slopes = []
        day0_values = []
        momentums = []
        r_squareds = []
        p_values = []
        volatilities = []
        accelerations = []

        for ev in events:
            if key not in ev.get("trends", {}):
                continue
            t = ev["trends"][key]
            slopes.append(t["slope"])
            if t["day0_value"] is not None:
                day0_values.append(t["day0_value"])
            if t["momentum"] is not None:
                momentums.append(t["momentum"])
            r_squareds.append(t["r_squared"])
            p_values.append(t["p_value"])
            if t["volatility"] is not None:
                volatilities.append(t["volatility"])
            if t["acceleration"] is not None:
                accelerations.append(t["acceleration"])

        if len(slopes) < 5:
            continue

        slopes_arr = np.array(slopes)
        # One-sample t-test: is the mean slope significantly different from 0?
        t_stat, t_pvalue = stats.ttest_1samp(slopes_arr, 0)

        # Sign consistency: what fraction of events show the same slope direction?
        positive_frac = np.sum(slopes_arr > 0) / len(slopes_arr)
        sign_consistency = max(positive_frac, 1 - positive_frac)
        dominant_direction = "increasing" if positive_frac > 0.5 else "decreasing"

        results[key] = {
            "n_events": len(slopes),
            "slope_mean": float(np.mean(slopes_arr)),
            "slope_std": float(np.std(slopes_arr)),
            "slope_median": float(np.median(slopes_arr)),
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pvalue),
            "significant_005": t_pvalue < 0.05,
            "significant_001": t_pvalue < 0.01,
            "sign_consistency": float(sign_consistency),
            "dominant_direction": dominant_direction,
            "positive_fraction": float(positive_frac),
            "mean_r_squared": float(np.mean(r_squareds)),
            "day0_mean": float(np.mean(day0_values)) if day0_values else None,
            "day0_std": float(np.std(day0_values)) if day0_values else None,
            "momentum_mean": float(np.mean(momentums)) if momentums else None,
            "momentum_std": float(np.std(momentums)) if momentums else None,
            "volatility_mean": float(np.mean(volatilities)) if volatilities else None,
            "acceleration_mean": float(np.mean(accelerations)) if accelerations else None,
        }

    return results


def compute_predictive_power(events, metric_keys):
    """Score each metric's predictive power using multiple criteria.

    Predictive power = how useful is this metric for forecasting marlin activity.
    Criteria:
    1. Trend significance (t-test p-value)
    2. Sign consistency (do most catches show same trend direction?)
    3. Trend linearity (mean R-squared)
    4. Low volatility relative to trend (signal-to-noise)
    """
    importance = analyze_parameter_importance(events, metric_keys)
    scored = {}

    for key, info in importance.items():
        # Score each criterion 0-1
        # 1. Significance: lower p-value = higher score
        sig_score = max(0, 1 - info["t_pvalue"] / 0.10)  # p<0.10 -> score>0

        # 2. Consistency: how often does the trend go the same way?
        consistency_score = (info["sign_consistency"] - 0.5) * 2  # 0.5->0, 1.0->1.0

        # 3. Linearity: how well does a line fit the 7-day trend?
        linearity_score = info["mean_r_squared"]

        # 4. Signal-to-noise: |mean slope| / slope_std
        if info["slope_std"] > 0:
            snr = abs(info["slope_mean"]) / info["slope_std"]
            snr_score = min(1.0, snr / 2.0)  # SNR of 2 = perfect score
        else:
            snr_score = 0

        # Composite predictive power
        predictive_power = (
            0.35 * sig_score +
            0.25 * consistency_score +
            0.20 * linearity_score +
            0.20 * snr_score
        )

        scored[key] = {
            "predictive_power": round(predictive_power, 4),
            "significance_score": round(sig_score, 4),
            "consistency_score": round(consistency_score, 4),
            "linearity_score": round(linearity_score, 4),
            "snr_score": round(snr_score, 4),
            **info,
        }

    # Sort by predictive power
    scored = dict(sorted(scored.items(), key=lambda x: -x[1]["predictive_power"]))
    return scored


# ---------------------------------------------------------------------------
# Prediction algorithm
# ---------------------------------------------------------------------------
def build_prediction_model(events, scored_metrics):
    """Build the prediction model from analyzed trends.

    The model defines:
    1. Ideal pre-catch conditions (what the ocean looks like before marlin appear)
    2. Trend signatures (what direction each parameter should be moving)
    3. Confidence thresholds
    """
    model = {
        "description": "Blue marlin activity prediction model for Perth Canyon, WA",
        "basis": f"Trained on {len(events)} historical blue marlin catch events",
        "lookback_days": 7,
        "parameters": {},
    }

    for key, info in scored_metrics.items():
        if info["predictive_power"] < 0.10:
            continue  # Skip very weak predictors

        param = {
            "predictive_power": info["predictive_power"],
            "weight": 0,  # will be normalized
            "condition_on_catch_day": {
                "mean": info["day0_mean"],
                "std": info["day0_std"],
            },
            "ideal_trend": {
                "direction": info["dominant_direction"],
                "slope_mean": info["slope_mean"],
                "slope_std": info["slope_std"],
            },
            "momentum": {
                "mean": info["momentum_mean"],
                "std": info["momentum_std"],
            },
            "significance": {
                "p_value": info["t_pvalue"],
                "sign_consistency": info["sign_consistency"],
            },
        }
        model["parameters"][key] = param

    # Normalize weights by predictive power
    total_power = sum(p["predictive_power"] for p in model["parameters"].values())
    if total_power > 0:
        for key in model["parameters"]:
            model["parameters"][key]["weight"] = round(
                model["parameters"][key]["predictive_power"] / total_power, 4
            )

    return model


def predict_marlin_activity(model, current_series):
    """Score current ocean conditions against the prediction model.

    current_series: list of 7 dicts (day -7 to day -1), each with metrics.

    Returns:
    - overall_score: 0-100 prediction score
    - parameter_scores: individual parameter match scores
    - confidence: how confident we are in this prediction
    - trend_matches: which trends match the historical pattern
    """
    if not current_series or len(current_series) < 3:
        return {"overall_score": 0, "confidence": 0, "error": "Insufficient data"}

    # Extract trends from current data
    param_scores = {}
    weights_used = {}
    confidences = []

    for key, param_info in model["parameters"].items():
        # Get time series for this parameter
        offsets = []
        values = []
        for i, day_data in enumerate(current_series):
            offset = i - len(current_series)  # negative offsets
            if day_data and key in day_data:
                v = day_data[key]
                if not np.isnan(v):
                    offsets.append(offset)
                    values.append(v)

        if len(values) < 3:
            continue

        offsets_arr = np.array(offsets)
        values_arr = np.array(values)

        # Current trend
        slope, _, r_value, _, _ = stats.linregress(offsets_arr, values_arr)
        current_mean = np.mean(values_arr)

        # --- Score 1: Trend direction match ---
        ideal_dir = param_info["ideal_trend"]["direction"]
        if ideal_dir == "increasing" and slope > 0:
            direction_score = 1.0
        elif ideal_dir == "decreasing" and slope < 0:
            direction_score = 1.0
        elif abs(slope) < abs(param_info["ideal_trend"]["slope_mean"]) * 0.1:
            direction_score = 0.5  # flat trend = partial match
        else:
            direction_score = 0.0  # wrong direction

        # --- Score 2: Trend magnitude match ---
        ideal_slope = param_info["ideal_trend"]["slope_mean"]
        slope_std = param_info["ideal_trend"]["slope_std"]
        if slope_std > 0:
            z_slope = abs(slope - ideal_slope) / slope_std
            magnitude_score = max(0, 1 - z_slope / 3)  # within 3 sigma
        else:
            magnitude_score = 0.5

        # --- Score 3: Absolute value match ---
        if param_info["condition_on_catch_day"]["mean"] is not None:
            ideal_val = param_info["condition_on_catch_day"]["mean"]
            val_std = param_info["condition_on_catch_day"]["std"] or 1.0
            # Use current mean (latest days) as proxy for "where it's heading"
            recent_vals = values_arr[-3:] if len(values_arr) >= 3 else values_arr
            current_latest = np.mean(recent_vals)
            z_val = abs(current_latest - ideal_val) / val_std
            value_score = max(0, 1 - z_val / 3)
        else:
            value_score = 0.5

        # Composite parameter score
        composite = 0.35 * direction_score + 0.25 * magnitude_score + 0.40 * value_score
        weight = param_info["weight"]

        param_scores[key] = {
            "score": round(composite * 100, 1),
            "direction_match": round(direction_score * 100, 1),
            "magnitude_match": round(magnitude_score * 100, 1),
            "value_match": round(value_score * 100, 1),
            "current_slope": round(slope, 6),
            "ideal_slope": round(ideal_slope, 6),
            "current_mean": round(current_mean, 4),
            "weight": round(weight, 4),
        }
        weights_used[key] = weight
        confidences.append(r_value**2)  # trend linearity as confidence

    if not param_scores:
        return {"overall_score": 0, "confidence": 0, "error": "No matching parameters"}

    # Weighted overall score
    total_weight = sum(weights_used.values())
    if total_weight > 0:
        overall = sum(
            param_scores[k]["score"] * weights_used[k] / total_weight
            for k in param_scores
        )
    else:
        overall = 0

    # Confidence based on data completeness and trend linearity
    data_completeness = len(param_scores) / max(1, len(model["parameters"]))
    mean_linearity = np.mean(confidences) if confidences else 0
    confidence = 0.5 * data_completeness + 0.5 * mean_linearity

    return {
        "overall_score": round(overall, 1),
        "confidence": round(confidence * 100, 1),
        "parameter_scores": param_scores,
        "n_parameters_matched": len(param_scores),
        "n_parameters_total": len(model["parameters"]),
    }


# ---------------------------------------------------------------------------
# Cross-validation: test model on historical catches
# ---------------------------------------------------------------------------
def cross_validate(events, scored_metrics):
    """Leave-one-out cross-validation of the prediction model.

    For each catch event, build model from all OTHER events, then predict
    on the held-out event.
    """
    predictions = []

    for i, held_out in enumerate(events):
        # Build model from remaining events
        remaining = [e for j, e in enumerate(events) if j != i]
        remaining_keys = set()
        for ev in remaining:
            remaining_keys.update(ev.get("trends", {}).keys())
        remaining_scored = compute_predictive_power(remaining, sorted(remaining_keys))
        model = build_prediction_model(remaining, remaining_scored)

        # Extract the held-out event's lookback series as "current data"
        current_series = []
        for d in held_out["days"][:-1]:  # exclude catch day (day 0)
            current_series.append(d["metrics"])

        result = predict_marlin_activity(model, current_series)
        result["catch_date"] = held_out["catch_date"]
        predictions.append(result)

    return predictions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(events, scored_metrics, model, cv_results):
    """Generate comprehensive analysis report."""
    lines = []
    lines.append("=" * 78)
    lines.append("BLUE MARLIN PREDICTION ANALYSIS - PERTH CANYON, WESTERN AUSTRALIA")
    lines.append("=" * 78)
    lines.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Catch events analyzed: {len(events)}")
    lines.append(f"Lookback window: 7 days prior to each catch")
    lines.append(f"Analysis region: {CANYON_BBOX}")
    lines.append("")

    # --- Section 1: Parameter Importance Ranking ---
    lines.append("-" * 78)
    lines.append("1. PARAMETER IMPORTANCE RANKING (by predictive power)")
    lines.append("-" * 78)
    lines.append(f"{'Parameter':<28} {'Power':>6} {'Sig':>5} {'Consist':>7} "
                 f"{'Linear':>6} {'SNR':>5} {'Direction':>12} {'p-value':>8}")
    lines.append("-" * 78)

    for key, info in scored_metrics.items():
        sig_marker = "***" if info["significant_001"] else ("** " if info["significant_005"] else "   ")
        lines.append(
            f"{key:<28} {info['predictive_power']:>5.3f} {sig_marker:>5} "
            f"{info['sign_consistency']:>6.1%} {info['linearity_score']:>6.3f} "
            f"{info['snr_score']:>5.3f} {info['dominant_direction']:>12} "
            f"{info['t_pvalue']:>8.4f}"
        )

    lines.append("")
    lines.append("Significance: *** p<0.01, ** p<0.05")
    lines.append("")

    # --- Section 2: Pre-Catch Trend Signatures ---
    lines.append("-" * 78)
    lines.append("2. PRE-CATCH TREND SIGNATURES (7-day patterns before marlin appear)")
    lines.append("-" * 78)

    # Group by category
    categories = {
        "SST (Sea Surface Temperature)": [k for k in scored_metrics if k.startswith("sst_")],
        "Currents": [k for k in scored_metrics if k.startswith("current_") or k.startswith("leeuwin") or k.startswith("convergence")],
        "Chlorophyll": [k for k in scored_metrics if k.startswith("chl_")],
        "Mixed Layer Depth": [k for k in scored_metrics if k.startswith("mld_")],
    }

    for cat_name, cat_keys in categories.items():
        cat_keys = [k for k in cat_keys if k in scored_metrics]
        if not cat_keys:
            continue
        lines.append(f"\n  {cat_name}:")
        for key in cat_keys:
            info = scored_metrics[key]
            trend_dir = "UP" if info["dominant_direction"] == "increasing" else "DOWN"
            slope_str = f"{info['slope_mean']:+.5f}/day"
            consist_str = f"{info['sign_consistency']:.0%} consistent"
            d0 = info.get("day0_mean")
            d0_str = f"catch-day mean: {d0:.4f}" if d0 is not None else "N/A"
            mom = info.get("momentum_mean")
            mom_str = f"momentum: {mom:+.5f}" if mom is not None else ""

            sig = ""
            if info["significant_001"]:
                sig = " [HIGHLY SIGNIFICANT]"
            elif info["significant_005"]:
                sig = " [SIGNIFICANT]"

            lines.append(f"    {key}:")
            lines.append(f"      Trend: {trend_dir} ({slope_str}, {consist_str}){sig}")
            lines.append(f"      {d0_str}  {mom_str}")

    lines.append("")

    # --- Section 3: Key Findings ---
    lines.append("-" * 78)
    lines.append("3. KEY FINDINGS")
    lines.append("-" * 78)

    significant_params = [k for k, v in scored_metrics.items() if v["significant_005"]]
    high_consistency = [k for k, v in scored_metrics.items() if v["sign_consistency"] > 0.65]

    lines.append(f"\n  Statistically significant trends (p<0.05): {len(significant_params)}")
    for k in significant_params:
        info = scored_metrics[k]
        lines.append(f"    - {k}: {info['dominant_direction']} (p={info['t_pvalue']:.4f})")

    lines.append(f"\n  High-consistency trends (>65% same direction): {len(high_consistency)}")
    for k in high_consistency:
        info = scored_metrics[k]
        lines.append(f"    - {k}: {info['sign_consistency']:.0%} {info['dominant_direction']}")

    # Top 5 predictors
    lines.append("\n  Top 5 predictive parameters:")
    for i, (key, info) in enumerate(list(scored_metrics.items())[:5]):
        lines.append(f"    {i+1}. {key} (power={info['predictive_power']:.3f}, "
                     f"weight={model['parameters'].get(key, {}).get('weight', 0):.3f})")

    lines.append("")

    # --- Section 4: Catch-Day Conditions ---
    lines.append("-" * 78)
    lines.append("4. OPTIMAL CATCH-DAY CONDITIONS (what the ocean looks like when marlin bite)")
    lines.append("-" * 78)

    for key, info in list(scored_metrics.items())[:10]:
        if info["day0_mean"] is not None and info["day0_std"] is not None:
            low = info["day0_mean"] - info["day0_std"]
            high = info["day0_mean"] + info["day0_std"]
            lines.append(f"  {key:<28}: {info['day0_mean']:>10.4f}  "
                         f"(range: {low:.4f} to {high:.4f})")

    lines.append("")

    # --- Section 5: Cross-Validation ---
    lines.append("-" * 78)
    lines.append("5. CROSS-VALIDATION (leave-one-out prediction accuracy)")
    lines.append("-" * 78)

    if cv_results:
        scores = [r["overall_score"] for r in cv_results if "error" not in r]
        confidences = [r["confidence"] for r in cv_results if "error" not in r]

        if scores:
            lines.append(f"  Events tested: {len(cv_results)}")
            lines.append(f"  Successful predictions: {len(scores)}")
            lines.append(f"  Mean prediction score: {np.mean(scores):.1f}%")
            lines.append(f"  Median prediction score: {np.median(scores):.1f}%")
            lines.append(f"  Min prediction score: {np.min(scores):.1f}%")
            lines.append(f"  Max prediction score: {np.max(scores):.1f}%")
            lines.append(f"  Mean confidence: {np.mean(confidences):.1f}%")
            lines.append(f"  Predictions >= 50%: {sum(1 for s in scores if s >= 50)}/{len(scores)} "
                         f"({sum(1 for s in scores if s >= 50)/len(scores):.0%})")
            lines.append(f"  Predictions >= 60%: {sum(1 for s in scores if s >= 60)}/{len(scores)} "
                         f"({sum(1 for s in scores if s >= 60)/len(scores):.0%})")

            lines.append(f"\n  Per-event results:")
            for r in sorted(cv_results, key=lambda x: x.get("overall_score", 0), reverse=True):
                if "error" in r:
                    lines.append(f"    {r['catch_date']}: ERROR - {r['error']}")
                else:
                    lines.append(f"    {r['catch_date']}: {r['overall_score']:5.1f}% "
                                 f"(confidence: {r['confidence']:.0f}%)")

    lines.append("")

    # --- Section 6: Prediction Algorithm ---
    lines.append("-" * 78)
    lines.append("6. PREDICTION ALGORITHM SUMMARY")
    lines.append("-" * 78)
    lines.append("""
  The prediction algorithm works by:

  1. COLLECT: Gather 7 days of ocean data (SST, currents, CHL, MLD) for the
     Perth Canyon region (114.8-115.6E, 31.5-32.3S).

  2. COMPUTE TRENDS: For each parameter, calculate:
     - Linear slope (rate of change per day)
     - Momentum (recent 3 days vs early 4 days)
     - Current absolute values

  3. SCORE: Compare each parameter against the historical pattern:
     - Direction match: Is the trend going the right way? (35% weight)
     - Magnitude match: Is the rate of change similar? (25% weight)
     - Value match: Are absolute values in the right range? (40% weight)

  4. COMBINE: Weighted average of all parameter scores, using predictive
     power as weights. Parameters with stronger historical signal get
     more influence.

  5. CONFIDENCE: Based on data completeness and trend linearity.
     Higher confidence when more parameters are available and trends
     are clear/linear.
""")

    lines.append("=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze lookback trends for marlin prediction")
    parser.add_argument("--predict", action="store_true",
                        help="Run prediction on latest available data")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (faster)")
    args = parser.parse_args()

    print("Loading blue marlin catches...")
    catches = load_blue_marlin_catches()
    catch_dates = sorted(set(c["date"] for c in catches))
    print(f"  {len(catches)} catches across {len(catch_dates)} unique dates")

    # --- Step 1: Build time series ---
    print("\nExtracting ocean data for 7-day lookback windows...")
    events = build_catch_timeseries(catch_dates)

    # Check data coverage
    n_with_data = sum(1 for ev in events
                      if any(d["metrics"] for d in ev["days"]))
    print(f"  Events with data: {n_with_data}/{len(events)}")

    # Filter to events that have at least 5 days of data
    good_events = []
    for ev in events:
        n_days_with_data = sum(1 for d in ev["days"] if d["metrics"])
        if n_days_with_data >= 5:
            good_events.append(ev)
    print(f"  Events with >= 5 days of data: {len(good_events)}")

    # --- Step 2: Compute trends ---
    print("\nComputing 7-day temporal trends...")
    metric_keys = compute_trends(good_events)
    print(f"  Metrics tracked: {len(metric_keys)}")
    for k in sorted(metric_keys):
        print(f"    - {k}")

    # --- Step 3: Statistical analysis ---
    print("\nAnalyzing parameter importance...")
    scored_metrics = compute_predictive_power(good_events, metric_keys)
    print(f"  Scored parameters: {len(scored_metrics)}")
    print(f"\n  Top predictors:")
    for i, (key, info) in enumerate(list(scored_metrics.items())[:8]):
        sig = "***" if info["significant_001"] else ("** " if info["significant_005"] else "   ")
        print(f"    {i+1}. {key:<28} power={info['predictive_power']:.3f} "
              f"{sig} {info['dominant_direction']}")

    # --- Step 4: Build prediction model ---
    print("\nBuilding prediction model...")
    model = build_prediction_model(good_events, scored_metrics)
    print(f"  Model parameters: {len(model['parameters'])}")

    # --- Step 5: Cross-validation ---
    cv_results = None
    if not args.no_cv:
        print(f"\nRunning leave-one-out cross-validation ({len(good_events)} folds)...")
        cv_results = cross_validate(good_events, scored_metrics)
        scores = [r["overall_score"] for r in cv_results if "error" not in r]
        if scores:
            print(f"  Mean CV score: {np.mean(scores):.1f}%")
            print(f"  Median CV score: {np.median(scores):.1f}%")
            print(f"  >= 50% accuracy: {sum(1 for s in scores if s >= 50)/len(scores):.0%}")

    # --- Step 6: Generate report ---
    print("\nGenerating analysis report...")
    report = generate_report(good_events, scored_metrics, model, cv_results)

    # Save outputs
    report_path = os.path.join(BASE_DIR, "trend_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    model_path = os.path.join(BASE_DIR, "prediction_model.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  Model saved to {model_path}")

    # Save raw trend data
    trend_data = []
    for ev in good_events:
        entry = {"catch_date": ev["catch_date"], "trends": {}}
        for key, t in ev.get("trends", {}).items():
            entry["trends"][key] = t
        trend_data.append(entry)
    trends_path = os.path.join(BASE_DIR, "trend_data.json")
    with open(trends_path, "w") as f:
        json.dump(trend_data, f, indent=2)
    print(f"  Trend data saved to {trends_path}")

    # Print report to console
    print("\n")
    print(report)


if __name__ == "__main__":
    main()
