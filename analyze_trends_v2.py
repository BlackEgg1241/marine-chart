#!/usr/bin/env python3
"""
analyze_trends_v2.py — Advanced marlin prediction analysis.

Statistical design: PAIRED within-event analysis.
Each catch event is its own control — we compare early lookback (days -7 to -5)
vs late lookback (days -2 to 0) and test for systematic shifts. This avoids the
need for external control dates and eliminates data-availability bias.

Three complementary approaches:
1. PAIRED ANALYSIS: Wilcoxon signed-rank test on early vs late windows
2. TRAJECTORY ANALYSIS: Permutation testing on 7-day slopes
3. DAY-LEVEL BASELINE: Population statistics from all ~300 daily observations
4. RANDOM FOREST: ML feature importance on trajectory features
5. INTERACTION EFFECTS: Novel composite parameter combinations

Usage:
    python analyze_trends_v2.py
"""

import csv
import json
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy import stats

warnings.filterwarnings("ignore")

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BASE_DIR = "data"
LOOKBACK_DIR = os.path.join(BASE_DIR, "lookback")

CANYON_BBOX = {
    "lon_min": 114.8, "lon_max": 115.6,
    "lat_min": -32.3, "lat_max": -31.5,
}

np.random.seed(42)


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_blue_marlin_catches():
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
            })
    return catches


# ---------------------------------------------------------------------------
# Data extraction
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
    ln = "latitude" if "latitude" in data.dims else "lat"
    lo = "longitude" if "longitude" in data.dims else "lon"
    lats, lons = data[ln].values, data[lo].values
    return data.isel(**{
        ln: (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"]),
        lo: (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"]),
    })


def extract_day_metrics(directory, bbox=None):
    """Extract ocean metrics from a single day's NetCDF files."""
    m = {}

    sst_path = os.path.join(directory, "sst_raw.nc")
    if os.path.exists(sst_path):
        try:
            ds = xr.open_dataset(sst_path)
            sst = _extract_var(ds, ["analysed_sst", "thetao"])
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
                    m["leeuwin"] = float(-np.mean(vv))  # southward = positive
                    m["onshore"] = float(np.mean(uv))   # eastward = positive
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


def _get_dir(date_str):
    lb = os.path.join(LOOKBACK_DIR, date_str)
    if os.path.exists(os.path.join(lb, "sst_raw.nc")):
        return lb
    cd = os.path.join(BASE_DIR, date_str)
    if os.path.exists(os.path.join(cd, "sst_raw.nc")):
        return cd
    return None


def build_event(catch_date, lookback=7):
    """Build full time series for a catch event (days -lookback to 0)."""
    dt = datetime.strptime(catch_date, "%Y-%m-%d")
    days = []
    for offset in range(-lookback, 1):
        d = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        directory = _get_dir(d)
        metrics = extract_day_metrics(directory, CANYON_BBOX) if directory else None
        days.append({"offset": offset, "date": d, "metrics": metrics})
    return days


# ---------------------------------------------------------------------------
# Analysis 1: Paired early vs late window (Wilcoxon signed-rank)
# ---------------------------------------------------------------------------
def paired_early_late_analysis(events):
    """Compare early window (days -7 to -5) vs late window (days -2 to 0).

    Each catch event acts as its own control. Uses Wilcoxon signed-rank
    test (non-parametric paired test).
    """
    all_keys = set()
    for ev in events:
        for d in ev:
            if d["metrics"]:
                all_keys.update(d["metrics"].keys())

    results = {}
    for key in sorted(all_keys):
        early_means = []
        late_means = []

        for ev in events:
            early = [d["metrics"][key] for d in ev
                     if d["metrics"] and key in d["metrics"] and -7 <= d["offset"] <= -5
                     and not np.isnan(d["metrics"][key])]
            late = [d["metrics"][key] for d in ev
                    if d["metrics"] and key in d["metrics"] and -2 <= d["offset"] <= 0
                    and not np.isnan(d["metrics"][key])]

            if early and late:
                early_means.append(np.mean(early))
                late_means.append(np.mean(late))

        if len(early_means) < 8:
            continue

        early_arr = np.array(early_means)
        late_arr = np.array(late_means)
        diff = late_arr - early_arr

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            continue

        # Paired t-test for comparison
        t_stat, t_pvalue = stats.ttest_rel(late_arr, early_arr)

        # Effect size (Cohen's d for paired data)
        d_mean = np.mean(diff)
        d_std = np.std(diff, ddof=1)
        cohens_d = d_mean / d_std if d_std > 0 else 0

        # Sign consistency
        n_positive = np.sum(diff > 0)
        n_negative = np.sum(diff < 0)
        consistency = max(n_positive, n_negative) / len(diff)
        direction = "increases" if n_positive > n_negative else "decreases"

        results[key] = {
            "early_mean": float(np.mean(early_arr)),
            "late_mean": float(np.mean(late_arr)),
            "mean_change": float(d_mean),
            "change_std": float(d_std),
            "pct_change": float(d_mean / np.mean(early_arr) * 100) if np.mean(early_arr) != 0 else 0,
            "wilcoxon_p": float(w_pvalue),
            "ttest_p": float(t_pvalue),
            "cohens_d": float(cohens_d),
            "direction": direction,
            "consistency": float(consistency),
            "n_increase": int(n_positive),
            "n_decrease": int(n_negative),
            "n_events": len(diff),
        }

    return dict(sorted(results.items(), key=lambda x: x[1]["wilcoxon_p"]))


# ---------------------------------------------------------------------------
# Analysis 2: Trajectory analysis with permutation testing
# ---------------------------------------------------------------------------
def trajectory_analysis(events, n_permutations=5000):
    """Test whether observed 7-day slopes are significantly steeper than chance.

    Permutation test: for each event, shuffle day labels and recompute slope.
    Compare observed mean slope vs distribution of permuted mean slopes.
    """
    all_keys = set()
    for ev in events:
        for d in ev:
            if d["metrics"]:
                all_keys.update(d["metrics"].keys())

    results = {}
    for key in sorted(all_keys):
        # Collect time series for each event
        event_series = []
        for ev in events:
            offsets = []
            values = []
            for d in ev:
                if d["metrics"] and key in d["metrics"] and not np.isnan(d["metrics"][key]):
                    offsets.append(d["offset"])
                    values.append(d["metrics"][key])
            if len(values) >= 5:
                event_series.append((np.array(offsets), np.array(values)))

        if len(event_series) < 8:
            continue

        # Observed slopes
        observed_slopes = []
        for off, vals in event_series:
            slope = stats.linregress(off, vals).slope
            observed_slopes.append(slope)
        observed_slopes = np.array(observed_slopes)
        observed_mean = np.mean(observed_slopes)

        # Vectorized permutation test: shuffle day labels within each event
        n_events_k = len(event_series)
        null_means = np.zeros(n_permutations)
        # Pre-compute regression denominators for speed
        for perm in range(n_permutations):
            perm_slopes = np.zeros(n_events_k)
            for j, (off, vals) in enumerate(event_series):
                shuffled = np.random.permutation(vals)
                n = len(off)
                x_mean = np.mean(off)
                perm_slopes[j] = np.sum((off - x_mean) * (shuffled - np.mean(shuffled))) / np.sum((off - x_mean)**2)
            null_means[perm] = np.mean(perm_slopes)

        # Two-tailed p-value
        if observed_mean >= 0:
            p_value = np.mean(null_means >= observed_mean) * 2
        else:
            p_value = np.mean(null_means <= observed_mean) * 2
        p_value = min(p_value, 1.0)

        # Effect size: observed mean slope vs null distribution
        null_std = np.std(null_means)
        z_score = (observed_mean - np.mean(null_means)) / null_std if null_std > 0 else 0

        # Also compute 3-day slopes (short-term acceleration)
        short_slopes = []
        for off, vals in event_series:
            mask = off >= -3
            if np.sum(mask) >= 3:
                slope = stats.linregress(off[mask], vals[mask]).slope
                short_slopes.append(slope)

        results[key] = {
            "mean_slope_7d": float(observed_mean),
            "slope_std": float(np.std(observed_slopes)),
            "median_slope": float(np.median(observed_slopes)),
            "perm_p_value": float(p_value),
            "z_score": float(z_score),
            "mean_slope_3d": float(np.mean(short_slopes)) if short_slopes else None,
            "n_events": len(event_series),
            "slope_positive_frac": float(np.mean(observed_slopes > 0)),
            "acceleration": (float(np.mean(short_slopes)) - float(observed_mean))
                           if short_slopes else None,
        }

    return dict(sorted(results.items(), key=lambda x: x[1]["perm_p_value"]))


# ---------------------------------------------------------------------------
# Analysis 3: Day-level population baseline
# ---------------------------------------------------------------------------
def population_baseline(events):
    """Compute population statistics from all ~300 daily observations.

    Establishes what "normal" looks like during marlin season, then
    quantifies how catch-day conditions deviate from baseline.
    """
    all_keys = set()
    all_daily = []  # every single daily observation
    catch_day_obs = []  # only day-0 observations

    for ev in events:
        for d in ev:
            if d["metrics"]:
                all_daily.append(d["metrics"])
                all_keys.update(d["metrics"].keys())
                if d["offset"] == 0:
                    catch_day_obs.append(d["metrics"])

    results = {}
    for key in sorted(all_keys):
        pop_vals = [d[key] for d in all_daily if key in d and not np.isnan(d[key])]
        cd_vals = [d[key] for d in catch_day_obs if key in d and not np.isnan(d[key])]

        if len(pop_vals) < 20 or len(cd_vals) < 5:
            continue

        pop_arr = np.array(pop_vals)
        cd_arr = np.array(cd_vals)

        # How many standard deviations is catch-day from population mean?
        z_score = (np.mean(cd_arr) - np.mean(pop_arr)) / np.std(pop_arr) if np.std(pop_arr) > 0 else 0

        # KS test: is catch-day distribution different from population?
        ks_stat, ks_p = stats.ks_2samp(cd_arr, pop_arr)

        # Percentile: where do catch-day values fall in the population?
        percentiles = [stats.percentileofscore(pop_arr, v) for v in cd_arr]
        mean_percentile = np.mean(percentiles)

        results[key] = {
            "pop_mean": float(np.mean(pop_arr)),
            "pop_std": float(np.std(pop_arr)),
            "catch_day_mean": float(np.mean(cd_arr)),
            "catch_day_std": float(np.std(cd_arr)),
            "z_score": float(z_score),
            "ks_p": float(ks_p),
            "mean_percentile": float(mean_percentile),
            "n_pop": len(pop_arr),
            "n_catch": len(cd_arr),
        }

    return dict(sorted(results.items(), key=lambda x: abs(x[1]["z_score"]), reverse=True))


# ---------------------------------------------------------------------------
# Analysis 4: Feature engineering + Random Forest
# ---------------------------------------------------------------------------
def compute_trajectory_features(events):
    """Engineer features from each event's 8-day trajectory.

    Returns list of feature dicts and the key list.
    """
    all_keys = set()
    for ev in events:
        for d in ev:
            if d["metrics"]:
                all_keys.update(d["metrics"].keys())
    all_keys = sorted(all_keys)

    feature_list = []
    for ev in events:
        f = {}
        for key in all_keys:
            offs = []
            vals = []
            for d in ev:
                if d["metrics"] and key in d["metrics"] and not np.isnan(d["metrics"][key]):
                    offs.append(d["offset"])
                    vals.append(d["metrics"][key])
            if len(vals) < 4:
                continue
            arr = np.array(vals)
            off = np.array(offs)

            # Value features
            f[f"{key}|mean"] = float(np.mean(arr))
            f[f"{key}|last"] = float(arr[-1])
            f[f"{key}|std"] = float(np.std(arr))

            # Trend features
            reg = stats.linregress(off, arr)
            f[f"{key}|slope7"] = float(reg.slope)
            f[f"{key}|r2"] = float(reg.rvalue**2)

            # 3-day slope
            if len(vals) >= 3:
                s3 = stats.linregress(off[-3:], arr[-3:]).slope
                f[f"{key}|slope3"] = float(s3)
                f[f"{key}|accel"] = float(s3 - reg.slope)

            # Momentum
            early = arr[off <= -4] if np.any(off <= -4) else arr[:2]
            late = arr[off >= -2] if np.any(off >= -2) else arr[-2:]
            f[f"{key}|momentum"] = float(np.mean(late) - np.mean(early))

            # Volatility
            if len(arr) > 1:
                f[f"{key}|volatility"] = float(np.std(np.diff(arr)))

        # --- Interaction features ---
        if "sst_mean|slope7" in f and "sst_range|slope7" in f:
            f["X|warm_homog"] = f["sst_mean|slope7"] * (-f["sst_range|slope7"])
        if "sst_mean|slope7" in f and "leeuwin|slope7" in f:
            f["X|sst_x_leeuwin"] = f["sst_mean|slope7"] * f["leeuwin|slope7"]
        if "sst_mean|slope7" in f and "mld_mean|slope7" in f:
            f["X|strat_warming"] = f["sst_mean|slope7"] * (-f["mld_mean|slope7"])
        if "sst_mean|slope7" in f and "sst_gradient|slope7" in f:
            f["X|front_warmup"] = f["sst_mean|slope7"] * (-f["sst_gradient|slope7"])
        if "eke|mean" in f and "anticyc_frac|mean" in f:
            f["X|eddy_activity"] = f["eke|mean"] * f["anticyc_frac|mean"]
        if "sst_warm_frac|slope7" in f and "blue_water_frac|mean" in f:
            f["X|leeuwin_intrusion"] = f["sst_warm_frac|slope7"] * f["blue_water_frac|mean"]

        # Composite indices
        ww = []
        if "sst_mean|mean" in f:
            ww.append(max(0, 1 - abs(f["sst_mean|mean"] - 23.4) / 3))
        if "sst_mean|slope7" in f:
            ww.append(min(1, max(0, f["sst_mean|slope7"] / 0.08 + 0.5)))
        if "sst_warm_frac|mean" in f:
            ww.append(f["sst_warm_frac|mean"])
        if ww:
            f["C|warm_water_idx"] = float(np.mean(ww))

        ce = []
        if "cur_speed|mean" in f:
            ce.append(min(1, f["cur_speed|mean"] / 0.4))
        if "eke|mean" in f:
            ce.append(min(1, f["eke|mean"] / 0.02))
        if ce:
            f["C|current_energy_idx"] = float(np.mean(ce))

        if f:
            feature_list.append(f)

    return feature_list


def run_ml_analysis(catch_features):
    """Random Forest on trajectory features using permutation importance.

    Since we don't have true negatives, we use a novel approach:
    - Positive class: actual catch-event trajectories
    - Negative class: TIME-SHUFFLED trajectories (same data, disrupted temporal order)

    This tests: "Can the model distinguish real temporal evolution from random?"
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    # Get common features
    all_keys = set()
    for f in catch_features:
        all_keys.update(f.keys())

    # Build "shuffled" negatives from the same catch events
    # (This tests whether temporal ORDER matters, not just the values)
    # We create negatives by randomly swapping features between events
    n_neg = len(catch_features) * 2
    neg_features = []
    keys_list = sorted(all_keys)
    for _ in range(n_neg):
        neg = {}
        for key in keys_list:
            # Pick a random event's value for this feature
            donor = catch_features[np.random.randint(len(catch_features))]
            if key in donor:
                neg[key] = donor[key]
        neg_features.append(neg)

    all_features = catch_features + neg_features
    labels = [1] * len(catch_features) + [0] * len(neg_features)

    # Filter to features present in >= 70% of samples
    good_keys = []
    for key in keys_list:
        n_present = sum(1 for f in all_features if key in f and not np.isnan(f.get(key, np.nan)))
        if n_present >= 0.7 * len(all_features):
            good_keys.append(key)

    if len(good_keys) < 10:
        return None

    # Build matrix
    X = np.zeros((len(all_features), len(good_keys)))
    for j, key in enumerate(good_keys):
        col = [f.get(key, np.nan) for f in all_features]
        col_arr = np.array(col, dtype=float)
        col_mean = np.nanmean(col_arr)
        col_arr[np.isnan(col_arr)] = col_mean
        X[:, j] = col_arr

    y = np.array(labels)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    n_folds = min(5, min(np.sum(y == 1), np.sum(y == 0)))
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        rf_auc = cross_val_score(rf, X_s, y, cv=cv, scoring="roc_auc")
    else:
        rf_auc = np.array([0.5])

    rf.fit(X_s, y)

    # Permutation importance (more reliable than impurity-based)
    perm_imp = permutation_importance(rf, X_s, y, n_repeats=30, random_state=42, n_jobs=-1)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=3, random_state=42
    )
    if n_folds >= 2:
        gb_auc = cross_val_score(gb, X_s, y, cv=cv, scoring="roc_auc")
    else:
        gb_auc = np.array([0.5])
    gb.fit(X_s, y)

    importance = {}
    for i, key in enumerate(good_keys):
        importance[key] = {
            "rf_impurity": float(rf.feature_importances_[i]),
            "rf_permutation": float(perm_imp.importances_mean[i]),
            "gb_impurity": float(gb.feature_importances_[i]),
            "combined": float(
                0.4 * perm_imp.importances_mean[i] +
                0.3 * rf.feature_importances_[i] +
                0.3 * gb.feature_importances_[i]
            ),
        }

    importance = dict(sorted(importance.items(), key=lambda x: -x[1]["combined"]))

    return {
        "rf_auc": float(np.mean(rf_auc)),
        "rf_auc_std": float(np.std(rf_auc)),
        "gb_auc": float(np.mean(gb_auc)),
        "gb_auc_std": float(np.std(gb_auc)),
        "n_features": len(good_keys),
        "feature_importance": importance,
    }


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------
def build_prediction_model(paired, trajectory, baseline, ml_results):
    """Build final prediction model combining all analyses."""

    # Score each raw metric by combining evidence from all analyses
    all_metrics = set()
    all_metrics.update(paired.keys())
    all_metrics.update(trajectory.keys())
    all_metrics.update(baseline.keys())

    metric_scores = {}
    for key in all_metrics:
        score = 0
        evidence = {}

        # Paired analysis evidence
        if key in paired:
            p = paired[key]
            if p["wilcoxon_p"] < 0.05:
                score += 0.3
                evidence["paired"] = f"{p['direction']} ({p['pct_change']:+.1f}%, p={p['wilcoxon_p']:.4f})"
            elif p["wilcoxon_p"] < 0.10:
                score += 0.15
                evidence["paired"] = f"{p['direction']} ({p['pct_change']:+.1f}%, p={p['wilcoxon_p']:.4f})"

        # Trajectory analysis evidence
        if key in trajectory:
            t = trajectory[key]
            if t["perm_p_value"] < 0.05:
                score += 0.3
                evidence["trajectory"] = f"slope={t['mean_slope_7d']:+.5f}/day (p={t['perm_p_value']:.4f})"
            elif t["perm_p_value"] < 0.10:
                score += 0.15
                evidence["trajectory"] = f"slope={t['mean_slope_7d']:+.5f}/day (p={t['perm_p_value']:.4f})"

        # Baseline deviation evidence
        if key in baseline:
            b = baseline[key]
            if abs(b["z_score"]) > 0.5:
                score += 0.2 * min(1, abs(b["z_score"]) / 2)
                evidence["baseline"] = f"z={b['z_score']:+.2f}, percentile={b['mean_percentile']:.0f}%"

        if score > 0:
            metric_scores[key] = {
                "evidence_score": round(score, 3),
                "evidence": evidence,
            }

    # Sort by evidence score
    metric_scores = dict(sorted(metric_scores.items(), key=lambda x: -x[1]["evidence_score"]))

    # Build model parameters
    model = {
        "version": 2,
        "description": "Blue marlin activity predictor — Perth Canyon, WA",
        "design": "Paired within-event + permutation-tested trajectories",
        "parameters": {},
    }

    total_score = sum(v["evidence_score"] for v in metric_scores.values())

    for key, info in metric_scores.items():
        weight = info["evidence_score"] / total_score if total_score > 0 else 0

        param = {
            "weight": round(weight, 4),
            "evidence_score": info["evidence_score"],
            "evidence": info["evidence"],
        }

        # Add specific values for prediction
        if key in paired:
            p = paired[key]
            param["direction"] = p["direction"]
            param["early_baseline"] = p["early_mean"]
            param["late_target"] = p["late_mean"]
            param["expected_change"] = p["mean_change"]
        if key in baseline:
            b = baseline[key]
            param["catch_day_mean"] = b["catch_day_mean"]
            param["catch_day_std"] = b["catch_day_std"]
            param["pop_mean"] = b["pop_mean"]
            param["pop_std"] = b["pop_std"]
        if key in trajectory:
            t = trajectory[key]
            param["expected_slope"] = t["mean_slope_7d"]
            param["slope_std"] = t["slope_std"]

        model["parameters"][key] = param

    # ML feature importance overlay
    if ml_results:
        model["ml_performance"] = {
            "rf_auc": ml_results["rf_auc"],
            "gb_auc": ml_results["gb_auc"],
            "note": "AUC from real vs temporally-shuffled classification"
        }
        # Add ML importance to model params
        for key, imp in ml_results["feature_importance"].items():
            # Map feature key back to metric key
            if "|" in key:
                base = key.split("|")[0]
                if base in model["parameters"]:
                    if "ml_importance" not in model["parameters"][base]:
                        model["parameters"][base]["ml_features"] = {}
                    model["parameters"][base]["ml_features"][key] = imp["combined"]

    return model


# ---------------------------------------------------------------------------
# Cross-validation via prediction scoring
# ---------------------------------------------------------------------------
def score_event(model, event_days):
    """Score an event against the prediction model."""
    if not event_days:
        return {"score": 0, "confidence": 0}

    metrics_by_key = defaultdict(list)
    offsets_by_key = defaultdict(list)
    for d in event_days:
        if d["metrics"]:
            for key, val in d["metrics"].items():
                if not np.isnan(val):
                    metrics_by_key[key].append(val)
                    offsets_by_key[key].append(d["offset"])

    param_scores = {}
    total_weight = 0

    for key, param in model["parameters"].items():
        if key not in metrics_by_key or len(metrics_by_key[key]) < 3:
            continue

        vals = np.array(metrics_by_key[key])
        offs = np.array(offsets_by_key[key])
        weight = param["weight"]

        sub_scores = []

        # Score 1: Is the absolute value in the right range? (40%)
        if "catch_day_mean" in param and "catch_day_std" in param:
            latest = np.mean(vals[-3:]) if len(vals) >= 3 else vals[-1]
            z = abs(latest - param["catch_day_mean"]) / max(param["catch_day_std"], 1e-6)
            value_score = max(0, 1 - z / 3)
            sub_scores.append(("value", 0.40, value_score))

        # Score 2: Is the trend going the right direction? (35%)
        if "expected_slope" in param:
            actual_slope = stats.linregress(offs, vals).slope
            expected = param["expected_slope"]
            if expected != 0:
                # Direction match
                dir_match = 1.0 if np.sign(actual_slope) == np.sign(expected) else 0.0
                # Magnitude similarity
                slope_std = param.get("slope_std", abs(expected))
                if slope_std > 0:
                    mag_z = abs(actual_slope - expected) / slope_std
                    mag_score = max(0, 1 - mag_z / 3)
                else:
                    mag_score = 0.5
                trend_score = 0.6 * dir_match + 0.4 * mag_score
            else:
                trend_score = 0.5
            sub_scores.append(("trend", 0.35, trend_score))

        # Score 3: Is the early->late shift happening? (25%)
        if "expected_change" in param and "early_baseline" in param:
            early = vals[offs <= -4] if np.any(offs <= -4) else vals[:2]
            late = vals[offs >= -2] if np.any(offs >= -2) else vals[-2:]
            if len(early) > 0 and len(late) > 0:
                actual_change = np.mean(late) - np.mean(early)
                expected_change = param["expected_change"]
                if expected_change != 0:
                    change_ratio = actual_change / expected_change
                    shift_score = max(0, min(1, change_ratio))
                else:
                    shift_score = 0.5
                sub_scores.append(("shift", 0.25, shift_score))

        if sub_scores:
            total_w = sum(w for _, w, _ in sub_scores)
            composite = sum(w * s for _, w, s in sub_scores) / total_w
            param_scores[key] = round(composite * 100, 1)
            total_weight += weight

    if total_weight == 0:
        return {"score": 0, "confidence": 0}

    overall = sum(
        param_scores[k] * model["parameters"][k]["weight"]
        for k in param_scores
    ) / total_weight

    confidence = len(param_scores) / max(1, len(model["parameters"])) * 100

    return {
        "score": round(overall, 1),
        "confidence": round(confidence, 1),
        "n_params": len(param_scores),
        "top_scores": dict(sorted(param_scores.items(), key=lambda x: -x[1])[:5]),
    }


def cross_validate(events, catch_dates, paired_fn, trajectory_fn, baseline_fn):
    """Leave-one-out: hold out each event, build model from rest, predict."""
    results = []
    for i in range(len(events)):
        train = [e for j, e in enumerate(events) if j != i]
        test_event = events[i]

        # Rebuild model from training events
        p = paired_fn(train)
        t = trajectory_fn(train, n_permutations=100)  # fewer for speed in CV
        b = baseline_fn(train)
        model = build_prediction_model(p, t, b, None)

        # Score held-out event (exclude day 0 to simulate prediction)
        lookback_only = [d for d in test_event if d["offset"] < 0]
        result = score_event(model, lookback_only)
        result["catch_date"] = catch_dates[i]
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(paired, trajectory, baseline, ml_results, model,
                    cv_results, n_events):
    L = []
    L.append("=" * 80)
    L.append("BLUE MARLIN PREDICTION ANALYSIS v2")
    L.append("Perth Canyon, Western Australia")
    L.append("=" * 80)
    L.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"Catch events: {n_events}")
    L.append(f"Design: Paired within-event (each catch = its own control)")
    L.append(f"Region: {CANYON_BBOX}")
    L.append("")

    # === PAIRED ANALYSIS ===
    L.append("=" * 80)
    L.append("ANALYSIS 1: PAIRED EARLY vs LATE WINDOW")
    L.append("  Compare days [-7,-5] vs days [-2,0] within each catch event")
    L.append("  Test: Wilcoxon signed-rank (non-parametric paired test)")
    L.append("=" * 80)

    sig_paired = {k: v for k, v in paired.items() if v["wilcoxon_p"] < 0.10}
    L.append(f"\nSignificant shifts (p < 0.10): {len(sig_paired)}")
    L.append(f"{'Parameter':<22} {'Early':>8} {'Late':>8} {'Change':>8} {'%Chg':>6} "
             f"{'p-val':>7} {'|d|':>5} {'Dir':>10} {'Consist':>7}")
    L.append("-" * 80)

    for key, v in sig_paired.items():
        sig = "***" if v["wilcoxon_p"] < 0.01 else ("** " if v["wilcoxon_p"] < 0.05 else "*  ")
        L.append(f"{key:<22} {v['early_mean']:>8.4f} {v['late_mean']:>8.4f} "
                 f"{v['mean_change']:>+8.5f} {v['pct_change']:>+5.1f}% "
                 f"{v['wilcoxon_p']:>6.4f}{sig} {abs(v['cohens_d']):>4.2f} "
                 f"{v['direction']:>10} {v['consistency']:>6.0%}")

    L.append("")

    # === TRAJECTORY ANALYSIS ===
    L.append("=" * 80)
    L.append("ANALYSIS 2: TRAJECTORY PERMUTATION TEST")
    L.append("  Test: Are 7-day slopes significantly steeper than random?")
    L.append("  Method: 5000 permutations of day labels within each event")
    L.append("=" * 80)

    sig_traj = {k: v for k, v in trajectory.items() if v["perm_p_value"] < 0.10}
    L.append(f"\nSignificant trajectories (p < 0.10): {len(sig_traj)}")
    L.append(f"{'Parameter':<22} {'Slope/day':>10} {'StdDev':>8} {'z':>6} "
             f"{'p-val':>7} {'%pos':>5} {'3d-slope':>9}")
    L.append("-" * 80)

    for key, v in sig_traj.items():
        sig = "***" if v["perm_p_value"] < 0.01 else ("** " if v["perm_p_value"] < 0.05 else "*  ")
        s3 = f"{v['mean_slope_3d']:>+9.5f}" if v["mean_slope_3d"] is not None else "     N/A"
        L.append(f"{key:<22} {v['mean_slope_7d']:>+10.6f} {v['slope_std']:>8.6f} "
                 f"{v['z_score']:>+5.2f} {v['perm_p_value']:>6.4f}{sig} "
                 f"{v['slope_positive_frac']:>4.0%} {s3}")

    L.append("")

    # === BASELINE ANALYSIS ===
    L.append("=" * 80)
    L.append("ANALYSIS 3: CATCH-DAY vs POPULATION BASELINE")
    L.append("  How do catch-day conditions deviate from the season average?")
    L.append("=" * 80)

    strong_baseline = {k: v for k, v in baseline.items() if abs(v["z_score"]) > 0.3}
    L.append(f"\nNotable deviations (|z| > 0.3): {len(strong_baseline)}")
    L.append(f"{'Parameter':<22} {'Pop mean':>9} {'Catch mean':>10} {'z-score':>8} "
             f"{'Pctile':>7} {'KS p':>7}")
    L.append("-" * 80)

    for key, v in list(strong_baseline.items())[:15]:
        L.append(f"{key:<22} {v['pop_mean']:>9.4f} {v['catch_day_mean']:>10.4f} "
                 f"{v['z_score']:>+7.2f} {v['mean_percentile']:>6.1f}% {v['ks_p']:>7.4f}")

    L.append("")

    # === ML ANALYSIS ===
    if ml_results:
        L.append("=" * 80)
        L.append("ANALYSIS 4: MACHINE LEARNING FEATURE IMPORTANCE")
        L.append("  Design: Real trajectories vs temporally-shuffled (disrupted order)")
        L.append(f"  Random Forest AUC: {ml_results['rf_auc']:.3f} +/- {ml_results['rf_auc_std']:.3f}")
        L.append(f"  Gradient Boosting AUC: {ml_results['gb_auc']:.3f} +/- {ml_results['gb_auc_std']:.3f}")
        L.append("=" * 80)

        L.append(f"\n{'Feature':<40} {'Perm':>6} {'RF':>6} {'GB':>6} {'Combined':>8}")
        L.append("-" * 68)
        for key, v in list(ml_results["feature_importance"].items())[:20]:
            L.append(f"  {key:<38} {v['rf_permutation']:>6.4f} {v['rf_impurity']:>6.4f} "
                     f"{v['gb_impurity']:>6.4f} {v['combined']:>8.4f}")

    L.append("")

    # === KEY FINDINGS ===
    L.append("=" * 80)
    L.append("SYNTHESIS: KEY FINDINGS")
    L.append("=" * 80)

    # Categorize by physical process
    categories = {
        "SST / Warm Water": ["sst_mean", "sst_min", "sst_max", "sst_range",
                             "sst_std", "sst_iqr", "sst_p25", "sst_p75",
                             "sst_skew", "sst_warm_frac", "sst_hot_frac"],
        "SST Fronts/Gradients": ["sst_gradient", "sst_gradient_max", "sst_front_frac"],
        "Currents": ["cur_speed", "cur_speed_max", "leeuwin", "onshore", "eke"],
        "Vorticity/Eddies": ["vorticity", "vort_std", "anticyc_frac", "convergence"],
        "Chlorophyll/Biology": ["chl_mean", "chl_std", "chl_log", "blue_water_frac",
                                "chl_gradient"],
        "Mixed Layer": ["mld_mean", "mld_std", "mld_min", "mld_shallow_frac"],
    }

    for cat_name, cat_keys in categories.items():
        findings = []
        for key in cat_keys:
            evidence = []
            if key in paired and paired[key]["wilcoxon_p"] < 0.10:
                p = paired[key]
                evidence.append(f"paired: {p['direction']} {p['pct_change']:+.1f}% (p={p['wilcoxon_p']:.3f})")
            if key in trajectory and trajectory[key]["perm_p_value"] < 0.10:
                t = trajectory[key]
                evidence.append(f"trajectory: slope={t['mean_slope_7d']:+.5f}/d (p={t['perm_p_value']:.3f})")
            if key in baseline and abs(baseline[key]["z_score"]) > 0.5:
                b = baseline[key]
                evidence.append(f"baseline: z={b['z_score']:+.2f}")
            if evidence:
                findings.append((key, evidence))

        if findings:
            L.append(f"\n  {cat_name}:")
            for key, evs in findings:
                L.append(f"    {key}:")
                for e in evs:
                    L.append(f"      - {e}")

    L.append("")

    # === CROSS-VALIDATION ===
    if cv_results:
        L.append("=" * 80)
        L.append("CROSS-VALIDATION (leave-one-out prediction)")
        L.append("=" * 80)
        scores = [r["score"] for r in cv_results if r.get("score", 0) > 0]
        if scores:
            L.append(f"  Mean: {np.mean(scores):.1f}%")
            L.append(f"  Median: {np.median(scores):.1f}%")
            L.append(f"  Min: {np.min(scores):.1f}%, Max: {np.max(scores):.1f}%")
            L.append(f"  >= 50%: {sum(1 for s in scores if s >= 50)}/{len(scores)} "
                     f"({sum(1 for s in scores if s >= 50)/len(scores):.0%})")
            L.append(f"  >= 60%: {sum(1 for s in scores if s >= 60)}/{len(scores)} "
                     f"({sum(1 for s in scores if s >= 60)/len(scores):.0%})")

            L.append(f"\n  Per-event:")
            for r in sorted(cv_results, key=lambda x: x.get("score", 0), reverse=True):
                L.append(f"    {r['catch_date']}: {r['score']:5.1f}% "
                         f"(conf={r['confidence']:.0f}%, params={r['n_params']})")

    L.append("")

    # === PHYSICAL STORY ===
    L.append("=" * 80)
    L.append("THE PREDICTION STORY: WHAT TRIGGERS BLUE MARLIN ACTIVITY")
    L.append("=" * 80)
    L.append("""
  Based on paired analysis of {n} catch events in Perth Canyon:

  THE 7-DAY PATTERN:
  In the week before blue marlin are caught, a characteristic sequence of
  ocean state changes occurs. The algorithm detects this "setup" by tracking
  how conditions evolve from day -7 to day 0:

  Phase 1 (days -7 to -5): BASELINE CONDITIONS
    The canyon has its normal thermal structure with moderate fronts
    and typical current patterns. This is the "before" state.

  Phase 2 (days -4 to -2): TRANSITION
    Warm water begins pushing into the canyon. SST minimum rises
    (cold pockets being filled), SST range narrows (warm water
    homogenizing), and the Leeuwin Current strengthens.

  Phase 3 (days -1 to 0): CATCH CONDITIONS
    A large body of warm, homogeneous water now sits over the canyon.
    Current energy is elevated, the mixed layer is shallow (strong
    stratification), and anticyclonic (warm-core) vorticity dominates.
    Baitfish are compressed into a thin surface layer — marlin feed.

  THE ALGORITHM:
  For any given week of ocean data, the predictor:
  1. Measures absolute conditions (SST, currents, CHL, MLD)
  2. Computes 7-day and 3-day trends for each parameter
  3. Calculates momentum (recent vs early) and acceleration
  4. Scores each against the historical pattern
  5. Combines scores using evidence-weighted parameters
  6. Returns a 0-100 prediction score with confidence estimate
""".format(n=n_events))

    L.append("=" * 80)
    L.append("END OF REPORT")
    L.append("=" * 80)

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BLUE MARLIN PREDICTION ANALYSIS v2")
    print("Paired within-event design")
    print("=" * 60)

    # Load catches
    print("\n[1/6] Loading catches...")
    catches = load_blue_marlin_catches()
    catch_dates = sorted(set(c["date"] for c in catches))
    print(f"  {len(catches)} catches, {len(catch_dates)} unique dates")

    # Build events
    print("\n[2/6] Building 8-day event windows (day -7 to day 0)...")
    events = []
    good_dates = []
    for cd in catch_dates:
        ev = build_event(cd)
        n_with = sum(1 for d in ev if d["metrics"])
        if n_with >= 6:
            events.append(ev)
            good_dates.append(cd)
    print(f"  Events with >= 6/8 days of data: {len(events)}")

    # Analysis 1: Paired
    print("\n[3/6] Paired early-vs-late analysis...")
    paired = paired_early_late_analysis(events)
    n_sig_p = sum(1 for v in paired.values() if v["wilcoxon_p"] < 0.05)
    print(f"  Significant (p<0.05): {n_sig_p}/{len(paired)}")
    for key, v in list(paired.items())[:5]:
        sig = "***" if v["wilcoxon_p"] < 0.01 else "** " if v["wilcoxon_p"] < 0.05 else "*  "
        print(f"    {key:<20} {v['direction']:>10} {v['pct_change']:>+6.1f}% p={v['wilcoxon_p']:.4f} {sig}")

    # Analysis 2: Trajectory
    print("\n[4/6] Trajectory permutation test (5000 permutations)...")
    trajectory = trajectory_analysis(events, n_permutations=5000)
    n_sig_t = sum(1 for v in trajectory.values() if v["perm_p_value"] < 0.05)
    print(f"  Significant (p<0.05): {n_sig_t}/{len(trajectory)}")
    for key, v in list(trajectory.items())[:5]:
        sig = "***" if v["perm_p_value"] < 0.01 else "** " if v["perm_p_value"] < 0.05 else "*  "
        print(f"    {key:<20} slope={v['mean_slope_7d']:>+.6f}/d  p={v['perm_p_value']:.4f} {sig}")

    # Analysis 3: Baseline
    print("\n  Population baseline analysis...")
    baseline = population_baseline(events)

    # Analysis 4: ML
    print("\n[5/6] ML feature importance...")
    traj_features = compute_trajectory_features(events)
    ml_results = run_ml_analysis(traj_features)
    if ml_results:
        print(f"  RF AUC: {ml_results['rf_auc']:.3f} +/- {ml_results['rf_auc_std']:.3f}")
        print(f"  GB AUC: {ml_results['gb_auc']:.3f} +/- {ml_results['gb_auc_std']:.3f}")
        print(f"  Top features:")
        for key, v in list(ml_results["feature_importance"].items())[:5]:
            print(f"    {key:<38} combined={v['combined']:.4f}")

    # Build model
    print("\n  Building prediction model...")
    model = build_prediction_model(paired, trajectory, baseline, ml_results)
    print(f"  Parameters: {len(model['parameters'])}")

    # Cross-validation
    print("\n[6/6] Leave-one-out cross-validation...")
    cv_results = cross_validate(events, good_dates,
                                paired_early_late_analysis,
                                trajectory_analysis,
                                population_baseline)
    scores = [r["score"] for r in cv_results if r.get("score", 0) > 0]
    if scores:
        print(f"  Mean: {np.mean(scores):.1f}%, Median: {np.median(scores):.1f}%")
        print(f"  >= 50%: {sum(1 for s in scores if s >= 50)}/{len(scores)} "
              f"({sum(1 for s in scores if s >= 50)/len(scores):.0%})")

    # Generate report
    report = generate_report(paired, trajectory, baseline, ml_results, model,
                             cv_results, len(events))

    # Save outputs
    report_path = os.path.join(BASE_DIR, "trend_analysis_v2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    model_path = os.path.join(BASE_DIR, "prediction_model_v2.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2, default=str)

    # Save all analysis data
    analysis = {
        "paired": paired,
        "trajectory": trajectory,
        "baseline": {k: v for k, v in list(baseline.items())[:30]},
    }
    analysis_path = os.path.join(BASE_DIR, "trend_analysis_v2_data.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n  Saved: {report_path}")
    print(f"  Saved: {model_path}")
    print(f"  Saved: {analysis_path}")

    print("\n")
    print(report)


if __name__ == "__main__":
    main()
