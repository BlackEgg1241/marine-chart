#!/usr/bin/env python3
"""
optimize_scoring.py — Use Optuna to find optimal scoring parameters.

Runs generate_blue_marlin_hotspots with different parameter combinations
and evaluates against 71 historical catch records. Optimizes for:
  1. Mean score at catch locations (primary)
  2. Percentage of catches scoring >= 50% (secondary)
  3. Minimum score floor (penalize very low outliers)

Usage:
    python optimize_scoring.py              # 200 trials
    python optimize_scoring.py --trials 500 # more trials
"""

import argparse
import csv
import os
import sys
import json
from datetime import datetime

import numpy as np
import optuna

# Suppress optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

import marlin_data
from marlin_data import generate_blue_marlin_hotspots

BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BASE_DIR = "data"


def ddm_to_dd(raw_str, negative=False):
    """Convert degrees.minutes string (e.g. '31.49') to decimal degrees."""
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches():
    catches = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            catches.append({
                "date": r["Release_Date"][:10], "lat": lat, "lon": lon,
                "species": r["Species_Name"],
                "tag": r["Tag_Number"],
            })
    return catches


def get_dated_dirs():
    """Find all dated dirs that have cached data."""
    dirs = {}
    for d in os.listdir(BASE_DIR):
        sst_path = os.path.join(BASE_DIR, d, "sst_raw.nc")
        if os.path.exists(sst_path) and len(d) == 10 and d[4] == '-':
            dirs[d] = os.path.join(BASE_DIR, d)
    return dirs


# Pre-load catches and find available dates (do this once)
ALL_CATCHES = load_catches()
DATED_DIRS = get_dated_dirs()
DATES_WITH_DATA = sorted(set(c["date"] for c in ALL_CATCHES if c["date"] in DATED_DIRS))
CATCHES_WITH_DATA = [c for c in ALL_CATCHES if c["date"] in DATED_DIRS]
print(f"Loaded {len(CATCHES_WITH_DATA)} catches across {len(DATES_WITH_DATA)} dates with cached data")

# Find bathy tif
TIF_PATH = os.path.join(BASE_DIR, "bathy_gmrt.tif")
if not os.path.exists(TIF_PATH):
    for d in DATED_DIRS.values():
        candidate = os.path.join(d, "bathy_gmrt.tif")
        if os.path.exists(candidate):
            import shutil
            shutil.copy2(candidate, TIF_PATH)
            break


def evaluate_params(params):
    """Run scoring for all dates and return metrics at catch locations."""
    # Patch marlin_data module with trial parameters
    marlin_data.BLUE_MARLIN_WEIGHTS = {
        "sst":          params["w_sst"],
        "sst_front":    params["w_sst_front"],
        "sst_intrusion":params["w_sst_intrusion"],
        "chl":          params["w_chl"],
        "ssh":          params["w_ssh"],
        "current":      params["w_current"],
        "convergence":  params["w_convergence"],
        "mld":          params["w_mld"],
        "o2":           params["w_o2"],
        "clarity":      params["w_clarity"],
        "ssta":         params["w_ssta"],
        "boundary":     params["w_boundary"],
    }

    # Store tunable params as module-level attributes for the scoring function
    marlin_data._opt_sst_optimal = params["sst_optimal"]
    marlin_data._opt_sst_sigma = params["sst_sigma"]
    marlin_data._opt_front_floor = params["front_floor"]
    marlin_data._opt_intrusion_threshold = params["intrusion_threshold"]
    marlin_data._opt_intrusion_baseline = params["intrusion_baseline"]
    marlin_data._opt_shelf_boost = params["shelf_boost"]
    marlin_data._opt_east_bonus = params["east_bonus"]
    marlin_data._opt_synergy_factor = params["synergy_factor"]
    marlin_data._opt_chl_optimal = params["chl_optimal"]
    marlin_data._opt_chl_sigma = params["chl_sigma"]
    marlin_data._opt_boundary_threshold = params["boundary_threshold"]
    marlin_data._opt_boundary_blend = params["boundary_blend"]
    marlin_data._opt_ssta_optimal = params["ssta_optimal"]
    marlin_data._opt_ssta_sigma = params["ssta_sigma"]

    # Suppress all print output from scoring pipeline
    import io
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()

    scores = []
    try:
        for date_str in DATES_WITH_DATA:
            dated_dir = DATED_DIRS[date_str]
            marlin_data.OUTPUT_DIR = dated_dir
            dated_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
            if not os.path.exists(dated_tif) and os.path.exists(TIF_PATH):
                import shutil
                shutil.copy2(TIF_PATH, dated_tif)

            try:
                result = generate_blue_marlin_hotspots(BBOX, tif_path=dated_tif, date_str=date_str)
            except Exception:
                continue

            if not result or not isinstance(result, dict):
                continue

            grid = result["grid"]
            glats = result["lats"]
            glons = result["lons"]

            date_catches = [c for c in CATCHES_WITH_DATA if c["date"] == date_str]
            for c in date_catches:
                yi = np.argmin(np.abs(glats - c["lat"]))
                xi = np.argmin(np.abs(glons - c["lon"]))
                val = float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0
                scores.append(val)
    finally:
        sys.stdout = _real_stdout

    if len(scores) == 0:
        return {"mean": 0, "median": 0, "pct50": 0, "pct70": 0, "min": 0}

    scores = np.array(scores)
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "pct50": float(np.sum(scores >= 0.5) / len(scores)),
        "pct70": float(np.sum(scores >= 0.7) / len(scores)),
        "min": float(np.min(scores)),
    }


def objective(trial):
    """Optuna objective: maximize composite of mean score + floor bonus."""

    # --- Weights (must sum to ~1.0) ---
    # Sample raw weights then normalize
    raw_sst = trial.suggest_float("w_sst", 0.12, 0.30)
    raw_sst_front = trial.suggest_float("w_sst_front", 0.05, 0.20)
    raw_sst_intrusion = trial.suggest_float("w_sst_intrusion", 0.02, 0.12)
    raw_chl = trial.suggest_float("w_chl", 0.01, 0.10)
    raw_ssh = trial.suggest_float("w_ssh", 0.05, 0.20)
    raw_current = trial.suggest_float("w_current", 0.08, 0.25)
    raw_convergence = trial.suggest_float("w_convergence", 0.03, 0.15)
    raw_mld = trial.suggest_float("w_mld", 0.02, 0.12)
    raw_o2 = trial.suggest_float("w_o2", 0.01, 0.05)
    raw_clarity = trial.suggest_float("w_clarity", 0.01, 0.05)
    raw_ssta = trial.suggest_float("w_ssta", 0.02, 0.10)
    raw_boundary = trial.suggest_float("w_boundary", 0.02, 0.15)

    total = (raw_sst + raw_sst_front + raw_sst_intrusion + raw_chl +
             raw_ssh + raw_current + raw_convergence + raw_mld +
             raw_o2 + raw_clarity + raw_ssta + raw_boundary)

    # --- Scoring function parameters ---
    params = {
        "w_sst":          round(raw_sst / total, 4),
        "w_sst_front":    round(raw_sst_front / total, 4),
        "w_sst_intrusion":round(raw_sst_intrusion / total, 4),
        "w_chl":          round(raw_chl / total, 4),
        "w_ssh":          round(raw_ssh / total, 4),
        "w_current":      round(raw_current / total, 4),
        "w_convergence":  round(raw_convergence / total, 4),
        "w_mld":          round(raw_mld / total, 4),
        "w_o2":           round(raw_o2 / total, 4),
        "w_clarity":      round(raw_clarity / total, 4),
        "w_ssta":         round(raw_ssta / total, 4),
        "w_boundary":     round(raw_boundary / total, 4),
        "sst_optimal":       trial.suggest_float("sst_optimal", 22.0, 25.0),
        "sst_sigma":         trial.suggest_float("sst_sigma", 1.5, 3.0),
        "front_floor":       trial.suggest_float("front_floor", 0.0, 0.3),
        "intrusion_threshold": trial.suggest_float("intrusion_threshold", 0.1, 0.6),
        "intrusion_baseline":  trial.suggest_float("intrusion_baseline", 0.0, 0.3),
        "shelf_boost":       trial.suggest_float("shelf_boost", 0.1, 0.6),
        "east_bonus":        trial.suggest_float("east_bonus", 0.0, 0.5),
        "synergy_factor":    trial.suggest_float("synergy_factor", 0.0, 0.8),
        "chl_optimal":       trial.suggest_float("chl_optimal", 0.10, 0.40),
        "chl_sigma":         trial.suggest_float("chl_sigma", 0.2, 0.8),
        "boundary_threshold": trial.suggest_float("boundary_threshold", 0.15, 0.50),
        "boundary_blend":    trial.suggest_float("boundary_blend", 0.3, 0.8),
        "ssta_optimal":      trial.suggest_float("ssta_optimal", 0.5, 2.0),
        "ssta_sigma":        trial.suggest_float("ssta_sigma", 0.5, 2.5),
    }

    metrics = evaluate_params(params)

    # Composite objective: prioritize mean score, bonus for high floor
    # and high pct70, penalty for very low minimum scores
    score = (
        0.6 * metrics["mean"] +
        0.2 * metrics["pct70"] +
        0.1 * metrics["pct50"] +
        0.1 * max(metrics["min"], 0)  # reward raising the floor
    )

    # Store metrics as user attributes for inspection
    for k, v in metrics.items():
        trial.set_user_attr(k, round(v, 4))

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (1=sequential)")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name="marlin_scoring",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with current hand-tuned values as first trial
    study.enqueue_trial({
        "w_sst": 0.22, "w_sst_front": 0.14, "w_sst_intrusion": 0.08,
        "w_chl": 0.05, "w_ssh": 0.15, "w_current": 0.15,
        "w_convergence": 0.09, "w_mld": 0.07, "w_o2": 0.025, "w_clarity": 0.025,
        "sst_optimal": 23.5, "sst_sigma": 2.0,
        "front_floor": 0.15,
        "intrusion_threshold": 0.25, "intrusion_baseline": 0.2,
        "shelf_boost": 0.35,
        "east_bonus": 0.3, "synergy_factor": 0.4,
        "chl_optimal": 0.20, "chl_sigma": 0.4,
    })

    print(f"\nStarting Optuna optimization with {args.trials} trials...")
    print(f"{'='*70}")

    def callback(study, trial):
        if trial.number % 5 == 0 or trial.value == study.best_value:
            m = trial.user_attrs
            best = study.best_trial.user_attrs
            marker = " <-- BEST" if trial.value == study.best_value else ""
            print(f"Trial {trial.number:3d}: obj={trial.value:.4f} "
                  f"mean={m['mean']:.0%} med={m['median']:.0%} "
                  f">=70%={m['pct70']:.0%} min={m['min']:.0%}{marker}")

    study.optimize(objective, n_trials=args.trials, callbacks=[callback],
                   n_jobs=args.jobs, show_progress_bar=False)

    # Report results
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE — {args.trials} trials")
    print(f"{'='*70}")
    print(f"Best objective: {best.value:.4f}")
    print(f"  Mean score:    {best.user_attrs['mean']:.1%}")
    print(f"  Median score:  {best.user_attrs['median']:.1%}")
    print(f"  Score >= 50%:  {best.user_attrs['pct50']:.1%}")
    print(f"  Score >= 70%:  {best.user_attrs['pct70']:.1%}")
    print(f"  Min score:     {best.user_attrs['min']:.1%}")

    # Reconstruct normalized weights
    bp = best.params
    raw_keys = ["w_sst", "w_sst_front", "w_sst_intrusion", "w_chl",
                "w_ssh", "w_current", "w_convergence", "w_mld",
                "w_o2", "w_clarity"]
    total = sum(bp[k] for k in raw_keys)

    print(f"\nOptimal weights (normalized to sum=1.0):")
    for k in raw_keys:
        print(f"  {k:20s}: {bp[k]/total:.4f}")

    print(f"\nOptimal parameters:")
    for k in sorted(bp.keys()):
        if not k.startswith("w_"):
            print(f"  {k:25s}: {bp[k]:.4f}")

    # Save results
    output = {
        "best_objective": best.value,
        "metrics": best.user_attrs,
        "weights": {k: round(bp[k] / total, 4) for k in raw_keys},
        "params": {k: round(bp[k], 4) for k in sorted(bp.keys()) if not k.startswith("w_")},
        "raw_weights": {k: round(bp[k], 4) for k in raw_keys},
        "n_trials": args.trials,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    out_path = os.path.join(BASE_DIR, "optuna_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print code snippet to paste into marlin_data.py
    print(f"\n{'='*70}")
    print("Copy-paste into marlin_data.py BLUE_MARLIN_WEIGHTS:")
    print("{'='*70}")
    for k in raw_keys:
        name = k[2:]  # strip w_ prefix
        val = bp[k] / total
        print(f'    "{name}": {val:>20.4f},')
    print(f"\nScoring parameters to update:")
    print(f"  optimal_temp = {bp['sst_optimal']:.1f}")
    print(f"  sst_sigma = {bp['sst_sigma']:.2f}")
    print(f"  front_floor = {bp['front_floor']:.2f}")
    print(f"  intrusion_threshold = {bp['intrusion_threshold']:.2f}")
    print(f"  intrusion_baseline = {bp['intrusion_baseline']:.2f}")
    print(f"  shelf_boost = {bp['shelf_boost']:.2f}")
    print(f"  east_bonus = {bp['east_bonus']:.2f}")
    print(f"  synergy_factor = {bp['synergy_factor']:.2f}")
    print(f"  chl_optimal = {bp['chl_optimal']:.2f}")
    print(f"  chl_sigma = {bp['chl_sigma']:.2f}")


if __name__ == "__main__":
    main()
