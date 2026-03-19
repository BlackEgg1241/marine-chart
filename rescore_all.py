#!/usr/bin/env python3
"""
Rescore all historical hotspot GeoJSONs with current weights.

Iterates data/YYYY-MM-DD/ (observations) and data/backtest/YYYY-MM-DD/ (backtest)
directories, regenerates blue_marlin_hotspots.geojson and ssta_contours.geojson
using the current BLUE_MARLIN_WEIGHTS, and updates backtest_results.json.

No data is fetched — only existing cached NetCDF files are used.

Usage:
    python rescore_all.py              # rescore everything
    python rescore_all.py --obs-only   # only observation dirs
    python rescore_all.py --bt-only    # only backtest dirs
"""

import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Zone bounds (same as backtest_habitat.py)
ZONE_W = 114.98
ZONE_E = 115.3333
ZONE_S = -32.1667
ZONE_N = -31.7287


def compute_zone_stats(grid, lats, lons):
    lat_mask = (lats >= ZONE_S) & (lats <= ZONE_N)
    lon_mask = (lons >= ZONE_W) & (lons <= ZONE_E)
    zone = grid[np.ix_(lat_mask, lon_mask)]
    if zone.size == 0:
        return None, None, None, 0
    valid = zone[~np.isnan(zone)]
    if valid.size == 0:
        return None, None, None, 0
    return (
        round(float(np.nanmax(valid)) * 100, 1),
        round(float(np.nanmean(valid)) * 100, 1),
        round(float(np.nanmedian(valid)) * 100, 1),
        int(valid.size),
    )


def compute_zone_subscores(sub_scores, weights, lats, lons):
    lat_mask = (lats >= ZONE_S) & (lats <= ZONE_N)
    lon_mask = (lons >= ZONE_W) & (lons <= ZONE_E)
    result = {}
    for var_name, grid in sub_scores.items():
        zone = grid[np.ix_(lat_mask, lon_mask)]
        valid = zone[~np.isnan(zone)] if zone.size > 0 else np.array([])
        if valid.size > 0:
            result[f"s_{var_name}"] = round(float(np.nanmean(valid)), 4)
        else:
            result[f"s_{var_name}"] = None
    for var_name, w in weights.items():
        result[f"w_{var_name}"] = w
    return result


SUB_ZONES = {
    "canyon": [114.95, -32.02, 115.15, -31.85],
    "pgfc": [115.15, -32.12, 115.35, -31.92],
    "north": [114.98, -31.90, 115.25, -31.73],
    "south": [115.05, -32.17, 115.25, -32.05],
}


def compute_subzone_stats(grid, lats, lons):
    result = {}
    for key, (w, s, e, n) in SUB_ZONES.items():
        lat_mask = (lats >= s) & (lats <= n)
        lon_mask = (lons >= w) & (lons <= e)
        zone = grid[np.ix_(lat_mask, lon_mask)]
        valid = zone[~np.isnan(zone)] if zone.size > 0 else np.array([])
        if valid.size > 0:
            result[f"sz_{key}_max"] = round(float(np.nanmax(valid)) * 100, 1)
            result[f"sz_{key}_mean"] = round(float(np.nanmean(valid)) * 100, 1)
        else:
            result[f"sz_{key}_max"] = None
            result[f"sz_{key}_mean"] = None
    return result


def find_dated_dirs(base_dir):
    """Find all YYYY-MM-DD subdirectories."""
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for d in sorted(os.listdir(base_dir)):
        if len(d) == 10 and d[4] == '-' and d[7] == '-':
            full = os.path.join(base_dir, d)
            if os.path.isdir(full):
                # Must have at least sst_raw.nc
                if os.path.exists(os.path.join(full, "sst_raw.nc")):
                    dirs.append((d, full))
    return dirs


def rescore_dir(date_str, dir_path, bbox, bathy_tif_fallback=None):
    """Rescore a single directory. Returns (zone_max, zone_mean, zone_median, cells, var_scores) or None."""
    import marlin_data

    marlin_data.OUTPUT_DIR = dir_path

    # Ensure bathymetry is available
    local_tif = os.path.join(dir_path, "bathy_gmrt.tif")
    if not os.path.exists(local_tif) and bathy_tif_fallback and os.path.exists(bathy_tif_fallback):
        shutil.copy2(bathy_tif_fallback, local_tif)

    tif = local_tif if os.path.exists(local_tif) else None

    # Generate SSTA contours
    sst_file = os.path.join(dir_path, "sst_raw.nc")
    if os.path.exists(sst_file):
        try:
            marlin_data.generate_ssta_contours(sst_file, date_str)
        except Exception as e:
            pass  # SSTA is optional

    # Generate hotspots
    result = marlin_data.generate_blue_marlin_hotspots(bbox, tif_path=tif, date_str=date_str)
    if result is None:
        return None

    stats = compute_zone_stats(result["grid"], result["lats"], result["lons"])
    var_scores = {}
    if result.get("sub_scores") and result.get("weights"):
        var_scores = compute_zone_subscores(
            result["sub_scores"], result["weights"],
            result["lats"], result["lons"])
    sz_scores = compute_subzone_stats(result["grid"], result["lats"], result["lons"])
    var_scores.update(sz_scores)

    return stats[0], stats[1], stats[2], stats[3], var_scores


def main():
    parser = argparse.ArgumentParser(description="Rescore all hotspot GeoJSONs with current weights")
    parser.add_argument("--obs-only", action="store_true", help="Only rescore observation dirs")
    parser.add_argument("--bt-only", action="store_true", help="Only rescore backtest dirs")
    parser.add_argument("--pred-only", action="store_true", help="Only rescore prediction dirs")
    args = parser.parse_args()

    import marlin_data
    bbox = dict(marlin_data.DEFAULT_BBOX)

    obs_dir = os.path.join(SCRIPT_DIR, "data")
    bt_dir = os.path.join(SCRIPT_DIR, "data", "backtest")
    pred_dir = os.path.join(SCRIPT_DIR, "data", "prediction")

    # Find a shared bathymetry file
    bathy_fallback = None
    for candidate in [
        os.path.join(bt_dir, "bathy_gmrt.tif"),
        os.path.join(obs_dir, "bathy_gmrt.tif"),
    ]:
        if os.path.exists(candidate):
            bathy_fallback = candidate
            break

    # Determine which sets to process (default: all)
    do_obs = not args.bt_only and not args.pred_only
    do_bt = not args.obs_only and not args.pred_only
    do_pred = not args.obs_only and not args.bt_only

    dirs_to_process = []
    if do_obs:
        obs_dirs = find_dated_dirs(obs_dir)
        dirs_to_process.extend(("obs", d, p) for d, p in obs_dirs)
    if do_bt:
        bt_dirs = find_dated_dirs(bt_dir)
        dirs_to_process.extend(("bt", d, p) for d, p in bt_dirs)
    if do_pred:
        pred_dirs = find_dated_dirs(pred_dir)
        dirs_to_process.extend(("pred", d, p) for d, p in pred_dirs)

    total = len(dirs_to_process)
    print(f"Rescoring {total} directories with current weights")
    print(f"Weights: {marlin_data.BLUE_MARLIN_WEIGHTS}")
    print()

    bt_results = []
    succeeded = 0
    failed = 0
    t0 = time.time()

    for i, (kind, date_str, dir_path) in enumerate(dirs_to_process):
        tag = {"obs": "OBS", "bt": " BT", "pred": "PRD"}[kind]
        print(f"[{i+1}/{total}] [{tag}] {date_str} ... ", end="", flush=True)

        try:
            result = rescore_dir(date_str, dir_path, bbox, bathy_fallback)
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            continue

        if result is None:
            print("SKIP (no SST)")
            failed += 1
            continue

        zone_max, zone_mean, zone_median, cells, var_scores = result
        print(f"max={zone_max}% mean={zone_mean}% ({cells} cells)")
        succeeded += 1

        # Collect backtest results
        if kind == "bt":
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            entry = {
                "date": date_str,
                "day_name": dt.strftime("%A"),
                "month": dt.strftime("%B"),
                "zone_max": zone_max,
                "zone_mean": zone_mean,
                "zone_median": zone_median,
                "zone_cells": cells,
            }
            if var_scores:
                entry.update(var_scores)
            bt_results.append(entry)

    elapsed = time.time() - t0
    print(f"\nDone: {succeeded} succeeded, {failed} failed in {elapsed:.0f}s")

    # Update backtest_results.json
    if bt_results:
        bt_results.sort(key=lambda x: x["date"])
        output_file = os.path.join(bt_dir, "backtest_results.json")
        out = {
            "description": "Blue marlin habitat backtest — zone scores for Accessible Trench Zone",
            "zone": {"lon_min": ZONE_W, "lon_max": ZONE_E, "lat_min": ZONE_S, "lat_max": ZONE_N},
            "weights": dict(marlin_data.BLUE_MARLIN_WEIGHTS),
            "rescored": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dates": bt_results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Updated {output_file} ({len(bt_results)} entries)")

    # Regenerate forecast summary if prediction GeoJSONs were rescored
    if do_pred:
        summary_script = os.path.join(SCRIPT_DIR, "generate_forecast_summary.py")
        if os.path.exists(summary_script):
            print("\nRegenerating forecast summary...")
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, summary_script],
                cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Forecast summary failed: {result.stderr[:200]}")


if __name__ == "__main__":
    main()
