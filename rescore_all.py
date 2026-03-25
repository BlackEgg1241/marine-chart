#!/usr/bin/env python3
"""
Rescore all historical hotspot GeoJSONs with current weights.

Parallelized: uses ProcessPoolExecutor to score multiple dates concurrently.

Usage:
    python rescore_all.py              # rescore everything
    python rescore_all.py --obs-only   # only observation dirs
    python rescore_all.py --bt-only    # only backtest dirs
    python rescore_all.py --workers 8  # control parallelism
"""

import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Zone bounds (Accessible Trench Zone — same as generate_forecast_summary.py)
ZONE_W = 114.98
ZONE_E = 115.3333
ZONE_S = -32.1667
ZONE_N = -31.7287

N_WORKERS = 12  # leave headroom for memory (~2GB per worker)

SUB_ZONES = {
    "canyon": [114.95, -32.02, 115.15, -31.85],
    "pgfc": [115.15, -32.12, 115.35, -31.92],
    "north": [114.98, -31.90, 115.25, -31.73],
    "south": [115.05, -32.17, 115.25, -32.05],
}


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
                if os.path.exists(os.path.join(full, "sst_raw.nc")):
                    dirs.append((d, full))
    return dirs


def _rescore_worker(args):
    """Worker: rescore a single date in its own process."""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    kind, date_str, dir_path, bbox, bathy_fallback, script_dir = args

    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import marlin_data

        marlin_data.OUTPUT_DIR = dir_path

        local_tif = os.path.join(dir_path, "bathy_gmrt.tif")
        if not os.path.exists(local_tif) and bathy_fallback and os.path.exists(bathy_fallback):
            shutil.copy2(bathy_fallback, local_tif)

        tif = local_tif if os.path.exists(local_tif) else None

        # SSTA contours
        sst_file = os.path.join(dir_path, "sst_raw.nc")
        if os.path.exists(sst_file):
            try:
                marlin_data.generate_ssta_contours(sst_file, date_str)
            except Exception:
                pass

        # Blue marlin hotspots
        result = marlin_data.generate_blue_marlin_hotspots(bbox, tif_path=tif, date_str=date_str)
        if result is None:
            return {'kind': kind, 'date': date_str, 'status': 'skip'}

        # Spanish Mackerel
        try:
            from species.spanish_mackerel import generate_spanish_mackerel_hotspots
            generate_spanish_mackerel_hotspots(bbox, tif_path=tif, date_str=date_str, output_dir=dir_path)
        except Exception:
            pass

        # SBT
        try:
            from species.southern_bluefin_tuna import generate_sbt_hotspots
            generate_sbt_hotspots(bbox, tif_path=tif, date_str=date_str, output_dir=dir_path)
        except Exception:
            pass

        stats = compute_zone_stats(result["grid"], result["lats"], result["lons"])
        var_scores = {}
        if result.get("sub_scores") and result.get("weights"):
            var_scores = compute_zone_subscores(
                result["sub_scores"], result["weights"],
                result["lats"], result["lons"])
        sz_scores = compute_subzone_stats(result["grid"], result["lats"], result["lons"])
        var_scores.update(sz_scores)

        return {
            'kind': kind, 'date': date_str, 'status': 'ok',
            'zone_max': stats[0], 'zone_mean': stats[1],
            'zone_median': stats[2], 'cells': stats[3],
            'var_scores': var_scores,
        }

    except Exception as e:
        return {'kind': kind, 'date': date_str, 'status': 'fail', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Rescore all hotspot GeoJSONs with current weights")
    parser.add_argument("--obs-only", action="store_true", help="Only rescore observation dirs")
    parser.add_argument("--bt-only", action="store_true", help="Only rescore backtest dirs")
    parser.add_argument("--pred-only", action="store_true", help="Only rescore prediction dirs")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help=f"Parallel workers (default {N_WORKERS})")
    args = parser.parse_args()

    import marlin_data
    bbox = dict(marlin_data.DEFAULT_BBOX)

    obs_dir = os.path.join(SCRIPT_DIR, "data")
    bt_dir = os.path.join(SCRIPT_DIR, "data", "backtest")
    pred_dir = os.path.join(SCRIPT_DIR, "data", "prediction")

    bathy_fallback = None
    for candidate in [
        os.path.join(bt_dir, "bathy_gmrt.tif"),
        os.path.join(obs_dir, "bathy_gmrt.tif"),
    ]:
        if os.path.exists(candidate):
            bathy_fallback = candidate
            break

    do_obs = not args.bt_only and not args.pred_only
    do_bt = not args.obs_only and not args.pred_only
    do_pred = not args.obs_only and not args.bt_only

    work_items = []
    if do_obs:
        for d, p in find_dated_dirs(obs_dir):
            work_items.append(("obs", d, p, bbox, bathy_fallback, SCRIPT_DIR))
    if do_bt:
        for d, p in find_dated_dirs(bt_dir):
            work_items.append(("bt", d, p, bbox, bathy_fallback, SCRIPT_DIR))
    if do_pred:
        for d, p in find_dated_dirs(pred_dir):
            work_items.append(("pred", d, p, bbox, bathy_fallback, SCRIPT_DIR))

    total = len(work_items)
    n_workers = min(args.workers, total)
    print(f"Rescoring {total} directories with {n_workers} workers", flush=True)
    print(f"Weights: {marlin_data.BLUE_MARLIN_WEIGHTS}", flush=True)
    print(flush=True)

    bt_results = []
    succeeded = 0
    failed = 0
    t0 = time.time()
    completed = 0

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_rescore_worker, item): item for item in work_items}
        for future in as_completed(futures):
            completed += 1
            r = future.result()
            tag = {"obs": "OBS", "bt": " BT", "pred": "PRD"}[r['kind']]

            if r['status'] == 'ok':
                succeeded += 1
                print(f"[{completed}/{total}] [{tag}] {r['date']} "
                      f"max={r['zone_max']}% mean={r['zone_mean']}% ({r['cells']} cells)", flush=True)

                if r['kind'] == 'bt':
                    dt = datetime.strptime(r['date'], "%Y-%m-%d")
                    entry = {
                        "date": r['date'],
                        "day_name": dt.strftime("%A"),
                        "month": dt.strftime("%B"),
                        "zone_max": r['zone_max'],
                        "zone_mean": r['zone_mean'],
                        "zone_median": r['zone_median'],
                        "zone_cells": r['cells'],
                    }
                    if r.get('var_scores'):
                        entry.update(r['var_scores'])
                    bt_results.append(entry)
            elif r['status'] == 'skip':
                failed += 1
                print(f"[{completed}/{total}] [{tag}] {r['date']} SKIP", flush=True)
            else:
                failed += 1
                print(f"[{completed}/{total}] [{tag}] {r['date']} FAILED: {r.get('error', '?')}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {succeeded} succeeded, {failed} failed in {elapsed:.0f}s")

    if bt_results:
        bt_results.sort(key=lambda x: x["date"])
        output_file = os.path.join(bt_dir, "backtest_results.json")
        out = {
            "description": "Blue marlin habitat backtest - zone scores for Accessible Trench Zone",
            "zone": {"lon_min": ZONE_W, "lon_max": ZONE_E, "lat_min": ZONE_S, "lat_max": ZONE_N},
            "weights": dict(marlin_data.BLUE_MARLIN_WEIGHTS),
            "rescored": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dates": bt_results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Updated {output_file} ({len(bt_results)} entries)")

    if do_pred:
        summary_script = os.path.join(SCRIPT_DIR, "generate_forecast_summary.py")
        if os.path.exists(summary_script):
            print("\nRegenerating forecast summary...")
            import subprocess
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
