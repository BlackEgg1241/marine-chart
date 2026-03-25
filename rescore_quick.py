#!/usr/bin/env python3
"""
Quick rescore: only catch dates with GPS coords + current 7-day forecast.
Much faster than rescore_all.py which processes all observation + backtest + prediction dirs.

Usage:
    python rescore_quick.py
"""

import os
import sys
import csv
import shutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def get_catch_dates():
    """Get unique dates from Export.csv — only records with GPS coordinates."""
    dates = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                species = r.get('Species_Name', r.get('species', '')).upper()
                if 'BLUE MARLIN' not in species:
                    continue
                # Only include records with GPS coordinates
                lat_raw = r.get('Latitude', r.get('lat', '')).strip()
                lon_raw = r.get('Longitude', r.get('lon', '')).strip()
                if not lat_raw or not lon_raw:
                    continue
                d = r.get('Release_Date', r.get('date', '')).strip()
                if d:
                    # Handle ISO format with time component
                    if 'T' in d:
                        d = d.split('T')[0]
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                        try:
                            dt = datetime.strptime(d, fmt)
                            dates.add(dt.strftime('%Y-%m-%d'))
                            break
                        except ValueError:
                            continue
    return sorted(dates)


def get_prediction_dates():
    """Get today + 6 days forecast dates (7 total)."""
    pred_dir = os.path.join(SCRIPT_DIR, "data", "prediction")
    if not os.path.isdir(pred_dir):
        return []
    today = datetime.now()
    dirs = []
    for i in range(7):
        d = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        full = os.path.join(pred_dir, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "sst_raw.nc")):
            dirs.append(d)
    return dirs


def rescore_dir(date_str, dir_path, bbox, bathy_fallback=None):
    import marlin_data
    marlin_data.OUTPUT_DIR = dir_path

    local_tif = os.path.join(dir_path, "bathy_gmrt.tif")
    if not os.path.exists(local_tif) and bathy_fallback and os.path.exists(bathy_fallback):
        shutil.copy2(bathy_fallback, local_tif)
    tif = local_tif if os.path.exists(local_tif) else None

    sst_file = os.path.join(dir_path, "sst_raw.nc")
    if os.path.exists(sst_file):
        try:
            marlin_data.generate_ssta_contours(sst_file, date_str)
        except Exception:
            pass

    result = marlin_data.generate_blue_marlin_hotspots(bbox, tif_path=tif, date_str=date_str)
    if result is None:
        return False

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

    return True


def _rescore_worker(args):
    """Worker for parallel rescoring. Runs in a spawned subprocess."""
    date_str, dir_path, bbox, bathy_fallback, label = args
    # Suppress stdout in workers to avoid interleaved output
    sys.stdout = open(os.devnull, 'w')
    try:
        ok = rescore_dir(date_str, dir_path, bbox, bathy_fallback)
        return (date_str, label, ok, None)
    except Exception as e:
        return (date_str, label, False, str(e))


def main():
    import marlin_data
    bbox = dict(marlin_data.DEFAULT_BBOX)

    # Find bathy fallback
    bathy_fallback = None
    for d in os.listdir(os.path.join(SCRIPT_DIR, "data")):
        tif = os.path.join(SCRIPT_DIR, "data", d, "bathy_gmrt.tif")
        if os.path.exists(tif):
            bathy_fallback = tif
            break

    # Catch dates
    catch_dates = get_catch_dates()
    catch_dirs = []
    for d in catch_dates:
        dp = os.path.join(SCRIPT_DIR, "data", d)
        if os.path.isdir(dp) and os.path.exists(os.path.join(dp, "sst_raw.nc")):
            catch_dirs.append((d, dp))

    # Prediction dates
    pred_dates = get_prediction_dates()
    pred_dirs = [(d, os.path.join(SCRIPT_DIR, "data", "prediction", d)) for d in pred_dates]

    total = len(catch_dirs) + len(pred_dirs)
    n_workers = min(os.cpu_count() or 1, total, 16)
    print(f"Quick rescore: {len(catch_dirs)} catch dates + {len(pred_dirs)} prediction dates = {total} total ({n_workers} workers)")
    t0 = time.time()

    done = 0
    failed = 0

    # Build work items
    work_items = []
    for d, dp in catch_dirs:
        work_items.append((d, dp, bbox, bathy_fallback, "CATCH"))
    for d, dp in pred_dirs:
        work_items.append((d, dp, bbox, bathy_fallback, "PRED"))

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_rescore_worker, item): item for item in work_items}
        for i, future in enumerate(as_completed(futures)):
            date_str, label, ok, err = future.result()
            if err:
                print(f"  [{i+1}/{total}] {label} {date_str} ... FAIL: {err}")
                failed += 1
            elif ok:
                print(f"  [{i+1}/{total}] {label} {date_str} ... OK")
                done += 1
            else:
                print(f"  [{i+1}/{total}] {label} {date_str} ... SKIP")

    elapsed = time.time() - t0
    print(f"\nDone: {done} rescored, {failed} failed in {elapsed:.0f}s")

    # Regenerate forecast summary
    if pred_dirs:
        try:
            import generate_forecast_summary
            generate_forecast_summary.main()
            print("Forecast summary regenerated")
        except Exception as e:
            print(f"Forecast summary failed: {e}")


if __name__ == "__main__":
    main()
