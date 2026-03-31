#!/usr/bin/env python3
"""
validate_null_dates.py — False-positive baseline for habitat scoring.

Generates random dates within marlin season (Jan-Apr) that are NOT near known
catch dates. Runs the scoring pipeline and samples scores at catch GPS locations
to establish what "no-catch" days look like vs "catch" days.

If random season dates score as high as catch dates, our discrimination is poor.

Usage:
    python validate_null_dates.py              # 50 random dates, use cached data
    python validate_null_dates.py --n-dates 100
    python validate_null_dates.py --fetch      # fetch missing ocean data
"""

import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime, timedelta

import numpy as np

import marlin_data
from marlin_data import generate_blue_marlin_hotspots

BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}

BASE_DIR = "data"
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")

# Zone bounds for zone-max scoring (Accessible Trench Zone)
ZONE_W, ZONE_E = 114.98, 115.3333
ZONE_S, ZONE_N = -32.1667, -31.7287


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def _parse_date(date_str):
    if "/" in date_str:
        parts = date_str.strip().split("/")
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    return date_str[:10]


def load_catch_dates():
    """Load all blue marlin catch dates + GPS locations."""
    catches = []
    seen = set()

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("Species_Name", "").strip().upper() != "BLUE MARLIN":
                    continue
                lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
                lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
                date = r["Release_Date"][:10]
                key = (date, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date, "lat": lat, "lon": lon})

    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("species", "").strip().upper() != "BLUE MARLIN":
                    continue
                lat_str = r.get("lat", "").strip()
                lon_str = r.get("lon", "").strip()
                if not lat_str or not lon_str:
                    continue
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    continue
                date = _parse_date(r["date"])
                key = (date, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date, "lat": lat, "lon": lon})

    return catches


def generate_null_dates(catch_dates, n_dates=50, year_range=(2000, 2025),
                        season_months=(1, 2, 3, 4), exclusion_days=3):
    """Generate random dates within marlin season, excluding near-catch dates."""
    catch_dt_set = set()
    for d in catch_dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            for offset in range(-exclusion_days, exclusion_days + 1):
                catch_dt_set.add((dt + timedelta(days=offset)).strftime("%Y-%m-%d"))
        except ValueError:
            pass

    candidates = []
    for year in range(year_range[0], year_range[1] + 1):
        for month in season_months:
            for day in range(1, 32):
                try:
                    dt = datetime(year, month, day)
                    ds = dt.strftime("%Y-%m-%d")
                    if ds not in catch_dt_set:
                        candidates.append(ds)
                except ValueError:
                    pass

    random.seed(42)  # reproducible
    return sorted(random.sample(candidates, min(n_dates, len(candidates))))


def main():
    parser = argparse.ArgumentParser(description="Null-date validation for false-positive baseline")
    parser.add_argument("--n-dates", type=int, default=50, help="Number of random null dates")
    parser.add_argument("--fetch", action="store_true", help="Fetch missing ocean data from CMEMS")
    args = parser.parse_args()

    catches = load_catch_dates()
    catch_dates = sorted(set(c["date"] for c in catches))
    catch_locs = [(c["lat"], c["lon"]) for c in catches]
    print(f"Loaded {len(catches)} blue marlin catches across {len(catch_dates)} dates")

    # Mean catch location for zone sampling
    mean_lat = np.mean([c["lat"] for c in catches])
    mean_lon = np.mean([c["lon"] for c in catches])
    print(f"Mean catch location: {mean_lon:.3f}E, {abs(mean_lat):.3f}S")

    null_dates = generate_null_dates(catch_dates, n_dates=args.n_dates)
    print(f"Generated {len(null_dates)} null dates (season, not near catches)\n")

    # Bathy tif
    tif_path = os.path.join(BASE_DIR, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        for d in os.listdir(BASE_DIR):
            candidate = os.path.join(BASE_DIR, d, "bathy_gmrt.tif")
            if os.path.exists(candidate):
                shutil.copy2(candidate, tif_path)
                break

    null_results = []
    catch_results = []

    # --- Score null dates ---
    print("=== SCORING NULL DATES ===")
    for di, date_str in enumerate(null_dates):
        dated_dir = os.path.join(BASE_DIR, date_str)

        # Check if data exists
        sst_file = os.path.join(dated_dir, "sst_raw.nc")
        if not os.path.exists(sst_file):
            # Try backtest dir
            bt_dir = os.path.join(BASE_DIR, "backtest", date_str)
            if os.path.exists(os.path.join(bt_dir, "sst_raw.nc")):
                dated_dir = bt_dir
                sst_file = os.path.join(bt_dir, "sst_raw.nc")
            elif args.fetch:
                os.makedirs(dated_dir, exist_ok=True)
                try:
                    from validate_scoring import fetch_all
                    fetch_all(date_str, dated_dir)
                    sst_file = os.path.join(dated_dir, "sst_raw.nc")
                except Exception as e:
                    print(f"  [{di+1}/{len(null_dates)}] {date_str} FETCH FAILED: {e}")
                    continue
            else:
                continue

        if not os.path.exists(sst_file):
            continue

        marlin_data.OUTPUT_DIR = dated_dir
        dated_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
        if not os.path.exists(dated_tif) and os.path.exists(tif_path):
            shutil.copy2(tif_path, dated_tif)

        try:
            grid_result = generate_blue_marlin_hotspots(BBOX, tif_path=dated_tif, date_str=date_str)
            if grid_result is None:
                continue

            grid = grid_result["grid"]
            glats = grid_result["lats"]
            glons = grid_result["lons"]

            # Zone-max score (Accessible Trench Zone)
            lat_mask = (glats >= ZONE_S) & (glats <= ZONE_N)
            lon_mask = (glons >= ZONE_W) & (glons <= ZONE_E)
            zone = grid[np.ix_(lat_mask, lon_mask)]
            zone_valid = zone[~np.isnan(zone)]
            zone_max = float(np.nanmax(zone_valid)) if len(zone_valid) > 0 else 0
            zone_mean = float(np.nanmean(zone_valid)) if len(zone_valid) > 0 else 0

            # Sample at all catch locations
            loc_scores = []
            for lat, lon in catch_locs:
                yi = np.argmin(np.abs(glats - lat))
                xi = np.argmin(np.abs(glons - lon))
                s = float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0
                loc_scores.append(s)

            mean_at_locs = np.mean(loc_scores) if loc_scores else 0
            pct_70 = sum(1 for s in loc_scores if s >= 0.7) / max(len(loc_scores), 1)

            null_results.append({
                "date": date_str,
                "zone_max": zone_max,
                "zone_mean": zone_mean,
                "mean_at_catch_locs": mean_at_locs,
                "pct_70_at_catch_locs": pct_70,
            })
            print(f"  [{di+1}/{len(null_dates)}] {date_str} zone_max={zone_max:.0%} "
                  f"mean_at_locs={mean_at_locs:.0%} >=70%={pct_70:.0%}")
        except Exception as e:
            print(f"  [{di+1}/{len(null_dates)}] {date_str} FAILED: {e}")

    # --- Score catch dates ---
    print("\n=== SCORING CATCH DATES ===")
    for di, date_str in enumerate(catch_dates):
        dated_dir = os.path.join(BASE_DIR, date_str)
        sst_file = os.path.join(dated_dir, "sst_raw.nc")
        if not os.path.exists(sst_file):
            continue

        marlin_data.OUTPUT_DIR = dated_dir
        dated_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
        if not os.path.exists(dated_tif) and os.path.exists(tif_path):
            shutil.copy2(tif_path, dated_tif)

        try:
            grid_result = generate_blue_marlin_hotspots(BBOX, tif_path=dated_tif, date_str=date_str)
            if grid_result is None:
                continue

            grid = grid_result["grid"]
            glats = grid_result["lats"]
            glons = grid_result["lons"]

            lat_mask = (glats >= ZONE_S) & (glats <= ZONE_N)
            lon_mask = (glons >= ZONE_W) & (glons <= ZONE_E)
            zone = grid[np.ix_(lat_mask, lon_mask)]
            zone_valid = zone[~np.isnan(zone)]
            zone_max = float(np.nanmax(zone_valid)) if len(zone_valid) > 0 else 0
            zone_mean = float(np.nanmean(zone_valid)) if len(zone_valid) > 0 else 0

            date_catches = [c for c in catches if c["date"] == date_str]
            loc_scores = []
            for c in date_catches:
                yi = np.argmin(np.abs(glats - c["lat"]))
                xi = np.argmin(np.abs(glons - c["lon"]))
                s = float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0
                loc_scores.append(s)

            mean_at_locs = np.mean(loc_scores) if loc_scores else 0
            pct_70 = sum(1 for s in loc_scores if s >= 0.7) / max(len(loc_scores), 1)

            catch_results.append({
                "date": date_str,
                "zone_max": zone_max,
                "zone_mean": zone_mean,
                "mean_at_catch_locs": mean_at_locs,
                "pct_70_at_catch_locs": pct_70,
            })
            print(f"  [{di+1}/{len(catch_dates)}] {date_str} zone_max={zone_max:.0%} "
                  f"mean_at_locs={mean_at_locs:.0%} >=70%={pct_70:.0%}")
        except Exception as e:
            print(f"  [{di+1}/{len(catch_dates)}] {date_str} FAILED: {e}")

    # --- Analysis ---
    print(f"\n{'='*70}")
    print(f"NULL-DATE VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Catch dates scored: {len(catch_results)}")
    print(f"Null dates scored:  {len(null_results)}")

    if catch_results and null_results:
        catch_zone_max = [r["zone_max"] for r in catch_results]
        null_zone_max = [r["zone_max"] for r in null_results]
        catch_at_locs = [r["mean_at_catch_locs"] for r in catch_results]
        null_at_locs = [r["mean_at_catch_locs"] for r in null_results]

        print(f"\n--- Zone Max Scores ---")
        print(f"  Catch dates:  mean={np.mean(catch_zone_max):.1%} "
              f"median={np.median(catch_zone_max):.1%} range={min(catch_zone_max):.1%}-{max(catch_zone_max):.1%}")
        print(f"  Null dates:   mean={np.mean(null_zone_max):.1%} "
              f"median={np.median(null_zone_max):.1%} range={min(null_zone_max):.1%}-{max(null_zone_max):.1%}")
        print(f"  Lift (catch - null): {np.mean(catch_zone_max) - np.mean(null_zone_max):+.1%}")

        print(f"\n--- Scores at Catch GPS Locations ---")
        print(f"  Catch dates:  mean={np.mean(catch_at_locs):.1%} "
              f"median={np.median(catch_at_locs):.1%}")
        print(f"  Null dates:   mean={np.mean(null_at_locs):.1%} "
              f"median={np.median(null_at_locs):.1%}")
        print(f"  Lift (catch - null): {np.mean(catch_at_locs) - np.mean(null_at_locs):+.1%}")

        # Mann-Whitney U test
        try:
            from scipy.stats import mannwhitneyu
            stat, p_zone = mannwhitneyu(catch_zone_max, null_zone_max, alternative="greater")
            stat, p_locs = mannwhitneyu(catch_at_locs, null_at_locs, alternative="greater")
            print(f"\n--- Statistical Tests (Mann-Whitney U, one-sided) ---")
            print(f"  Zone max:       p={p_zone:.4f} {'***' if p_zone < 0.001 else '**' if p_zone < 0.01 else '*' if p_zone < 0.05 else 'ns'}")
            print(f"  At catch locs:  p={p_locs:.4f} {'***' if p_locs < 0.001 else '**' if p_locs < 0.01 else '*' if p_locs < 0.05 else 'ns'}")
        except ImportError:
            print("  (scipy not available for statistical test)")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(catch_at_locs)**2 + np.std(null_at_locs)**2) / 2)
        cohens_d = (np.mean(catch_at_locs) - np.mean(null_at_locs)) / pooled_std if pooled_std > 0 else 0
        print(f"\n  Cohen's d (at catch locs): {cohens_d:.2f} "
              f"({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

        # False positive rate
        catch_70_pct = np.mean([r["pct_70_at_catch_locs"] for r in catch_results])
        null_70_pct = np.mean([r["pct_70_at_catch_locs"] for r in null_results])
        print(f"\n--- False Positive Rate ---")
        print(f"  Catch dates >=70% at catch locs: {catch_70_pct:.0%}")
        print(f"  Null dates >=70% at catch locs:  {null_70_pct:.0%}")
        print(f"  If null >=70% is high, model lacks spatial discrimination")

    # Save results
    output = {
        "description": "Null-date validation — false positive baseline",
        "n_catch_dates": len(catch_results),
        "n_null_dates": len(null_results),
        "catch_results": catch_results,
        "null_results": null_results,
    }
    output_file = os.path.join(BASE_DIR, "null_date_validation.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
