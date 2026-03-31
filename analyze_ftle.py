#!/usr/bin/env python3
"""
analyze_ftle.py — Analyze FTLE structure at catch locations.
Checks: catch scores, percentiles, peak proximity, gradient at catch,
edge vs peak behavior, and comparison to other current-derived features.
"""

import csv
import os
import sys
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches():
    catches = []
    seen = set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Species_Name"].strip().upper() != "BLUE MARLIN":
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
                sp = r.get("species", "").strip().upper()
                if sp != "BLUE MARLIN":
                    continue
                lat_str = r.get("lat", "").strip()
                lon_str = r.get("lon", "").strip()
                if not lat_str or not lon_str:
                    continue
                try:
                    lat, lon = float(lat_str), float(lon_str)
                except ValueError:
                    continue
                date_raw = r["date"]
                if "/" in date_raw:
                    parts = date_raw.strip().split("/")
                    date = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                else:
                    date = date_raw[:10]
                key = (date, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date, "lat": lat, "lon": lon})
    return catches


def analyze_date(args):
    date_str, date_catches = args
    import marlin_data
    from marlin_data import generate_blue_marlin_hotspots

    dated_dir = os.path.join("data", date_str)
    if not os.path.exists(dated_dir):
        return None

    marlin_data.OUTPUT_DIR = dated_dir
    marlin_data.BLUE_MARLIN_WEIGHTS['ftle'] = 0.08  # force enable

    tif_path = os.path.join(dated_dir, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        tif_path = os.path.join("data", "bathy_gmrt.tif")

    # Redirect stdout to suppress scoring output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = generate_blue_marlin_hotspots(BBOX, tif_path=tif_path, date_str=date_str)
    except Exception:
        sys.stdout = old_stdout
        return None
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    sub_scores = result["sub_scores"]
    if "ftle" not in sub_scores:
        print(f"  {date_str}: no FTLE data")
        return None

    lats = result["lats"]
    lons = result["lons"]
    ftle_grid = sub_scores["ftle"]
    land = np.isnan(ftle_grid)
    ocean = ~land

    # Also grab other current-derived features for comparison
    shear_grid = sub_scores.get("current_shear")
    ow_grid = sub_scores.get("okubo_weiss")
    upwell_grid = sub_scores.get("upwelling_edge")
    vert_grid = sub_scores.get("vertical_velocity")

    date_results = []
    for c in date_catches:
        clat, clon = c["lat"], c["lon"]
        yi = np.argmin(np.abs(lats - clat))
        xi = np.argmin(np.abs(lons - clon))

        ftle_val = ftle_grid[yi, xi]
        if np.isnan(ftle_val):
            continue

        # Ocean stats
        ocean_vals = ftle_grid[ocean]
        catch_pctile = np.sum(ocean_vals <= ftle_val) / len(ocean_vals) * 100

        # Gradient at catch (spatial derivative)
        gy, gx = np.gradient(np.nan_to_num(ftle_grid, nan=0))
        grad_mag = np.sqrt(gx**2 + gy**2)
        catch_grad = grad_mag[yi, xi]
        ocean_grad = grad_mag[ocean]
        grad_pctile = np.sum(ocean_grad <= catch_grad) / len(ocean_grad) * 100

        # Peak within 5nm and 10nm
        nm_per_deg = 60.0
        for radius_nm in [5, 10]:
            r_deg = radius_nm / nm_per_deg
            lat_mask = np.abs(lats - clat) <= r_deg
            lon_mask = np.abs(lons - clon) <= r_deg
            local = ftle_grid[np.ix_(lat_mask, lon_mask)]
            local_ocean = local[~np.isnan(local)]
            if len(local_ocean) > 0:
                local_peak = np.max(local_ocean)
                local_mean = np.mean(local_ocean)
                # Find peak location
                local_full = ftle_grid.copy()
                local_full[~np.ix_(lat_mask, lon_mask)[0]] = -999
                local_full[land] = -999
                peak_yx = np.unravel_index(np.argmax(local_full), local_full.shape)
                peak_lat = lats[peak_yx[0]]
                peak_lon = lons[peak_yx[1]]
                dist_nm = np.sqrt(((clat - peak_lat) * nm_per_deg)**2 +
                                  ((clon - peak_lon) * nm_per_deg * np.cos(np.radians(clat)))**2)
                # Direction to peak
                dlat = peak_lat - clat
                dlon = peak_lon - clon
                angle = np.degrees(np.arctan2(dlon * np.cos(np.radians(clat)), dlat))
                if angle < 0:
                    angle += 360
                dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                dir_label = dirs[int((angle + 22.5) % 360 / 45)]

                if radius_nm == 5:
                    peak_5nm = local_peak
                    dist_5nm = dist_nm
                    ratio_5nm = ftle_val / local_peak if local_peak > 0 else 0
                    dir_5nm = dir_label
                    mean_5nm = local_mean
                else:
                    peak_10nm = local_peak
                    dist_10nm = dist_nm
                    ratio_10nm = ftle_val / local_peak if local_peak > 0 else 0
                    dir_10nm = dir_label
            else:
                if radius_nm == 5:
                    peak_5nm = dist_5nm = ratio_5nm = mean_5nm = 0
                    dir_5nm = "?"
                else:
                    peak_10nm = dist_10nm = ratio_10nm = 0
                    dir_10nm = "?"

        # Compare with other current-derived features
        comparisons = {}
        for name, grid in [("shear", shear_grid), ("okubo_weiss", ow_grid),
                           ("upwelling", upwell_grid), ("vert_vel", vert_grid)]:
            if grid is not None and not np.isnan(grid[yi, xi]):
                comparisons[name] = float(grid[yi, xi])

        date_results.append({
            "date": date_str,
            "lat": clat, "lon": clon,
            "ftle_score": float(ftle_val),
            "ftle_pctile": float(catch_pctile),
            "catch_gradient": float(catch_grad),
            "grad_pctile": float(grad_pctile),
            "peak_5nm": float(peak_5nm),
            "dist_5nm": float(dist_5nm),
            "ratio_5nm": float(ratio_5nm),
            "dir_5nm": dir_5nm,
            "mean_5nm": float(mean_5nm),
            "peak_10nm": float(peak_10nm),
            "dist_10nm": float(dist_10nm),
            "ratio_10nm": float(ratio_10nm),
            "dir_10nm": dir_10nm,
            "other_features": comparisons,
        })

    if date_results:
        print(f"  {date_str}: {len(date_results)} catches, "
              f"FTLE scores: {', '.join(f'{r['ftle_score']:.0%}' for r in date_results)}")
    return date_results


def main():
    catches = load_catches()
    print(f"Loaded {len(catches)} blue marlin catches")

    # Group by date
    by_date = {}
    for c in catches:
        by_date.setdefault(c["date"], []).append(c)

    # Filter to dates with data dirs
    valid = [(d, cs) for d, cs in sorted(by_date.items())
             if os.path.exists(os.path.join("data", d))]
    print(f"{len(valid)} dates with data directories")

    all_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for result in pool.map(analyze_date, valid):
            if result:
                all_results.extend(result)

    if not all_results:
        print("No FTLE results!")
        return

    print(f"\n{'='*70}")
    print(f"FTLE ANALYSIS — {len(all_results)} catches across {len(set(r['date'] for r in all_results))} dates")
    print(f"{'='*70}")

    scores = [r["ftle_score"] for r in all_results]
    pctiles = [r["ftle_pctile"] for r in all_results]
    grads = [r["catch_gradient"] for r in all_results]
    grad_pcts = [r["grad_pctile"] for r in all_results]
    dists_5 = [r["dist_5nm"] for r in all_results]
    ratios_5 = [r["ratio_5nm"] for r in all_results]
    dists_10 = [r["dist_10nm"] for r in all_results]
    ratios_10 = [r["ratio_10nm"] for r in all_results]

    print(f"\n--- FTLE Score at Catch ---")
    print(f"  Mean:   {np.mean(scores):.1%}")
    print(f"  Median: {np.median(scores):.1%}")
    print(f"  Min:    {np.min(scores):.1%}")
    print(f"  Max:    {np.max(scores):.1%}")
    print(f"  Std:    {np.std(scores):.1%}")

    print(f"\n--- Catch Percentile (vs ocean) ---")
    print(f"  Mean:   {np.mean(pctiles):.0f}th")
    print(f"  Median: {np.median(pctiles):.0f}th")
    print(f"  >= 50th: {sum(1 for p in pctiles if p >= 50)}/{len(pctiles)} ({sum(1 for p in pctiles if p >= 50)/len(pctiles):.0%})")
    print(f"  >= 70th: {sum(1 for p in pctiles if p >= 70)}/{len(pctiles)} ({sum(1 for p in pctiles if p >= 70)/len(pctiles):.0%})")

    print(f"\n--- Gradient at Catch (edge structure) ---")
    print(f"  Mean gradient:     {np.mean(grads):.4f}")
    print(f"  Median gradient:   {np.median(grads):.4f}")
    print(f"  Mean grad pctile:  {np.mean(grad_pcts):.0f}th")
    print(f"  Median grad pctile:{np.median(grad_pcts):.0f}th")
    print(f"  >= 50th: {sum(1 for p in grad_pcts if p >= 50)}/{len(grad_pcts)} ({sum(1 for p in grad_pcts if p >= 50)/len(grad_pcts):.0%})")

    print(f"\n--- Peak Proximity (5nm radius) ---")
    print(f"  Mean distance to local peak:  {np.mean(dists_5):.1f} nm")
    print(f"  Median distance:              {np.median(dists_5):.1f} nm")
    print(f"  Mean catch/peak ratio:        {np.mean(ratios_5):.1%}")
    print(f"  Median catch/peak ratio:      {np.median(ratios_5):.1%}")
    # Direction distribution
    dirs_5 = [r["dir_5nm"] for r in all_results]
    dir_counts = {}
    for d in dirs_5:
        dir_counts[d] = dir_counts.get(d, 0) + 1
    print(f"  Peak direction from catch:")
    for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        n = dir_counts.get(d, 0)
        print(f"    {d:3s}: {n:2d} ({n/len(dirs_5):.0%})")

    print(f"\n--- Peak Proximity (10nm radius) ---")
    print(f"  Mean distance to local peak:  {np.mean(dists_10):.1f} nm")
    print(f"  Median distance:              {np.median(dists_10):.1f} nm")
    print(f"  Mean catch/peak ratio:        {np.mean(ratios_10):.1%}")
    print(f"  Median catch/peak ratio:      {np.median(ratios_10):.1%}")

    # Score distribution buckets
    print(f"\n--- Score Distribution ---")
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        n = sum(1 for s in scores if lo <= s < hi)
        print(f"  {lo:.0%}-{hi:.0%}: {n} ({n/len(scores):.0%})")

    # Edge vs Peak behavior
    print(f"\n--- Edge vs Peak Behavior ---")
    high_score = [r for r in all_results if r["ftle_score"] >= 0.7]
    mid_score = [r for r in all_results if 0.3 <= r["ftle_score"] < 0.7]
    low_score = [r for r in all_results if r["ftle_score"] < 0.3]
    print(f"  High (>=0.7): {len(high_score)} catches ({len(high_score)/len(all_results):.0%})")
    print(f"  Mid (0.3-0.7): {len(mid_score)} catches ({len(mid_score)/len(all_results):.0%})")
    print(f"  Low (<0.3):    {len(low_score)} catches ({len(low_score)/len(all_results):.0%})")
    if high_score:
        print(f"  High-score avg gradient pctile: {np.mean([r['grad_pctile'] for r in high_score]):.0f}th")
    if mid_score:
        print(f"  Mid-score avg gradient pctile:  {np.mean([r['grad_pctile'] for r in mid_score]):.0f}th")

    # Correlation with other current-derived features
    print(f"\n--- Correlation with Other Current Features ---")
    for feat_name in ["shear", "okubo_weiss", "upwelling", "vert_vel"]:
        pairs = [(r["ftle_score"], r["other_features"].get(feat_name))
                 for r in all_results if feat_name in r["other_features"]]
        if len(pairs) >= 5:
            ftle_vals = [p[0] for p in pairs]
            other_vals = [p[1] for p in pairs]
            corr = np.corrcoef(ftle_vals, other_vals)[0, 1]
            print(f"  FTLE vs {feat_name:15s}: r={corr:+.3f} (n={len(pairs)})")

    # Summary verdict
    print(f"\n{'='*70}")
    print(f"VERDICT:")
    median_pct = np.median(pctiles)
    median_score = np.median(scores)
    median_grad_pct = np.median(grad_pcts)
    if median_pct >= 60 and median_score >= 0.5:
        print(f"  PEAK feature — catches at high FTLE values (median {median_score:.0%}, {median_pct:.0f}th pctile)")
    elif median_pct >= 40 and median_grad_pct >= 55:
        print(f"  EDGE feature — catches at FTLE transitions (median score {median_score:.0%}, gradient {median_grad_pct:.0f}th pctile)")
    elif median_pct < 40:
        print(f"  WEAK/NO signal — catches not preferentially at FTLE features (median {median_pct:.0f}th pctile)")
    else:
        print(f"  MIXED — median score {median_score:.0%}, pctile {median_pct:.0f}th, gradient {median_grad_pct:.0f}th")
    print(f"{'='*70}")

    # Save results
    with open("data/ftle_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to data/ftle_analysis.json")


if __name__ == "__main__":
    main()
