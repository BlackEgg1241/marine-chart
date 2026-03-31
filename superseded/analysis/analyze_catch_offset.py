#!/usr/bin/env python3
"""
analyze_catch_offset.py — Investigate why blue marlin catches sit on the EDGES
of hotspot zones rather than in the centers.

Hypothesis: marlin hunt the edges/transitions of oceanographic features,
not the optimal peaks. This script quantifies the offset pattern.
"""

import csv
import os
import sys
import shutil
from collections import defaultdict

import numpy as np

import marlin_data
from marlin_data import generate_blue_marlin_hotspots

BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")
BASE_DIR = "data"


def ddm_to_dd(raw_str, negative=False):
    """Convert degrees.minutes string (e.g. '31.49') to decimal degrees."""
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


def load_catches():
    catches = []
    seen = set()

    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            wt = float(r["Weight"]) if r["Weight"] else None
            ln = float(r["Length"]) if r["Length"] and r["Length"] != "0" else None
            date = r["Release_Date"][:10]
            key = (date, round(lat, 4), round(lon, 4))
            if key not in seen:
                seen.add(key)
                catches.append({
                    "date": date, "lat": lat, "lon": lon,
                    "species": r["Species_Name"], "weight": wt, "length": ln,
                    "tag": r["Tag_Number"],
                })

    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
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
                    catches.append({
                        "date": date, "lat": lat, "lon": lon,
                        "species": r.get("species", ""),
                        "weight": None, "length": None,
                        "tag": r.get("tag", ""),
                    })

    return [c for c in catches if c["species"].strip().upper() == "BLUE MARLIN"]


def nm_to_deg(nm):
    """Convert nautical miles to approximate degrees at Perth latitude."""
    return nm / 60.0  # 1 degree ~ 60 nm


def find_peak_within_radius(grid, lats, lons, yi, xi, radius_nm):
    """Find peak score within radius_nm of pixel (yi, xi). Returns (peak_val, peak_yi, peak_xi, dist_nm)."""
    radius_deg = nm_to_deg(radius_nm)
    lat_center = lats[yi]
    lon_center = lons[xi]

    # Lon correction for latitude
    cos_lat = np.cos(np.radians(abs(lat_center)))

    # Find pixel ranges
    lat_range = np.abs(lats - lat_center) <= radius_deg
    lon_range = np.abs(lons - lon_center) <= (radius_deg / cos_lat)

    lat_indices = np.where(lat_range)[0]
    lon_indices = np.where(lon_range)[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return np.nan, yi, xi, 0.0

    best_val = -1
    best_yi, best_xi = yi, xi
    for li in lat_indices:
        for lj in lon_indices:
            # Check actual distance in nm
            dlat = (lats[li] - lat_center) * 60.0
            dlon = (lons[lj] - lon_center) * 60.0 * cos_lat
            dist = np.sqrt(dlat**2 + dlon**2)
            if dist <= radius_nm:
                val = grid[li, lj]
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    best_yi = li
                    best_xi = lj

    if best_val < 0:
        return np.nan, yi, xi, 0.0

    # Distance to peak in nm
    dlat = (lats[best_yi] - lat_center) * 60.0
    dlon = (lons[best_xi] - lon_center) * 60.0 * cos_lat
    dist_to_peak = np.sqrt(dlat**2 + dlon**2)

    return best_val, best_yi, best_xi, dist_to_peak


def direction_to_peak(lats, lons, yi, xi, peak_yi, peak_xi):
    """Return compass bearing from catch to peak."""
    if yi == peak_yi and xi == peak_xi:
        return "SAME"
    dlat = lats[peak_yi] - lats[yi]
    dlon = lons[peak_xi] - lons[xi]
    angle = np.degrees(np.arctan2(dlon, dlat)) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((angle + 22.5) / 45) % 8
    return dirs[idx]


def distance_to_contour(grid, lats, lons, yi, xi, contour_val, search_radius_px=50):
    """Find distance (nm) from pixel to nearest pixel at the given contour value."""
    cos_lat = np.cos(np.radians(abs(lats[yi])))
    best_dist = 999.0

    y_lo = max(0, yi - search_radius_px)
    y_hi = min(grid.shape[0], yi + search_radius_px)
    x_lo = max(0, xi - search_radius_px)
    x_hi = min(grid.shape[1], xi + search_radius_px)

    sub = grid[y_lo:y_hi, x_lo:x_hi]

    # Find cells that cross the contour value (within 0.02 of contour)
    diff = np.abs(sub - contour_val)
    crossings = np.argwhere(diff < 0.02)

    for cy, cx in crossings:
        abs_y = cy + y_lo
        abs_x = cx + x_lo
        dlat = (lats[abs_y] - lats[yi]) * 60.0
        dlon = (lons[abs_x] - lons[xi]) * 60.0 * cos_lat
        d = np.sqrt(dlat**2 + dlon**2)
        if d < best_dist:
            best_dist = d

    return best_dist if best_dist < 999 else np.nan


def main():
    print("=" * 70, flush=True)
    print("CATCH OFFSET ANALYSIS — Why catches sit on hotspot edges", flush=True)
    print("=" * 70, flush=True)

    catches = load_catches()
    dates = sorted(set(c["date"] for c in catches))
    print(f"\nLoaded {len(catches)} blue marlin catches across {len(dates)} dates", flush=True)

    # Only process dates that have existing data
    tif_path = os.path.join(BASE_DIR, "bathy_gmrt.tif")

    # Collect results
    all_catch_data = []
    feature_gradients = defaultdict(list)  # feature_name -> list of gradient magnitudes at catches
    feature_gradients_ocean = defaultdict(list)  # ocean-wide average gradient per feature

    processed = 0
    skipped = 0

    for di, date_str in enumerate(dates):
        dated_dir = os.path.join(BASE_DIR, date_str)
        sst_path = os.path.join(dated_dir, "sst_raw.nc")

        if not os.path.exists(sst_path):
            skipped += 1
            continue

        date_catches = [c for c in catches if c["date"] == date_str]
        print(f"\n[{di+1}/{len(dates)}] {date_str} -- {len(date_catches)} catches", flush=True)

        # Set output dir for marlin_data
        marlin_data.OUTPUT_DIR = dated_dir

        # Copy bathy tif if needed
        dated_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
        if not os.path.exists(dated_tif) and os.path.exists(tif_path):
            shutil.copy2(tif_path, dated_tif)

        try:
            result = generate_blue_marlin_hotspots(BBOX, tif_path=dated_tif)
        except Exception as e:
            print(f"  Scoring failed: {e}", flush=True)
            skipped += 1
            continue

        if not result or not isinstance(result, dict):
            skipped += 1
            continue

        grid = result["grid"]
        lats = result["lats"]
        lons = result["lons"]
        sub_scores = result["sub_scores"]
        processed += 1

        # Compute gradient of composite score
        grad_y, grad_x = np.gradient(grid)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        # Compute per-feature gradients
        feat_grad_mags = {}
        for fname, farr in sub_scores.items():
            fy, fx = np.gradient(farr)
            feat_grad_mags[fname] = np.sqrt(fy**2 + fx**2)
            # Ocean-wide average gradient (excluding NaN and land)
            valid = feat_grad_mags[fname][~np.isnan(feat_grad_mags[fname])]
            if len(valid) > 0:
                feature_gradients_ocean[fname].append(float(np.nanmean(valid)))

        for c in date_catches:
            yi = np.argmin(np.abs(lats - c["lat"]))
            xi = np.argmin(np.abs(lons - c["lon"]))

            # Check bounds
            if yi < 0 or yi >= grid.shape[0] or xi < 0 or xi >= grid.shape[1]:
                continue

            catch_score = float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0.0
            catch_gradient = float(grad_mag[yi, xi]) if not np.isnan(grad_mag[yi, xi]) else 0.0

            # Find peaks at 3nm, 5nm, 10nm
            peak_3, pyi3, pxi3, dist_3 = find_peak_within_radius(grid, lats, lons, yi, xi, 3)
            peak_5, pyi5, pxi5, dist_5 = find_peak_within_radius(grid, lats, lons, yi, xi, 5)
            peak_10, pyi10, pxi10, dist_10 = find_peak_within_radius(grid, lats, lons, yi, xi, 10)

            ratio_3 = catch_score / peak_3 if peak_3 > 0 and not np.isnan(peak_3) else np.nan
            ratio_5 = catch_score / peak_5 if peak_5 > 0 and not np.isnan(peak_5) else np.nan
            ratio_10 = catch_score / peak_10 if peak_10 > 0 and not np.isnan(peak_10) else np.nan

            dir_10 = direction_to_peak(lats, lons, yi, xi, pyi10, pxi10)

            # Distance to contour lines
            dist_70 = distance_to_contour(grid, lats, lons, yi, xi, 0.70)
            dist_80 = distance_to_contour(grid, lats, lons, yi, xi, 0.80)
            dist_90 = distance_to_contour(grid, lats, lons, yi, xi, 0.90)

            # Per-feature gradients at catch location
            catch_feat_grads = {}
            for fname, gm in feat_grad_mags.items():
                val = float(gm[yi, xi]) if not np.isnan(gm[yi, xi]) else 0.0
                catch_feat_grads[fname] = val
                feature_gradients[fname].append(val)

            entry = {
                "date": c["date"], "lat": c["lat"], "lon": c["lon"],
                "catch_score": catch_score,
                "catch_gradient": catch_gradient,
                "peak_3nm": peak_3, "peak_5nm": peak_5, "peak_10nm": peak_10,
                "ratio_3nm": ratio_3, "ratio_5nm": ratio_5, "ratio_10nm": ratio_10,
                "dist_to_peak_3nm": dist_3, "dist_to_peak_5nm": dist_5, "dist_to_peak_10nm": dist_10,
                "dir_to_peak_10nm": dir_10,
                "dist_contour_70": dist_70, "dist_contour_80": dist_80, "dist_contour_90": dist_90,
                "feat_grads": catch_feat_grads,
            }
            all_catch_data.append(entry)
            print(f"  {c['lon']:.2f}E {abs(c['lat']):.2f}S: "
                  f"score={catch_score:.0%} peak10={peak_10:.0%} "
                  f"ratio={ratio_10:.2f} grad={catch_gradient:.4f} dir={dir_10}", flush=True)

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Dates processed: {processed}, skipped: {skipped}", flush=True)
    print(f"Catch records analyzed: {len(all_catch_data)}", flush=True)

    if not all_catch_data:
        print("No data to analyze!", flush=True)
        return

    # --- 1. Catch Score Distribution ---
    scores = [d["catch_score"] for d in all_catch_data]
    print(f"\n--- CATCH SCORE DISTRIBUTION ---", flush=True)
    print(f"  Mean:   {np.mean(scores):.1%}", flush=True)
    print(f"  Median: {np.median(scores):.1%}", flush=True)
    print(f"  Std:    {np.std(scores):.1%}", flush=True)
    print(f"  Min:    {np.min(scores):.1%}", flush=True)
    print(f"  Max:    {np.max(scores):.1%}", flush=True)

    # --- 2. Catch-to-Peak Ratio ---
    print(f"\n--- CATCH-TO-PEAK RATIO (lower = more offset from peak) ---", flush=True)
    for radius, key in [(3, "ratio_3nm"), (5, "ratio_5nm"), (10, "ratio_10nm")]:
        vals = [d[key] for d in all_catch_data if not np.isnan(d[key])]
        if vals:
            print(f"  {radius}nm radius:", flush=True)
            print(f"    Mean ratio:   {np.mean(vals):.3f}", flush=True)
            print(f"    Median ratio: {np.median(vals):.3f}", flush=True)
            print(f"    Std:          {np.std(vals):.3f}", flush=True)
            below_80 = sum(1 for v in vals if v < 0.80)
            below_90 = sum(1 for v in vals if v < 0.90)
            print(f"    Ratio < 0.80: {below_80}/{len(vals)} ({below_80/len(vals)*100:.0f}%)", flush=True)
            print(f"    Ratio < 0.90: {below_90}/{len(vals)} ({below_90/len(vals)*100:.0f}%)", flush=True)

    # --- 3. Distance to Nearest Peak ---
    print(f"\n--- DISTANCE TO NEAREST PEAK (nm) ---", flush=True)
    for radius, key in [(3, "dist_to_peak_3nm"), (5, "dist_to_peak_5nm"), (10, "dist_to_peak_10nm")]:
        vals = [d[key] for d in all_catch_data if not np.isnan(d.get(f"peak_{radius}nm", np.nan))]
        if vals:
            print(f"  {radius}nm search:", flush=True)
            print(f"    Mean dist to peak:   {np.mean(vals):.2f} nm", flush=True)
            print(f"    Median dist to peak: {np.median(vals):.2f} nm", flush=True)

    # --- 4. Direction to Peak ---
    print(f"\n--- DIRECTION FROM CATCH TO NEAREST PEAK (10nm) ---", flush=True)
    dir_counts = defaultdict(int)
    for d in all_catch_data:
        dir_counts[d["dir_to_peak_10nm"]] += 1
    for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "SAME"]:
        cnt = dir_counts.get(direction, 0)
        bar = "#" * cnt
        print(f"  {direction:4s}: {cnt:3d} {bar}", flush=True)

    # --- 5. Gradient at Catch Locations ---
    grads = [d["catch_gradient"] for d in all_catch_data]
    print(f"\n--- COMPOSITE SCORE GRADIENT AT CATCH LOCATIONS ---", flush=True)
    print(f"  Mean gradient:   {np.mean(grads):.5f}", flush=True)
    print(f"  Median gradient: {np.median(grads):.5f}", flush=True)
    print(f"  Std gradient:    {np.std(grads):.5f}", flush=True)
    print(f"  (Higher gradient = steeper transition = more 'edge-like')", flush=True)

    # --- 6. Per-Feature Gradient Analysis ---
    print(f"\n--- PER-FEATURE GRADIENT AT CATCH vs OCEAN AVERAGE ---", flush=True)
    print(f"  {'Feature':<20s} {'Catch Grad':>12s} {'Ocean Grad':>12s} {'Ratio':>8s}  Interpretation", flush=True)
    print(f"  {'-'*75}", flush=True)

    feat_ratios = {}
    for fname in sorted(feature_gradients.keys()):
        catch_vals = feature_gradients[fname]
        ocean_vals = feature_gradients_ocean.get(fname, [])
        if catch_vals and ocean_vals:
            cmean = np.mean(catch_vals)
            omean = np.mean(ocean_vals)
            ratio = cmean / omean if omean > 0 else 0
            feat_ratios[fname] = ratio
            interp = "EDGE FEATURE" if ratio > 1.5 else ("edge-like" if ratio > 1.1 else "not edge-driven")
            print(f"  {fname:<20s} {cmean:>12.5f} {omean:>12.5f} {ratio:>8.2f}  {interp}", flush=True)

    if feat_ratios:
        print(f"\n  Top edge features (highest catch/ocean gradient ratio):", flush=True)
        for fname, ratio in sorted(feat_ratios.items(), key=lambda x: -x[1])[:5]:
            print(f"    {fname}: {ratio:.2f}x", flush=True)

    # --- 7. Contour Distance Analysis ---
    print(f"\n--- DISTANCE TO SCORE CONTOURS vs DISTANCE TO PEAK ---", flush=True)
    for contour_val, key in [(0.70, "dist_contour_70"), (0.80, "dist_contour_80"), (0.90, "dist_contour_90")]:
        vals = [d[key] for d in all_catch_data if not np.isnan(d[key])]
        if vals:
            print(f"  {contour_val:.0%} contour:", flush=True)
            print(f"    Mean dist to contour:   {np.mean(vals):.2f} nm", flush=True)
            print(f"    Median dist to contour: {np.median(vals):.2f} nm", flush=True)
            print(f"    Catches ON contour (<0.5nm): {sum(1 for v in vals if v < 0.5)}/{len(vals)} "
                  f"({sum(1 for v in vals if v < 0.5)/len(vals)*100:.0f}%)", flush=True)

    peak_dists = [d["dist_to_peak_10nm"] for d in all_catch_data
                  if not np.isnan(d.get("peak_10nm", np.nan))]
    contour80_dists = [d["dist_contour_80"] for d in all_catch_data if not np.isnan(d["dist_contour_80"])]
    if peak_dists and contour80_dists:
        print(f"\n  COMPARISON:", flush=True)
        print(f"    Mean dist to PEAK (10nm search):  {np.mean(peak_dists):.2f} nm", flush=True)
        print(f"    Mean dist to 80% CONTOUR:         {np.mean(contour80_dists):.2f} nm", flush=True)
        if np.mean(contour80_dists) < np.mean(peak_dists):
            print(f"    -> Catches are CLOSER to contours than to peaks!", flush=True)
        else:
            print(f"    -> Catches are closer to peaks than to contours", flush=True)

    # --- 8. Ratio Distribution ---
    print(f"\n--- CATCH-TO-PEAK RATIO DISTRIBUTION (10nm) ---", flush=True)
    ratios = [d["ratio_10nm"] for d in all_catch_data if not np.isnan(d["ratio_10nm"])]
    if ratios:
        bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.001)]
        for lo, hi in bins:
            cnt = sum(1 for r in ratios if lo <= r < hi)
            bar = "#" * (cnt * 2)
            label = f"{lo:.2f}-{hi:.2f}"
            print(f"  {label:12s}: {cnt:3d} ({cnt/len(ratios)*100:4.0f}%) {bar}", flush=True)

    # --- 9. Recommendations ---
    print(f"\n{'='*70}", flush=True)
    print("RECOMMENDATIONS FOR SCORING ADJUSTMENTS", flush=True)
    print("=" * 70, flush=True)

    mean_ratio_10 = np.mean(ratios) if ratios else 1.0
    if mean_ratio_10 < 0.85:
        print("""
  1. STRONG EDGE EFFECT DETECTED
     Catches consistently score below nearby peaks (mean ratio {:.2f}).
     Marlin appear to hunt at zone boundaries, not zone centers.

  2. Consider adding an EDGE BONUS: boost pixels where composite gradient
     is high (transitions between good and great habitat). This rewards
     the ecotone effect where prey accumulates at oceanographic boundaries.

  3. Consider SMOOTHING REDUCTION: the current scoring may over-smooth,
     pushing peaks away from the actual feature edges. Reducing Gaussian
     smoothing sigma would keep peaks closer to true feature boundaries.

  4. Consider a GRADIENT-WEIGHTED SCORE: composite = base_score * (1 + k * gradient)
     where k tunes how much edge preference is rewarded.
""".format(mean_ratio_10), flush=True)
    elif mean_ratio_10 < 0.95:
        print("""
  1. MODERATE EDGE EFFECT DETECTED
     Catches are somewhat offset from peaks (mean ratio {:.2f}).
     Some edge preference exists but catches are reasonably well-centered.

  2. A mild edge bonus might improve accuracy: boost pixels with
     moderate composite gradient to account for ecotone hunting.

  3. Current smoothing may be slightly excessive — consider reducing
     by 10-20% to sharpen zone boundaries.
""".format(mean_ratio_10), flush=True)
    else:
        print("""
  1. MINIMAL EDGE EFFECT
     Catches are well-centered in hotspot zones (mean ratio {:.2f}).
     Current scoring accurately predicts catch locations at zone cores.

  2. No edge-based adjustments recommended at this time.
""".format(mean_ratio_10), flush=True)

    # Feature-specific recommendations
    if feat_ratios:
        edge_features = [(f, r) for f, r in feat_ratios.items() if r > 1.3]
        if edge_features:
            print("  EDGE-DOMINANT FEATURES (gradient at catch >> ocean average):", flush=True)
            for fname, ratio in sorted(edge_features, key=lambda x: -x[1]):
                print(f"    - {fname}: {ratio:.2f}x ocean average gradient", flush=True)
            print("  These features show the strongest edge-hunting signal.", flush=True)
            print("  Consider boosting score where these features have HIGH gradient", flush=True)
            print("  rather than optimal values.", flush=True)

    print(f"\nAnalysis complete.", flush=True)


if __name__ == "__main__":
    main()
