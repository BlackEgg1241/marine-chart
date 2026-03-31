#!/usr/bin/env python3
"""
analyze_valley_pattern.py - Investigate whether blue marlin catches sit in
scoring VALLEYS (between peaks) rather than at scoring peaks themselves.

Tests:
1. Local score topology at each catch (valley/saddle detection)
2. Sub-score gradient convergence analysis
3. Transition density as alternative predictor
4. Score variance as alternative predictor
5. Multi-feature edge overlap analysis
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, laplace

# Fix Windows cp1252 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlin_data

BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}

# Catch dates and positions (lat, lon) - already decimal degrees
CATCHES = {
    "2001-03-03": [(-31.8167, 115.2833)],
    "2001-03-05": [(-32.0667, 115.2333)],
    "2001-03-10": [(-31.9333, 115.25)],
    "2001-03-28": [(-32.0667, 115.2333)],
    "2001-04-15": [(-32.0667, 115.2333)],
    "2002-02-07": [(-32.0333, 115.1667)],
    "2002-03-02": [(-32.0333, 115.1667)],
    "2002-03-23": [(-32.0667, 115.2333), (-31.9333, 115.25)],
    "2003-03-03": [(-32.0667, 115.2333)],
    "2003-03-15": [(-31.8833, 115.1833)],
    "2003-03-26": [(-32.0667, 115.2333)],
    "2004-02-28": [(-31.9333, 115.35)],
    "2006-03-10": [(-32.0667, 115.2333)],
    "2007-03-13": [(-32.1, 115.1667)],
    "2011-02-09": [(-32.2667, 115.1333)],
    "2011-03-04": [(-32.0833, 115.1833)],
    "2012-01-19": [(-31.9667, 115.15)],
    "2012-01-28": [(-32.0, 115.0)],
    "2012-02-12": [(-32.0667, 115.05)],
    "2012-12-30": [(-32.1333, 115.1)],
    "2013-02-23": [(-31.9667, 115.2167)],
    "2015-02-08": [(-31.9167, 114.9833)],
    "2015-02-24": [(-31.9667, 115.1667)],
    "2015-02-28": [(-31.9167, 115.2167)],
    "2015-03-02": [(-31.9667, 115.2333)],
    "2015-03-29": [(-32.0, 115.2)],
    "2015-04-03": [(-32.1, 115.1667)],
    "2016-01-16": [(-31.8833, 115.0833)],
    "2016-02-20": [(-31.9333, 115.0667)],
    "2016-03-07": [(-32.1, 115.2)],
    "2017-02-04": [(-31.8833, 115.0667)],
    "2017-02-13": [(-31.9667, 115.1333)],
    "2017-02-19": [(-31.7333, 114.95)],
    "2017-03-06": [(-32.0667, 115.1667)],
    "2021-02-14": [(-31.8833, 115.1833)],
    "2021-04-06": [(-31.7167, 114.95)],
    "2022-02-19": [(-32.0667, 115.2333)],
    "2022-12-29": [(-32.0833, 115.1833)],
    "2025-03-03": [(-31.9926, 115.2129)],
    "2025-03-04": [(-31.9904, 115.1901)],
    "2026-03-18": [(-31.9702, 115.2114), (-32.0613, 115.1458)],
}


def find_pixel(lats, lons, lat, lon):
    """Find nearest grid pixel to a lat/lon coordinate."""
    ri = np.argmin(np.abs(lats - lat))
    ci = np.argmin(np.abs(lons - lon))
    return ri, ci


def count_higher_neighbors(grid, ri, ci):
    """Count how many of the 8 neighbors have higher scores than center."""
    ny, nx = grid.shape
    center = grid[ri, ci]
    if np.isnan(center):
        return -1, 0
    count = 0
    total = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r2, c2 = ri + dr, ci + dc
            if 0 <= r2 < ny and 0 <= c2 < nx and not np.isnan(grid[r2, c2]):
                total += 1
                if grid[r2, c2] > center:
                    count += 1
    return count, total


def compute_laplacian_at(grid, ri, ci):
    """Compute discrete Laplacian at a point. Positive = concave up = valley."""
    ny, nx = grid.shape
    if ri < 1 or ri >= ny - 1 or ci < 1 or ci >= nx - 1:
        return np.nan
    center = grid[ri, ci]
    if np.isnan(center):
        return np.nan
    neighbors = [grid[ri-1, ci], grid[ri+1, ci], grid[ri, ci-1], grid[ri, ci+1]]
    if any(np.isnan(n) for n in neighbors):
        return np.nan
    return sum(neighbors) - 4 * center


def compute_transition_density(sub_scores, weights, ny, nx):
    """Sum of absolute gradient magnitudes across all weighted sub_scores."""
    td = np.zeros((ny, nx))
    for name, arr in sub_scores.items():
        w = weights.get(name, 0)
        if w == 0 or arr is None:
            continue
        filled = arr.copy()
        filled[np.isnan(filled)] = 0
        gx = sobel(filled, axis=1)
        gy = sobel(filled, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        td += w * grad_mag
    return td


def compute_score_variance(grid, radius_pixels=3):
    """Compute local variance of composite score within a radius."""
    from scipy.ndimage import uniform_filter
    filled = grid.copy()
    filled[np.isnan(filled)] = 0
    mask = (~np.isnan(grid)).astype(float)
    size = 2 * radius_pixels + 1
    local_mean = uniform_filter(filled, size=size)
    local_sq_mean = uniform_filter(filled**2, size=size)
    local_count = uniform_filter(mask, size=size)
    local_count = np.maximum(local_count, 1e-10)
    variance = local_sq_mean / local_count - (local_mean / local_count)**2
    variance = np.maximum(variance, 0)
    variance[np.isnan(grid)] = np.nan
    return variance


def count_transitioning_features(sub_scores, weights, ny, nx):
    """Count how many sub_scores have gradient above their 75th percentile."""
    grad_mags = {}
    thresholds = {}
    for name, arr in sub_scores.items():
        w = weights.get(name, 0)
        if w == 0 or arr is None:
            continue
        filled = arr.copy()
        filled[np.isnan(filled)] = 0
        gx = sobel(filled, axis=1)
        gy = sobel(filled, axis=0)
        gm = np.sqrt(gx**2 + gy**2)
        grad_mags[name] = gm
        valid_vals = gm[~np.isnan(arr) & (gm > 0)]
        if len(valid_vals) > 0:
            thresholds[name] = np.percentile(valid_vals, 75)
        else:
            thresholds[name] = 0

    count_grid = np.zeros((ny, nx))
    for name, gm in grad_mags.items():
        count_grid += (gm > thresholds[name]).astype(float)
    return count_grid


def analyze_gradient_convergence(sub_scores, weights, ri, ci, ny, nx):
    """Check if sub_score gradients point INWARD at catch location.
    Returns number of features with gradients pointing toward the catch."""
    inward_count = 0
    total_features = 0
    for name, arr in sub_scores.items():
        w = weights.get(name, 0)
        if w == 0 or arr is None:
            continue
        if ri < 1 or ri >= ny - 1 or ci < 1 or ci >= nx - 1:
            continue
        filled = arr.copy()
        filled[np.isnan(filled)] = 0
        # Gradient at catch pixel
        gy = (filled[ri+1, ci] - filled[ri-1, ci]) / 2
        gx = (filled[ri, ci+1] - filled[ri, ci-1]) / 2
        grad_mag = np.sqrt(gx**2 + gy**2)
        if grad_mag < 1e-6:
            continue
        total_features += 1
        # Check surrounding: are neighbors' gradients pointing TOWARD this pixel?
        inward = 0
        checked = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = ri + dr, ci + dc
            if 0 <= r2 < ny-1 and r2 >= 1 and 0 <= c2 < nx-1 and c2 >= 1:
                ngy = (filled[r2+1, c2] - filled[r2-1, c2]) / 2
                ngx = (filled[r2, c2+1] - filled[r2, c2-1]) / 2
                # Does this neighbor's gradient point toward (ri, ci)?
                # Direction from neighbor to center
                dir_r = ri - r2  # +1 if neighbor is above
                dir_c = ci - c2
                # Dot product: positive means gradient points toward center
                dot = ngy * dir_r + ngx * dir_c
                checked += 1
                if dot > 0:
                    inward += 1
        if checked > 0 and inward >= 3:  # 3 of 4 directions point inward
            inward_count += 1
    return inward_count, total_features


def main():
    print("=" * 80, flush=True)
    print("VALLEY PATTERN ANALYSIS", flush=True)
    print("Investigating: do catches sit in scoring valleys between peaks?", flush=True)
    print("=" * 80, flush=True)

    # Collect results across all dates
    all_catch_scores = []
    all_catch_higher_neighbors = []
    all_catch_laplacians = []
    all_catch_td = []
    all_catch_variance = []
    all_catch_feat_transitioning = []
    all_catch_gradient_convergence = []

    all_peak_scores = []
    all_peak_td = []
    all_peak_variance = []
    all_peak_feat_transitioning = []

    dates_processed = 0
    catches_analyzed = 0
    valley_count = 0
    saddle_count = 0

    # Sub-score analysis for valleys
    valley_low_features = {}  # feature -> count of times it's LOW at valley catch
    valley_high_nearby = {}   # feature -> count of times it's HIGH nearby

    sorted_dates = sorted(CATCHES.keys())

    for date_str in sorted_dates:
        catch_positions = CATCHES[date_str]
        data_dir = os.path.join("data", date_str)

        if not os.path.exists(data_dir):
            print(f"\n[SKIP] {date_str}: no data directory", flush=True)
            continue

        sst_file = os.path.join(data_dir, "sst_raw.nc")
        if not os.path.exists(sst_file):
            print(f"\n[SKIP] {date_str}: no SST data", flush=True)
            continue

        print(f"\n{'─'*60}", flush=True)
        print(f"Processing {date_str} ({len(catch_positions)} catches)...", flush=True)

        # Set OUTPUT_DIR so marlin_data reads from the right directory
        marlin_data.OUTPUT_DIR = data_dir

        tif_path = os.path.join(data_dir, "bathy_gmrt.tif")
        if not os.path.exists(tif_path):
            tif_path = None

        try:
            result = marlin_data.generate_blue_marlin_hotspots(
                BBOX, tif_path=tif_path, date_str=date_str
            )
        except Exception as e:
            print(f"[ERROR] {date_str}: {e}", flush=True)
            continue

        if result is None:
            print(f"[SKIP] {date_str}: hotspot generation returned None", flush=True)
            continue

        grid = result["grid"]
        lats = result["lats"]
        lons = result["lons"]
        sub_scores = result["sub_scores"]
        weights = result["weights"]
        ny, nx = grid.shape

        # Compute derived grids
        td = compute_transition_density(sub_scores, weights, ny, nx)

        # Radius in pixels for ~3nm: grid step is ~0.02 deg, 1 deg lat ~ 60nm
        grid_step = abs(lons[1] - lons[0]) if nx > 1 else 0.02
        radius_px = max(1, int(round(3.0 / (grid_step * 60))))
        variance = compute_score_variance(grid, radius_pixels=radius_px)
        feat_trans = count_transitioning_features(sub_scores, weights, ny, nx)

        # Find peaks (top 5% of score) for comparison
        valid_mask = ~np.isnan(grid)
        if np.sum(valid_mask) == 0:
            continue
        threshold_95 = np.nanpercentile(grid[valid_mask], 95)
        peak_mask = valid_mask & (grid >= threshold_95)

        peak_scores = grid[peak_mask]
        peak_td_vals = td[peak_mask]
        peak_var_vals = variance[peak_mask]
        peak_ft_vals = feat_trans[peak_mask]

        all_peak_scores.extend(peak_scores.tolist())
        all_peak_td.extend(peak_td_vals.tolist())
        all_peak_variance.extend(peak_var_vals[~np.isnan(peak_var_vals)].tolist())
        all_peak_feat_transitioning.extend(peak_ft_vals.tolist())

        dates_processed += 1

        for lat, lon in catch_positions:
            ri, ci = find_pixel(lats, lons, lat, lon)

            # Check bounds
            if ri < 0 or ri >= ny or ci < 0 or ci >= nx:
                print(f"  Catch ({lat:.4f}, {lon:.4f}): OUT OF BOUNDS", flush=True)
                continue

            score_at_catch = grid[ri, ci]
            if np.isnan(score_at_catch):
                print(f"  Catch ({lat:.4f}, {lon:.4f}): NaN score (land/missing)", flush=True)
                continue

            catches_analyzed += 1

            # 1. Topology analysis
            higher_n, total_n = count_higher_neighbors(grid, ri, ci)
            lap = compute_laplacian_at(grid, ri, ci)
            is_valley = higher_n >= 6 and not np.isnan(lap) and lap > 0
            is_saddle = higher_n >= 5 and not np.isnan(lap)

            if is_valley:
                valley_count += 1
            elif is_saddle:
                saddle_count += 1

            all_catch_scores.append(score_at_catch)
            all_catch_higher_neighbors.append(higher_n)
            if not np.isnan(lap):
                all_catch_laplacians.append(lap)

            # Transition density at catch
            td_at_catch = td[ri, ci]
            all_catch_td.append(td_at_catch)

            # Variance at catch
            var_at_catch = variance[ri, ci]
            if not np.isnan(var_at_catch):
                all_catch_variance.append(var_at_catch)

            # Feature transitioning count at catch
            ft_at_catch = feat_trans[ri, ci]
            all_catch_feat_transitioning.append(ft_at_catch)

            # Gradient convergence
            inward, total_feat = analyze_gradient_convergence(
                sub_scores, weights, ri, ci, ny, nx
            )
            all_catch_gradient_convergence.append((inward, total_feat))

            # Percentile of catch score within this date's grid
            pctile = np.sum(grid[valid_mask] <= score_at_catch) / np.sum(valid_mask) * 100

            # Sub-score analysis for valleys
            if is_valley or is_saddle:
                for name, arr in sub_scores.items():
                    w = weights.get(name, 0)
                    if w == 0 or arr is None:
                        continue
                    val_at_catch = arr[ri, ci]
                    if np.isnan(val_at_catch):
                        continue
                    # Check if LOW at catch but HIGH nearby
                    neighbors_vals = []
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue
                            r2, c2 = ri + dr, ci + dc
                            if 0 <= r2 < ny and 0 <= c2 < nx and not np.isnan(arr[r2, c2]):
                                neighbors_vals.append(arr[r2, c2])
                    if len(neighbors_vals) > 0:
                        neighbor_mean = np.mean(neighbors_vals)
                        if val_at_catch < neighbor_mean * 0.9:  # 10%+ lower
                            valley_low_features[name] = valley_low_features.get(name, 0) + 1
                        if neighbor_mean > np.nanpercentile(arr[valid_mask], 70):
                            valley_high_nearby[name] = valley_high_nearby.get(name, 0) + 1

            label = "VALLEY" if is_valley else ("SADDLE" if is_saddle else "normal")
            lap_str = f"{lap:.4f}" if not np.isnan(lap) else "NaN"
            var_str = f"{var_at_catch:.6f}" if not np.isnan(var_at_catch) else "NaN"
            print(f"  Catch ({lat:.4f}, {lon:.4f}): score={score_at_catch:.3f} "
                  f"pctile={pctile:.0f}% higher_neighbors={higher_n}/{total_n} "
                  f"lap={lap_str:>8} "
                  f"td={td_at_catch:.4f} var={var_str} "
                  f"feat_trans={ft_at_catch:.0f} [{label}]", flush=True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY RESULTS", flush=True)
    print("=" * 80, flush=True)

    print(f"\nDates processed: {dates_processed}", flush=True)
    print(f"Catches analyzed: {catches_analyzed}", flush=True)

    if catches_analyzed == 0:
        print("No catches could be analyzed. Exiting.", flush=True)
        return

    # 1. Valley/Saddle detection
    print(f"\n{'─'*60}", flush=True)
    print("1. LOCAL SCORE TOPOLOGY", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"Valleys (6+ higher neighbors + positive Laplacian): "
          f"{valley_count}/{catches_analyzed} = {valley_count/catches_analyzed*100:.1f}%", flush=True)
    print(f"Saddle points (5+ higher neighbors): "
          f"{saddle_count}/{catches_analyzed} = {saddle_count/catches_analyzed*100:.1f}%", flush=True)
    print(f"Valley + Saddle combined: "
          f"{(valley_count+saddle_count)}/{catches_analyzed} = "
          f"{(valley_count+saddle_count)/catches_analyzed*100:.1f}%", flush=True)

    hn = np.array(all_catch_higher_neighbors)
    print(f"\nHigher-neighbor count distribution at catches:", flush=True)
    for n in range(9):
        c = np.sum(hn == n)
        if c > 0:
            print(f"  {n} higher neighbors: {c} catches ({c/len(hn)*100:.1f}%)", flush=True)
    print(f"  Mean higher neighbors: {np.mean(hn):.2f} (4.0 = flat, >4 = valley tendency)", flush=True)

    if len(all_catch_laplacians) > 0:
        laps = np.array(all_catch_laplacians)
        print(f"\nLaplacian at catches: mean={np.mean(laps):.4f} "
              f"(positive=valley, negative=peak)", flush=True)
        print(f"  Positive Laplacian (valley): {np.sum(laps > 0)}/{len(laps)} = "
              f"{np.sum(laps > 0)/len(laps)*100:.1f}%", flush=True)
        print(f"  Negative Laplacian (peak): {np.sum(laps < 0)}/{len(laps)} = "
              f"{np.sum(laps < 0)/len(laps)*100:.1f}%", flush=True)

    # 2. Sub-score profile at valleys
    print(f"\n{'─'*60}", flush=True)
    print("2. SUB-SCORE PROFILE AT VALLEY/SADDLE CATCHES", flush=True)
    print(f"{'─'*60}", flush=True)
    if valley_low_features:
        print("Features that are LOW at catch but HIGH nearby:", flush=True)
        for name, count in sorted(valley_low_features.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} times", flush=True)
    if valley_high_nearby:
        print("\nFeatures with HIGH values in neighborhood:", flush=True)
        for name, count in sorted(valley_high_nearby.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} times", flush=True)

    # 3. Gradient convergence
    print(f"\n{'─'*60}", flush=True)
    print("3. GRADIENT CONVERGENCE AT CATCHES", flush=True)
    print(f"{'─'*60}", flush=True)
    if all_catch_gradient_convergence:
        gc = np.array([x[0] for x in all_catch_gradient_convergence])
        gt = np.array([x[1] for x in all_catch_gradient_convergence])
        print(f"Features with inward-pointing gradients at catches:", flush=True)
        print(f"  Mean: {np.mean(gc):.2f} of {np.mean(gt):.1f} total features", flush=True)
        print(f"  Catches with 3+ convergent features: {np.sum(gc >= 3)}/{len(gc)} = "
              f"{np.sum(gc >= 3)/len(gc)*100:.1f}%", flush=True)

    # 4. Transition density comparison
    print(f"\n{'─'*60}", flush=True)
    print("4. TRANSITION DENSITY (sum of all sub-score gradients)", flush=True)
    print(f"{'─'*60}", flush=True)
    catch_td = np.array(all_catch_td)
    peak_td = np.array(all_peak_td)
    print(f"At catches: mean={np.mean(catch_td):.4f} median={np.median(catch_td):.4f}", flush=True)
    print(f"At peaks:   mean={np.mean(peak_td):.4f} median={np.median(peak_td):.4f}", flush=True)
    td_ratio = np.mean(catch_td) / max(np.mean(peak_td), 1e-10)
    print(f"Ratio (catch/peak): {td_ratio:.2f}x", flush=True)
    if td_ratio > 1.0:
        print("  -> Catches have HIGHER transition density than peaks!", flush=True)
        print("  -> Transition density is a BETTER predictor of catch locations", flush=True)
    else:
        print("  -> Peaks have higher transition density", flush=True)

    # 5. Score variance comparison
    print(f"\n{'─'*60}", flush=True)
    print("5. SCORE VARIANCE (heterogeneity within ~3nm radius)", flush=True)
    print(f"{'─'*60}", flush=True)
    catch_var = np.array(all_catch_variance)
    peak_var = np.array(all_peak_variance)
    if len(catch_var) > 0 and len(peak_var) > 0:
        print(f"At catches: mean={np.mean(catch_var):.6f} median={np.median(catch_var):.6f}", flush=True)
        print(f"At peaks:   mean={np.mean(peak_var):.6f} median={np.median(peak_var):.6f}", flush=True)
        var_ratio = np.mean(catch_var) / max(np.mean(peak_var), 1e-10)
        print(f"Ratio (catch/peak): {var_ratio:.2f}x", flush=True)
        if var_ratio > 1.0:
            print("  -> Catches are in MORE heterogeneous areas!", flush=True)
            print("  -> Score variance is a BETTER predictor", flush=True)
        else:
            print("  -> Peaks are in more heterogeneous areas", flush=True)

    # 6. Multi-feature edge overlap
    print(f"\n{'─'*60}", flush=True)
    print("6. MULTI-FEATURE EDGE OVERLAP (features transitioning simultaneously)", flush=True)
    print(f"{'─'*60}", flush=True)
    catch_ft = np.array(all_catch_feat_transitioning)
    peak_ft = np.array(all_peak_feat_transitioning)
    print(f"At catches: mean={np.mean(catch_ft):.2f} median={np.median(catch_ft):.1f}", flush=True)
    print(f"At peaks:   mean={np.mean(peak_ft):.2f} median={np.median(peak_ft):.1f}", flush=True)
    ft_ratio = np.mean(catch_ft) / max(np.mean(peak_ft), 1e-10)
    print(f"Ratio (catch/peak): {ft_ratio:.2f}x", flush=True)
    if ft_ratio > 1.0:
        print("  -> Catches have MORE features transitioning!", flush=True)
        print("  -> Multi-feature edges are BETTER predictors", flush=True)
    else:
        print("  -> Peaks have more features transitioning", flush=True)

    # 7. Catch score percentile analysis
    print(f"\n{'─'*60}", flush=True)
    print("7. CATCH SCORE DISTRIBUTION", flush=True)
    print(f"{'─'*60}", flush=True)
    cs = np.array(all_catch_scores)
    print(f"Catch scores: mean={np.mean(cs):.3f} median={np.median(cs):.3f} "
          f"std={np.std(cs):.3f}", flush=True)
    print(f"  min={np.min(cs):.3f} max={np.max(cs):.3f}", flush=True)
    ps = np.array(all_peak_scores)
    print(f"Peak scores:  mean={np.mean(ps):.3f} median={np.median(ps):.3f}", flush=True)
    print(f"Score gap (peak - catch): {np.mean(ps) - np.mean(cs):.3f}", flush=True)

    # RECOMMENDATIONS
    print(f"\n{'='*80}", flush=True)
    print("RECOMMENDATIONS", flush=True)
    print(f"{'='*80}", flush=True)

    valley_pct = (valley_count + saddle_count) / catches_analyzed * 100

    print(f"\n1. VALLEY/SADDLE PATTERN: {valley_pct:.0f}% of catches are in valleys/saddles", flush=True)
    if valley_pct > 40:
        print("   -> CONFIRMED: Catches cluster in scoring valleys, not peaks", flush=True)
        print("   -> The scoring system rewards feature PEAKS but marlin hunt at EDGES", flush=True)
    elif valley_pct > 25:
        print("   -> MODERATE: Significant minority of catches in valleys", flush=True)
    else:
        print("   -> WEAK: Most catches are NOT in valleys", flush=True)

    print(f"\n2. TRANSITION DENSITY: {td_ratio:.2f}x at catches vs peaks", flush=True)
    if td_ratio > 1.1:
        print("   -> STRONG SIGNAL: Transition density predicts catches better than score", flush=True)
        print("   -> RECOMMENDATION: Add transition_density as a scoring feature or", flush=True)
        print("      replace composite score with gradient-weighted score", flush=True)
    elif td_ratio > 0.9:
        print("   -> NEUTRAL: Similar transition density at catches and peaks", flush=True)

    if len(catch_var) > 0 and len(peak_var) > 0:
        print(f"\n3. SCORE VARIANCE: {var_ratio:.2f}x at catches vs peaks", flush=True)
        if var_ratio > 1.1:
            print("   -> STRONG SIGNAL: Catches prefer heterogeneous zones", flush=True)
            print("   -> RECOMMENDATION: Add local_variance as a multiplicative boost", flush=True)

    print(f"\n4. FEATURE EDGES: {ft_ratio:.2f}x features transitioning at catches vs peaks", flush=True)
    if ft_ratio > 1.1:
        print("   -> STRONG SIGNAL: Catches sit where multiple features change simultaneously", flush=True)
        print("   -> RECOMMENDATION: Add multi-feature edge count as scoring boost", flush=True)

    print(f"\n5. SPECIFIC SCORING CHANGES RECOMMENDED:", flush=True)
    if valley_pct > 25 or td_ratio > 1.0:
        print("   a) Add 'transition_zone' feature: sum of sub-score gradient magnitudes", flush=True)
        print("      Weight: 0.08-0.12 (take from SST/CHL which are spatially uniform)", flush=True)
        print("   b) Add 'edge_count' multiplier: boost pixels where 3+ features have", flush=True)
        print("      above-median gradients (x1.10 boost)", flush=True)
        print("   c) Reduce emphasis on individual feature MAGNITUDE, increase emphasis", flush=True)
        print("      on feature GRADIENT (edges/transitions)", flush=True)
        print("   d) Consider Laplacian-based penalty: reduce score at local maxima,", flush=True)
        print("      boost score at local minima (saddle-point boost)", flush=True)
    else:
        print("   The valley pattern is not strong enough to warrant major changes.", flush=True)
        print("   Current scoring may be adequate with minor tuning.", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
