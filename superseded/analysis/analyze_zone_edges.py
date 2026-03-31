#!/usr/bin/env python3
"""
analyze_zone_edges.py — Deep investigation into why blue marlin catches
sit on zone edges rather than zone centers.

Decomposes scores at catch locations vs nearby peaks to identify which
features are pulling score maxima away from actual catch positions.
"""

import csv
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")
SEARCH_RADIUS_NM = 5.0
PROFILE_RADIUS_NM = 10.0
NM_TO_DEG_LAT = 1.0 / 60.0  # 1nm ~ 1/60 degree latitude
NM_TO_DEG_LON_AT_32S = 1.0 / (60.0 * np.cos(np.radians(32)))  # at ~32S


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

    # GFAA Export.csv (DDM format)
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Species_Name"].strip().upper() != "BLUE MARLIN":
                continue
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            date = _parse_date(r["Release_Date"][:10])
            key = (date, round(lat, 4), round(lon, 4))
            if key not in seen:
                seen.add(key)
                catches.append({"date": date, "lat": lat, "lon": lon})

    # all_catches.csv (decimal degrees)
    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                species = r.get("species", "").strip().upper()
                if species != "BLUE MARLIN":
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


def haversine_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_grid_index(lats, lons, lat, lon):
    """Find nearest grid indices for a lat/lon."""
    ri = np.argmin(np.abs(lats - lat))
    ci = np.argmin(np.abs(lons - lon))
    return ri, ci


def main():
    print("=" * 80, flush=True)
    print("BLUE MARLIN ZONE EDGE ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    # Load catches
    catches = load_catches()
    print(f"\nLoaded {len(catches)} blue marlin catches", flush=True)

    # Group by date
    dates = {}
    for c in catches:
        dates.setdefault(c["date"], []).append(c)
    print(f"Across {len(dates)} unique dates", flush=True)

    # Check which dates have pre-computed data
    import marlin_data
    from marlin_data import generate_blue_marlin_hotspots, BLUE_MARLIN_WEIGHTS

    available_dates = []
    for date_str in sorted(dates.keys()):
        date_dir = os.path.join("data", date_str)
        geojson = os.path.join(date_dir, "blue_marlin_hotspots.geojson")
        sst_path = os.path.join(date_dir, "sst_raw.nc")
        # We need the raw data to regenerate sub_scores
        if os.path.exists(sst_path):
            available_dates.append(date_str)

    print(f"\nDates with raw data available: {len(available_dates)}/{len(dates)}", flush=True)
    if not available_dates:
        # Try backtest dirs
        for date_str in sorted(dates.keys()):
            bt_dir = os.path.join("data", "backtest", date_str)
            if os.path.exists(os.path.join(bt_dir, "sst_raw.nc")):
                available_dates.append(date_str)
        print(f"Dates with backtest data: {len(available_dates)}", flush=True)

    if not available_dates:
        print("ERROR: No dates with raw NetCDF data found. Cannot decompose scores.", flush=True)
        return

    # --- Accumulators for aggregate statistics ---
    all_feature_deltas = {}       # feature -> list of (catch_score - peak_score)
    all_weighted_deltas = {}      # feature -> list of weighted deltas
    all_gradient_directions = {}  # feature -> list of (dx, dy) gradient at catch
    all_band_catch = []           # band_count at catch locations
    all_band_peak = []            # band_count at peak locations
    all_shelf_catch = []          # shelf_break score at catch
    all_shelf_peak = []           # shelf_break score at peak
    all_floor_catch = []          # whether feature floor was active at catch
    all_floor_peak = []           # whether feature floor was active at peak
    all_compass_profiles = {d: [] for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
    all_band_type_distances = {}  # band_type -> list of distances from catch
    all_catch_scores = []
    all_peak_scores = []
    all_peak_distances = []
    all_offset_directions = []    # (dx, dy) from catch to peak
    n_processed = 0
    n_skipped = 0

    # Compass direction vectors (dlat, dlon per nm)
    compass = {
        "N":  ( NM_TO_DEG_LAT,  0),
        "NE": ( NM_TO_DEG_LAT,  NM_TO_DEG_LON_AT_32S),
        "E":  ( 0,               NM_TO_DEG_LON_AT_32S),
        "SE": (-NM_TO_DEG_LAT,  NM_TO_DEG_LON_AT_32S),
        "S":  (-NM_TO_DEG_LAT,  0),
        "SW": (-NM_TO_DEG_LAT, -NM_TO_DEG_LON_AT_32S),
        "W":  ( 0,              -NM_TO_DEG_LON_AT_32S),
        "NW": ( NM_TO_DEG_LAT, -NM_TO_DEG_LON_AT_32S),
    }
    # Normalize diagonal directions
    for d in ["NE", "SE", "SW", "NW"]:
        v = compass[d]
        mag = np.sqrt(v[0]**2 + v[1]**2)
        compass[d] = (v[0] * NM_TO_DEG_LAT / mag, v[1] * NM_TO_DEG_LON_AT_32S / mag)
        # Actually keep it simpler: per-nm step
        compass[d] = (v[0] / np.sqrt(2) * (1/NM_TO_DEG_LAT) * NM_TO_DEG_LAT,
                       v[1] / np.sqrt(2) * (1/NM_TO_DEG_LON_AT_32S) * NM_TO_DEG_LON_AT_32S)

    # Redefine compass more cleanly: unit vector in degrees per nm
    compass = {
        "N":  ( NM_TO_DEG_LAT,  0),
        "NE": ( NM_TO_DEG_LAT / np.sqrt(2),  NM_TO_DEG_LON_AT_32S / np.sqrt(2)),
        "E":  ( 0,               NM_TO_DEG_LON_AT_32S),
        "SE": (-NM_TO_DEG_LAT / np.sqrt(2),  NM_TO_DEG_LON_AT_32S / np.sqrt(2)),
        "S":  (-NM_TO_DEG_LAT,  0),
        "SW": (-NM_TO_DEG_LAT / np.sqrt(2), -NM_TO_DEG_LON_AT_32S / np.sqrt(2)),
        "W":  ( 0,              -NM_TO_DEG_LON_AT_32S),
        "NW": ( NM_TO_DEG_LAT / np.sqrt(2), -NM_TO_DEG_LON_AT_32S / np.sqrt(2)),
    }

    for date_str in available_dates:
        catch_list = dates[date_str]

        # Determine data directory
        date_dir = os.path.join("data", date_str)
        bt_dir = os.path.join("data", "backtest", date_str)
        if os.path.exists(os.path.join(date_dir, "sst_raw.nc")):
            work_dir = date_dir
        elif os.path.exists(os.path.join(bt_dir, "sst_raw.nc")):
            work_dir = bt_dir
        else:
            n_skipped += len(catch_list)
            continue

        # Point marlin_data.OUTPUT_DIR to the data directory
        old_output = marlin_data.OUTPUT_DIR
        marlin_data.OUTPUT_DIR = work_dir

        print(f"\n--- {date_str} ({len(catch_list)} catches, dir={work_dir}) ---", flush=True)

        try:
            result = generate_blue_marlin_hotspots(BBOX, date_str=date_str)
        except Exception as e:
            print(f"  SKIP: {e}", flush=True)
            marlin_data.OUTPUT_DIR = old_output
            n_skipped += len(catch_list)
            continue
        finally:
            marlin_data.OUTPUT_DIR = old_output

        grid = result["grid"]           # final smoothed composite score
        lats = result["lats"]
        lons = result["lons"]
        sub_scores = result["sub_scores"]
        weights = result["weights"]
        band_count = result["band_count"]
        band_mean = result["band_mean"]
        ny, nx = grid.shape

        for catch in catch_list:
            clat, clon = catch["lat"], catch["lon"]

            # Check if catch is within grid
            if clat < lats.min() or clat > lats.max() or clon < lons.min() or clon > lons.max():
                n_skipped += 1
                continue

            ci, cj = find_grid_index(lats, lons, clat, clon)
            catch_score = grid[ci, cj]

            if np.isnan(catch_score):
                n_skipped += 1
                continue

            # --- Find peak within 5nm ---
            search_dlat = SEARCH_RADIUS_NM * NM_TO_DEG_LAT
            search_dlon = SEARCH_RADIUS_NM * NM_TO_DEG_LON_AT_32S
            ri_min = max(0, np.argmin(np.abs(lats - (clat - search_dlat))))
            ri_max = min(ny - 1, np.argmin(np.abs(lats - (clat + search_dlat))))
            ci_min = max(0, np.argmin(np.abs(lons - (clon - search_dlon))))
            ci_max = min(nx - 1, np.argmin(np.abs(lons - (clon + search_dlon))))

            if ri_min > ri_max:
                ri_min, ri_max = ri_max, ri_min

            region = grid[ri_min:ri_max+1, ci_min:ci_max+1]
            if region.size == 0 or np.all(np.isnan(region)):
                n_skipped += 1
                continue

            # Find peak in region
            region_filled = np.where(np.isnan(region), -999, region)
            peak_local = np.unravel_index(np.argmax(region_filled), region.shape)
            pi = ri_min + peak_local[0]
            pj = ci_min + peak_local[1]
            peak_score = grid[pi, pj]

            if np.isnan(peak_score):
                n_skipped += 1
                continue

            peak_lat = lats[pi]
            peak_lon = lons[pj]
            dist_nm = haversine_nm(clat, clon, peak_lat, peak_lon)

            all_catch_scores.append(catch_score)
            all_peak_scores.append(peak_score)
            all_peak_distances.append(dist_nm)

            # Offset direction from catch to peak
            dlat_offset = peak_lat - clat
            dlon_offset = peak_lon - clon
            all_offset_directions.append((dlat_offset, dlon_offset))

            # --- 1. Score Decomposition: sub_scores at catch vs peak ---
            for feat_name, feat_arr in sub_scores.items():
                w = weights.get(feat_name, 0)
                cs = feat_arr[ci, cj] if not np.isnan(feat_arr[ci, cj]) else 0
                ps = feat_arr[pi, pj] if not np.isnan(feat_arr[pi, pj]) else 0

                delta = cs - ps  # positive = catch scores higher
                all_feature_deltas.setdefault(feat_name, []).append(delta)
                all_weighted_deltas.setdefault(feat_name, []).append(delta * w)

            # --- 2. Feature gradient direction at catch ---
            for feat_name, feat_arr in sub_scores.items():
                if ci > 0 and ci < ny - 1 and cj > 0 and cj < nx - 1:
                    dy = feat_arr[ci+1, cj] - feat_arr[ci-1, cj]
                    dx = feat_arr[ci, cj+1] - feat_arr[ci, cj-1]
                    if not np.isnan(dy) and not np.isnan(dx):
                        all_gradient_directions.setdefault(feat_name, []).append((dx, dy))

            # --- 3. Band count at catch vs peak ---
            bc_catch = band_count[ci, cj] if not np.isnan(band_count[ci, cj]) else 0
            bc_peak = band_count[pi, pj] if not np.isnan(band_count[pi, pj]) else 0
            all_band_catch.append(bc_catch)
            all_band_peak.append(bc_peak)

            # --- 4. Shelf break at catch vs peak ---
            if "shelf_break" in sub_scores:
                sb_c = sub_scores["shelf_break"][ci, cj]
                sb_p = sub_scores["shelf_break"][pi, pj]
                if not np.isnan(sb_c):
                    all_shelf_catch.append(sb_c)
                if not np.isnan(sb_p):
                    all_shelf_peak.append(sb_p)

            # --- 5. Feature floor analysis ---
            # The floor lifts cells to 0.40 near feature lines.
            # Check if the pre-floor score was lower at catch (floor lifted it)
            # We can approximate by checking if score is near the floor value
            all_floor_catch.append(1 if abs(catch_score - 0.62) < 0.05 else 0)
            all_floor_peak.append(1 if abs(peak_score - 0.62) < 0.05 else 0)

            # --- 6. Compass profile ---
            for direction, (dlat_step, dlon_step) in compass.items():
                profile = []
                for dist in range(0, int(PROFILE_RADIUS_NM) + 1):
                    sample_lat = clat + dist * dlat_step
                    sample_lon = clon + dist * dlon_step
                    if (sample_lat < lats.min() or sample_lat > lats.max() or
                        sample_lon < lons.min() or sample_lon > lons.max()):
                        profile.append(np.nan)
                        continue
                    si, sj = find_grid_index(lats, lons, sample_lat, sample_lon)
                    profile.append(grid[si, sj])
                all_compass_profiles[direction].append(profile)

            # --- 7. Band type distances ---
            # Check individual band layers by looking at band_count contributions
            # We don't have individual band layers saved, but we can check band_count
            # at increasing distances from catch
            # (handled in aggregate below)

            n_processed += 1

        # End of catch loop for this date

    # ======================================================================
    # AGGREGATE RESULTS
    # ======================================================================
    print("\n" + "=" * 80, flush=True)
    print(f"RESULTS: {n_processed} catches processed, {n_skipped} skipped", flush=True)
    print("=" * 80, flush=True)

    if n_processed == 0:
        print("No catches could be processed. Exiting.", flush=True)
        return

    # --- Overall score comparison ---
    print("\n--- 1. SCORE AT CATCH vs NEAREST PEAK (within 5nm) ---", flush=True)
    print(f"  Mean catch score:  {np.nanmean(all_catch_scores):.4f}", flush=True)
    print(f"  Mean peak score:   {np.nanmean(all_peak_scores):.4f}", flush=True)
    print(f"  Mean deficit:      {np.nanmean(np.array(all_peak_scores) - np.array(all_catch_scores)):.4f}", flush=True)
    print(f"  Mean peak distance:{np.nanmean(all_peak_distances):.2f} nm", flush=True)
    print(f"  Median peak dist:  {np.nanmedian(all_peak_distances):.2f} nm", flush=True)

    # --- 1. Feature decomposition ---
    print("\n--- 1b. FEATURE SCORE DECOMPOSITION (catch - peak) ---", flush=True)
    print(f"  {'Feature':<20s} {'Weight':>6s} {'MeanDelta':>10s} {'WtdDelta':>10s} {'CatchHigher%':>12s} {'PeakHigher%':>12s}", flush=True)
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*12}", flush=True)

    # Sort by weighted delta (most negative = biggest pull toward peak)
    feat_summary = []
    for feat_name in sorted(all_feature_deltas.keys()):
        deltas = all_feature_deltas[feat_name]
        w_deltas = all_weighted_deltas[feat_name]
        mean_d = np.nanmean(deltas)
        mean_wd = np.nanmean(w_deltas)
        w = weights.get(feat_name, 0)
        catch_higher = sum(1 for d in deltas if d > 0.01) / len(deltas) * 100
        peak_higher = sum(1 for d in deltas if d < -0.01) / len(deltas) * 100
        feat_summary.append((feat_name, w, mean_d, mean_wd, catch_higher, peak_higher))

    feat_summary.sort(key=lambda x: x[3])  # sort by weighted delta ascending (most negative first)
    for name, w, md, mwd, ch, ph in feat_summary:
        print(f"  {name:<20s} {w:>6.2f} {md:>+10.4f} {mwd:>+10.4f} {ch:>11.0f}% {ph:>11.0f}%", flush=True)

    print("\n  INTERPRETATION: Negative WtdDelta = feature scores HIGHER at peak than catch", flush=True)
    print("  -> These features PULL the zone center AWAY from the catch location", flush=True)

    # --- 2. Gradient direction analysis ---
    print("\n--- 2. FEATURE GRADIENT DIRECTION AT CATCH ---", flush=True)
    print(f"  {'Feature':<20s} {'MeanGradX(E)':>12s} {'MeanGradY(N)':>12s} {'Direction':>12s} {'Consistency':>12s}", flush=True)
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", flush=True)

    for feat_name in sorted(all_gradient_directions.keys()):
        grads = all_gradient_directions[feat_name]
        if not grads:
            continue
        dx_arr = np.array([g[0] for g in grads])
        dy_arr = np.array([g[1] for g in grads])
        mean_dx = np.nanmean(dx_arr)
        mean_dy = np.nanmean(dy_arr)
        mag = np.sqrt(mean_dx**2 + mean_dy**2)
        if mag > 1e-6:
            angle = np.degrees(np.arctan2(mean_dy, mean_dx))
            # Convert to compass: 0=E, 90=N
            if angle < 0:
                angle += 360
            # Consistency: mean vector length / mean of individual magnitudes
            individual_mags = np.sqrt(dx_arr**2 + dy_arr**2)
            consistency = mag / (np.nanmean(individual_mags) + 1e-10) * 100
            direction_name = _angle_to_compass(angle)
        else:
            angle = 0
            consistency = 0
            direction_name = "none"

        print(f"  {feat_name:<20s} {mean_dx:>+12.4f} {mean_dy:>+12.4f} {direction_name:>12s} {consistency:>11.0f}%", flush=True)

    # --- 3. Band count analysis ---
    print("\n--- 3. BAND COUNT AT CATCH vs PEAK ---", flush=True)
    print(f"  Mean bands at catch: {np.mean(all_band_catch):.2f}", flush=True)
    print(f"  Mean bands at peak:  {np.mean(all_band_peak):.2f}", flush=True)
    print(f"  Catch 0 bands: {sum(1 for b in all_band_catch if b < 0.5)}/{len(all_band_catch)} ({sum(1 for b in all_band_catch if b < 0.5)/max(len(all_band_catch),1)*100:.0f}%)", flush=True)
    print(f"  Peak  0 bands: {sum(1 for b in all_band_peak if b < 0.5)}/{len(all_band_peak)} ({sum(1 for b in all_band_peak if b < 0.5)/max(len(all_band_peak),1)*100:.0f}%)", flush=True)
    print(f"  Catch 1 band:  {sum(1 for b in all_band_catch if 0.5 <= b < 1.5)}/{len(all_band_catch)}", flush=True)
    print(f"  Peak  1 band:  {sum(1 for b in all_band_peak if 0.5 <= b < 1.5)}/{len(all_band_peak)}", flush=True)
    print(f"  Catch 2+ bands:{sum(1 for b in all_band_catch if b >= 1.5)}/{len(all_band_catch)}", flush=True)
    print(f"  Peak  2+ bands:{sum(1 for b in all_band_peak if b >= 1.5)}/{len(all_band_peak)}", flush=True)
    bc_deficit = np.mean(all_band_peak) - np.mean(all_band_catch)
    print(f"  Band deficit (peak-catch): {bc_deficit:+.2f}", flush=True)
    if bc_deficit > 0.3:
        print(f"  -> SIGNIFICANT: Peaks have more bands than catches. Band positions may be offset.", flush=True)
    elif bc_deficit > 0.1:
        print(f"  -> MODERATE: Peaks tend to sit on more band lines than catches.", flush=True)
    else:
        print(f"  -> SMALL: Band count similar at catches and peaks.", flush=True)

    # --- 4. Shelf break multiplier analysis ---
    print("\n--- 4. SHELF BREAK MULTIPLIER ANALYSIS ---", flush=True)
    if all_shelf_catch and all_shelf_peak:
        print(f"  Mean shelf_break score at catch: {np.mean(all_shelf_catch):.4f}", flush=True)
        print(f"  Mean shelf_break score at peak:  {np.mean(all_shelf_peak):.4f}", flush=True)
        shelf_boost = 0.20  # _opt_shelf_boost default
        print(f"  Multiplicative boost at catch: x{1.0 + shelf_boost * np.mean(all_shelf_catch):.3f}", flush=True)
        print(f"  Multiplicative boost at peak:  x{1.0 + shelf_boost * np.mean(all_shelf_peak):.3f}", flush=True)
        sb_delta = np.mean(all_shelf_peak) - np.mean(all_shelf_catch)
        if sb_delta > 0.05:
            print(f"  -> SHELF PULL: Peaks sit closer to shelf break than catches (+{sb_delta:.3f})", flush=True)
            print(f"     The multiplicative boost is amplifying this offset.", flush=True)
        else:
            print(f"  -> Shelf break not a significant offset driver (delta={sb_delta:+.3f})", flush=True)
    else:
        print(f"  No shelf break data available.", flush=True)

    # --- 5. Feature floor analysis ---
    print("\n--- 5. FEATURE FLOOR ANALYSIS ---", flush=True)
    floor_catch_pct = sum(all_floor_catch) / max(len(all_floor_catch), 1) * 100
    floor_peak_pct = sum(all_floor_peak) / max(len(all_floor_peak), 1) * 100
    print(f"  Catches near floor (score ~0.62): {sum(all_floor_catch)}/{len(all_floor_catch)} ({floor_catch_pct:.0f}%)", flush=True)
    print(f"  Peaks near floor (score ~0.62):   {sum(all_floor_peak)}/{len(all_floor_peak)} ({floor_peak_pct:.0f}%)", flush=True)
    if floor_catch_pct > 20:
        print(f"  -> WARNING: {floor_catch_pct:.0f}% of catches are sitting AT the floor level.", flush=True)
        print(f"     This suggests catches are on feature lines but the floor is capping their score.", flush=True)
    else:
        print(f"  -> Feature floor not creating artificial catch-level capping.", flush=True)

    # --- 6. Compass profile ---
    print("\n--- 6. SCORE GRADIENT PROFILE (avg score at distance from catch) ---", flush=True)
    print(f"  {'Dist(nm)':>8s}", end="", flush=True)
    for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        print(f" {d:>6s}", end="", flush=True)
    print(f" {'Mean':>6s}", flush=True)

    for dist_idx in range(0, int(PROFILE_RADIUS_NM) + 1):
        print(f"  {dist_idx:>7d} ", end="", flush=True)
        row_vals = []
        for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            profiles = all_compass_profiles[d]
            vals = [p[dist_idx] for p in profiles if dist_idx < len(p) and not np.isnan(p[dist_idx])]
            if vals:
                mean_val = np.mean(vals)
                row_vals.append(mean_val)
                print(f" {mean_val:.4f}", end="", flush=True)
            else:
                print(f"    nan", end="", flush=True)
        if row_vals:
            print(f" {np.mean(row_vals):.4f}", flush=True)
        else:
            print(f"    nan", flush=True)

    # Find which directions score increases most
    print("\n  Score change from catch (dist=0) to 3nm:", flush=True)
    dir_changes = {}
    for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        profiles = all_compass_profiles[d]
        at_0 = [p[0] for p in profiles if len(p) > 0 and not np.isnan(p[0])]
        at_3 = [p[3] for p in profiles if len(p) > 3 and not np.isnan(p[3])]
        if at_0 and at_3:
            change = np.mean(at_3) - np.mean(at_0)
            dir_changes[d] = change
            print(f"    {d:>3s}: {change:+.4f}", flush=True)

    if dir_changes:
        best_dir = max(dir_changes, key=dir_changes.get)
        worst_dir = min(dir_changes, key=dir_changes.get)
        print(f"  -> Score increases most toward {best_dir} ({dir_changes[best_dir]:+.4f})", flush=True)
        print(f"  -> Score decreases most toward {worst_dir} ({dir_changes[worst_dir]:+.4f})", flush=True)

    # --- 7. Offset direction from catch to peak ---
    print("\n--- 7. SYSTEMATIC OFFSET DIRECTION (catch -> peak) ---", flush=True)
    if all_offset_directions:
        dlats = [d[0] for d in all_offset_directions]
        dlons = [d[1] for d in all_offset_directions]
        mean_dlat = np.mean(dlats)
        mean_dlon = np.mean(dlons)
        # Convert to nm
        mean_dlat_nm = mean_dlat / NM_TO_DEG_LAT
        mean_dlon_nm = mean_dlon / NM_TO_DEG_LON_AT_32S
        offset_mag_nm = np.sqrt(mean_dlat_nm**2 + mean_dlon_nm**2)
        angle = np.degrees(np.arctan2(mean_dlat_nm, mean_dlon_nm))
        if angle < 0:
            angle += 360
        print(f"  Mean offset: {mean_dlat_nm:+.2f}nm N/S, {mean_dlon_nm:+.2f}nm E/W", flush=True)
        print(f"  Magnitude: {offset_mag_nm:.2f}nm", flush=True)
        print(f"  Direction: {_angle_to_compass(angle)} ({angle:.0f} deg)", flush=True)

        # Breakdown by quadrant
        quad_counts = {"NE": 0, "NW": 0, "SE": 0, "SW": 0}
        for dlat, dlon in all_offset_directions:
            if dlat >= 0 and dlon >= 0: quad_counts["NE"] += 1
            elif dlat >= 0 and dlon < 0: quad_counts["NW"] += 1
            elif dlat < 0 and dlon >= 0: quad_counts["SE"] += 1
            else: quad_counts["SW"] += 1
        total = sum(quad_counts.values())
        print(f"  Quadrant distribution:", flush=True)
        for q, n in sorted(quad_counts.items(), key=lambda x: -x[1]):
            print(f"    {q}: {n}/{total} ({n/total*100:.0f}%)", flush=True)

    # --- 8. Per-Feature Contribution Delta (weighted) ---
    print("\n--- 8. PER-FEATURE CONTRIBUTION DELTA (weighted_catch - weighted_peak) ---", flush=True)
    print(f"  Features ranked by TOTAL impact (sum of weighted deltas across all catches):", flush=True)
    print(f"  {'Feature':<20s} {'TotalWtdDelta':>14s} {'MeanWtdDelta':>14s} {'Impact':>10s}", flush=True)
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*10}", flush=True)

    impact_list = []
    for feat_name in all_weighted_deltas:
        wd = all_weighted_deltas[feat_name]
        total = np.sum(wd)
        mean = np.mean(wd)
        impact_list.append((feat_name, total, mean))

    impact_list.sort(key=lambda x: x[1])  # most negative first
    for name, total, mean in impact_list:
        impact = "PULLS AWAY" if total < -0.1 else "PULLS TO" if total > 0.1 else "NEUTRAL"
        print(f"  {name:<20s} {total:>+14.4f} {mean:>+14.4f} {impact:>10s}", flush=True)

    # --- Summary & Recommendations ---
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY & RECOMMENDATIONS", flush=True)
    print("=" * 80, flush=True)

    # Top 3 features pulling peaks away
    pull_away = [(n, t, m) for n, t, m in impact_list if t < -0.05]
    if pull_away:
        print("\n  TOP FEATURES PULLING PEAKS AWAY FROM CATCHES:", flush=True)
        for i, (name, total, mean) in enumerate(pull_away[:5]):
            w = weights.get(name, 0)
            print(f"    {i+1}. {name} (weight={w:.2f}): avg delta = {mean:+.4f} per catch", flush=True)

    # Band count issue
    if bc_deficit > 0.3:
        print(f"\n  BAND POSITION ISSUE:", flush=True)
        print(f"    Peaks have {bc_deficit:.1f} more bands on average than catches.", flush=True)
        print(f"    Consider: Widening band width, or reducing 0-band penalty.", flush=True)

    # Directional bias
    if all_offset_directions:
        if offset_mag_nm > 0.5:
            print(f"\n  SYSTEMATIC DIRECTIONAL BIAS:", flush=True)
            print(f"    Peaks are {offset_mag_nm:.1f}nm {_angle_to_compass(angle)} of catches.", flush=True)
            if abs(mean_dlon_nm) > abs(mean_dlat_nm):
                if mean_dlon_nm > 0:
                    print(f"    -> Scoring pulls OFFSHORE (east). Shelf/depth features may be too strong.", flush=True)
                else:
                    print(f"    -> Scoring pulls INSHORE (west). Shelf break proximity may be too strong.", flush=True)
            else:
                if mean_dlat_nm > 0:
                    print(f"    -> Scoring pulls NORTH. SST gradient or current direction may bias.", flush=True)
                else:
                    print(f"    -> Scoring pulls SOUTH. Canyon head features may dominate.", flush=True)

    print("\n  DONE.", flush=True)


def _angle_to_compass(angle_deg):
    """Convert angle (0=E, 90=N) to compass label."""
    dirs = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW",
            "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
    idx = int((angle_deg + 11.25) / 22.5) % 16
    return dirs[idx]


if __name__ == "__main__":
    main()
