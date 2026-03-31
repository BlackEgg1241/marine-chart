#!/usr/bin/env python3
"""Analyze feature values in a 10nm radius around each catch location.

Parallelized: each date scored in its own process.

Reports:
1. Absolute feature values at each catch
2. Direction & distance to local peak
3. Which features differ most between catch and peak
4. Feature correlations and combination patterns
5. Quadrant scoring (N/S/E/W of catch)
"""
import csv, io, os, sys, json
import numpy as np
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES = os.path.join(SCRIPT_DIR, 'data', 'all_catches.csv')
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}

N_WORKERS = 12


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches():
    catches = []
    seen = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if 'BLUE MARLIN' not in r.get('Species_Name', '').upper():
                    continue
                lat_raw = r.get('Latitude', '').strip()
                lon_raw = r.get('Longitude', '').strip()
                if not lat_raw or not lon_raw:
                    continue
                lat = ddm_to_dd(lat_raw.rstrip('SsNn'), negative='S' in lat_raw.upper())
                lon = ddm_to_dd(lon_raw.rstrip('EeWw'), negative='W' in lon_raw.upper())
                d = r.get('Release_Date', '').strip()
                if 'T' in d:
                    d = d.split('T')[0]
                for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(d, fmt)
                        ds = dt.strftime('%Y-%m-%d')
                        key = (ds, round(lat, 4), round(lon, 4))
                        if key not in seen:
                            seen.add(key)
                            catches.append({'date': ds, 'lat': lat, 'lon': lon})
                        break
                    except ValueError:
                        continue
    if os.path.exists(ALL_CATCHES):
        with open(ALL_CATCHES, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if 'BLUE MARLIN' not in r.get('species', '').upper():
                    continue
                lat_raw = r.get('lat', '').strip()
                lon_raw = r.get('lon', '').strip()
                if not lat_raw or not lon_raw:
                    continue
                try:
                    lat, lon = float(lat_raw), float(lon_raw)
                except ValueError:
                    continue
                d = r.get('date', '').strip()
                if 'T' in d:
                    d = d.split('T')[0]
                for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(d, fmt)
                        ds = dt.strftime('%Y-%m-%d')
                        key = (ds, round(lat, 4), round(lon, 4))
                        if key not in seen:
                            seen.add(key)
                            catches.append({'date': ds, 'lat': lat, 'lon': lon})
                        break
                    except ValueError:
                        continue
    return catches


def nm_to_deg_lat(nm):
    return nm / 60.0


def nm_to_deg_lon(nm, lat):
    return nm / (60.0 * np.cos(np.radians(lat)))


def _analyze_worker(args):
    """Worker: analyze catches for a single date."""
    date_str, date_catches, script_dir, bbox = args

    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        sys.path.insert(0, script_dir)
        import importlib
        import marlin_data
        importlib.reload(marlin_data)

        ddir = os.path.join(script_dir, 'data', date_str)
        if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
            return []

        marlin_data.OUTPUT_DIR = ddir
        tif = os.path.join(ddir, 'bathy_gmrt.tif')

        result = marlin_data.generate_blue_marlin_hotspots(
            bbox, tif_path=tif if os.path.exists(tif) else None,
            date_str=date_str)

        if result is None:
            return []

        grid = result['grid']
        lats = result['lats']
        lons = result['lons']
        sub_scores = result.get('sub_scores', {})

        records = []
        for c in date_catches:
            ci_lat = np.argmin(np.abs(lats - c['lat']))
            ci_lon = np.argmin(np.abs(lons - c['lon']))
            catch_score = grid[ci_lat, ci_lon]
            if np.isnan(catch_score):
                continue

            # 10nm radius box
            dlat = nm_to_deg_lat(10)
            dlon = nm_to_deg_lon(10, c['lat'])
            lat_mask = (lats >= c['lat'] - dlat) & (lats <= c['lat'] + dlat)
            lon_mask = (lons >= c['lon'] - dlon) & (lons <= c['lon'] + dlon)
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                continue

            local_grid = grid[np.ix_(lat_idx, lon_idx)]
            local_lats = lats[lat_idx]
            local_lons = lons[lon_idx]

            valid_local = np.where(~np.isnan(local_grid))
            if len(valid_local[0]) == 0:
                continue

            peak_idx = np.argmax(local_grid[valid_local])
            peak_li = valid_local[0][peak_idx]
            peak_lo = valid_local[1][peak_idx]
            peak_score = local_grid[peak_li, peak_lo]
            peak_lat = local_lats[peak_li]
            peak_lon = local_lons[peak_lo]

            dlat_peak = peak_lat - c['lat']
            dlon_peak = peak_lon - c['lon']
            dist_nm = np.sqrt((dlat_peak * 60)**2 + (dlon_peak * 60 * np.cos(np.radians(c['lat'])))**2)

            if dist_nm < 0.5:
                direction = "AT_CATCH"
            else:
                angle = np.degrees(np.arctan2(dlon_peak, dlat_peak))
                if angle < 0:
                    angle += 360
                dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                direction = dirs[int((angle + 22.5) / 45) % 8]

            peak_global_li = lat_idx[peak_li]
            peak_global_lo = lon_idx[peak_lo]

            feature_comparison = {}
            for fname, fgrid in sub_scores.items():
                if fgrid is None or fgrid.shape != grid.shape:
                    continue
                catch_val = fgrid[ci_lat, ci_lon]
                peak_val = fgrid[peak_global_li, peak_global_lo]
                if not np.isnan(catch_val) and not np.isnan(peak_val):
                    feature_comparison[fname] = {
                        'at_catch': round(float(catch_val), 3),
                        'at_peak': round(float(peak_val), 3),
                        'diff': round(float(peak_val - catch_val), 3)
                    }

            catch_local_li = np.argmin(np.abs(local_lats - c['lat']))
            catch_local_lo = np.argmin(np.abs(local_lons - c['lon']))

            quadrants = {}
            q_slices = {
                'N': local_grid[:catch_local_li, :],
                'S': local_grid[catch_local_li + 1:, :],
                'W_offshore': local_grid[:, :catch_local_lo],
                'E_shoreward': local_grid[:, catch_local_lo + 1:],
            }
            for qname, qgrid in q_slices.items():
                valid = qgrid[~np.isnan(qgrid)]
                quadrants[qname] = round(float(np.mean(valid)) * 100, 1) if len(valid) > 0 else None

            feat_by_quadrant = {}
            for fname, fgrid in sub_scores.items():
                if fgrid is None or fgrid.shape != grid.shape:
                    continue
                local_feat = fgrid[np.ix_(lat_idx, lon_idx)]
                fq = {}
                for qname in ['N', 'S', 'W_offshore', 'E_shoreward']:
                    if qname == 'N':
                        sl = local_feat[:catch_local_li, :]
                    elif qname == 'S':
                        sl = local_feat[catch_local_li + 1:, :]
                    elif qname == 'W_offshore':
                        sl = local_feat[:, :catch_local_lo]
                    else:
                        sl = local_feat[:, catch_local_lo + 1:]
                    valid = sl[~np.isnan(sl)]
                    fq[qname] = round(float(np.mean(valid)), 3) if len(valid) > 0 else None
                feat_by_quadrant[fname] = fq

            gradients = {}
            for step_nm in [2, 5, 10]:
                step_lat = nm_to_deg_lat(step_nm)
                step_lon = nm_to_deg_lon(step_nm, c['lat'])
                for dname, dlat_s, dlon_s in [('N', step_lat, 0), ('S', -step_lat, 0),
                                               ('W', 0, -step_lon), ('E', 0, step_lon)]:
                    tgt_lat = c['lat'] + dlat_s
                    tgt_lon = c['lon'] + dlon_s
                    ti_lat = np.argmin(np.abs(lats - tgt_lat))
                    ti_lon = np.argmin(np.abs(lons - tgt_lon))
                    tgt_val = grid[ti_lat, ti_lon]
                    if not np.isnan(tgt_val):
                        gradients[f"{dname}_{step_nm}nm"] = round(float(tgt_val - catch_score) * 100, 1)

            records.append({
                'date': date_str,
                'catch_lat': round(c['lat'], 4),
                'catch_lon': round(c['lon'], 4),
                'catch_score': round(float(catch_score) * 100, 1),
                'peak_score': round(float(peak_score) * 100, 1),
                'peak_direction': direction,
                'peak_dist_nm': round(dist_nm, 1),
                'peak_lat': round(float(peak_lat), 4),
                'peak_lon': round(float(peak_lon), 4),
                'quadrant_means': quadrants,
                'features_at_catch': {f: v['at_catch'] for f, v in feature_comparison.items()},
                'features_at_peak': {f: v['at_peak'] for f, v in feature_comparison.items()},
                'feature_diffs': {f: v['diff'] for f, v in feature_comparison.items()},
                'feat_by_quadrant': feat_by_quadrant,
                'gradients': gradients,
            })

        return records

    except Exception:
        return []


def analyze():
    catches = load_catches()
    print(f"Loaded {len(catches)} catches")

    # Group by date
    by_date = defaultdict(list)
    for c in catches:
        by_date[c['date']].append(c)

    # Build work items
    work_items = [(date_str, date_catches, SCRIPT_DIR, BBOX)
                  for date_str, date_catches in sorted(by_date.items())]

    print(f"Processing {len(work_items)} dates with {N_WORKERS} workers...")

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_analyze_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            completed += 1
            date_results = future.result()
            for rec in date_results:
                results.append(rec)
                print(f"  [{completed}/{len(work_items)}] {rec['date']}: "
                      f"catch={rec['catch_score']}% peak={rec['peak_score']}% "
                      f"dir={rec['peak_direction']} dist={rec['peak_dist_nm']:.1f}nm",
                      flush=True)

    # =====================================================================
    # SUMMARY ANALYSIS
    # =====================================================================
    print(f"\n{'=' * 90}")
    print(f"SUMMARY: {len(results)} catches analyzed")
    print(f"{'=' * 90}")

    # 1. Peak direction histogram
    dir_counts = defaultdict(int)
    for r in results:
        dir_counts[r['peak_direction']] += 1
    print(f"\n--- PEAK DIRECTION FROM CATCH (within 10nm) ---")
    for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'AT_CATCH']:
        if d in dir_counts:
            bar = '#' * dir_counts[d]
            print(f"  {d:>10s}: {dir_counts[d]:>3d} {bar}")

    dists = [r['peak_dist_nm'] for r in results]
    print(f"\n  Peak distance: mean={np.mean(dists):.1f}nm median={np.median(dists):.1f}nm")

    # 2. Quadrant scoring
    print(f"\n--- QUADRANT SCORING (mean score in each direction from catch) ---")
    for q in ['N', 'S', 'W_offshore', 'E_shoreward']:
        vals = [r['quadrant_means'][q] for r in results if r['quadrant_means'].get(q) is not None]
        if vals:
            print(f"  {q:>15s}: mean={np.mean(vals):.1f}% median={np.median(vals):.1f}% "
                  f"std={np.std(vals):.1f}%")

    # 3. Absolute feature values at catch locations
    print(f"\n--- ABSOLUTE FEATURE VALUES AT CATCH LOCATIONS ---")
    all_features = set()
    for r in results:
        all_features.update(r['features_at_catch'].keys())

    for fname in sorted(all_features):
        vals = [r['features_at_catch'][fname] for r in results if fname in r['features_at_catch']]
        if vals:
            print(f"  {fname:>20s}: mean={np.mean(vals):.3f} med={np.median(vals):.3f} "
                  f"min={np.min(vals):.3f} max={np.max(vals):.3f} std={np.std(vals):.3f}")

    # 4. Feature differences (peak vs catch)
    print(f"\n--- FEATURE DIFFERENCES (peak minus catch) ---")
    print(f"  {'feature':>20s}  {'catch':>7s}  {'peak':>7s}  {'diff':>7s}  {'interpretation'}")
    for fname in sorted(all_features):
        catch_vals = [r['features_at_catch'][fname] for r in results if fname in r['features_at_catch']]
        peak_vals = [r['features_at_peak'][fname] for r in results if fname in r['features_at_peak']]
        diffs = [r['feature_diffs'][fname] for r in results if fname in r['feature_diffs']]
        if diffs:
            mean_diff = np.mean(diffs)
            interp = "peak HIGHER" if mean_diff > 0.01 else ("peak LOWER" if mean_diff < -0.01 else "similar")
            print(f"  {fname:>20s}  {np.mean(catch_vals):>7.3f}  {np.mean(peak_vals):>7.3f}  "
                  f"{mean_diff:>+7.3f}  {interp}")

    # 5. Feature quadrant analysis
    print(f"\n--- FEATURE VALUES BY QUADRANT (W_offshore vs E_shoreward) ---")
    print(f"  {'feature':>20s}  {'W_offshore':>10s}  {'E_shoreward':>11s}  {'W-E diff':>8s}  {'W bias?'}")
    for fname in sorted(all_features):
        w_vals = [r['feat_by_quadrant'][fname]['W_offshore']
                  for r in results if fname in r['feat_by_quadrant']
                  and r['feat_by_quadrant'][fname].get('W_offshore') is not None]
        e_vals = [r['feat_by_quadrant'][fname]['E_shoreward']
                  for r in results if fname in r['feat_by_quadrant']
                  and r['feat_by_quadrant'][fname].get('E_shoreward') is not None]
        if w_vals and e_vals:
            w_mean = np.mean(w_vals)
            e_mean = np.mean(e_vals)
            diff = w_mean - e_mean
            bias = "YES - pulls W" if diff > 0.02 else ("pulls E" if diff < -0.02 else "neutral")
            print(f"  {fname:>20s}  {w_mean:>10.3f}  {e_mean:>11.3f}  {diff:>+8.3f}  {bias}")

    # 6. Gradient analysis
    print(f"\n--- SCORE GRADIENT FROM CATCH (+ = score increases, - = score decreases) ---")
    for step in [2, 5, 10]:
        print(f"  At {step}nm:")
        for d in ['N', 'S', 'W', 'E']:
            key = f"{d}_{step}nm"
            vals = [r['gradients'][key] for r in results if key in r['gradients']]
            if vals:
                print(f"    {d:>5s}: mean={np.mean(vals):>+5.1f}% median={np.median(vals):>+5.1f}%")

    # 7. Combination analysis
    print(f"\n--- FEATURE COMBINATION PATTERNS AT CATCHES ---")
    high_threshold = 0.7
    combo_counts = defaultdict(int)
    for r in results:
        high_feats = sorted([f for f, v in r['features_at_catch'].items() if v >= high_threshold])
        if len(high_feats) >= 2:
            for i in range(len(high_feats)):
                for j in range(i + 1, len(high_feats)):
                    combo_counts[(high_feats[i], high_feats[j])] += 1

    print(f"  Feature pairs both >= {high_threshold} at catch (top 15):")
    for (f1, f2), count in sorted(combo_counts.items(), key=lambda x: -x[1])[:15]:
        pct = count / len(results) * 100
        print(f"    {f1:>20s} + {f2:<20s}: {count:>3d}/{len(results)} ({pct:.0f}%)")

    # 8. Most misaligned catches
    print(f"\n--- MOST MISALIGNED CATCHES (peak > 3nm from catch) ---")
    for r in sorted(results, key=lambda x: -x['peak_dist_nm']):
        if r['peak_dist_nm'] < 3:
            break
        top_pull = max(r['feature_diffs'].items(), key=lambda x: x[1]) if r['feature_diffs'] else ('?', 0)
        print(f"  {r['date']} dist={r['peak_dist_nm']:.1f}nm dir={r['peak_direction']:>2s} "
              f"catch={r['catch_score']}% peak={r['peak_score']}% "
              f"top_pull={top_pull[0]}({top_pull[1]:+.3f})")

    # Save
    with open('data/catch_surroundings_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to data/catch_surroundings_analysis.json")


if __name__ == '__main__':
    analyze()
