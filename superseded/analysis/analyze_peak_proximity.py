#!/usr/bin/env python3
"""Peak Zone Proximity to Catch Analysis.

For each catch location, examines:
1. Distance from catch to nearest peak zone at multiple thresholds (top 5%, 10%, 20%, 30%)
2. Whether catch is INSIDE or OUTSIDE each peak zone
3. Distance to zone centroid vs zone edge
4. Peak zone size (area in nm^2) and shape
5. Score profile: radial average from catch outward
6. Per-feature contribution at catch vs at peak
7. Aggregate statistics and directional bias

Parallelized across dates.
"""
import csv, io, os, sys, json
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import label, center_of_mass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES = os.path.join(SCRIPT_DIR, 'data', 'all_catches.csv')
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}

N_WORKERS = 10

# Thresholds: top N% of ocean scores define "peak zones"
PEAK_THRESHOLDS = [5, 10, 20, 30]

# Radial profile distances (nm)
RADIAL_STEPS = [0.5, 1, 2, 3, 5, 7, 10]


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
        with open(CSV_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sp = (r.get("Species_Name") or r.get("Species", "")).strip().upper()
                if "BLUE MARLIN" not in sp:
                    continue
                lat_raw = r.get("Latitude", "").strip()
                lon_raw = r.get("Longitude", "").strip()
                date_raw = (r.get("Release_Date") or r.get("Date", "")).strip()
                if not lat_raw or not lon_raw or not date_raw:
                    continue
                lat = ddm_to_dd(lat_raw.replace("S", ""), negative=True)
                lon = ddm_to_dd(lon_raw.replace("E", ""))
                date_str = date_raw[:10]
                key = (date_str, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date_str, "lat": lat, "lon": lon})
    if os.path.exists(ALL_CATCHES):
        with open(ALL_CATCHES) as f:
            for r in csv.DictReader(f):
                sp = r.get("species", "").strip().upper()
                if "BLUE MARLIN" not in sp and "MARLIN" not in sp:
                    continue
                if not r.get("lat") or not r.get("lon") or not r.get("date"):
                    continue
                lat = float(r["lat"])
                lon = float(r["lon"])
                date_str = r["date"]
                key = (date_str, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date_str, "lat": lat, "lon": lon})
    return catches


def nm_to_deg_lat(nm):
    return nm / 60.0


def nm_to_deg_lon(nm, lat):
    return nm / (60.0 * np.cos(np.radians(lat)))


def haversine_nm(lat1, lon1, lat2, lon2):
    """Approximate distance in nm using equirectangular projection."""
    dlat = (lat2 - lat1) * 60
    dlon = (lon2 - lon1) * 60 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(dlat**2 + dlon**2)


def pixel_area_nm2(lat, dlat_deg, dlon_deg):
    """Area of a single pixel in nm^2."""
    h_nm = dlat_deg * 60
    w_nm = dlon_deg * 60 * np.cos(np.radians(lat))
    return abs(h_nm * w_nm)


def _worker(args):
    """Analyze peak zone proximity for all catches on one date."""
    date_str, date_catches, script_dir, bbox = args

    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        sys.path.insert(0, script_dir)
        import marlin_data

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
        weights = result.get('weights', {})

        # Grid spacing
        dlat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.01
        dlon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.01

        # Ocean mask (non-NaN, non-zero)
        ocean = ~np.isnan(grid) & (grid > 0)
        ocean_scores = grid[ocean]
        if len(ocean_scores) < 50:
            return []

        # Compute percentile thresholds
        thresholds = {}
        for pct in PEAK_THRESHOLDS:
            thresholds[pct] = np.percentile(ocean_scores, 100 - pct)

        records = []
        for c in date_catches:
            ci_lat = np.argmin(np.abs(lats - c['lat']))
            ci_lon = np.argmin(np.abs(lons - c['lon']))
            catch_score = grid[ci_lat, ci_lon]
            if np.isnan(catch_score):
                continue

            catch_percentile = float(np.sum(ocean_scores <= catch_score) / len(ocean_scores) * 100)

            rec = {
                'date': date_str,
                'lat': round(c['lat'], 4),
                'lon': round(c['lon'], 4),
                'catch_score': round(float(catch_score) * 100, 1),
                'catch_percentile': round(catch_percentile, 1),
                'zones': {},
                'radial_profile': {},
                'features_catch_vs_peak': {},
            }

            # --- Per-threshold zone analysis ---
            for pct in PEAK_THRESHOLDS:
                thresh = thresholds[pct]
                zone_mask = ocean & (grid >= thresh)

                # Label connected components
                labeled, n_zones = label(zone_mask)

                # Which zone (if any) contains the catch?
                catch_label = labeled[ci_lat, ci_lon]
                inside = catch_label > 0

                # Find nearest zone pixel if outside
                if inside:
                    # Catch is inside a peak zone
                    zone_pixels = np.argwhere(labeled == catch_label)
                    zone_lats = lats[zone_pixels[:, 0]]
                    zone_lons = lons[zone_pixels[:, 1]]

                    # Zone centroid
                    centroid = center_of_mass(zone_mask, labeled, catch_label)
                    cent_lat = lats[int(round(centroid[0]))]
                    cent_lon = lons[int(round(centroid[1]))]
                    dist_centroid = haversine_nm(c['lat'], c['lon'], cent_lat, cent_lon)

                    # Distance to zone edge (min distance to non-zone neighbor)
                    dist_edge = 0.0
                    zone_scores = grid[labeled == catch_label]
                    zone_area = len(zone_pixels) * pixel_area_nm2(
                        np.mean(zone_lats), dlat, dlon)
                    zone_max = float(np.max(zone_scores))
                    zone_mean = float(np.mean(zone_scores))

                    # Peak pixel within this zone
                    peak_idx = np.argmax(zone_scores)
                    peak_lat = zone_lats[peak_idx]
                    peak_lon = zone_lons[peak_idx]
                    dist_peak = haversine_nm(c['lat'], c['lon'], peak_lat, peak_lon)

                    # Catch position: how far through the zone from edge to center
                    # 0 = at edge, 1 = at centroid
                    if dist_centroid + dist_edge > 0:
                        zone_position = 1.0 - (dist_centroid / (dist_centroid + max(dist_edge, 0.1)))
                    else:
                        zone_position = 1.0

                else:
                    # Catch is outside — find nearest peak zone pixel
                    zone_pixels = np.argwhere(zone_mask)
                    if len(zone_pixels) == 0:
                        continue

                    # Vectorized distance to all zone pixels
                    z_lats = lats[zone_pixels[:, 0]]
                    z_lons = lons[zone_pixels[:, 1]]
                    dists = np.sqrt(
                        ((z_lats - c['lat']) * 60)**2 +
                        ((z_lons - c['lon']) * 60 * np.cos(np.radians(c['lat'])))**2
                    )
                    nearest_idx = np.argmin(dists)
                    dist_edge = float(dists[nearest_idx])

                    # Nearest zone label
                    near_li = zone_pixels[nearest_idx, 0]
                    near_lo = zone_pixels[nearest_idx, 1]
                    nearest_label = labeled[near_li, near_lo]

                    # Zone properties
                    this_zone = np.argwhere(labeled == nearest_label)
                    zone_lats_z = lats[this_zone[:, 0]]
                    zone_lons_z = lons[this_zone[:, 1]]
                    zone_scores = grid[labeled == nearest_label]
                    zone_area = len(this_zone) * pixel_area_nm2(
                        np.mean(zone_lats_z), dlat, dlon)
                    zone_max = float(np.max(zone_scores))
                    zone_mean = float(np.mean(zone_scores))

                    centroid = center_of_mass(zone_mask, labeled, nearest_label)
                    cent_lat = lats[int(round(centroid[0]))]
                    cent_lon = lons[int(round(centroid[1]))]
                    dist_centroid = haversine_nm(c['lat'], c['lon'], cent_lat, cent_lon)

                    # Peak pixel
                    peak_idx = np.argmax(zone_scores)
                    peak_lat = zone_lats_z[peak_idx]
                    peak_lon = zone_lons_z[peak_idx]
                    dist_peak = haversine_nm(c['lat'], c['lon'], peak_lat, peak_lon)
                    zone_position = 0.0

                # Direction to peak
                dlat_p = peak_lat - c['lat']
                dlon_p = peak_lon - c['lon']
                if dist_peak < 0.5:
                    peak_dir = "AT_CATCH"
                else:
                    angle = np.degrees(np.arctan2(dlon_p, dlat_p))
                    if angle < 0:
                        angle += 360
                    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    peak_dir = dirs[int((angle + 22.5) / 45) % 8]

                rec['zones'][f'top_{pct}pct'] = {
                    'inside': bool(inside),
                    'dist_edge_nm': round(dist_edge, 2),
                    'dist_centroid_nm': round(dist_centroid, 1),
                    'dist_peak_nm': round(dist_peak, 1),
                    'peak_direction': peak_dir,
                    'zone_area_nm2': round(zone_area, 1),
                    'zone_max_score': round(zone_max * 100, 1),
                    'zone_mean_score': round(zone_mean * 100, 1),
                    'zone_position': round(zone_position, 2),
                    'n_zones_total': n_zones,
                }

            # --- Radial score profile from catch ---
            for r_nm in RADIAL_STEPS:
                r_lat = nm_to_deg_lat(r_nm)
                r_lon = nm_to_deg_lon(r_nm, c['lat'])
                # Sample 16 points around the catch at this radius
                angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ring_scores = []
                for a in angles:
                    s_lat = c['lat'] + r_lat * np.sin(a)
                    s_lon = c['lon'] + r_lon * np.cos(a)
                    si = np.argmin(np.abs(lats - s_lat))
                    sj = np.argmin(np.abs(lons - s_lon))
                    if 0 <= si < len(lats) and 0 <= sj < len(lons):
                        v = grid[si, sj]
                        if not np.isnan(v):
                            ring_scores.append(float(v))
                if ring_scores:
                    rec['radial_profile'][f'{r_nm}nm'] = {
                        'mean': round(np.mean(ring_scores) * 100, 1),
                        'max': round(np.max(ring_scores) * 100, 1),
                        'min': round(np.min(ring_scores) * 100, 1),
                    }

            # --- LOCAL PEAK ANALYSIS (within 10nm) ---
            for search_radius in [5, 10]:
                r_lat = nm_to_deg_lat(search_radius)
                r_lon = nm_to_deg_lon(search_radius, c['lat'])
                lat_mask = (lats >= c['lat'] - r_lat) & (lats <= c['lat'] + r_lat)
                lon_mask = (lons >= c['lon'] - r_lon) & (lons <= c['lon'] + r_lon)
                lat_idx = np.where(lat_mask)[0]
                lon_idx = np.where(lon_mask)[0]

                if len(lat_idx) == 0 or len(lon_idx) == 0:
                    continue

                local_grid = grid[np.ix_(lat_idx, lon_idx)]
                local_lats = lats[lat_idx]
                local_lons = lons[lon_idx]

                valid = ~np.isnan(local_grid) & (local_grid > 0)
                if not np.any(valid):
                    continue

                # Find local peak
                local_valid_idx = np.argwhere(valid)
                peak_i = local_valid_idx[np.argmax(local_grid[valid])]
                peak_li, peak_lo = peak_i[0], peak_i[1]
                local_peak_score = float(local_grid[peak_li, peak_lo])
                local_peak_lat = float(local_lats[peak_li])
                local_peak_lon = float(local_lons[peak_lo])

                dist_to_local_peak = haversine_nm(c['lat'], c['lon'],
                                                   local_peak_lat, local_peak_lon)

                # Direction to local peak
                dlp = local_peak_lat - c['lat']
                dlo = local_peak_lon - c['lon']
                if dist_to_local_peak < 0.5:
                    local_peak_dir = "AT_CATCH"
                else:
                    ang = np.degrees(np.arctan2(dlo, dlp))
                    if ang < 0:
                        ang += 360
                    dirs8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    local_peak_dir = dirs8[int((ang + 22.5) / 45) % 8]

                # Score ratio: catch / local peak
                score_ratio = float(catch_score) / local_peak_score if local_peak_score > 0 else 0

                # Local percentile of catch within this radius
                local_ocean = local_grid[valid]
                local_pctl = float(np.sum(local_ocean <= catch_score) / len(local_ocean) * 100)

                # How many pixels in top-10% of local area?
                local_top10_thresh = np.percentile(local_ocean, 90)
                local_top10_mask = valid & (local_grid >= local_top10_thresh)
                local_top10_area = np.sum(local_top10_mask) * pixel_area_nm2(
                    c['lat'], dlat, dlon)

                # Is catch inside local top-10%?
                catch_local_li = np.argmin(np.abs(local_lats - c['lat']))
                catch_local_lo = np.argmin(np.abs(local_lons - c['lon']))
                catch_in_local_top10 = bool(local_top10_mask[catch_local_li, catch_local_lo])

                # Per-feature comparison: catch vs local peak
                local_feat_diff = {}
                if sub_scores:
                    pk_gi = lat_idx[peak_li]
                    pk_gj = lon_idx[peak_lo]
                    for fname, fgrid in sub_scores.items():
                        if fgrid is None or fgrid.shape != grid.shape:
                            continue
                        if '_raw' in fname:
                            continue
                        w = weights.get(fname, 0)
                        if w <= 0:
                            continue
                        cv = fgrid[ci_lat, ci_lon]
                        pv = fgrid[pk_gi, pk_gj]
                        if not np.isnan(cv) and not np.isnan(pv):
                            local_feat_diff[fname] = {
                                'catch': round(float(cv), 3),
                                'peak': round(float(pv), 3),
                                'diff': round(float(pv - cv), 3),
                                'weight': round(float(w), 3),
                                'weighted_diff': round(float((pv - cv) * w), 4),
                            }

                rec[f'local_{search_radius}nm'] = {
                    'peak_score': round(local_peak_score * 100, 1),
                    'dist_to_peak_nm': round(dist_to_local_peak, 1),
                    'peak_direction': local_peak_dir,
                    'score_ratio': round(score_ratio, 3),
                    'local_percentile': round(local_pctl, 1),
                    'catch_in_local_top10': catch_in_local_top10,
                    'local_top10_area_nm2': round(local_top10_area, 1),
                    'peak_lat': round(local_peak_lat, 4),
                    'peak_lon': round(local_peak_lon, 4),
                    'feature_diffs': local_feat_diff,
                }

            # --- Feature contribution: catch vs nearest top-10% peak ---
            z10 = rec['zones'].get('top_10pct', {})
            if z10 and sub_scores:
                # Find the peak pixel for top-10% zone
                thresh10 = thresholds[10]
                zmask10 = ocean & (grid >= thresh10)
                z_pix = np.argwhere(zmask10)
                if len(z_pix) > 0:
                    z_lats = lats[z_pix[:, 0]]
                    z_lons = lons[z_pix[:, 1]]
                    dists = np.sqrt(
                        ((z_lats - c['lat']) * 60)**2 +
                        ((z_lons - c['lon']) * 60 * np.cos(np.radians(c['lat'])))**2
                    )
                    # Find peak of nearest zone
                    near5 = np.argsort(dists)[:max(1, len(dists)//10)]
                    best = near5[np.argmax(grid[z_pix[near5, 0], z_pix[near5, 1]])]
                    pk_li, pk_lo = z_pix[best]

                    for fname, fgrid in sub_scores.items():
                        if fgrid is None or fgrid.shape != grid.shape:
                            continue
                        if '_raw' in fname:
                            continue
                        w = weights.get(fname, 0)
                        if w <= 0:
                            continue
                        cv = fgrid[ci_lat, ci_lon]
                        pv = fgrid[pk_li, pk_lo]
                        if not np.isnan(cv) and not np.isnan(pv):
                            rec['features_catch_vs_peak'][fname] = {
                                'catch': round(float(cv), 3),
                                'peak': round(float(pv), 3),
                                'diff': round(float(pv - cv), 3),
                                'weight': round(float(w), 3),
                                'weighted_diff': round(float((pv - cv) * w), 4),
                            }

            records.append(rec)
        return records

    except Exception:
        import traceback
        traceback.print_exc()
        return []


def main():
    catches = load_catches()
    by_date = defaultdict(list)
    for c in catches:
        by_date[c['date']].append(c)

    work_items = [(d, cs, SCRIPT_DIR, BBOX) for d, cs in sorted(by_date.items())]
    print(f"Peak Zone Proximity Analysis: {len(catches)} catches across {len(work_items)} dates")
    print(f"Thresholds: top {PEAK_THRESHOLDS}%")
    print(f"Workers: {N_WORKERS}\n")

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            completed += 1
            date_results = future.result()
            for rec in date_results:
                results.append(rec)
                z10 = rec['zones'].get('top_10pct', {})
                status = "INSIDE" if z10.get('inside') else f"{z10.get('dist_edge_nm', '?')}nm away"
                print(f"  [{completed}/{len(work_items)}] {rec['date']}: "
                      f"score={rec['catch_score']}% pctl={rec['catch_percentile']:.0f}% "
                      f"top10%: {status}",
                      flush=True)

    if not results:
        print("No results!")
        return

    # Save raw results
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(SCRIPT_DIR, 'data', 'peak_proximity_results.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    N = len(results)
    print(f"\n{'='*90}")
    print(f"PEAK ZONE PROXIMITY ANALYSIS — {N} catches")
    print(f"{'='*90}")

    # --- 1. Zone containment at each threshold ---
    print(f"\n{'-'*90}")
    print(f"1. ZONE CONTAINMENT: Is the catch inside a peak zone?")
    print(f"{'-'*90}")
    print(f"  {'Threshold':>12s}  {'Inside':>7s}  {'Outside':>8s}  {'% Inside':>9s}  "
          f"{'Avg dist if out':>15s}  {'Med dist if out':>15s}")
    for pct in PEAK_THRESHOLDS:
        key = f'top_{pct}pct'
        inside = [r for r in results if r['zones'].get(key, {}).get('inside', False)]
        outside = [r for r in results if key in r['zones'] and not r['zones'][key]['inside']]
        pct_in = len(inside) / N * 100
        if outside:
            out_dists = [r['zones'][key]['dist_edge_nm'] for r in outside]
            avg_d = np.mean(out_dists)
            med_d = np.median(out_dists)
        else:
            avg_d = med_d = 0
        print(f"  Top {pct:>2d}%       {len(inside):>5d}    {len(outside):>5d}      "
              f"{pct_in:>5.1f}%        {avg_d:>7.1f} nm        {med_d:>7.1f} nm")

    # --- 2. Distance to peak (within nearest zone) ---
    print(f"\n{'-'*90}")
    print(f"2. DISTANCE TO PEAK PIXEL (nearest zone at each threshold)")
    print(f"{'-'*90}")
    print(f"  {'Threshold':>12s}  {'Mean':>6s}  {'Median':>7s}  {'P25':>5s}  {'P75':>5s}  "
          f"{'Max':>5s}  {'<1nm':>5s}  {'<3nm':>5s}  {'<5nm':>5s}")
    for pct in PEAK_THRESHOLDS:
        key = f'top_{pct}pct'
        dists = [r['zones'][key]['dist_peak_nm'] for r in results if key in r['zones']]
        if not dists:
            continue
        d = np.array(dists)
        print(f"  Top {pct:>2d}%       {np.mean(d):>5.1f}   {np.median(d):>6.1f}  "
              f"{np.percentile(d,25):>5.1f} {np.percentile(d,75):>5.1f}  "
              f"{np.max(d):>5.1f}  "
              f"{np.sum(d<1)/len(d)*100:>4.0f}% {np.sum(d<3)/len(d)*100:>4.0f}% "
              f"{np.sum(d<5)/len(d)*100:>4.0f}%")

    # --- 3. Direction to peak ---
    print(f"\n{'-'*90}")
    print(f"3. DIRECTION TO NEAREST PEAK (top 10% zones)")
    print(f"{'-'*90}")
    dir_counts = defaultdict(int)
    for r in results:
        z = r['zones'].get('top_10pct', {})
        if z:
            dir_counts[z.get('peak_direction', '?')] += 1
    for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'AT_CATCH']:
        if d in dir_counts:
            bar = '#' * int(dir_counts[d] / N * 60)
            print(f"  {d:>10s}: {dir_counts[d]:>3d} ({dir_counts[d]/N*100:>4.0f}%) {bar}")

    # Offshore (W, NW, SW) vs Shoreward (E, NE, SE)
    offshore_dirs = {'W', 'NW', 'SW'}
    shore_dirs = {'E', 'NE', 'SE'}
    n_off = sum(dir_counts.get(d, 0) for d in offshore_dirs)
    n_shore = sum(dir_counts.get(d, 0) for d in shore_dirs)
    n_ns = sum(dir_counts.get(d, 0) for d in ['N', 'S'])
    n_at = dir_counts.get('AT_CATCH', 0)
    print(f"\n  Offshore (W/NW/SW): {n_off} ({n_off/N*100:.0f}%)")
    print(f"  Shoreward (E/NE/SE): {n_shore} ({n_shore/N*100:.0f}%)")
    print(f"  Along-shelf (N/S): {n_ns} ({n_ns/N*100:.0f}%)")
    print(f"  At catch: {n_at} ({n_at/N*100:.0f}%)")

    # --- 4. Zone size and characteristics ---
    print(f"\n{'-'*90}")
    print(f"4. PEAK ZONE CHARACTERISTICS (top 10%)")
    print(f"{'-'*90}")
    areas = [r['zones']['top_10pct']['zone_area_nm2'] for r in results if 'top_10pct' in r['zones']]
    z_maxs = [r['zones']['top_10pct']['zone_max_score'] for r in results if 'top_10pct' in r['zones']]
    z_means = [r['zones']['top_10pct']['zone_mean_score'] for r in results if 'top_10pct' in r['zones']]
    n_zones = [r['zones']['top_10pct']['n_zones_total'] for r in results if 'top_10pct' in r['zones']]
    print(f"  Zone area:  mean={np.mean(areas):.0f} nm2, median={np.median(areas):.0f} nm2, "
          f"range={np.min(areas):.0f}-{np.max(areas):.0f} nm2")
    print(f"  Zone max:   mean={np.mean(z_maxs):.0f}%, range={np.min(z_maxs):.0f}-{np.max(z_maxs):.0f}%")
    print(f"  Zone mean:  mean={np.mean(z_means):.0f}%")
    print(f"  # of top-10% zones per date: mean={np.mean(n_zones):.0f}, "
          f"range={np.min(n_zones)}-{np.max(n_zones)}")

    # --- 5. Radial score profile ---
    print(f"\n{'-'*90}")
    print(f"5. RADIAL SCORE PROFILE (average score at distance from catch)")
    print(f"{'-'*90}")
    catch_scores = [r['catch_score'] for r in results]
    print(f"  {'Distance':>10s}  {'Mean':>6s}  {'vs catch':>9s}  {'% of catch':>10s}")
    print(f"  {'0 (catch)':>10s}  {np.mean(catch_scores):>5.1f}%  {'---':>9s}  {'100%':>10s}")
    for r_nm in RADIAL_STEPS:
        key = f'{r_nm}nm'
        vals = [r['radial_profile'][key]['mean'] for r in results if key in r['radial_profile']]
        if vals:
            mean_ring = np.mean(vals)
            diff = mean_ring - np.mean(catch_scores)
            ratio = mean_ring / np.mean(catch_scores) * 100 if np.mean(catch_scores) > 0 else 0
            print(f"  {key:>10s}  {mean_ring:>5.1f}%  {diff:>+8.1f}%  {ratio:>9.0f}%")

    # --- 6. Catches by score percentile band ---
    print(f"\n{'-'*90}")
    print(f"6. CATCH SCORE PERCENTILE DISTRIBUTION")
    print(f"{'-'*90}")
    pctls = [r['catch_percentile'] for r in results]
    bands = [(90, 100, "Top 10%"), (80, 90, "80-90%"), (70, 80, "70-80%"),
             (60, 70, "60-70%"), (50, 60, "50-60%"), (0, 50, "Bottom 50%")]
    for lo, hi, label in bands:
        cnt = sum(1 for p in pctls if lo <= p < hi) if hi < 100 else sum(1 for p in pctls if lo <= p <= hi)
        bar = '#' * int(cnt / N * 50)
        print(f"  {label:>12s}: {cnt:>3d} ({cnt/N*100:>4.0f}%) {bar}")
    print(f"\n  Mean percentile: {np.mean(pctls):.0f}%")
    print(f"  Median percentile: {np.median(pctls):.0f}%")

    # --- 7. LOCAL PEAK ANALYSIS (5nm and 10nm) ---
    for search_r in [5, 10]:
        key = f'local_{search_r}nm'
        local_results = [r for r in results if key in r]
        if not local_results:
            continue

        print(f"\n{'-'*90}")
        print(f"7{'a' if search_r == 5 else 'b'}. LOCAL PEAK ANALYSIS (within {search_r}nm of catch)")
        print(f"{'-'*90}")

        l_dists = [r[key]['dist_to_peak_nm'] for r in local_results]
        l_scores = [r[key]['peak_score'] for r in local_results]
        l_ratios = [r[key]['score_ratio'] for r in local_results]
        l_pctls = [r[key]['local_percentile'] for r in local_results]
        l_in_top10 = [r for r in local_results if r[key]['catch_in_local_top10']]

        print(f"  Catches analyzed: {len(local_results)}")
        print(f"\n  Distance to local peak ({search_r}nm radius):")
        print(f"    Mean:   {np.mean(l_dists):.1f} nm")
        print(f"    Median: {np.median(l_dists):.1f} nm")
        print(f"    P25:    {np.percentile(l_dists, 25):.1f} nm")
        print(f"    P75:    {np.percentile(l_dists, 75):.1f} nm")
        d_arr = np.array(l_dists)
        for thresh in [1, 2, 3, 5]:
            if thresh <= search_r:
                pct = np.sum(d_arr < thresh) / len(d_arr) * 100
                print(f"    < {thresh}nm:  {np.sum(d_arr < thresh)}/{len(d_arr)} ({pct:.0f}%)")

        print(f"\n  Score at catch vs local peak:")
        print(f"    Catch mean:     {np.mean([r['catch_score'] for r in local_results]):.1f}%")
        print(f"    Local peak:     {np.mean(l_scores):.1f}%")
        print(f"    Score ratio:    {np.mean(l_ratios):.3f} (1.0 = catch IS the peak)")
        print(f"    Catch/peak=1.0: {sum(1 for r in l_ratios if r >= 0.999)}/{len(local_results)} "
              f"({sum(1 for r in l_ratios if r >= 0.999)/len(local_results)*100:.0f}%)")
        print(f"    Catch/peak>=0.95: {sum(1 for r in l_ratios if r >= 0.95)}/{len(local_results)} "
              f"({sum(1 for r in l_ratios if r >= 0.95)/len(local_results)*100:.0f}%)")
        print(f"    Catch/peak>=0.90: {sum(1 for r in l_ratios if r >= 0.90)}/{len(local_results)} "
              f"({sum(1 for r in l_ratios if r >= 0.90)/len(local_results)*100:.0f}%)")

        print(f"\n  Local percentile (catch rank within {search_r}nm):")
        print(f"    Mean:   {np.mean(l_pctls):.0f}%")
        print(f"    Median: {np.median(l_pctls):.0f}%")
        print(f"    In local top 10%: {len(l_in_top10)}/{len(local_results)} "
              f"({len(l_in_top10)/len(local_results)*100:.0f}%)")

        # Direction to local peak
        print(f"\n  Direction to local peak:")
        ld_counts = defaultdict(int)
        for r in local_results:
            ld_counts[r[key]['peak_direction']] += 1
        for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'AT_CATCH']:
            if d in ld_counts:
                bar = '#' * int(ld_counts[d] / len(local_results) * 50)
                print(f"    {d:>10s}: {ld_counts[d]:>3d} ({ld_counts[d]/len(local_results)*100:>4.0f}%) {bar}")

        l_off = sum(ld_counts.get(d, 0) for d in ['W', 'NW', 'SW'])
        l_shore = sum(ld_counts.get(d, 0) for d in ['E', 'NE', 'SE'])
        l_ns = sum(ld_counts.get(d, 0) for d in ['N', 'S'])
        l_at = ld_counts.get('AT_CATCH', 0)
        print(f"    Offshore (W/NW/SW): {l_off} ({l_off/len(local_results)*100:.0f}%)")
        print(f"    Shoreward (E/NE/SE): {l_shore} ({l_shore/len(local_results)*100:.0f}%)")
        print(f"    Along-shelf (N/S): {l_ns} ({l_ns/len(local_results)*100:.0f}%)")
        print(f"    At catch: {l_at} ({l_at/len(local_results)*100:.0f}%)")

        # Feature differences (local peak)
        print(f"\n  Features driving local peak offset:")
        lf_diffs = defaultdict(list)
        lf_catch = defaultdict(list)
        lf_peak = defaultdict(list)
        lf_wdiffs = defaultdict(list)
        for r in local_results:
            for fname, fdata in r[key].get('feature_diffs', {}).items():
                lf_diffs[fname].append(fdata['diff'])
                lf_catch[fname].append(fdata['catch'])
                lf_peak[fname].append(fdata['peak'])
                lf_wdiffs[fname].append(fdata['weighted_diff'])

        print(f"    {'Feature':>20s}  {'W':>5s}  {'Catch':>6s}  {'Peak':>6s}  "
              f"{'Diff':>6s}  {'W*Diff':>7s}")
        lf_sorted = sorted(lf_wdiffs.keys(), key=lambda f: abs(np.mean(lf_wdiffs[f])), reverse=True)
        for fname in lf_sorted:
            mc = np.mean(lf_catch[fname])
            mp = np.mean(lf_peak[fname])
            md = np.mean(lf_diffs[fname])
            mw = np.mean(lf_wdiffs[fname])
            w_vals = [r[key]['feature_diffs'][fname]['weight']
                      for r in local_results if fname in r[key].get('feature_diffs', {})]
            w = np.mean(w_vals) if w_vals else 0
            print(f"    {fname:>20s}  {w:>5.3f}  {mc:>6.3f}  {mp:>6.3f}  "
                  f"{md:>+6.3f}  {mw:>+7.4f}")

    # --- 8. Per-catch detail: local 10nm ---
    print(f"\n{'-'*90}")
    print(f"8. PER-CATCH DETAIL (local 10nm peak)")
    print(f"{'-'*90}")
    local_key = 'local_10nm'
    has_local = [r for r in results if local_key in r]
    by_ratio = sorted(has_local, key=lambda r: r[local_key]['score_ratio'])
    print(f"  {'Date':>12s}  {'Score':>6s}  {'Peak':>6s}  {'Ratio':>6s}  "
          f"{'Dist':>5s}  {'Dir':>5s}  {'LclPctl':>7s}  {'Top diff feature'}")
    for r in by_ratio[:15]:
        loc = r[local_key]
        fdiffs = loc.get('feature_diffs', {})
        top_driver = max(fdiffs.items(), key=lambda x: abs(x[1]['weighted_diff']),
                         default=('none', {'diff': 0}))
        print(f"  {r['date']:>12s}  {r['catch_score']:>5.0f}%  {loc['peak_score']:>5.0f}%  "
              f"{loc['score_ratio']:>5.3f}  {loc['dist_to_peak_nm']:>4.1f}nm "
              f"{loc['peak_direction']:>4s}  {loc['local_percentile']:>6.0f}%  "
              f"{top_driver[0]}({top_driver[1]['diff']:+.2f})")

    # --- 9. Feature differences driving peak offset (global) ---
    print(f"\n{'-'*90}")
    print(f"9. FEATURE CONTRIBUTIONS: Catch vs Nearest Top-10% Peak (global)")
    print(f"{'-'*90}")
    feat_diffs = defaultdict(list)
    feat_catch = defaultdict(list)
    feat_peak = defaultdict(list)
    feat_wdiffs = defaultdict(list)
    for r in results:
        for fname, fdata in r.get('features_catch_vs_peak', {}).items():
            feat_diffs[fname].append(fdata['diff'])
            feat_catch[fname].append(fdata['catch'])
            feat_peak[fname].append(fdata['peak'])
            feat_wdiffs[fname].append(fdata['weighted_diff'])

    print(f"  {'Feature':>20s}  {'W':>5s}  {'Catch':>6s}  {'Peak':>6s}  "
          f"{'Diff':>6s}  {'W*Diff':>7s}  {'Pulls peak?'}")
    sorted_feats = sorted(feat_wdiffs.keys(), key=lambda f: abs(np.mean(feat_wdiffs[f])), reverse=True)
    for fname in sorted_feats:
        mc = np.mean(feat_catch[fname])
        mp = np.mean(feat_peak[fname])
        md = np.mean(feat_diffs[fname])
        mw = np.mean(feat_wdiffs[fname])
        # Find approximate weight
        w_vals = [r['features_catch_vs_peak'][fname]['weight']
                  for r in results if fname in r.get('features_catch_vs_peak', {})]
        w = np.mean(w_vals) if w_vals else 0
        pull = "YES - peak higher" if md > 0.03 else ("catch higher" if md < -0.03 else "~equal")
        print(f"  {fname:>20s}  {w:>5.3f}  {mc:>6.3f}  {mp:>6.3f}  "
              f"{md:>+6.3f}  {mw:>+7.4f}  {pull}")

    # --- 10. Worst misaligned catches ---
    print(f"\n{'-'*90}")
    print(f"10. MOST MISALIGNED CATCHES (furthest from top-10% peak)")
    print(f"{'-'*90}")
    by_dist = sorted(results, key=lambda r: r['zones'].get('top_10pct', {}).get('dist_peak_nm', 0),
                     reverse=True)
    print(f"  {'Date':>12s}  {'Score':>6s}  {'Pctl':>5s}  {'Dist':>6s}  {'Dir':>5s}  "
          f"{'Inside':>6s}  {'Top features driving offset'}")
    for r in by_dist[:10]:
        z = r['zones'].get('top_10pct', {})
        # Top 3 features with biggest weighted diff
        fdiffs = r.get('features_catch_vs_peak', {})
        top_drivers = sorted(fdiffs.items(), key=lambda x: abs(x[1]['weighted_diff']), reverse=True)[:3]
        drivers_str = ", ".join(f"{f}({d['diff']:+.2f})" for f, d in top_drivers)
        print(f"  {r['date']:>12s}  {r['catch_score']:>5.0f}%  {r['catch_percentile']:>4.0f}%  "
              f"{z.get('dist_peak_nm',0):>5.1f}nm {z.get('peak_direction','?'):>4s}  "
              f"{'Y' if z.get('inside') else 'N':>6s}  {drivers_str}")

    # --- 11. Best aligned catches ---
    print(f"\n{'-'*90}")
    print(f"11. BEST ALIGNED CATCHES (inside top-10% zone, closest to peak)")
    print(f"{'-'*90}")
    inside_results = [r for r in results if r['zones'].get('top_10pct', {}).get('inside', False)]
    by_dist_asc = sorted(inside_results, key=lambda r: r['zones']['top_10pct']['dist_peak_nm'])
    print(f"  {'Date':>12s}  {'Score':>6s}  {'Pctl':>5s}  {'Dist':>6s}  {'Zone area':>10s}")
    for r in by_dist_asc[:10]:
        z = r['zones']['top_10pct']
        print(f"  {r['date']:>12s}  {r['catch_score']:>5.0f}%  {r['catch_percentile']:>4.0f}%  "
              f"{z['dist_peak_nm']:>5.1f}nm  {z['zone_area_nm2']:>8.0f} nm2")

    # --- 12. Score vs distance correlation ---
    print(f"\n{'-'*90}")
    print(f"12. CORRELATIONS")
    print(f"{'-'*90}")
    scores = np.array([r['catch_score'] for r in results])
    d10 = np.array([r['zones'].get('top_10pct', {}).get('dist_peak_nm', np.nan) for r in results])
    valid = ~np.isnan(d10)
    if np.sum(valid) > 5:
        corr = np.corrcoef(scores[valid], d10[valid])[0, 1]
        print(f"  Catch score vs dist-to-peak (top 10%): r = {corr:.3f}")
        print(f"  (Negative = higher-scoring catches are closer to peaks = good alignment)")

    areas10 = np.array([r['zones'].get('top_10pct', {}).get('zone_area_nm2', np.nan) for r in results])
    valid2 = ~np.isnan(areas10) & ~np.isnan(d10)
    if np.sum(valid2) > 5:
        corr2 = np.corrcoef(areas10[valid2], d10[valid2])[0, 1]
        print(f"  Zone area vs dist-to-peak: r = {corr2:.3f}")
        print(f"  (Negative = larger zones have catches closer to peak)")

    print(f"\n{'='*90}")
    print(f"ANALYSIS COMPLETE — saved raw data to data/peak_proximity_results.json")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
