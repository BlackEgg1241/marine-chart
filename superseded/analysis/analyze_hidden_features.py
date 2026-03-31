"""Analysis of 'hidden' scoring features at blue marlin catch locations.

Studies front_corridor, convergence, current_shear, upwelling_edge, and
chl_curvature — the 5 features that previously had no visible overlay.

Analyses:
  1. Score distributions at catch locations vs random ocean control points
  2. Proximity to new contour lines (GeoJSON overlays)
  3. Feature co-occurrence — which combos fire together at catches
  4. Sensitivity — how much would scores change if each feature were zeroed
  5. Spatial clustering — do catches cluster near feature hotspots
"""
import marlin_data, numpy as np, os, csv, json
from datetime import datetime
from collections import defaultdict
from itertools import combinations

NM_PER_DEG_LAT = 60.0

def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R_nm * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def min_dist_to_features(lat, lon, features):
    best = float('inf')
    for feat in features:
        coords = feat['geometry']['coordinates']
        if feat['geometry']['type'] == 'MultiLineString':
            for line in coords:
                for c in line:
                    d = haversine_nm(lat, lon, c[1], c[0])
                    if d < best: best = d
        else:
            for c in coords:
                d = haversine_nm(lat, lon, c[1], c[0])
                if d < best: best = d
    return best if best < 999 else None

# Load catches
with open('data/all_catches.csv', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if 'BLUE MARLIN' in r.get('species', '').upper()]

catches = []
for r in rows:
    lat, lon = r.get('lat', '').strip(), r.get('lon', '').strip()
    if not lat or not lon: continue
    for fmt in ['%d/%m/%Y', '%Y-%m-%d']:
        try:
            dt = datetime.strptime(r['date'].strip(), fmt)
            catches.append({'date': dt.strftime('%Y-%m-%d'), 'lat': float(lat), 'lon': float(lon),
                            'tag': r.get('tag', ''), 'type': r.get('type', '')})
            break
        except: continue

by_date = defaultdict(list)
for c in catches: by_date[c['date']].append(c)

BBOX = [114.5, -32.5, 115.6, -31.5]
FEATURES = ['front_corridor', 'convergence', 'current_shear', 'upwelling_edge', 'chl_curvature']
FEATURE_WEIGHTS = {
    'front_corridor': 0.10, 'convergence': 0.02, 'current_shear': 0.04,
    'upwelling_edge': 0.04, 'chl_curvature': 0.02,
}

# New overlay GeoJSON files
OVERLAY_FILES = {
    'front_corridor': 'front_corridors.geojson',
    'convergence': 'convergence_zones.geojson',
    'current_shear': 'shear_zones.geojson',
    'upwelling_edge': 'upwelling_edges.geojson',
    'chl_curvature': 'chl_curvature.geojson',
}

# ---------- ANALYSIS ----------
catch_scores = []      # per-catch feature scores
control_scores = []    # random ocean control points
proximity_data = []    # distance to nearest overlay contour
n_control = 20         # control points per date

print(f"Analysing {len(catches)} catches across {len(by_date)} dates...")
print()

for date in sorted(by_date.keys()):
    ddir = os.path.join('data', date)
    if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
        ddir = os.path.join('data', 'prediction', date)
    if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
        continue

    tif = os.path.join(ddir, 'bathy_gmrt.tif')
    marlin_data.OUTPUT_DIR = ddir
    result = marlin_data.generate_blue_marlin_hotspots(
        BBOX, tif_path=tif if os.path.exists(tif) else None, date_str=date)
    grid = result['grid']
    lats, lons = result['lats'], result['lons']
    ss = result['sub_scores']

    # Load overlay GeoJSONs for proximity analysis
    overlays = {}
    for feat, fname in OVERLAY_FILES.items():
        path = os.path.join(ddir, fname)
        if os.path.exists(path):
            with open(path) as f:
                gj = json.load(f)
                overlays[feat] = gj['features']

    # Catch scores
    for c in by_date[date]:
        li = np.argmin(np.abs(lats - c['lat']))
        lo = np.argmin(np.abs(lons - c['lon']))
        total = float(grid[li, lo]) if not np.isnan(grid[li, lo]) else 0

        row = {'date': date, 'tag': c['tag'], 'lat': c['lat'], 'lon': c['lon'],
               'total_score': round(total, 3), 'is_catch': True}

        for feat in FEATURES:
            if feat in ss and isinstance(ss[feat], np.ndarray):
                val = ss[feat][li, lo]
                row[feat] = round(float(val), 4) if not np.isnan(val) else 0
            else:
                row[feat] = 0

        # Proximity to overlay contours
        for feat, features in overlays.items():
            if features:
                dist = min_dist_to_features(c['lat'], c['lon'], features)
                row[f'{feat}_dist_nm'] = round(dist, 2) if dist else None

        # Active feature count (score > 0.3)
        row['n_active'] = sum(1 for feat in FEATURES if row.get(feat, 0) > 0.3)

        catch_scores.append(row)

    # Control points: random valid ocean cells
    valid_mask = ~np.isnan(grid) & (grid > 0)
    valid_idx = np.argwhere(valid_mask)
    if len(valid_idx) > n_control:
        chosen = valid_idx[np.random.choice(len(valid_idx), n_control, replace=False)]
    else:
        chosen = valid_idx

    for idx in chosen:
        ri, ci = idx
        total = float(grid[ri, ci])
        row = {'date': date, 'lat': float(lats[ri]), 'lon': float(lons[ci]),
               'total_score': round(total, 3), 'is_catch': False}
        for feat in FEATURES:
            if feat in ss and isinstance(ss[feat], np.ndarray):
                val = ss[feat][ri, ci]
                row[feat] = round(float(val), 4) if not np.isnan(val) else 0
            else:
                row[feat] = 0
        row['n_active'] = sum(1 for feat in FEATURES if row.get(feat, 0) > 0.3)
        control_scores.append(row)

    print(f'  {date}: {len(by_date[date])} catches processed', flush=True)

print(f'\nTotal: {len(catch_scores)} catch points, {len(control_scores)} control points')

# =====================================================================
# 1. SCORE DISTRIBUTIONS: CATCH vs CONTROL
# =====================================================================
print(f'\n{"="*90}')
print(f'1. FEATURE SCORE DISTRIBUTIONS — CATCH vs CONTROL')
print(f'{"="*90}')
print(f'{"Feature":<20s} {"Wt":>4s} | {"Catch Mean":>10s} {"Catch Med":>10s} | {"Ctrl Mean":>10s} {"Ctrl Med":>10s} | {"Lift":>6s} {"p>0.3":>6s}')
print('-' * 90)

for feat in FEATURES:
    w = FEATURE_WEIGHTS[feat]
    c_vals = [r[feat] for r in catch_scores]
    o_vals = [r[feat] for r in control_scores]
    c_mean, c_med = np.mean(c_vals), np.median(c_vals)
    o_mean, o_med = np.mean(o_vals), np.median(o_vals)
    lift = (c_mean / o_mean - 1) * 100 if o_mean > 0 else 0
    pct_active = sum(1 for v in c_vals if v > 0.3) / len(c_vals) * 100
    print(f'{feat:<20s} {w:>4.2f} | {c_mean:>10.3f} {c_med:>10.3f} | '
          f'{o_mean:>10.3f} {o_med:>10.3f} | {lift:>+5.0f}% {pct_active:>5.0f}%')

# =====================================================================
# 2. PROXIMITY TO NEW OVERLAY CONTOURS
# =====================================================================
print(f'\n{"="*90}')
print(f'2. CATCH PROXIMITY TO NEW OVERLAY CONTOURS')
print(f'{"="*90}')
print(f'{"Overlay":<20s} {"N":>4s} {"Mean":>7s} {"Med":>7s} {"Min":>7s} {"<1nm":>6s} {"<2nm":>6s} {"<3nm":>6s} {"<5nm":>6s}')
print('-' * 90)

for feat in FEATURES:
    dist_key = f'{feat}_dist_nm'
    vals = [r[dist_key] for r in catch_scores if r.get(dist_key) is not None]
    if not vals: continue
    n = len(vals)
    within = lambda t: sum(1 for v in vals if v <= t)
    print(f'{feat:<20s} {n:>4d} {np.mean(vals):>7.2f} {np.median(vals):>7.2f} {min(vals):>7.2f} '
          f'{within(1):>3d}/{n:<2d} {within(2):>3d}/{n:<2d} {within(3):>3d}/{n:<2d} {within(5):>3d}/{n:<2d}')

# =====================================================================
# 3. FEATURE CO-OCCURRENCE AT CATCHES
# =====================================================================
print(f'\n{"="*90}')
print(f'3. FEATURE CO-OCCURRENCE AT CATCH LOCATIONS (score > 0.3)')
print(f'{"="*90}')

# Single features
print(f'\n  Single features active at catches:')
for feat in FEATURES:
    n_active = sum(1 for r in catch_scores if r[feat] > 0.3)
    pct = n_active / len(catch_scores) * 100
    print(f'    {feat:<20s}: {n_active:>3d}/{len(catch_scores)} ({pct:.0f}%)')

# Pairs
print(f'\n  Feature pairs (both > 0.3) at catches vs control:')
print(f'  {"Pair":<42s} {"Catch":>8s} {"Control":>8s} {"Lift":>8s}')
print(f'  {"-"*70}')
pair_lifts = []
for f1, f2 in combinations(FEATURES, 2):
    c_both = sum(1 for r in catch_scores if r[f1] > 0.3 and r[f2] > 0.3)
    o_both = sum(1 for r in control_scores if r[f1] > 0.3 and r[f2] > 0.3)
    c_pct = c_both / len(catch_scores) * 100
    o_pct = o_both / len(control_scores) * 100 if control_scores else 0
    lift = c_pct - o_pct
    pair_lifts.append((f1, f2, c_pct, o_pct, lift))

pair_lifts.sort(key=lambda x: x[4], reverse=True)
for f1, f2, c_pct, o_pct, lift in pair_lifts:
    bar = '#' * int(abs(lift))
    print(f'  {f1+" + "+f2:<42s} {c_pct:>7.1f}% {o_pct:>7.1f}% {lift:>+7.1f}% {bar}')

# Active count distribution
print(f'\n  Features active (>0.3) per catch:')
for n in range(6):
    c_n = sum(1 for r in catch_scores if r['n_active'] == n)
    o_n = sum(1 for r in control_scores if r['n_active'] == n)
    c_pct = c_n / len(catch_scores) * 100
    o_pct = o_n / len(control_scores) * 100 if control_scores else 0
    bar_c = '#' * int(c_pct / 2)
    bar_o = '.' * int(o_pct / 2)
    print(f'    {n} active: catch {c_pct:>5.1f}% {bar_c}')
    print(f'             ctrl  {o_pct:>5.1f}% {bar_o}')

# =====================================================================
# 4. SENSITIVITY: WEIGHTED CONTRIBUTION AT CATCHES
# =====================================================================
print(f'\n{"="*90}')
print(f'4. WEIGHTED CONTRIBUTION — HOW MUCH EACH FEATURE MOVES THE SCORE')
print(f'{"="*90}')
print(f'  Each feature contributes score * weight to the total. This shows')
print(f'  the absolute and relative contribution at catch locations.')
print()
print(f'  {"Feature":<20s} {"Weight":>6s} {"MeanScore":>10s} {"Contrib":>8s} {"% of Total":>10s} {"If Zeroed":>10s}')
print(f'  {"-"*70}')

total_contrib = sum(FEATURE_WEIGHTS[f] * np.mean([r[f] for r in catch_scores]) for f in FEATURES)
for feat in FEATURES:
    w = FEATURE_WEIGHTS[feat]
    mean_s = np.mean([r[feat] for r in catch_scores])
    contrib = w * mean_s
    pct = contrib / total_contrib * 100 if total_contrib > 0 else 0
    # Impact if zeroed: how much would mean total score drop?
    mean_total = np.mean([r['total_score'] for r in catch_scores])
    drop = contrib / mean_total * 100 if mean_total > 0 else 0
    print(f'  {feat:<20s} {w:>6.2f} {mean_s:>10.3f} {contrib:>8.4f} {pct:>9.1f}% {drop:>9.1f}%')

print(f'\n  Total hidden feature contribution: {total_contrib:.4f} '
      f'({total_contrib / np.mean([r["total_score"] for r in catch_scores]) * 100:.1f}% of mean catch score)')

# =====================================================================
# 5. CORRELATION MATRIX — WHICH FEATURES MOVE TOGETHER
# =====================================================================
print(f'\n{"="*90}')
print(f'5. FEATURE CORRELATION AT CATCH LOCATIONS')
print(f'{"="*90}')
print(f'  Pearson correlation between feature scores at catch points.')
print(f'  High correlation = features fire together (potentially redundant).')
print(f'  Low correlation = independent signal (valuable diversity).')
print()

# Build correlation matrix
feat_arrays = {f: np.array([r[f] for r in catch_scores]) for f in FEATURES}
print(f'  {"":>20s}', end='')
for f in FEATURES:
    print(f' {f[:8]:>8s}', end='')
print()

for f1 in FEATURES:
    print(f'  {f1:<20s}', end='')
    for f2 in FEATURES:
        if f1 == f2:
            print(f' {"1.00":>8s}', end='')
        else:
            v1, v2 = feat_arrays[f1], feat_arrays[f2]
            if np.std(v1) > 0 and np.std(v2) > 0:
                corr = np.corrcoef(v1, v2)[0, 1]
                print(f' {corr:>8.2f}', end='')
            else:
                print(f' {"n/a":>8s}', end='')
    print()

# =====================================================================
# 6. HIGH-SCORE CATCHES: WHICH FEATURES DRIVE TOP CATCHES
# =====================================================================
print(f'\n{"="*90}')
print(f'6. FEATURE PROFILES OF TOP vs BOTTOM CATCHES')
print(f'{"="*90}')

sorted_catches = sorted(catch_scores, key=lambda r: r['total_score'], reverse=True)
n_top = max(1, len(sorted_catches) // 4)
n_bot = n_top
top = sorted_catches[:n_top]
bot = sorted_catches[-n_bot:]

print(f'  Top {n_top} catches (mean score {np.mean([r["total_score"] for r in top]):.3f}) vs '
      f'Bottom {n_bot} (mean score {np.mean([r["total_score"] for r in bot]):.3f})')
print()
print(f'  {"Feature":<20s} {"Top Mean":>10s} {"Bot Mean":>10s} {"Delta":>8s} {"Signal":>8s}')
print(f'  {"-"*60}')

for feat in FEATURES:
    t_mean = np.mean([r[feat] for r in top])
    b_mean = np.mean([r[feat] for r in bot])
    delta = t_mean - b_mean
    signal = '+++' if delta > 0.15 else '++' if delta > 0.08 else '+' if delta > 0.03 else '=' if delta > -0.03 else '-'
    print(f'  {feat:<20s} {t_mean:>10.3f} {b_mean:>10.3f} {delta:>+8.3f} {signal:>8s}')

# =====================================================================
# 7. PER-CATCH DETAIL TABLE
# =====================================================================
print(f'\n{"="*90}')
print(f'7. PER-CATCH DETAIL (sorted by total score)')
print(f'{"="*90}')
print(f'  {"Date":<12s} {"Tag":<10s} {"Score":>5s} {"Corr":>5s} {"Conv":>5s} {"Shear":>5s} {"Upwl":>5s} {"ChlC":>5s} {"#Act":>4s}', end='')
# Proximity columns if available
has_prox = any(r.get('front_corridor_dist_nm') is not None for r in catch_scores)
if has_prox:
    print(f' | {"dCorr":>5s} {"dConv":>5s} {"dShear":>5s} {"dUpwl":>5s} {"dChlC":>5s}', end='')
print()
print(f'  {"-"*85}')

for r in sorted_catches:
    print(f'  {r["date"]:<12s} {r.get("tag",""):<10s} {r["total_score"]:>5.2f} '
          f'{r["front_corridor"]:>5.2f} {r["convergence"]:>5.2f} {r["current_shear"]:>5.2f} '
          f'{r["upwelling_edge"]:>5.2f} {r["chl_curvature"]:>5.2f} {r["n_active"]:>4d}', end='')
    if has_prox:
        for feat in FEATURES:
            dk = f'{feat}_dist_nm'
            v = r.get(dk)
            print(f' {v:>6.1f}' if v is not None else f' {"n/a":>6s}', end='')
    print()

# Save results
out = {'catch_scores': catch_scores, 'control_scores': control_scores}
with open('data/hidden_feature_analysis.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nRaw data saved to data/hidden_feature_analysis.json')
