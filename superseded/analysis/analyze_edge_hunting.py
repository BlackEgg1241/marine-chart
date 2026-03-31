"""Edge-hunting analysis: do blue marlin sit ON features or on the EDGE of features?

For each scoring feature, compares the local score at the catch location
to the peak score within ~3nm. A low ratio (local/peak) means the catch
is on the boundary, not inside the feature — classic edge-hunting behaviour.
"""
import marlin_data, numpy as np, os, csv, json
from datetime import datetime
from collections import defaultdict

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

# All features to analyse
ALL_FEATURES = [
    'sst', 'sst_front', 'front_corridor', 'chl', 'chl_curvature',
    'ssh', 'mld', 'current', 'convergence', 'current_shear',
    'upwelling_edge', 'shelf_break', 'depth',
]

# Collect edge data
edge_data = {f: [] for f in ALL_FEATURES}
SEARCH_RADIUS = 3  # grid cells (~3nm)

print(f"Analysing edge-hunting across {len(ALL_FEATURES)} features, {len(catches)} catches...")

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
    ny, nx = grid.shape

    for c in by_date[date]:
        li = np.argmin(np.abs(lats - c['lat']))
        lo = np.argmin(np.abs(lons - c['lon']))

        for feat in ALL_FEATURES:
            if feat not in ss or not isinstance(ss[feat], np.ndarray):
                continue
            fgrid = ss[feat]
            local = float(fgrid[li, lo]) if not np.isnan(fgrid[li, lo]) else 0

            # Peak within search radius
            r1, r2 = max(0, li - SEARCH_RADIUS), min(ny, li + SEARCH_RADIUS + 1)
            c1, c2 = max(0, lo - SEARCH_RADIUS), min(nx, lo + SEARCH_RADIUS + 1)
            nearby = fgrid[r1:r2, c1:c2]
            valid_nearby = nearby[~np.isnan(nearby)]
            if len(valid_nearby) == 0:
                continue

            peak = float(np.max(valid_nearby))
            mean_nearby = float(np.mean(valid_nearby))
            # Also compute the gradient magnitude at catch location
            grad_y, grad_x = np.gradient(np.where(np.isnan(fgrid), 0, fgrid))
            grad_mag = np.sqrt(grad_x[li, lo]**2 + grad_y[li, lo]**2)

            ratio = local / peak if peak > 0.05 else None  # avoid div-by-tiny

            edge_data[feat].append({
                'date': date, 'tag': c['tag'],
                'local': local, 'peak': peak, 'mean_nearby': mean_nearby,
                'ratio': ratio, 'gradient': float(grad_mag),
            })

    print(f'  {date} done', flush=True)

# =====================================================================
# RESULTS
# =====================================================================
print(f'\n{"="*100}')
print(f'EDGE-HUNTING ANALYSIS: LOCAL SCORE vs PEAK WITHIN 3nm')
print(f'{"="*100}')
print(f'  Ratio < 0.3 = deep edge (catch far from feature peak)')
print(f'  Ratio 0.3-0.7 = boundary zone')
print(f'  Ratio > 0.7 = inside the feature')
print()

# Summary table
print(f'{"Feature":<18s} {"N":>4s} {"LocalMean":>9s} {"PeakMean":>9s} {"Ratio":>6s} '
      f'{"<30%":>5s} {"<50%":>5s} {"<70%":>5s} {">70%":>5s} {"GradMean":>8s} {"Pattern":>12s}')
print('-' * 100)

feature_patterns = {}
for feat in ALL_FEATURES:
    data = edge_data[feat]
    if not data: continue
    # Filter to catches where feature is active nearby (peak > 0.3)
    active = [d for d in data if d['peak'] > 0.3]
    if not active: continue

    with_ratio = [d for d in active if d['ratio'] is not None]
    if not with_ratio: continue

    n = len(with_ratio)
    local_mean = np.mean([d['local'] for d in with_ratio])
    peak_mean = np.mean([d['peak'] for d in with_ratio])
    ratio_mean = np.mean([d['ratio'] for d in with_ratio])
    grad_mean = np.mean([d['gradient'] for d in with_ratio])

    pct_deep = sum(1 for d in with_ratio if d['ratio'] < 0.3) / n * 100
    pct_edge = sum(1 for d in with_ratio if d['ratio'] < 0.5) / n * 100
    pct_mod = sum(1 for d in with_ratio if d['ratio'] < 0.7) / n * 100
    pct_inside = sum(1 for d in with_ratio if d['ratio'] >= 0.7) / n * 100

    # Classify pattern
    if pct_mod > 70:
        pattern = "EDGE HUNTER"
    elif pct_mod > 50:
        pattern = "EDGE PREF"
    elif pct_inside > 70:
        pattern = "INSIDE"
    else:
        pattern = "MIXED"

    feature_patterns[feat] = pattern

    print(f'{feat:<18s} {n:>4d} {local_mean:>9.3f} {peak_mean:>9.3f} {ratio_mean:>6.2f} '
          f'{pct_deep:>4.0f}% {pct_edge:>4.0f}% {pct_mod:>4.0f}% {pct_inside:>4.0f}% '
          f'{grad_mean:>8.4f} {pattern:>12s}')

# =====================================================================
# DETAILED BREAKDOWN PER FEATURE
# =====================================================================
print(f'\n{"="*100}')
print(f'DETAILED RATIO DISTRIBUTIONS')
print(f'{"="*100}')

for feat in ALL_FEATURES:
    data = edge_data[feat]
    if not data: continue
    active = [d for d in data if d['peak'] > 0.3 and d['ratio'] is not None]
    if len(active) < 5: continue

    ratios = [d['ratio'] for d in active]
    print(f'\n  {feat.upper()} (n={len(active)}, pattern={feature_patterns.get(feat, "?")})')
    print(f'    Ratio: mean={np.mean(ratios):.2f}  med={np.median(ratios):.2f}  '
          f'min={min(ratios):.2f}  max={max(ratios):.2f}  std={np.std(ratios):.2f}')

    # Histogram
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f'    {"Ratio Range":<14s} {"Count":>5s} {"Pct":>5s} {"Bar":<30s}')
    for lo, hi in bins:
        count = sum(1 for r in ratios if lo <= r < hi)
        pct = count / len(ratios) * 100
        bar = '#' * int(pct / 2)
        label = 'EDGE' if hi <= 0.5 else 'BOUNDARY' if hi <= 0.7 else 'INSIDE'
        print(f'    {lo:.1f}-{hi:.1f}  {label:<8s} {count:>5d} {pct:>4.0f}% {bar}')

# =====================================================================
# GRADIENT AT CATCH: DO CATCHES SIT ON STEEP GRADIENTS?
# =====================================================================
print(f'\n{"="*100}')
print(f'GRADIENT STRENGTH AT CATCH LOCATIONS')
print(f'{"="*100}')
print(f'  High gradient at catch = catch sits on a sharp feature boundary')
print(f'  (Gradient of score field at catch location vs ocean average)')
print()
print(f'{"Feature":<18s} {"CatchGrad":>9s} {"OceanGrad":>9s} {"Lift":>6s} {"Interpretation":>20s}')
print('-' * 70)

for feat in ALL_FEATURES:
    data = edge_data[feat]
    if not data: continue
    active = [d for d in data if d['peak'] > 0.3]
    if not active: continue

    catch_grad = np.mean([d['gradient'] for d in active])

    # Compare to random control gradient (approximate from all data)
    all_grads = [d['gradient'] for d in data]
    ocean_grad = np.mean(all_grads)

    lift = (catch_grad / ocean_grad - 1) * 100 if ocean_grad > 0 else 0
    interp = "BOUNDARY SEEKER" if lift > 30 else "MODERATE EDGE" if lift > 10 else "NO PREFERENCE" if lift > -10 else "SMOOTH ZONE"

    print(f'{feat:<18s} {catch_grad:>9.4f} {ocean_grad:>9.4f} {lift:>+5.0f}% {interp:>20s}')

# =====================================================================
# CROSS-FEATURE EDGE PROFILE
# =====================================================================
print(f'\n{"="*100}')
print(f'CROSS-FEATURE EDGE PROFILE PER CATCH')
print(f'{"="*100}')
print(f'  How many features show edge-hunting (ratio<0.5) at each catch?')
print()

# Build per-catch edge profile
catch_edge_counts = defaultdict(lambda: {'n_edge': 0, 'n_inside': 0, 'n_active': 0, 'features': {}})
for feat in ALL_FEATURES:
    for d in edge_data[feat]:
        if d['peak'] <= 0.3 or d['ratio'] is None: continue
        key = (d['date'], d['tag'])
        catch_edge_counts[key]['n_active'] += 1
        catch_edge_counts[key]['features'][feat] = d['ratio']
        if d['ratio'] < 0.5:
            catch_edge_counts[key]['n_edge'] += 1
        elif d['ratio'] >= 0.7:
            catch_edge_counts[key]['n_inside'] += 1

# Distribution
print(f'  {"Edge features":>14s} {"Count":>6s} {"Pct":>5s}')
for n in range(max(c['n_edge'] for c in catch_edge_counts.values()) + 1):
    count = sum(1 for c in catch_edge_counts.values() if c['n_edge'] == n)
    pct = count / len(catch_edge_counts) * 100
    print(f'  {n:>14d} {count:>6d} {pct:>4.0f}%')

mean_edge = np.mean([c['n_edge'] for c in catch_edge_counts.values()])
mean_inside = np.mean([c['n_inside'] for c in catch_edge_counts.values()])
mean_active = np.mean([c['n_active'] for c in catch_edge_counts.values()])
print(f'\n  Average per catch: {mean_active:.1f} features active, '
      f'{mean_edge:.1f} on edge, {mean_inside:.1f} inside')
print(f'  Edge ratio: {mean_edge/mean_active*100:.0f}% of active features show edge-hunting')

# =====================================================================
# IMPLICATIONS FOR SCORING
# =====================================================================
print(f'\n{"="*100}')
print(f'IMPLICATIONS FOR SCORING MODEL')
print(f'{"="*100}')
print()
for feat in ALL_FEATURES:
    pat = feature_patterns.get(feat)
    if pat == "EDGE HUNTER":
        print(f'  {feat:<18s}: EDGE HUNTER -> Consider scoring the GRADIENT of this feature')
        print(f'                      (reward the boundary, not the peak value)')
    elif pat == "EDGE PREF":
        print(f'  {feat:<18s}: EDGE PREF  -> Moderate edge effect. Current scoring OK but')
        print(f'                      could benefit from gradient bonus')
    elif pat == "INSIDE":
        print(f'  {feat:<18s}: INSIDE     -> Catches sit inside this feature. Current')
        print(f'                      value-based scoring is correct')
    elif pat == "MIXED":
        print(f'  {feat:<18s}: MIXED      -> No clear edge/inside pattern. Current scoring OK')

# Save raw data
out = {feat: edge_data[feat] for feat in ALL_FEATURES if edge_data[feat]}
with open('data/edge_hunting_analysis.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nRaw data saved to data/edge_hunting_analysis.json')
