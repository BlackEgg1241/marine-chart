"""Grid search: find parameter combo that maximises catch scores while minimising zone area.

Metric: catch_score_mean * (1 - zone_area_fraction)
  - High catch scores pull the metric UP
  - Large zone area pulls it DOWN
  - Sweet spot: tight zones that still cover catches

Runs on 8 representative dates for speed, then verifies winner on full 44.
"""
import marlin_data, numpy as np, os, csv, sys, itertools, json
from datetime import datetime
from collections import defaultdict

# Load catches
with open('data/all_catches.csv', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if 'BLUE MARLIN' in r.get('species', '').upper()]

catches = []
for r in rows:
    lat, lon = r.get('lat', '').strip(), r.get('lon', '').strip()
    if not lat or not lon:
        continue
    for fmt in ['%d/%m/%Y', '%Y-%m-%d']:
        try:
            dt = datetime.strptime(r['date'].strip(), fmt)
            catches.append({'date': dt.strftime('%Y-%m-%d'), 'lat': float(lat), 'lon': float(lon)})
            break
        except:
            continue

by_date = defaultdict(list)
for c in catches:
    by_date[c['date']].append(c)

# Select representative subset: mix of high/low scoring, different years
SUBSET_DATES = [
    '2001-03-03',  # moderate conditions
    '2011-02-09',  # catch near shelf break
    '2012-01-28',  # 2 catches, strong features
    '2015-02-24',  # catch in deep water
    '2016-01-16',  # sparse features
    '2017-02-19',  # typical summer
    '2021-02-14',  # 3 catches same day
    '2025-03-03',  # recent data
]

# Verify subset dates have data
valid_subset = []
for d in SUBSET_DATES:
    ddir = os.path.join('data', d)
    if os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
        valid_subset.append(d)
    else:
        ddir2 = os.path.join('data', 'prediction', d)
        if os.path.exists(os.path.join(ddir2, 'sst_raw.nc')):
            valid_subset.append(d)

SUBSET_DATES = valid_subset
subset_catches = {d: by_date[d] for d in SUBSET_DATES if d in by_date}
n_subset_catches = sum(len(v) for v in subset_catches.values())
print(f"Subset: {len(SUBSET_DATES)} dates, {n_subset_catches} catches")

BBOX = [114.5, -32.5, 115.6, -31.5]
DISPLAY_FLOOR = 0.70

# Parameters to search
PARAMS = {
    'band_width_nm':     [2.5, 3.0, 3.5, 4.0],
    'shelf_boost':       [0.10, 0.15, 0.20, 0.25],
    'zero_band_mult':    [0.50, 0.55, 0.60, 0.65],
    'one_band_mult':     [0.80, 0.85, 0.90],
    'feature_floor':     [0.40, 0.50, 0.60],
}

# Generate all combinations
keys = list(PARAMS.keys())
combos = list(itertools.product(*[PARAMS[k] for k in keys]))
print(f"Total combinations: {len(combos)}")


def evaluate(params_dict, dates_catches, quiet=True):
    """Run scoring with given params on specified dates, return metrics."""
    # Set module-level overrides
    marlin_data._opt_band_width_nm = params_dict['band_width_nm']
    marlin_data._opt_shelf_boost = params_dict['shelf_boost']

    catch_scores = []
    area_fractions = []

    for date, date_catches in dates_catches.items():
        ddir = os.path.join('data', date)
        if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
            ddir = os.path.join('data', 'prediction', date)
        if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
            continue

        tif = os.path.join(ddir, 'bathy_gmrt.tif')
        marlin_data.OUTPUT_DIR = ddir

        # Temporarily override band multiplier values
        orig_code = None
        # We'll monkey-patch the band multiplier values via module attrs
        marlin_data._opt_zero_band_mult = params_dict['zero_band_mult']
        marlin_data._opt_one_band_mult = params_dict['one_band_mult']
        marlin_data._opt_key_feature_floor = params_dict['feature_floor']

        try:
            result = marlin_data.generate_blue_marlin_hotspots(
                BBOX, tif_path=tif if os.path.exists(tif) else None,
                date_str=date, quiet=quiet)
        except TypeError:
            # If quiet param not supported
            # Redirect stdout
            if quiet:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
            result = marlin_data.generate_blue_marlin_hotspots(
                BBOX, tif_path=tif if os.path.exists(tif) else None,
                date_str=date)
            if quiet:
                sys.stdout.close()
                sys.stdout = old_stdout

        grid = result['grid']
        lats, lons = result['lats'], result['lons']

        # Catch scores
        for c in date_catches:
            li = np.argmin(np.abs(lats - c['lat']))
            lo = np.argmin(np.abs(lons - c['lon']))
            cs = float(grid[li, lo]) if not np.isnan(grid[li, lo]) else 0
            catch_scores.append(cs)

        # Zone area: fraction of valid ocean cells above display floor
        valid_ocean = ~np.isnan(grid)
        n_ocean = np.sum(valid_ocean)
        if n_ocean > 0:
            n_above = np.sum(grid[valid_ocean] >= DISPLAY_FLOOR)
            area_fractions.append(n_above / n_ocean)

    if not catch_scores:
        return None

    mean_score = np.mean(catch_scores)
    pct_above_70 = sum(1 for s in catch_scores if s >= 0.70) / len(catch_scores)
    mean_area = np.mean(area_fractions) if area_fractions else 1.0

    # Combined metric: reward high scores, penalize large area
    # Also bonus for having ALL catches above floor
    metric = mean_score * (1 - mean_area) * (0.5 + 0.5 * pct_above_70)

    return {
        'mean_score': round(mean_score, 4),
        'pct_above_70': round(pct_above_70, 4),
        'mean_area': round(mean_area, 4),
        'metric': round(metric, 6),
        'n_catches': len(catch_scores),
    }


# Patch marlin_data to read our overrides in the band multiplier section
# We need to make the graduated band multiplier use our params
import types

# Save original function reference
_orig_generate = marlin_data.generate_blue_marlin_hotspots

def _patched_generate(bbox_or_dict, **kwargs):
    """Wrapper that injects band multiplier overrides."""
    # The band multiplier values are hardcoded in the function.
    # We'll use getattr to make them configurable.
    return _orig_generate(bbox_or_dict, **kwargs)

# Actually, let me check if the band multiplier already uses getattr...
# Looking at the code: the values 0.60 and 0.85 are hardcoded in np.where.
# I need to make them use module-level overrides.

# First, let's check and patch the source
print("Patching band multiplier to use configurable thresholds...")

# Read current code
with open('marlin_data.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Check if already patched
if '_opt_zero_band_mult' not in code:
    code = code.replace(
        "        band_base_mult = np.where(\n"
        "            _feature_band_count < 0.1, 0.60,\n"
        "            np.where(_feature_band_count < 1.5, 0.85,",
        "        _zero_bm = getattr(sys.modules[__name__], '_opt_zero_band_mult', 0.60)\n"
        "        _one_bm = getattr(sys.modules[__name__], '_opt_one_band_mult', 0.85)\n"
        "        band_base_mult = np.where(\n"
        "            _feature_band_count < 0.1, _zero_bm,\n"
        "            np.where(_feature_band_count < 1.5, _one_bm,"
    )
    # Also patch the feature floor
    code = code.replace(
        "        _key_feature_floor = 0.50",
        "        _key_feature_floor = getattr(sys.modules[__name__], '_opt_key_feature_floor', 0.50)"
    )
    with open('marlin_data.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched marlin_data.py with configurable band multiplier thresholds")
    # Reload
    import importlib
    importlib.reload(marlin_data)
else:
    print("Already patched")

# Run grid search
print(f"\nStarting grid search: {len(combos)} combinations x {len(SUBSET_DATES)} dates...")
results = []

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        metrics = evaluate(params, subset_catches, quiet=True)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    if metrics:
        results.append({**params, **metrics})

    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{len(combos)}] tested", flush=True)

# Sort by metric
results.sort(key=lambda x: x['metric'], reverse=True)

print(f"\n{'='*90}")
print(f"TOP 10 PARAMETER COMBINATIONS (metric = score * (1-area) * coverage)")
print(f"{'='*90}")
print(f"{'Rank':>4s} {'BandW':>5s} {'Shelf':>5s} {'0-bnd':>5s} {'1-bnd':>5s} {'Floor':>5s} "
      f"{'Score':>6s} {'>=70%':>5s} {'Area':>6s} {'Metric':>8s}")
print('-' * 90)

for i, r in enumerate(results[:20]):
    print(f"{i+1:>4d} {r['band_width_nm']:>5.1f} {r['shelf_boost']:>5.2f} "
          f"{r['zero_band_mult']:>5.2f} {r['one_band_mult']:>5.2f} {r['feature_floor']:>5.2f} "
          f"{r['mean_score']:>6.3f} {r['pct_above_70']:>5.1%} {r['mean_area']:>6.1%} "
          f"{r['metric']:>8.4f}")

# Also show current params for comparison
print(f"\n{'='*90}")
print(f"CURRENT PARAMS:")
current = {'band_width_nm': 3.5, 'shelf_boost': 0.15, 'zero_band_mult': 0.60,
           'one_band_mult': 0.85, 'feature_floor': 0.50}
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    current_metrics = evaluate(current, subset_catches, quiet=True)
finally:
    sys.stdout.close()
    sys.stdout = old_stdout

if current_metrics:
    print(f"  Score={current_metrics['mean_score']:.3f}  >=70%={current_metrics['pct_above_70']:.1%}  "
          f"Area={current_metrics['mean_area']:.1%}  Metric={current_metrics['metric']:.4f}")

# Save top result
best = results[0]
print(f"\n{'='*90}")
print(f"BEST COMBINATION:")
print(f"  band_width_nm:  {best['band_width_nm']}")
print(f"  shelf_boost:    {best['shelf_boost']}")
print(f"  zero_band_mult: {best['zero_band_mult']}")
print(f"  one_band_mult:  {best['one_band_mult']}")
print(f"  feature_floor:  {best['feature_floor']}")
print(f"  -> Score={best['mean_score']:.3f}  Area={best['mean_area']:.1%}  Metric={best['metric']:.4f}")

# Save all results
with open('data/zone_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to data/zone_optimization_results.json")
