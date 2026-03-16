"""Composite predictor analysis: CHL + spatial heterogeneity for marlin season prediction."""
import json, csv, os
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
from scipy import stats

# ── Load backtest data ──────────────────────────────────────────────────────
bt = json.loads(Path('data/backtest/backtest_results.json').read_text())

# ── Load all blue marlin catches (same approach as analyze_preseason.py) ────
catch_dates = []
geo = json.loads(open('data/marlin_catches.geojson').read())
for f in geo['features']:
    p = f['properties']
    if p['species'] == 'BLUE MARLIN':
        catch_dates.append(p['date'])

rows = list(csv.DictReader(open('data/all_catches.csv')))
for r in rows:
    if r['species'] == 'BLUE MARLIN':
        try:
            dt = datetime.strptime(r['date'], '%d/%m/%Y')
            d = dt.strftime('%Y-%m-%d')
            if d not in catch_dates:
                catch_dates.append(d)
        except:
            pass

# ── Per-season stats (Jan-Apr) ──────────────────────────────────────────────
season_catches = defaultdict(int)
season_first_catch = {}  # year -> earliest date string

for d in catch_dates:
    dt = datetime.strptime(d, '%Y-%m-%d')
    if dt.month <= 4:
        year = dt.year
        season_catches[year] += 1
        if year not in season_first_catch or d < season_first_catch[year]:
            season_first_catch[year] = d

# Compute day-of-year for first catch (days from Jan 1)
season_first_doy = {}
for year, dstr in season_first_catch.items():
    dt = datetime.strptime(dstr, '%Y-%m-%d')
    doy = (dt - datetime(year, 1, 1)).days + 1
    season_first_doy[year] = doy

# ── Build pre-season Oct-Nov metrics from backtest ──────────────────────────
# Pre-season Oct-Nov of year Y predicts season Y+1
preseason = defaultdict(lambda: defaultdict(list))

for e in bt['dates']:
    dt = datetime.strptime(e['date'], '%Y-%m-%d')
    if dt.month in [10, 11]:
        target_year = dt.year + 1  # predicts next year's season
        for key in ['s_chl', 'zone_max', 'zone_mean', 's_sst', 's_ssh', 's_mld',
                     's_sst_front', 's_current', 's_convergence']:
            if e.get(key) is not None:
                preseason[target_year][key].append(e[key])

# Compute averages and derived metrics
yearly_metrics = {}
for year in sorted(preseason.keys()):
    m = {}
    for key in preseason[year]:
        m[key] = np.mean(preseason[year][key])
    # zone_spread = zone_max - zone_mean (spatial heterogeneity proxy)
    if 'zone_max' in m and 'zone_mean' in m:
        m['zone_spread'] = m['zone_max'] - m['zone_mean']
    yearly_metrics[year] = m

# Years that have both CHL and zone data
valid_years = sorted(y for y in yearly_metrics if 's_chl' in yearly_metrics[y]
                     and 'zone_spread' in yearly_metrics[y])

# ── Helper: normalize to ranks (0-1) ───────────────────────────────────────
def rank_normalize(values):
    """Return rank-normalized values (0 to 1). Handles ties with average rank."""
    arr = np.array(values, dtype=float)
    ranks = stats.rankdata(arr)
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(arr)

# ── Helper: leave-one-out classification ────────────────────────────────────
def loo_classify(x_vals, y_binary):
    """Leave-one-out: predict class based on whether left-out x is above/below
    the mean of the remaining training samples' x for positive class.
    Returns accuracy."""
    x = np.array(x_vals)
    y = np.array(y_binary)
    n = len(x)
    correct = 0
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        # Threshold = median of training x for positive class
        pos_vals = x[train_mask & (y == 1)]
        neg_vals = x[train_mask & (y == 0)]
        if len(pos_vals) == 0 or len(neg_vals) == 0:
            continue
        threshold = (np.mean(pos_vals) + np.mean(neg_vals)) / 2
        # Determine direction: are positives higher or lower?
        pred_positive = x[i] >= threshold if np.mean(pos_vals) > np.mean(neg_vals) else x[i] <= threshold
        if pred_positive == (y[i] == 1):
            correct += 1
    return correct / n if n > 0 else 0

# ══════════════════════════════════════════════════════════════════════════════
# YEAR-BY-YEAR TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 110)
print("YEAR-BY-YEAR TABLE: Pre-season (Oct-Nov) metrics vs season outcomes")
print("=" * 110)
print(f"\n{'Year':>4s} {'Catches':>7s} {'1st Catch':>12s} {'DOY':>5s} | "
      f"{'CHL':>7s} {'ZoneMax':>7s} {'ZoneMn':>7s} {'Spread':>7s}")
print("-" * 110)

for year in valid_years:
    m = yearly_metrics[year]
    c = season_catches.get(year, 0)
    first = season_first_catch.get(year, 'N/A')
    doy = season_first_doy.get(year, '')
    doy_str = f"{doy:5d}" if doy else "    -"
    # CHL is INVERTED: lower score = higher actual chlorophyll = better
    print(f"{year:>4d} {c:>7d} {first:>12s} {doy_str} | "
          f"{m.get('s_chl', float('nan')):7.3f} "
          f"{m.get('zone_max', float('nan')):7.1f} "
          f"{m.get('zone_mean', float('nan')):7.1f} "
          f"{m.get('zone_spread', float('nan')):7.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIRST CAPTURE TIMING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("FIRST CAPTURE TIMING ANALYSIS")
print(f"{'='*110}")

# Years with catches (have first-catch DOY) AND have pre-season data
timing_years = [y for y in valid_years if y in season_first_doy]
count_years = valid_years  # all valid years (include zero-catch for count analysis)

print(f"\nYears with catches and pre-season data: {len(timing_years)}")
print(f"Years with pre-season data (all, incl zero-catch): {len(count_years)}")

# ── Correlations with first-catch DOY ───────────────────────────────────────
print(f"\n--- Correlations with first-catch day-of-year ---")
print(f"  (Negative rho = earlier first catch = better)")
print(f"\n{'Metric':>15s} | {'Spearman rho':>12s} | {'p-value':>8s} | {'n':>3s}")
print("-" * 55)

for metric_name in ['s_chl', 'zone_spread', 'zone_max', 'zone_mean']:
    x = [yearly_metrics[y][metric_name] for y in timing_years if metric_name in yearly_metrics[y]]
    y_doy = [season_first_doy[y] for y in timing_years if metric_name in yearly_metrics[y]]
    if len(x) < 5:
        continue
    rho, p = stats.spearmanr(x, y_doy)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.1 else "")))
    print(f"{metric_name:>15s} | rho={rho:+.3f}    | {p:8.4f} {sig:3s} | {len(x):3d}")

# ── Correlations with total catches ─────────────────────────────────────────
print(f"\n--- Correlations with total catches (Jan-Apr) ---")
print(f"  Note: CHL sub-score is INVERTED (lower = more chlorophyll = better)")
print(f"\n{'Metric':>15s} | {'Spearman rho':>12s} | {'p-value':>8s} | {'n':>3s}")
print("-" * 55)

for metric_name in ['s_chl', 'zone_spread', 'zone_max', 'zone_mean']:
    x = [yearly_metrics[y][metric_name] for y in count_years if metric_name in yearly_metrics[y]]
    y_c = [season_catches.get(y, 0) for y in count_years if metric_name in yearly_metrics[y]]
    if len(x) < 5:
        continue
    rho, p = stats.spearmanr(x, y_c)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.1 else "")))
    print(f"{metric_name:>15s} | rho={rho:+.3f}    | {p:8.4f} {sig:3s} | {len(x):3d}")

# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE PREDICTOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("COMPOSITE PREDICTOR TESTS")
print(f"{'='*110}")

# Build predictor arrays for valid years
chl_vals = np.array([yearly_metrics[y]['s_chl'] for y in count_years])
spread_vals = np.array([yearly_metrics[y]['zone_spread'] for y in count_years])
zmax_vals = np.array([yearly_metrics[y]['zone_max'] for y in count_years])
catches_arr = np.array([season_catches.get(y, 0) for y in count_years])

# CHL is INVERTED: lower score = higher chlorophyll = better for marlin
# So we NEGATE chl for composites (higher composite = better conditions)
chl_inverted = -chl_vals  # negate so higher = more chlorophyll

# Rank-normalize each component
chl_ranks = rank_normalize(chl_inverted)
spread_ranks = rank_normalize(spread_vals)
zmax_ranks = rank_normalize(zmax_vals)

# Composite predictors
predictors = {
    'CHL alone (inverted)': chl_inverted,
    'zone_spread alone': spread_vals,
    'zone_max alone': zmax_vals,
    'CHL + spread (rank avg)': (chl_ranks + spread_ranks) / 2,
    'CHL + spread + zmax (rank avg)': (chl_ranks + spread_ranks + zmax_ranks) / 3,
    'CHL + spread (2:1 weighted)': (2 * chl_ranks + spread_ranks) / 3,
}

# Binary targets
good_season = (catches_arr >= 3).astype(int)
# For timing: only years with catches
timing_mask = np.array([y in season_first_doy for y in count_years])
doy_arr = np.array([season_first_doy.get(y, 999) for y in count_years])
median_doy = np.median(doy_arr[timing_mask])
# "Early start" = first catch before Feb 15 (DOY 46)
feb15_doy = (date(2000, 2, 15) - date(2000, 1, 1)).days + 1  # = 46
early_start = (doy_arr < feb15_doy).astype(int)

print(f"\nGood season threshold: >= 3 catches ({good_season.sum()} of {len(good_season)} years)")
print(f"Early start threshold: first catch before Feb 15 (DOY {feb15_doy})")
print(f"Median first-catch DOY: {median_doy:.0f}")

print(f"\n{'Predictor':>35s} | {'rho(catches)':>12s} {'p':>7s} | "
      f"{'rho(DOY)':>10s} {'p':>7s} | "
      f"{'LOO good':>8s} | {'LOO early':>9s}")
print("-" * 110)

for name, pred in predictors.items():
    # Correlation with total catches
    rho_c, p_c = stats.spearmanr(pred, catches_arr)
    sig_c = "***" if p_c < 0.001 else ("**" if p_c < 0.01 else ("*" if p_c < 0.05 else ("~" if p_c < 0.1 else "")))

    # Correlation with first-catch DOY (only years with catches)
    pred_timing = pred[timing_mask]
    doy_timing = doy_arr[timing_mask]
    if len(pred_timing) >= 5:
        rho_d, p_d = stats.spearmanr(pred_timing, doy_timing)
        sig_d = "***" if p_d < 0.001 else ("**" if p_d < 0.01 else ("*" if p_d < 0.05 else ("~" if p_d < 0.1 else "")))
        doy_str = f"rho={rho_d:+.3f} {p_d:6.4f}{sig_d:3s}"
    else:
        doy_str = "      N/A        "

    # LOO classification: good season (>= 3 catches)
    loo_good = loo_classify(pred, good_season)

    # LOO classification: early start (only years with catches)
    if timing_mask.sum() >= 5:
        loo_early = loo_classify(pred_timing, early_start[timing_mask])
    else:
        loo_early = float('nan')

    print(f"{name:>35s} | rho={rho_c:+.3f} {p_c:6.4f}{sig_c:3s} | "
          f"{doy_str} | "
          f"{loo_good:7.1%} | "
          f"{loo_early:8.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# BINARY ANALYSIS: Early vs Late seasons
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("BINARY ANALYSIS: Early seasons (first catch before median) vs Late seasons")
print(f"{'='*110}")

early_years = [y for y in timing_years if season_first_doy[y] < median_doy]
late_years = [y for y in timing_years if season_first_doy[y] >= median_doy]

print(f"\nEarly seasons ({len(early_years)}): {early_years}")
print(f"  First catch DOYs: {[season_first_doy[y] for y in early_years]}")
print(f"Late seasons  ({len(late_years)}): {late_years}")
print(f"  First catch DOYs: {[season_first_doy[y] for y in late_years]}")

print(f"\n{'Metric':>15s} | {'Early mean':>10s} | {'Late mean':>10s} | {'Diff':>8s} | {'Mann-W p':>8s}")
print("-" * 75)

for metric_name in ['s_chl', 'zone_spread', 'zone_max', 'zone_mean']:
    early_vals = [yearly_metrics[y][metric_name] for y in early_years if metric_name in yearly_metrics[y]]
    late_vals = [yearly_metrics[y][metric_name] for y in late_years if metric_name in yearly_metrics[y]]
    if len(early_vals) < 3 or len(late_vals) < 3:
        continue
    em = np.mean(early_vals)
    lm = np.mean(late_vals)
    u_stat, u_p = stats.mannwhitneyu(early_vals, late_vals, alternative='two-sided')
    sig = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else ("~" if u_p < 0.1 else "")))
    print(f"{metric_name:>15s} | {em:10.4f} | {lm:10.4f} | {em-lm:+8.4f} | {u_p:8.4f} {sig}")

# ══════════════════════════════════════════════════════════════════════════════
# JANUARY vs FEBRUARY+ first catch prediction
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("JANUARY vs FEBRUARY+ FIRST CATCH PREDICTION")
print(f"{'='*110}")

jan_years = [y for y in timing_years if datetime.strptime(season_first_catch[y], '%Y-%m-%d').month == 1]
feb_plus_years = [y for y in timing_years if datetime.strptime(season_first_catch[y], '%Y-%m-%d').month >= 2]

print(f"\nJanuary first catch ({len(jan_years)}): {jan_years}")
print(f"  Dates: {[season_first_catch[y] for y in jan_years]}")
print(f"February+ first catch ({len(feb_plus_years)}): {feb_plus_years}")

print(f"\n{'Metric':>15s} | {'Jan mean':>10s} | {'Feb+ mean':>10s} | {'Diff':>8s} | {'Mann-W p':>8s}")
print("-" * 75)

for metric_name in ['s_chl', 'zone_spread', 'zone_max', 'zone_mean']:
    jan_vals = [yearly_metrics[y][metric_name] for y in jan_years if metric_name in yearly_metrics[y]]
    feb_vals = [yearly_metrics[y][metric_name] for y in feb_plus_years if metric_name in yearly_metrics[y]]
    if len(jan_vals) < 2 or len(feb_vals) < 3:
        continue
    jm = np.mean(jan_vals)
    fm = np.mean(feb_vals)
    u_stat, u_p = stats.mannwhitneyu(jan_vals, feb_vals, alternative='two-sided')
    sig = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else ("~" if u_p < 0.1 else "")))
    print(f"{metric_name:>15s} | {jm:10.4f} | {fm:10.4f} | {jm-fm:+8.4f} | {u_p:8.4f} {sig}")

# ══════════════════════════════════════════════════════════════════════════════
# PCA-BASED COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("PCA-BASED COMPOSITE (CHL inverted + zone_spread)")
print(f"{'='*110}")

# Standardize
chl_std = (chl_inverted - np.mean(chl_inverted)) / np.std(chl_inverted)
spread_std = (spread_vals - np.mean(spread_vals)) / np.std(spread_vals)

# 2x2 covariance matrix
X = np.column_stack([chl_std, spread_std])
cov_mat = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

pc1 = X @ eigenvectors[:, 0]
var_explained = eigenvalues[0] / eigenvalues.sum()

print(f"\nPC1 loadings: CHL_inv={eigenvectors[0,0]:.3f}, spread={eigenvectors[1,0]:.3f}")
print(f"Variance explained by PC1: {var_explained:.1%}")

# Correlate PC1 with outcomes
rho_c, p_c = stats.spearmanr(pc1, catches_arr)
sig_c = "***" if p_c < 0.001 else ("**" if p_c < 0.01 else ("*" if p_c < 0.05 else ("~" if p_c < 0.1 else "")))
print(f"PC1 vs catches: rho={rho_c:+.3f}, p={p_c:.4f} {sig_c}")

pc1_timing = pc1[timing_mask]
doy_timing = doy_arr[timing_mask]
if len(pc1_timing) >= 5:
    rho_d, p_d = stats.spearmanr(pc1_timing, doy_timing)
    sig_d = "***" if p_d < 0.001 else ("**" if p_d < 0.01 else ("*" if p_d < 0.05 else ("~" if p_d < 0.1 else "")))
    print(f"PC1 vs first-catch DOY: rho={rho_d:+.3f}, p={p_d:.4f} {sig_d}")

loo_good_pca = loo_classify(pc1, good_season)
print(f"PC1 LOO accuracy (good season): {loo_good_pca:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("RECOMMENDATION")
print(f"{'='*110}")

# Collect all results for comparison
all_predictors = list(predictors.keys()) + ['PCA(CHL+spread)']
all_catch_rhos = []
for name, pred in predictors.items():
    rho, p = stats.spearmanr(pred, catches_arr)
    all_catch_rhos.append((name, rho, p))
rho_pca, p_pca = stats.spearmanr(pc1, catches_arr)
all_catch_rhos.append(('PCA(CHL+spread)', rho_pca, p_pca))

# Sort by absolute rho
all_catch_rhos.sort(key=lambda x: -abs(x[1]))

print(f"\nRanking by |Spearman rho| with total catches:")
for i, (name, rho, p) in enumerate(all_catch_rhos):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.1 else "")))
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {i+1}. {name:>35s}: rho={rho:+.3f}, p={p:.4f} {sig}{marker}")

print(f"\nNote: CHL sub-score is INVERTED in the backtest data.")
print(f"  Lower s_chl = higher actual chlorophyll = more productive water = better for marlin.")
print(f"  In composites above, CHL has been negated so higher composite = better conditions.")
print(f"\nInterpretation guide:")
print(f"  - Positive rho with catches = predictor increases with more catches (good)")
print(f"  - Negative rho with DOY = predictor increases with earlier first catch (good)")
print(f"  - LOO accuracy above 50% = better than random for classification")
