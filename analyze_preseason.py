"""Analyze Nov-Dec pre-season factors that predict marlin season strength."""
import json, csv, os
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats

# Load data
bt = json.loads(Path('data/backtest/backtest_results.json').read_text())

# Load all blue marlin catches
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

# Count catches per season
season_catches = defaultdict(int)
for d in catch_dates:
    dt = datetime.strptime(d, '%Y-%m-%d')
    if dt.month <= 4:
        season_catches[dt.year] += 1

# Sub-score variables
sub_vars = ['s_sst', 's_ssh', 's_current', 's_sst_front', 's_chl', 's_mld',
            's_convergence', 's_o2', 's_clarity', 's_sst_intrusion']

# Build per-year metrics for multiple pre-season windows
windows = {
    'Oct': [10],
    'Nov': [11],
    'Dec': [12],
    'Oct-Nov': [10, 11],
    'Nov-Dec': [11, 12],
    'Oct-Dec': [10, 11, 12],
    'Sep-Oct': [9, 10],
    'Aug-Oct': [8, 9, 10],
}

results = {}
for window_name, months in windows.items():
    yearly = defaultdict(lambda: defaultdict(list))
    for e in bt['dates']:
        if 's_sst' not in e:
            continue
        dt = datetime.strptime(e['date'], '%Y-%m-%d')
        if dt.month in months:
            # Pre-season months predict NEXT year's season
            target = dt.year + 1
            for v in sub_vars:
                if e.get(v) is not None:
                    yearly[target][v].append(e[v])
            if e.get('zone_mean') is not None:
                yearly[target]['zone_mean'].append(e['zone_mean'])
            if e.get('zone_max') is not None:
                yearly[target]['zone_max'].append(e['zone_max'])
    results[window_name] = yearly

# Correlation analysis
print("=" * 80)
print("PRE-SEASON PREDICTOR ANALYSIS: What Nov-Dec conditions predict marlin season?")
print("=" * 80)
print(f"\nBlue marlin catches per season (Jan-Apr):")
all_years = sorted(set(y for w in results.values() for y in w.keys()))
for y in all_years:
    c = season_catches.get(y, 0)
    label = "***" if c >= 5 else ("**" if c >= 3 else ("*" if c >= 1 else ""))
    print(f"  {y}: {c:2d} {label}")

print(f"\n{'='*80}")
print(f"CORRELATIONS: Pre-season metric vs next season catches")
print(f"{'='*80}")

best_results = []

for window_name in ['Nov-Dec', 'Oct-Dec', 'Oct-Nov', 'Aug-Oct', 'Oct', 'Nov', 'Dec']:
    yearly = results[window_name]
    print(f"\n--- Window: {window_name} ---")

    metrics = sub_vars + ['zone_mean', 'zone_max']
    for metric in metrics:
        x_vals = []
        y_vals = []
        for year in sorted(yearly.keys()):
            vals = yearly[year].get(metric, [])
            if vals:
                x_vals.append(np.mean(vals))
                y_vals.append(season_catches.get(year, 0))

        if len(x_vals) < 10:
            continue

        rho, p = stats.spearmanr(x_vals, y_vals)
        r_pearson, p_pearson = stats.pearsonr(x_vals, y_vals)

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if p < 0.1:
            print(f"  {metric:>15s}: rho={rho:+.3f} p={p:.4f} {sig}  (r={r_pearson:+.3f}, n={len(x_vals)})")
            best_results.append((p, rho, metric, window_name, len(x_vals)))

# Binary analysis: catch years vs no-catch years
print(f"\n{'='*80}")
print(f"BINARY ANALYSIS: Catch years vs zero-catch years (Nov-Dec window)")
print(f"{'='*80}")

yearly_nd = results['Nov-Dec']
catch_years = []
zero_years = []
for year in sorted(yearly_nd.keys()):
    c = season_catches.get(year, 0)
    if c > 0:
        catch_years.append(year)
    else:
        zero_years.append(year)

print(f"\nCatch years ({len(catch_years)}): {catch_years}")
print(f"Zero years  ({len(zero_years)}): {zero_years}")

metrics = sub_vars + ['zone_mean', 'zone_max']
print(f"\n{'Metric':>15s} | {'Catch Mean':>10s} | {'Zero Mean':>10s} | {'Diff':>8s} | {'Cohen d':>8s} | {'Mann-W p':>8s} | {'Direction'}")
print("-" * 100)

for metric in metrics:
    catch_vals = []
    zero_vals = []
    for year in catch_years:
        vals = yearly_nd[year].get(metric, [])
        if vals:
            catch_vals.append(np.mean(vals))
    for year in zero_years:
        vals = yearly_nd[year].get(metric, [])
        if vals:
            zero_vals.append(np.mean(vals))

    if len(catch_vals) < 5 or len(zero_vals) < 5:
        continue

    cm = np.mean(catch_vals)
    zm = np.mean(zero_vals)
    pooled_std = np.sqrt((np.var(catch_vals) * (len(catch_vals)-1) + np.var(zero_vals) * (len(zero_vals)-1)) / (len(catch_vals) + len(zero_vals) - 2))
    d = (cm - zm) / pooled_std if pooled_std > 0 else 0
    u_stat, u_p = stats.mannwhitneyu(catch_vals, zero_vals, alternative='two-sided')

    direction = "HIGHER before catches" if cm > zm else "LOWER before catches"
    sig = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else ("~" if u_p < 0.1 else "")))

    print(f"{metric:>15s} | {cm:10.4f} | {zm:10.4f} | {cm-zm:+8.4f} | {d:+8.3f} | {u_p:8.4f} {sig:3s} | {direction}")

# Detailed year-by-year table
print(f"\n{'='*80}")
print(f"YEAR-BY-YEAR TABLE: Pre-season (Nov-Dec) averages")
print(f"{'='*80}")
print(f"\n{'Year':>4s} {'Catches':>7s} | {'SST':>7s} {'SSH':>7s} {'Current':>7s} {'Front':>7s} {'CHL':>7s} {'MLD':>7s} {'ZoneMn':>7s}")
print("-" * 80)

for year in sorted(yearly_nd.keys()):
    c = season_catches.get(year, 0)
    row = [f"{year:>4d}", f"{c:>7d}"]
    for metric in ['s_sst', 's_ssh', 's_current', 's_sst_front', 's_chl', 's_mld', 'zone_mean']:
        vals = yearly_nd[year].get(metric, [])
        if vals:
            row.append(f"{np.mean(vals):7.3f}")
        else:
            row.append(f"{'N/A':>7s}")
    print(f"{row[0]} {row[1]} | {' '.join(row[2:])}")

# Multi-window best predictors summary
print(f"\n{'='*80}")
print(f"TOP PREDICTORS (all windows, p < 0.1)")
print(f"{'='*80}")
best_results.sort(key=lambda x: x[0])
for p, rho, metric, window, n in best_results[:20]:
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "~"))
    print(f"  {window:>8s} {metric:>15s}: rho={rho:+.3f}  p={p:.4f} {sig}  (n={n})")

# Strong season analysis (>= 4 catches)
print(f"\n{'='*80}")
print(f"STRONG SEASON PREDICTION (>= 4 catches)")
print(f"{'='*80}")

strong_years = [y for y in sorted(yearly_nd.keys()) if season_catches.get(y, 0) >= 4]
weak_years = [y for y in sorted(yearly_nd.keys()) if season_catches.get(y, 0) < 4]

print(f"\nStrong seasons ({len(strong_years)}): {strong_years}")
print(f"Weak seasons   ({len(weak_years)}): {weak_years}")

print(f"\n{'Metric':>15s} | {'Strong':>8s} | {'Weak':>8s} | {'Cohen d':>8s} | {'AUC':>6s} | {'p':>8s}")
print("-" * 75)

for metric in metrics:
    strong_vals = [np.mean(yearly_nd[y].get(metric, [])) for y in strong_years if yearly_nd[y].get(metric)]
    weak_vals = [np.mean(yearly_nd[y].get(metric, [])) for y in weak_years if yearly_nd[y].get(metric)]

    if len(strong_vals) < 3 or len(weak_vals) < 3:
        continue

    sm = np.mean(strong_vals)
    wm = np.mean(weak_vals)
    pooled_std = np.sqrt((np.var(strong_vals)*(len(strong_vals)-1) + np.var(weak_vals)*(len(weak_vals)-1)) / (len(strong_vals)+len(weak_vals)-2))
    d = (sm - wm) / pooled_std if pooled_std > 0 else 0

    # AUC
    all_vals = [(v, 1) for v in strong_vals] + [(v, 0) for v in weak_vals]
    all_vals.sort(key=lambda x: x[0])
    n_pos = len(strong_vals)
    n_neg = len(weak_vals)
    auc = sum(1 for i, (vi, li) in enumerate(all_vals) if li == 1 for j, (vj, lj) in enumerate(all_vals) if lj == 0 and vi > vj) / (n_pos * n_neg)

    u_stat, u_p = stats.mannwhitneyu(strong_vals, weak_vals, alternative='two-sided')
    sig = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else ("~" if u_p < 0.1 else "")))

    print(f"{metric:>15s} | {sm:8.4f} | {wm:8.4f} | {d:+8.3f} | {auc:6.3f} | {u_p:8.4f} {sig}")

# Rate of change analysis: is Nov-Dec HIGHER than Sep-Oct?
print(f"\n{'='*80}")
print(f"TRAJECTORY ANALYSIS: Nov-Dec minus Sep-Oct (warming trend)")
print(f"{'='*80}")

yearly_so = results['Sep-Oct']
print(f"\n{'Metric':>15s} | {'Catch rho':>9s} | {'p':>8s} | {'Zero rho':>9s}")
print("-" * 55)

for metric in sub_vars + ['zone_mean']:
    deltas = []
    catches_list = []
    for year in sorted(yearly_nd.keys()):
        nd_vals = yearly_nd[year].get(metric, [])
        so_vals = yearly_so[year].get(metric, [])
        if nd_vals and so_vals:
            delta = np.mean(nd_vals) - np.mean(so_vals)
            deltas.append(delta)
            catches_list.append(season_catches.get(year, 0))

    if len(deltas) < 10:
        continue

    rho, p = stats.spearmanr(deltas, catches_list)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.1 else "")))
    if p < 0.15:
        print(f"  {metric:>15s} | rho={rho:+.3f} | {p:8.4f} {sig:3s}")
