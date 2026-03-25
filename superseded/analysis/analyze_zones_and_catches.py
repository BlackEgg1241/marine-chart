"""Analyze marlin zones, validate predictions at GPS catch points,
and check zone correlation for social media catches without coordinates."""

import json, csv, os, sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from scipy import stats

# ── Load catches ──────────────────────────────────────────────────────────
geo = json.loads(Path('data/marlin_catches.geojson').read_text())
rows = list(csv.DictReader(open('data/all_catches.csv')))

# GPS catches (tagged, have coordinates)
gps_catches = []
for r in rows:
    if r['species'] != 'BLUE MARLIN':
        continue
    if r['lat'] and r['lon'] and r['type'] == 'tagged':
        gps_catches.append({
            'date': datetime.strptime(r['date'], '%d/%m/%Y').strftime('%Y-%m-%d'),
            'lat': float(r['lat']),
            'lon': float(r['lon']),
            'weight': float(r['weight_kg']) if r['weight_kg'] else None,
            'length': float(r['length_cm']) if r['length_cm'] and float(r['length_cm']) > 0 else None,
            'tag': r['tag'],
        })

# Social catches (no GPS, have quantity)
social_catches = []
for r in rows:
    if r['species'] != 'BLUE MARLIN':
        continue
    if not r['lat'] and not r['lon']:
        qty = int(r['Quantity']) if r['Quantity'] else 0
        social_catches.append({
            'date': datetime.strptime(r['date'], '%d/%m/%Y').strftime('%Y-%m-%d'),
            'quantity': qty,
            'event': r['Event'],
            'boat': r['Boat'],
            'source': r['source'],
            'type': r['type'],
        })

print(f"GPS blue marlin catches: {len(gps_catches)}")
print(f"Social blue marlin reports: {len(social_catches)} ({sum(s['quantity'] for s in social_catches)} fish)")

# ── Current zone bounds ──────────────────────────────────────────────────
ZONE_W, ZONE_E = 114.98, 115.3333
ZONE_S, ZONE_N = -32.1667, -31.7287

# ═════════════════════════════════════════════════════════════════════════
# PART 1: GPS CATCH SPATIAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 1: GPS CATCH SPATIAL DISTRIBUTION (n={len(gps_catches)})")
print(f"{'='*80}")

lats = [c['lat'] for c in gps_catches]
lons = [c['lon'] for c in gps_catches]

print(f"\nCurrent zone: lat [{ZONE_S:.4f}, {ZONE_N:.4f}] lon [{ZONE_W:.4f}, {ZONE_E:.4f}]")
print(f"Catch extent: lat [{min(lats):.4f}, {max(lats):.4f}] lon [{min(lons):.4f}, {max(lons):.4f}]")
print(f"Catch center: lat {np.mean(lats):.4f}, lon {np.mean(lons):.4f}")
print(f"Catch median: lat {np.median(lats):.4f}, lon {np.median(lons):.4f}")
print(f"Catch std:    lat {np.std(lats):.4f}, lon {np.std(lons):.4f}")

# Count in/out of zone
in_zone = sum(1 for c in gps_catches
              if ZONE_S <= c['lat'] <= ZONE_N and ZONE_W <= c['lon'] <= ZONE_E)
print(f"\nIn current zone: {in_zone}/{len(gps_catches)} ({100*in_zone/len(gps_catches):.0f}%)")

# Which catches are outside?
outside = [(c['date'], c['lat'], c['lon'], c['tag']) for c in gps_catches
           if not (ZONE_S <= c['lat'] <= ZONE_N and ZONE_W <= c['lon'] <= ZONE_E)]
if outside:
    print(f"\nOutside zone ({len(outside)}):")
    for d, la, lo, t in outside:
        reasons = []
        if la < ZONE_S: reasons.append(f"too far south by {ZONE_S - la:.4f} deg")
        if la > ZONE_N: reasons.append(f"too far north by {la - ZONE_N:.4f} deg")
        if lo < ZONE_W: reasons.append(f"too far west by {ZONE_W - lo:.4f} deg")
        if lo > ZONE_E: reasons.append(f"too far east by {lo - ZONE_E:.4f} deg")
        print(f"  {d} [{lo:.4f}, {la:.4f}] {t}: {', '.join(reasons)}")

# Spatial clustering - divide zone into quadrants
mid_lat = (ZONE_S + ZONE_N) / 2  # ~-31.95
mid_lon = (ZONE_W + ZONE_E) / 2  # ~115.16
print(f"\nQuadrant analysis (split at lat={mid_lat:.3f}, lon={mid_lon:.3f}):")
quads = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0}
for c in gps_catches:
    ns = 'N' if c['lat'] > mid_lat else 'S'
    ew = 'W' if c['lon'] < mid_lon else 'E'
    quads[ns+ew] += 1
for q in ['NW', 'NE', 'SW', 'SE']:
    print(f"  {q}: {quads[q]:2d} catches ({100*quads[q]/len(gps_catches):.0f}%)")

# Grid-based hotspot density (0.05 deg cells)
print(f"\nCatch density grid (0.05 deg cells):")
cell_size = 0.05
cell_counts = defaultdict(int)
for c in gps_catches:
    cell_lat = round(c['lat'] / cell_size) * cell_size
    cell_lon = round(c['lon'] / cell_size) * cell_size
    cell_counts[(cell_lat, cell_lon)] += 1

# Sort by count
top_cells = sorted(cell_counts.items(), key=lambda x: -x[1])[:15]
print(f"  {'Cell center':>24s}  {'Count':>5s}  {'Pct':>5s}  Near mark")

# Known marks for reference
marks = {
    'PGFC': (115.2333, -32.0667), 'Perth Canyon Head': (115.08, -31.92),
    'Rottnest Trench': (114.98, -32.01), 'FURUNO': (115.2667, -31.9667),
    'Club Marine': (115.3333, -32.05), 'Compleat Angler': (115.2, -31.9167),
    'Fibrelite': (115.1667, -32.1667), 'Woodman Pt 5': (115.1193, -32.1191),
    'North Metro 04': (115.1754, -31.7287), 'Two Rocks Canyon': (114.854, -31.701),
}

for (la, lo), cnt in top_cells:
    nearest = min(marks.items(), key=lambda m: ((m[1][1]-la)**2 + (m[1][0]-lo)**2)**0.5)
    dist_nm = ((nearest[1][1]-la)**2 + (nearest[1][0]-lo)**2)**0.5 * 60
    print(f"  [{lo:8.4f}, {la:8.4f}]  {cnt:5d}  {100*cnt/len(gps_catches):4.0f}%  {nearest[0]} ({dist_nm:.1f}nm)")

# Monthly distribution
print(f"\nMonthly catch distribution:")
month_counts = defaultdict(int)
for c in gps_catches:
    m = int(c['date'][5:7])
    month_counts[m] += 1
for m in [1, 2, 3, 4]:
    print(f"  {['Jan','Feb','Mar','Apr'][m-1]}: {month_counts[m]:2d} ({100*month_counts[m]/len(gps_catches):.0f}%)")

# Early vs late season spatial shift
early = [c for c in gps_catches if int(c['date'][5:7]) <= 2]
late = [c for c in gps_catches if int(c['date'][5:7]) >= 3]
if early and late:
    print(f"\nSeasonal spatial shift (early=Jan-Feb vs late=Mar-Apr):")
    print(f"  Early (n={len(early)}): mean lat {np.mean([c['lat'] for c in early]):.4f}, "
          f"mean lon {np.mean([c['lon'] for c in early]):.4f}")
    print(f"  Late  (n={len(late)}):  mean lat {np.mean([c['lat'] for c in late]):.4f}, "
          f"mean lon {np.mean([c['lon'] for c in late]):.4f}")
    lat_diff = np.mean([c['lat'] for c in early]) - np.mean([c['lat'] for c in late])
    lon_diff = np.mean([c['lon'] for c in early]) - np.mean([c['lon'] for c in late])
    print(f"  Shift: {abs(lat_diff)*60:.1f}nm {'north' if lat_diff > 0 else 'south'}, "
          f"{abs(lon_diff)*60:.1f}nm {'east' if lon_diff > 0 else 'west'}")

# ═════════════════════════════════════════════════════════════════════════
# PART 2: HABITAT SCORE VALIDATION AT GPS CATCH LOCATIONS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 2: HABITAT SCORE AT GPS CATCH LOCATIONS")
print(f"{'='*80}")

# Try to load existing validation results
val_path = 'data/validation_results.csv'
if os.path.exists(val_path):
    val_rows = list(csv.DictReader(open(val_path)))
    blue_val = [r for r in val_rows if r.get('species', '').upper() == 'BLUE MARLIN'
                and r.get('hotspot_score')]
    print(f"\nExisting validation results: {len(blue_val)} blue marlin scores")

    scores = [float(r['hotspot_score']) * 100 for r in blue_val if float(r['hotspot_score']) > 0]
    print(f"  Mean: {np.mean(scores):.1f}%  Median: {np.median(scores):.1f}%")
    print(f"  Range: {min(scores):.1f}% - {max(scores):.1f}%")
    print(f"  >= 70%: {sum(1 for s in scores if s >= 70)}/{len(scores)} ({100*sum(1 for s in scores if s >= 70)/len(scores):.0f}%)")
    print(f"  >= 80%: {sum(1 for s in scores if s >= 80)}/{len(scores)} ({100*sum(1 for s in scores if s >= 80)/len(scores):.0f}%)")

    # Sub-score breakdown at catch points
    sub_vars = ['s_sst', 's_ssh', 's_current', 's_sst_front', 's_chl', 's_mld',
                's_convergence', 's_o2', 's_clarity', 's_sst_intrusion']
    print(f"\n  Sub-score averages at catch locations:")
    print(f"  {'Variable':>16s}  {'Mean':>6s}  {'Med':>6s}  {'Min':>6s}  {'Max':>6s}  {'n':>4s}")
    print(f"  {'-'*55}")
    for v in sub_vars:
        vals = [float(r[v]) for r in blue_val if r.get(v) and r[v] != '']
        if vals:
            print(f"  {v:>16s}  {np.mean(vals):.4f}  {np.median(vals):.4f}  "
                  f"{min(vals):.4f}  {max(vals):.4f}  {len(vals):4d}")
else:
    print(f"\n  No validation_results.csv found - skipping point validation")

# ═════════════════════════════════════════════════════════════════════════
# PART 3: BACKTEST ZONE SCORES ON CATCH DATES
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 3: ZONE SCORES ON CATCH DATES (backtest lookup)")
print(f"{'='*80}")

bt = json.loads(Path('data/backtest/backtest_results.json').read_text())
bt_by_date = {e['date']: e for e in bt['dates']}

# For each catch date, find nearest backtest date
def nearest_bt(date_str, bt_dates, max_days=4):
    """Find nearest backtest entry within max_days."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    best = None
    best_dist = 999
    for bd in bt_dates:
        bdt = datetime.strptime(bd, '%Y-%m-%d')
        dist = abs((dt - bdt).days)
        if dist < best_dist:
            best_dist = dist
            best = bd
    return best if best_dist <= max_days else None

bt_dates = sorted(bt_by_date.keys())

# GPS catches vs zone scores
print(f"\nGPS catch dates matched to backtest zone scores:")
print(f"  {'Date':>10s}  {'ZoneMax':>7s}  {'ZoneMn':>7s}  {'s_sst':>6s}  {'s_ssh':>6s}  "
      f"{'s_chl':>6s}  {'s_curr':>6s}  {'s_front':>7s}  {'BT date':>10s}")
print(f"  {'-'*85}")

gps_zone_scores = []
gps_unmatched = []
for c in gps_catches:
    bd = nearest_bt(c['date'], bt_dates)
    if bd and bt_by_date[bd].get('zone_max') is not None:
        e = bt_by_date[bd]
        gps_zone_scores.append(e['zone_max'])
        sst = f"{e.get('s_sst', 0):.3f}" if e.get('s_sst') is not None else '  N/A'
        ssh = f"{e.get('s_ssh', 0):.3f}" if e.get('s_ssh') is not None else '  N/A'
        chl = f"{e.get('s_chl', 0):.3f}" if e.get('s_chl') is not None else '  N/A'
        cur = f"{e.get('s_current', 0):.3f}" if e.get('s_current') is not None else '  N/A'
        frt = f"{e.get('s_sst_front', 0):.3f}" if e.get('s_sst_front') is not None else '  N/A'
        match = '' if bd == c['date'] else f" (~{bd})"
        print(f"  {c['date']:>10s}  {e['zone_max']:7.1f}  {e['zone_mean']:7.1f}  "
              f"{sst}  {ssh}  {chl}  {cur}  {frt:>7s}{match}")
    else:
        gps_unmatched.append(c['date'])

if gps_zone_scores:
    print(f"\n  Zone-max on GPS catch dates: mean {np.mean(gps_zone_scores):.1f}%, "
          f"median {np.median(gps_zone_scores):.1f}%, min {min(gps_zone_scores):.1f}%")

if gps_unmatched:
    print(f"  Unmatched dates: {gps_unmatched}")

# ═════════════════════════════════════════════════════════════════════════
# PART 4: SOCIAL MEDIA CATCH VALIDATION (zone-level)
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 4: SOCIAL MEDIA CATCHES vs ZONE SCORES")
print(f"{'='*80}")

# Only include social catches that actually had fish (quantity > 0 or Release type)
social_with_fish = [s for s in social_catches if s['quantity'] > 0 or s['type'] == 'Release']
social_no_fish = [s for s in social_catches if s['quantity'] == 0 and s['type'] != 'Release']

print(f"\nSocial reports with fish: {len(social_with_fish)} ({sum(s['quantity'] for s in social_with_fish)} fish)")
print(f"Social reports no fish:   {len(social_no_fish)}")

print(f"\n  {'Date':>10s}  {'Qty':>3s}  {'ZoneMax':>7s}  {'ZoneMn':>7s}  {'s_sst':>6s}  "
      f"{'s_ssh':>6s}  {'s_chl':>6s}  {'Event/Boat':>20s}  {'BT date':>10s}")
print(f"  {'-'*100}")

social_zone_scores = []
social_nofish_scores = []

for s in social_with_fish + social_no_fish:
    bd = nearest_bt(s['date'], bt_dates)
    is_fish = s in social_with_fish
    if bd and bt_by_date[bd].get('zone_max') is not None:
        e = bt_by_date[bd]
        if is_fish:
            social_zone_scores.append(e['zone_max'])
        else:
            social_nofish_scores.append(e['zone_max'])
        sst = f"{e.get('s_sst', 0):.3f}" if e.get('s_sst') is not None else '  N/A'
        ssh = f"{e.get('s_ssh', 0):.3f}" if e.get('s_ssh') is not None else '  N/A'
        chl = f"{e.get('s_chl', 0):.3f}" if e.get('s_chl') is not None else '  N/A'
        label = s['event'] or s['boat'] or s['source']
        fish_marker = '*' if not is_fish else ' '
        match = '' if bd == s['date'] else f" (~{bd})"
        print(f" {fish_marker}{s['date']:>10s}  {s['quantity']:3d}  {e['zone_max']:7.1f}  {e['zone_mean']:7.1f}  "
              f"{sst}  {ssh}  {chl}  {label:>20s}{match}")

print(f"\n  * = no fish caught on that trip")

if social_zone_scores:
    print(f"\n  Zone-max on CATCH days:   mean {np.mean(social_zone_scores):.1f}%, "
          f"median {np.median(social_zone_scores):.1f}%, min {min(social_zone_scores):.1f}%")
if social_nofish_scores:
    print(f"  Zone-max on NO-FISH days: mean {np.mean(social_nofish_scores):.1f}%, "
          f"median {np.median(social_nofish_scores):.1f}%, min {min(social_nofish_scores):.1f}%")

# ═════════════════════════════════════════════════════════════════════════
# PART 5: CATCH vs NON-CATCH ZONE SCORE COMPARISON
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 5: CATCH DAYS vs RANDOM SEASON DAYS (zone score comparison)")
print(f"{'='*80}")

# All catch dates (GPS + social with fish)
all_catch_dates = set()
for c in gps_catches:
    all_catch_dates.add(c['date'])
for s in social_with_fish:
    all_catch_dates.add(s['date'])

# Season days = Jan-Apr of all years
season_bt = [(e['date'], e) for e in bt['dates']
             if e.get('zone_max') is not None
             and int(e['date'][5:7]) in [1, 2, 3, 4]]

catch_zone = []
non_catch_zone = []
for d, e in season_bt:
    # Check if any catch within 2 days of this backtest date
    dt = datetime.strptime(d, '%Y-%m-%d')
    is_catch_window = any(abs((dt - datetime.strptime(cd, '%Y-%m-%d')).days) <= 2
                          for cd in all_catch_dates)
    if is_catch_window:
        catch_zone.append(e['zone_max'])
    else:
        non_catch_zone.append(e['zone_max'])

print(f"\nSeason backtest days (Jan-Apr): {len(season_bt)}")
print(f"  Near catch: {len(catch_zone)}, Non-catch: {len(non_catch_zone)}")
print(f"\n  Zone-max near catches: mean {np.mean(catch_zone):.1f}%, median {np.median(catch_zone):.1f}%")
print(f"  Zone-max non-catches:  mean {np.mean(non_catch_zone):.1f}%, median {np.median(non_catch_zone):.1f}%")

u_stat, u_p = stats.mannwhitneyu(catch_zone, non_catch_zone, alternative='greater')
print(f"  Mann-Whitney (catch > non-catch): U={u_stat:.0f}, p={u_p:.4f}")

# Sub-score comparison
print(f"\n  Sub-score comparison (catch window vs non-catch, Jan-Apr):")
sub_vars = ['s_sst', 's_ssh', 's_current', 's_sst_front', 's_chl', 's_mld',
            's_convergence', 's_o2', 's_clarity', 's_sst_intrusion']
print(f"  {'Variable':>16s}  {'Catch':>7s}  {'Non-catch':>9s}  {'Diff':>7s}  {'p-value':>8s}")
print(f"  {'-'*55}")

for v in sub_vars:
    catch_vals = []
    non_vals = []
    for d, e in season_bt:
        if e.get(v) is None:
            continue
        dt = datetime.strptime(d, '%Y-%m-%d')
        is_catch_window = any(abs((dt - datetime.strptime(cd, '%Y-%m-%d')).days) <= 2
                              for cd in all_catch_dates)
        if is_catch_window:
            catch_vals.append(e[v])
        else:
            non_vals.append(e[v])
    if catch_vals and non_vals:
        cm, nm = np.mean(catch_vals), np.mean(non_vals)
        _, p = stats.mannwhitneyu(catch_vals, non_vals, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ('~' if p < 0.1 else '')))
        print(f"  {v:>16s}  {cm:7.4f}  {nm:9.4f}  {cm-nm:+7.4f}  {p:8.4f} {sig}")

# ═════════════════════════════════════════════════════════════════════════
# PART 6: ZONE REFINEMENT - DO WE NEED SUB-ZONES?
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 6: ZONE REFINEMENT ANALYSIS")
print(f"{'='*80}")

# Distance from canyon head (115.08, -31.92) - the topographic attractor
canyon_head = (115.08, -31.92)
dists = []
for c in gps_catches:
    d = ((c['lon'] - canyon_head[0])**2 + (c['lat'] - canyon_head[1])**2)**0.5 * 60
    dists.append(d)

print(f"\nDistance from Perth Canyon Head ({canyon_head[0]}, {canyon_head[1]}):")
print(f"  Mean: {np.mean(dists):.1f}nm  Median: {np.median(dists):.1f}nm  "
      f"Max: {max(dists):.1f}nm  Std: {np.std(dists):.1f}nm")
print(f"  50% within: {np.percentile(dists, 50):.1f}nm")
print(f"  75% within: {np.percentile(dists, 75):.1f}nm")
print(f"  90% within: {np.percentile(dists, 90):.1f}nm")

# Check if a tighter zone captures catches better
tight_zones = [
    ("Current zone", ZONE_W, ZONE_E, ZONE_S, ZONE_N),
    ("Core (PGFC area)", 115.10, 115.30, -32.10, -31.85),
    ("Canyon focus", 114.95, 115.25, -32.10, -31.88),
    ("Wide shelf", 114.90, 115.40, -32.20, -31.70),
    ("North shift", 114.98, 115.30, -32.05, -31.72),
]

print(f"\nAlternative zone capture rates:")
print(f"  {'Zone':>20s}  {'Catches':>8s}  {'Pct':>5s}  {'Area ratio':>10s}")
print(f"  {'-'*50}")
current_area = (ZONE_E - ZONE_W) * (ZONE_N - ZONE_S)
for name, w, e, s, n in tight_zones:
    inside = sum(1 for c in gps_catches if s <= c['lat'] <= n and w <= c['lon'] <= e)
    area_ratio = ((e - w) * (n - s)) / current_area
    print(f"  {name:>20s}  {inside:3d}/{len(gps_catches):2d}  {100*inside/len(gps_catches):4.0f}%  {area_ratio:10.2f}x")

# Catch centroid by year (track movement over time)
print(f"\nCatch centroid by year:")
by_year = defaultdict(list)
for c in gps_catches:
    y = int(c['date'][:4])
    by_year[y].append(c)

for y in sorted(by_year.keys()):
    catches = by_year[y]
    mean_lat = np.mean([c['lat'] for c in catches])
    mean_lon = np.mean([c['lon'] for c in catches])
    print(f"  {y}: [{mean_lon:.4f}, {mean_lat:.4f}] (n={len(catches)})")

# ═════════════════════════════════════════════════════════════════════════
# PART 7: SCORING ALGORITHM PERFORMANCE FOR SOCIAL MEDIA ERA (2023+)
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 7: PREDICTION PERFORMANCE FOR SOCIAL ERA (2023+)")
print(f"{'='*80}")

# For each social catch date, what was the zone score?
# Compare to the full season average for that year
for year in [2023, 2024, 2025, 2026]:
    year_catches = [s for s in social_with_fish if s['date'].startswith(str(year))]
    year_gps = [c for c in gps_catches if c['date'].startswith(str(year))]
    all_year = year_catches + [{'date': c['date'], 'quantity': 1} for c in year_gps]

    if not all_year:
        continue

    print(f"\n--- {year} Season ---")

    # Season average zone score (Jan-Apr)
    season_scores = [e['zone_max'] for e in bt['dates']
                     if e['date'].startswith(str(year))
                     and int(e['date'][5:7]) in [1, 2, 3, 4]
                     and e.get('zone_max') is not None]

    # Catch date zone scores
    catch_dates_scores = []
    for entry in all_year:
        bd = nearest_bt(entry['date'], bt_dates)
        if bd and bt_by_date[bd].get('zone_max') is not None:
            catch_dates_scores.append((entry['date'], entry.get('quantity', 1),
                                       bt_by_date[bd]['zone_max']))

    if season_scores:
        print(f"  Season avg zone-max: {np.mean(season_scores):.1f}% "
              f"(n={len(season_scores)} samples)")
    if catch_dates_scores:
        weighted_avg = np.average([s for _, _, s in catch_dates_scores],
                                   weights=[max(q, 1) for _, q, _ in catch_dates_scores])
        print(f"  Catch day zone-max:  {weighted_avg:.1f}% "
              f"(n={len(catch_dates_scores)} dates, weighted by fish count)")

        # List individual dates
        for d, q, s in sorted(catch_dates_scores):
            fish_text = f"{q} fish" if q > 1 else "1 fish"
            above = "ABOVE" if s > np.mean(season_scores) else "BELOW"
            print(f"    {d}: {s:.1f}% ({fish_text}) - {above} season avg")

# ═════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"SUMMARY & RECOMMENDATIONS")
print(f"{'='*80}")

# Quick stats
all_zone_scores = catch_zone  # from Part 5
if gps_zone_scores:
    print(f"\n1. SCORING VALIDATION:")
    print(f"   GPS catches (n={len(gps_zone_scores)}): zone-max mean={np.mean(gps_zone_scores):.1f}%, "
          f"min={min(gps_zone_scores):.1f}%")
    if social_zone_scores:
        print(f"   Social catches (n={len(social_zone_scores)}): zone-max mean={np.mean(social_zone_scores):.1f}%, "
              f"min={min(social_zone_scores):.1f}%")

print(f"\n2. ZONE COVERAGE: {in_zone}/{len(gps_catches)} GPS catches ({100*in_zone/len(gps_catches):.0f}%) in current zone")
if outside:
    print(f"   {len(outside)} catches outside zone - consider adjusting bounds")

print(f"\n3. CATCH vs NON-CATCH:")
print(f"   Zone-max {np.mean(catch_zone):.1f}% (catch) vs {np.mean(non_catch_zone):.1f}% (non-catch)")
print(f"   Mann-Whitney p={u_p:.4f}")
