"""Correlate catch records from all_catches.csv against backtest habitat data."""
import csv, json, numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats

# Load backtest data
bt = json.load(open("data/backtest/backtest_results.json"))
bt_by_date = {e["date"]: e for e in bt["dates"] if e.get("zone_mean") is not None}

# Load CSV catches
catch_dates = []
with open("data/all_catches.csv", newline="") as f:
    for row in csv.DictReader(f):
        qty = int(row.get("Quantity") or 0)
        if qty <= 0:
            continue
        parts = row["date"].split("/")
        iso = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        sp = row.get("species", "").upper()
        catch_dates.append({"date": iso, "qty": qty, "species": sp, "type": row.get("type", "")})

# Load absence dates
absence_dates = []
with open("data/all_catches.csv", newline="") as f:
    for row in csv.DictReader(f):
        qty = int(row.get("Quantity") or 0)
        if qty > 0:
            continue
        parts = row["date"].split("/")
        iso = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        absence_dates.append(iso)

# Load annual catches
pmc = json.load(open("data/perth_marlin_catches.json"))
catch_by_year = {y["year"]: y["total"] for y in pmc["years"]}

# Build weekly habitat per year
weekly = defaultdict(list)
for r in bt["dates"]:
    if r.get("zone_mean") is None:
        continue
    dt = datetime.strptime(r["date"], "%Y-%m-%d")
    year = dt.year
    doy = dt.timetuple().tm_yday
    weekly[year].append((doy, r))


def find_nearest(iso_date, max_gap=4):
    """Find nearest backtest entry within max_gap days."""
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    best = None
    best_diff = 999
    for delta in range(0, max_gap + 1):
        for d in [dt + timedelta(days=delta), dt - timedelta(days=delta)]:
            key = d.strftime("%Y-%m-%d")
            if key in bt_by_date and delta < best_diff:
                best = bt_by_date[key]
                best_diff = delta
    return best, best_diff


# ── Analysis 1: Catch dates vs habitat scores ──
print("=" * 70)
print("CATCH DATE vs HABITAT SCORE ANALYSIS")
print("=" * 70)

catch_scores = []
miss = 0
for c in catch_dates:
    best, gap = find_nearest(c["date"])
    if best:
        catch_scores.append({
            "date": c["date"], "qty": c["qty"], "species": c["species"],
            "zone_max": best["zone_max"], "zone_mean": best["zone_mean"],
            "gap_days": gap
        })
    else:
        miss += 1

print(f"\nMatched {len(catch_scores)}/{len(catch_dates)} catch records to backtest dates ({miss} no match)")
if catch_scores:
    maxs = [s["zone_max"] for s in catch_scores]
    means = [s["zone_mean"] for s in catch_scores]
    print(f"\nHabitat scores on CATCH dates:")
    print(f"  Zone Max:  mean={np.mean(maxs):.1f}%, median={np.median(maxs):.1f}%, min={np.min(maxs):.1f}%, max={np.max(maxs):.1f}%")
    print(f"  Zone Mean: mean={np.mean(means):.1f}%, median={np.median(means):.1f}%, min={np.min(means):.1f}%, max={np.max(means):.1f}%")

# ── Analysis 2: Absence dates vs habitat scores ──
absence_scores = []
for iso in absence_dates:
    best, gap = find_nearest(iso)
    if best:
        absence_scores.append({"date": iso, "zone_max": best["zone_max"], "zone_mean": best["zone_mean"]})

if absence_scores:
    abs_maxs = [s["zone_max"] for s in absence_scores]
    abs_means = [s["zone_mean"] for s in absence_scores]
    print(f"\nHabitat scores on NO-FISH dates ({len(absence_scores)} matched):")
    print(f"  Zone Max:  mean={np.mean(abs_maxs):.1f}%, median={np.median(abs_maxs):.1f}%")
    print(f"  Zone Mean: mean={np.mean(abs_means):.1f}%, median={np.median(abs_means):.1f}%")

    # Mann-Whitney test
    u_stat, p_val = stats.mannwhitneyu(means, abs_means, alternative="greater")
    print(f"\n  Mann-Whitney (catch > absence): U={u_stat:.0f}, p={p_val:.4f}")
    print(f"  Catch mean habitat: {np.mean(means):.1f}% vs Absence: {np.mean(abs_means):.1f}% (diff={np.mean(means)-np.mean(abs_means):+.1f}%)")

# ── Analysis 3: Per-variable sub-scores on catch vs absence dates ──
print("\n" + "=" * 70)
print("PER-VARIABLE SUB-SCORES: CATCH vs ABSENCE DATES")
print("=" * 70)

var_names = ["s_sst", "s_ssh", "s_mld", "s_chl", "s_o2", "s_clarity",
             "s_sst_front", "s_sst_intrusion", "s_current", "s_convergence"]

catch_vars = defaultdict(list)
absence_vars = defaultdict(list)

for c in catch_dates:
    best, gap = find_nearest(c["date"])
    if best and "s_sst" in best:
        for v in var_names:
            val = best.get(v)
            if val is not None:
                catch_vars[v].append(val)

for iso in absence_dates:
    best, gap = find_nearest(iso)
    if best and "s_sst" in best:
        for v in var_names:
            val = best.get(v)
            if val is not None:
                absence_vars[v].append(val)

print(f"\nCatch dates with sub-scores: {len(catch_vars.get('s_sst', []))}")
print(f"Absence dates with sub-scores: {len(absence_vars.get('s_sst', []))}")
print(f"\n{'Variable':<20} {'Catch Mean':>10} {'Absence Mean':>12} {'Diff':>8} {'p-value':>10} {'Signal':>8}")
print("-" * 70)
for v in var_names:
    if catch_vars[v] and absence_vars[v]:
        cm = np.mean(catch_vars[v])
        am = np.mean(absence_vars[v])
        diff = cm - am
        try:
            u, p = stats.mannwhitneyu(catch_vars[v], absence_vars[v], alternative="two-sided")
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        except:
            p = 1.0
            sig = ""
        print(f"{v:<20} {cm:>10.3f} {am:>12.3f} {diff:>+8.3f} {p:>10.4f} {sig:>8}")
    else:
        print(f"{v:<20}  insufficient data (catch={len(catch_vars[v])}, absence={len(absence_vars[v])})")

# ── Analysis 4: Season-level correlation ──
print("\n" + "=" * 70)
print("SEASON HABITAT vs ANNUAL CATCHES (Spearman)")
print("=" * 70)

years_with_data = sorted(set(weekly.keys()) & set(catch_by_year.keys()))
years_sub = [y for y in years_with_data if any("s_sst" in r for _, r in weekly[y])]

print(f"\nYears with backtest data: {len(years_with_data)}")
print(f"Years with sub-scores: {len(years_sub)}")

# Overall habitat mean vs catches
hab_means = []
catches = []
for y in years_with_data:
    season = [(doy, r) for doy, r in weekly[y] if doy <= 121]  # Jan-Apr peak
    if len(season) < 4:
        continue
    hab_means.append(np.mean([r["zone_mean"] for _, r in season]))
    catches.append(catch_by_year[y])

rho, p = stats.spearmanr(hab_means, catches)
print(f"\nPeak season (Jan-Apr) habitat mean vs catches: rho={rho:.3f}, p={p:.4f}")

# Full year
hab_full = []
catches_full = []
for y in years_with_data:
    all_pts = [(doy, r) for doy, r in weekly[y]]
    if len(all_pts) < 8:
        continue
    hab_full.append(np.mean([r["zone_mean"] for _, r in all_pts]))
    catches_full.append(catch_by_year[y])

if hab_full:
    rho, p = stats.spearmanr(hab_full, catches_full)
    print(f"Full year habitat mean vs catches: rho={rho:.3f}, p={p:.4f}")

# Per-variable seasonal means
if years_sub:
    print(f"\nPer-variable peak season means vs catches (n={len(years_sub)} years):")
    print(f"{'Variable':<20} {'rho':>8} {'p-value':>10} {'Signal':>8}")
    print("-" * 50)
    for v in var_names:
        var_means = []
        var_catches = []
        for y in years_sub:
            season = [(doy, r) for doy, r in weekly[y] if doy <= 121 and v in r]
            if len(season) < 3:
                continue
            var_means.append(np.mean([r[v] for _, r in season]))
            var_catches.append(catch_by_year[y])
        if len(var_means) >= 5:
            rho, p = stats.spearmanr(var_means, var_catches)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            print(f"{v:<20} {rho:>+8.3f} {p:>10.4f} {sig:>8}")
        else:
            print(f"{v:<20}  insufficient ({len(var_means)} years)")

# ── Analysis 5: Monthly catch distribution ──
print("\n" + "=" * 70)
print("MONTHLY CATCH DISTRIBUTION (from CSV)")
print("=" * 70)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_catches = defaultdict(int)
for c in catch_dates:
    m = int(c["date"][5:7])
    monthly_catches[m] += c["qty"]
for m in range(1, 13):
    bar = "#" * (monthly_catches[m] // 2)
    print(f"  {month_names[m-1]}: {monthly_catches[m]:>3} {bar}")

# ── Analysis 6: Per-catch detail (dates with scores) ──
print("\n" + "=" * 70)
print("INDIVIDUAL CATCH DATES WITH HABITAT SCORES")
print("=" * 70)
print(f"\n{'Date':<12} {'Qty':>4} {'Species':<18} {'ZoneMax':>8} {'ZoneMean':>9} {'Gap':>4}")
print("-" * 60)
for s in sorted(catch_scores, key=lambda x: x["date"]):
    sp_short = s["species"].replace(" MARLIN", "")
    print(f"{s['date']:<12} {s['qty']:>4} {sp_short:<18} {s['zone_max']:>7.1f}% {s['zone_mean']:>8.1f}% {s['gap_days']:>3}d")
