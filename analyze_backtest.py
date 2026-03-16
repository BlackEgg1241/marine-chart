"""
Backtest analysis and seasonal tracker visualizations.

Generates:
1. Early-season signal vs catches scatter plot
2. Seasonal overlay: current season vs historical (good/bad year bands)
3. Season progress tracker: where does this season rank?

Usage:
    python analyze_backtest.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict
from datetime import datetime
from scipy import stats

# Load data
bt = json.load(open("data/backtest/backtest_results.json"))
valid = [r for r in bt["dates"] if r.get("zone_mean") is not None]
catches = json.load(open("data/perth_marlin_catches.json"))
catch_by_year = {y["year"]: y for y in catches["years"]}

# Build weekly time series per year (day_of_year -> score)
weekly = defaultdict(list)
for r in valid:
    year = int(r["date"][:4])
    dt = datetime.strptime(r["date"], "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    weekly[year].append((doy, r["zone_mean"]))
for y in weekly:
    weekly[y].sort()

years = sorted(set(weekly.keys()) & set(catch_by_year.keys()))
current_year = 2026

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.size": 10,
})

GOOD_COLOR = "#22c55e"
MID_COLOR = "#fbbf24"
POOR_COLOR = "#ef4444"
CURRENT_COLOR = "#a855f7"
BAND_GOOD = "#22c55e"
BAND_ALL = "#64748b"

# ── Load all catch records with dates ──────────────────────────
catch_records = []
import csv
try:
    with open("data/all_catches.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qty = int(row.get("Quantity") or 0)
            if qty <= 0:
                continue  # skip absence records
            # Parse DD/MM/YYYY to YYYY-MM-DD
            parts = row["date"].split("/")
            iso_date = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            sp = row.get("species", "BLUE MARLIN").upper()
            for _ in range(qty):
                catch_records.append({"date": iso_date, "species": sp})
except FileNotFoundError:
    pass


# ══════════════════════════════════════════════════════════════
# FIGURE 1: Early-Season Signal vs Catches
# ══════════════════════════════════════════════════════════════
def plot_early_season_signal():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Intra-Season Trajectory Signals vs Marlin Catches (2010-2025)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Compute features
    data = []
    for y in years:
        season = [(doy, s) for doy, s in weekly[y] if doy <= 182]  # Jan-Jun
        if len(season) < 8:
            continue
        doys = np.array([x[0] for x in season])
        scores = np.array([x[1] for x in season])
        early = scores[doys <= 59]   # Jan-Feb
        mid = scores[(doys >= 60) & (doys <= 121)]  # Mar-Apr
        if len(early) == 0 or len(mid) == 0:
            continue
        jump = float(np.mean(mid) - np.mean(early))
        peak_doy = int(doys[np.argmax(scores)])
        peak_week = peak_doy // 7 + 1
        data.append({
            "year": y, "jump": jump, "peak_week": peak_week,
            "early": float(np.mean(early)), "mid": float(np.mean(mid)),
            "catches": catch_by_year[y]["total"],
            "blue": catch_by_year[y]["blue"],
        })

    # Normalize for composite signal
    jumps = [d["jump"] for d in data]
    peaks = [d["peak_week"] for d in data]
    j_mean, j_std = np.mean(jumps), np.std(jumps)
    p_mean, p_std = np.mean(peaks), np.std(peaks)
    for d in data:
        d["signal"] = -0.5 * (d["jump"] - j_mean) / j_std - 0.5 * (d["peak_week"] - p_mean) / p_std

    # ── Panel 1: Jump vs Catches ──
    ax = axes[0]
    for d in data:
        color = CURRENT_COLOR if d["year"] == 2025 else ("#38bdf8" if d["catches"] >= 6 else "#94a3b8")
        size = 40 + d["catches"] * 4
        ax.scatter(d["jump"], d["catches"], s=size, c=color, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(str(d["year"]), (d["jump"], d["catches"]),
                    fontsize=7, ha="center", va="bottom", color="#94a3b8",
                    xytext=(0, 5), textcoords="offset points")
    # Trend line (excl 2025)
    x_no25 = [d["jump"] for d in data if d["year"] != 2025]
    y_no25 = [d["catches"] for d in data if d["year"] != 2025]
    z = np.polyfit(x_no25, y_no25, 1)
    xline = np.linspace(min(x_no25) - 0.5, max(x_no25) + 0.5, 50)
    ax.plot(xline, np.polyval(z, xline), "--", color="#fbbf24", alpha=0.5, linewidth=1)
    rho, p = stats.spearmanr(x_no25, y_no25)
    ax.set_xlabel("Early-to-Mid Jump (%)")
    ax.set_ylabel("Annual Catches")
    ax.set_title(f"Jan-Feb to Mar-Apr Jump\nrho={rho:.2f}, p={p:.3f} (excl 2025)", fontsize=10)
    ax.axvline(0, color="#475569", linewidth=0.5, linestyle=":")
    ax.grid(True, linewidth=0.3)

    # ── Panel 2: Peak Week vs Catches ──
    ax = axes[1]
    for d in data:
        color = CURRENT_COLOR if d["year"] == 2025 else ("#38bdf8" if d["catches"] >= 6 else "#94a3b8")
        size = 40 + d["catches"] * 4
        ax.scatter(d["peak_week"], d["catches"], s=size, c=color, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(str(d["year"]), (d["peak_week"], d["catches"]),
                    fontsize=7, ha="center", va="bottom", color="#94a3b8",
                    xytext=(0, 5), textcoords="offset points")
    x_no25 = [d["peak_week"] for d in data if d["year"] != 2025]
    y_no25 = [d["catches"] for d in data if d["year"] != 2025]
    z = np.polyfit(x_no25, y_no25, 1)
    xline = np.linspace(min(x_no25) - 1, max(x_no25) + 1, 50)
    ax.plot(xline, np.polyval(z, xline), "--", color="#fbbf24", alpha=0.5, linewidth=1)
    rho, p = stats.spearmanr(x_no25, y_no25)
    ax.set_xlabel("Peak Week (1=Jan, 26=Jun)")
    ax.set_title(f"Peak Timing\nrho={rho:.2f}, p={p:.3f} (excl 2025)", fontsize=10)
    ax.grid(True, linewidth=0.3)

    # ── Panel 3: Combined Signal vs Catches ──
    ax = axes[2]
    for d in data:
        color = CURRENT_COLOR if d["year"] == 2025 else (GOOD_COLOR if d["catches"] >= 6 else (POOR_COLOR if d["catches"] == 0 else MID_COLOR))
        size = 50 + d["catches"] * 4
        ax.scatter(d["signal"], d["catches"], s=size, c=color, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(str(d["year"]), (d["signal"], d["catches"]),
                    fontsize=7, ha="center", va="bottom", color="#e2e8f0",
                    xytext=(0, 6), textcoords="offset points")
    # Trend line
    x_all = [d["signal"] for d in data]
    y_all = [d["catches"] for d in data]
    z = np.polyfit(x_all, y_all, 1)
    xline = np.linspace(min(x_all) - 0.3, max(x_all) + 0.3, 50)
    ax.plot(xline, np.polyval(z, xline), "-", color="#a855f7", alpha=0.6, linewidth=1.5)
    rho, p = stats.spearmanr(x_all, y_all)
    ax.set_xlabel("Early-Season Signal (higher = earlier peak)")
    ax.set_title(f"Combined Signal\nrho={rho:.2f}, p={p:.3f}", fontsize=10)
    ax.grid(True, linewidth=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GOOD_COLOR, markersize=8, label=">=6 catches"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MID_COLOR, markersize=8, label="1-5 catches"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POOR_COLOR, markersize=8, label="0 catches"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CURRENT_COLOR, markersize=8, label="2025"),
    ]
    axes[2].legend(handles=legend_elements, loc="upper left", fontsize=8,
                   facecolor="#1e293b", edgecolor="#475569")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("data/backtest/signal_vs_catches.png", dpi=150, bbox_inches="tight")
    print("Saved: data/backtest/signal_vs_catches.png")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Season Overlay — Current vs Historical
# ══════════════════════════════════════════════════════════════
def plot_season_overlay():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Month labels for x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Categorize years
    good_years = [y for y in years if catch_by_year[y]["total"] >= 6]
    poor_years = [y for y in years if catch_by_year[y]["total"] == 0]
    mid_years = [y for y in years if 0 < catch_by_year[y]["total"] < 6]

    # Interpolate all years to common DOY grid
    doy_grid = np.arange(1, 366)

    def interp_year(y):
        pts = weekly[y]
        if len(pts) < 4:
            return None
        doys = np.array([p[0] for p in pts])
        scores = np.array([p[1] for p in pts])
        return np.interp(doy_grid, doys, scores, left=np.nan, right=np.nan)

    # Compute percentile bands from all historical years (2010-2024)
    hist_years = [y for y in years if y <= 2024]
    hist_matrix = []
    for y in hist_years:
        interp = interp_year(y)
        if interp is not None:
            hist_matrix.append(interp)
    hist_matrix = np.array(hist_matrix)

    p10 = np.nanpercentile(hist_matrix, 10, axis=0)
    p25 = np.nanpercentile(hist_matrix, 25, axis=0)
    p50 = np.nanpercentile(hist_matrix, 50, axis=0)
    p75 = np.nanpercentile(hist_matrix, 75, axis=0)
    p90 = np.nanpercentile(hist_matrix, 90, axis=0)

    # Draw bands
    ax.fill_between(doy_grid, p10, p90, alpha=0.08, color="#94a3b8", label="10th-90th pctile")
    ax.fill_between(doy_grid, p25, p75, alpha=0.15, color="#64748b", label="25th-75th pctile")
    ax.plot(doy_grid, p50, color="#64748b", linewidth=1, alpha=0.6, label="Median (2010-2024)")

    # Draw good years lightly
    for y in good_years:
        if y == 2025:
            continue
        interp = interp_year(y)
        if interp is not None:
            ax.plot(doy_grid, interp, color=GOOD_COLOR, alpha=0.2, linewidth=0.7)
    # Label one
    ax.plot([], [], color=GOOD_COLOR, alpha=0.4, linewidth=1, label=f"Good years (>=6 catches)")

    # Draw poor years lightly
    for y in poor_years:
        interp = interp_year(y)
        if interp is not None:
            ax.plot(doy_grid, interp, color=POOR_COLOR, alpha=0.15, linewidth=0.7)
    ax.plot([], [], color=POOR_COLOR, alpha=0.3, linewidth=1, label="Zero-catch years")

    # Draw 2025 (record year)
    interp_2025 = interp_year(2025)
    if interp_2025 is not None:
        mask_2025 = ~np.isnan(interp_2025)
        ax.plot(doy_grid[mask_2025], interp_2025[mask_2025], color=CURRENT_COLOR,
                linewidth=2, alpha=0.9, label="2025 (38 catches)", zorder=5)

    # Draw current year (2026)
    if current_year in weekly and len(weekly[current_year]) > 2:
        interp_cur = interp_year(current_year)
        if interp_cur is not None:
            mask_cur = ~np.isnan(interp_cur)
            ax.plot(doy_grid[mask_cur], interp_cur[mask_cur], color="#38bdf8",
                    linewidth=2.5, alpha=1.0, label=f"{current_year} (current)", zorder=6)
            # Mark latest point
            last_doy = max(d for d, s in weekly[current_year])
            last_score = [s for d, s in weekly[current_year] if d == last_doy][0]
            ax.scatter([last_doy], [last_score], s=80, c="#38bdf8", edgecolors="white",
                       linewidths=1.5, zorder=7)
            ax.annotate(f"{last_score:.0f}%", (last_doy, last_score),
                        fontsize=9, fontweight="bold", color="#38bdf8",
                        xytext=(8, 5), textcoords="offset points")

    # ── Overlay actual catch dates ──
    # Build interpolated scores per year for catch lookup
    year_interps = {}
    for y in set(weekly.keys()):
        interp = interp_year(y)
        if interp is not None:
            year_interps[y] = interp

    species_markers = {"BLUE MARLIN": "o", "STRIPED MARLIN": "s", "BLACK MARLIN": "D"}
    species_colors = {"BLUE MARLIN": "#38bdf8", "STRIPED MARLIN": "#22c55e", "BLACK MARLIN": "#1e1e1e"}
    species_edge = {"BLUE MARLIN": "white", "STRIPED MARLIN": "white", "BLACK MARLIN": "white"}

    # Group catches by unique (year, doy) to stack multiples
    from collections import Counter
    catch_points = []
    for c in catch_records:
        dt_c = datetime.strptime(c["date"], "%Y-%m-%d")
        y_c = dt_c.year
        doy_c = dt_c.timetuple().tm_yday
        sp = c["species"]
        if y_c in year_interps and 1 <= doy_c <= 365:
            score = year_interps[y_c][doy_c - 1]
            if not np.isnan(score):
                catch_points.append((doy_c, score, sp, y_c))

    # Plot catch markers
    for sp_name, marker in species_markers.items():
        pts = [(d, s) for d, s, sp, y in catch_points if sp == sp_name]
        if pts:
            doys, scores = zip(*pts)
            ax.scatter(doys, scores, marker=marker, s=45, c=species_colors[sp_name],
                       edgecolors=species_edge[sp_name], linewidths=0.8, alpha=0.85,
                       zorder=8, label=f"Catch: {sp_name.title()}")

    # Season markers
    ax.axvspan(1, 182, alpha=0.03, color="#22c55e")  # Marlin season
    ax.text(91, 42, "MARLIN SEASON", ha="center", fontsize=9, color="#22c55e", alpha=0.5, fontweight="bold")

    # Gridlines at 5kn intervals equivalent
    for level in [50, 60, 70, 80, 90]:
        ax.axhline(level, color="#ffffff", linewidth=0.2, alpha=0.15)

    ax.set_xlim(1, 365)
    ax.set_ylim(40, 100)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Zone Mean Habitat Score (%)")
    ax.set_title("Perth Canyon Habitat Score — Season Overlay (2010-2026)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, facecolor="#1e293b", edgecolor="#475569", ncol=2)
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig("data/backtest/season_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: data/backtest/season_overlay.png")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Season Progress Tracker
# ══════════════════════════════════════════════════════════════
def plot_season_tracker():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{current_year} Season Tracker — Perth Canyon Habitat",
                 fontsize=14, fontweight="bold", y=0.98)

    # Build season stats up to current DOY
    today_doy = datetime.now().timetuple().tm_yday
    season_doy_max = min(today_doy, 182)  # Cap at end of June

    def season_stats(y, max_doy=182):
        pts = [(d, s) for d, s in weekly[y] if d <= max_doy]
        if not pts:
            return None
        scores = [s for _, s in pts]
        doys = [d for d, _ in pts]
        early = [s for d, s in pts if d <= 59]
        mid = [s for d, s in pts if 60 <= d <= 121]
        return {
            "mean": np.mean(scores),
            "early_mean": np.mean(early) if early else None,
            "mid_mean": np.mean(mid) if mid else None,
            "peak": max(scores),
            "peak_doy": doys[np.argmax(scores)],
            "weeks_above_90": sum(1 for s in scores if s >= 90),
            "min": min(scores),
            "n": len(scores),
        }

    # ── Panel 1: Season Mean Bar Chart ──
    ax = axes[0, 0]
    bar_data = []
    for y in years:
        st = season_stats(y, max_doy=season_doy_max if y == current_year else 182)
        if st and st["n"] >= 3:
            bar_data.append((y, st["mean"], catch_by_year.get(y, {}).get("total", 0)))
    # Add current year
    if current_year in weekly:
        st = season_stats(current_year, max_doy=season_doy_max)
        if st and st["n"] >= 3:
            bar_data.append((current_year, st["mean"], 0))

    bar_data.sort(key=lambda x: x[1], reverse=True)
    ys = [str(d[0]) for d in bar_data]
    means = [d[1] for d in bar_data]
    colors = []
    for d in bar_data:
        if d[0] == current_year:
            colors.append("#38bdf8")
        elif d[0] == 2025:
            colors.append(CURRENT_COLOR)
        elif d[2] >= 6:
            colors.append(GOOD_COLOR)
        elif d[2] == 0:
            colors.append(POOR_COLOR)
        else:
            colors.append(MID_COLOR)

    bars = ax.barh(range(len(bar_data)), means, color=colors, alpha=0.8, edgecolor="#1e293b")
    ax.set_yticks(range(len(bar_data)))
    ax.set_yticklabels(ys, fontsize=8)
    ax.set_xlabel("Season Mean Habitat Score (%)")
    title_suffix = f" (to DOY {season_doy_max})" if current_year in weekly else ""
    ax.set_title(f"Season Mean Ranking{title_suffix}", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(70, 97)
    ax.grid(True, axis="x", linewidth=0.3)
    # Add values
    for i, (y, m, c) in enumerate(bar_data):
        ax.text(m + 0.3, i, f"{m:.1f}%", va="center", fontsize=7, color="#e2e8f0")

    # ── Panel 2: Monthly Comparison Heatmap ──
    ax = axes[0, 1]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    all_years_sorted = sorted(years + ([current_year] if current_year in weekly else []))

    # Build monthly means matrix
    monthly_data = []
    year_labels = []
    for y in all_years_sorted:
        row = []
        for m_idx, m_start, m_end in [(0,1,31), (1,32,59), (2,60,90), (3,91,120), (4,121,151), (5,152,182)]:
            vals = [s for d, s in weekly[y] if m_start <= d <= m_end]
            row.append(np.mean(vals) if vals else np.nan)
        monthly_data.append(row)
        year_labels.append(str(y))

    matrix = np.array(monthly_data)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=60, vmax=97)
    ax.set_xticks(range(6))
    ax.set_xticklabels(months)
    ax.set_yticks(range(len(year_labels)))
    ax.set_yticklabels(year_labels, fontsize=7)
    ax.set_title("Monthly Habitat Scores (Jan-Jun)", fontsize=10)

    # Add text values
    for i in range(len(year_labels)):
        for j in range(6):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "black" if val > 80 else "white"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Habitat %")

    # ── Panel 3: Cumulative weeks above 90% ──
    ax = axes[1, 0]
    doy_range = range(1, 183)
    for y in years:
        pts = sorted(weekly[y])
        cumul = []
        count = 0
        pi = 0
        for doy in doy_range:
            while pi < len(pts) and pts[pi][0] <= doy:
                if pts[pi][1] >= 90:
                    count += 1
                pi += 1
            cumul.append(count)
        color = GOOD_COLOR if catch_by_year[y]["total"] >= 6 else (POOR_COLOR if catch_by_year[y]["total"] == 0 else "#64748b")
        alpha = 0.5 if catch_by_year[y]["total"] >= 6 else 0.2
        lw = 1 if catch_by_year[y]["total"] >= 6 else 0.7
        if y == 2025:
            color = CURRENT_COLOR
            alpha = 0.9
            lw = 2
        ax.plot(list(doy_range), cumul, color=color, alpha=alpha, linewidth=lw)

    # Current year
    if current_year in weekly:
        pts = sorted(weekly[current_year])
        cumul = []
        count = 0
        pi = 0
        for doy in doy_range:
            while pi < len(pts) and pts[pi][0] <= doy:
                if pts[pi][1] >= 90:
                    count += 1
                pi += 1
            cumul.append(count)
        ax.plot(list(doy_range), cumul, color="#38bdf8", linewidth=2.5, alpha=1.0, zorder=5)
        # Mark latest
        last_doy = min(max(d for d, s in weekly[current_year]), 182)
        ax.scatter([last_doy], [cumul[last_doy - 1]], s=60, c="#38bdf8",
                   edgecolors="white", linewidths=1.5, zorder=6)

    ax.set_xlim(1, 182)
    ax.set_xticks([1, 32, 60, 91, 121, 152])
    ax.set_xticklabels(months)
    ax.set_ylabel("Cumulative Weeks >= 90%")
    ax.set_title("Season Accumulation of High-Score Weeks", fontsize=10)
    ax.grid(True, linewidth=0.3)

    # ── Panel 4: Season Scorecard ──
    ax = axes[1, 1]
    ax.axis("off")

    if current_year in weekly:
        cur_st = season_stats(current_year, max_doy=season_doy_max)
        # Compare to historical seasons at same point in year
        hist_at_doy = []
        for y in years:
            st = season_stats(y, max_doy=season_doy_max)
            if st and st["n"] >= 3:
                hist_at_doy.append((y, st))

        cur_mean = cur_st["mean"] if cur_st else 0
        hist_means = [st["mean"] for _, st in hist_at_doy]
        percentile = stats.percentileofscore(hist_means, cur_mean)

        # Best matching historical years
        similarities = [(y, abs(st["mean"] - cur_mean), st["mean"], catch_by_year.get(y, {}).get("total", 0))
                        for y, st in hist_at_doy]
        similarities.sort(key=lambda x: x[1])

        lines = [
            f"{current_year} SEASON SCORECARD",
            f"(through day {season_doy_max}, {cur_st['n']} weekly samples)",
            "",
            f"Season Mean:     {cur_mean:.1f}%",
            f"Season Peak:     {cur_st['peak']:.1f}%  (day {cur_st['peak_doy']})",
            f"Season Min:      {cur_st['min']:.1f}%",
            f"Weeks >= 90%:    {cur_st['weeks_above_90']}",
            "",
            f"Historical Rank: {percentile:.0f}th percentile",
            f"(vs {len(hist_means)} prior seasons at same point)",
            "",
            "Most Similar Seasons:",
        ]
        for y, diff, mean, catches in similarities[:4]:
            lines.append(f"  {y}: {mean:.1f}% mean -> {catches} catches")

        for i, line in enumerate(lines):
            weight = "bold" if i == 0 else "normal"
            size = 12 if i == 0 else (9 if i == 1 else 10)
            color = "#38bdf8" if i == 0 else ("#94a3b8" if i == 1 else "#e2e8f0")
            ax.text(0.05, 0.95 - i * 0.065, line, transform=ax.transAxes,
                    fontsize=size, fontweight=weight, color=color,
                    fontfamily="monospace", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("data/backtest/season_tracker.png", dpi=150, bbox_inches="tight")
    print("Saved: data/backtest/season_tracker.png")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Annual Catch Bars + Habitat Trend
# ══════════════════════════════════════════════════════════════
def plot_catches_vs_habitat():
    fig, ax1 = plt.subplots(figsize=(14, 5))

    bar_years = sorted(catch_by_year.keys())
    catches_vals = [catch_by_year[y]["total"] for y in bar_years]
    blue_vals = [catch_by_year[y]["blue"] for y in bar_years]

    # Bars: total catches
    x = np.arange(len(bar_years))
    bars = ax1.bar(x, catches_vals, color="#38bdf8", alpha=0.6, label="Total catches", width=0.7)
    ax1.bar(x, blue_vals, color="#2563eb", alpha=0.8, label="Blue marlin", width=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in bar_years], rotation=45)
    ax1.set_ylabel("Annual Catches", color="#38bdf8")
    ax1.tick_params(axis="y", labelcolor="#38bdf8")

    # Overlay: season mean habitat
    ax2 = ax1.twinx()
    season_means = []
    season_years = []
    for y in bar_years:
        if y in weekly:
            pts = [(d, s) for d, s in weekly[y] if d <= 182]
            if pts:
                season_means.append(np.mean([s for _, s in pts]))
                season_years.append(y)

    x2 = [list(bar_years).index(y) for y in season_years]
    ax2.plot(x2, season_means, "o-", color=CURRENT_COLOR, linewidth=1.5, markersize=5,
             alpha=0.9, label="Season habitat mean", zorder=3)
    ax2.set_ylabel("Season Mean Habitat (%)", color=CURRENT_COLOR)
    ax2.tick_params(axis="y", labelcolor=CURRENT_COLOR)
    ax2.set_ylim(80, 97)

    # Combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="#2563eb", alpha=0.8, label="Blue marlin"),
        plt.Rectangle((0, 0), 1, 1, fc="#38bdf8", alpha=0.6, label="Other marlin"),
        Line2D([0], [0], marker="o", color=CURRENT_COLOR, linewidth=1.5, markersize=5, label="Season habitat mean"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8,
               facecolor="#1e293b", edgecolor="#475569")

    ax1.set_title("Perth Marlin Catches vs Season Habitat Score (2010-2025)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", linewidth=0.3)

    plt.tight_layout()
    plt.savefig("data/backtest/catches_vs_habitat.png", dpi=150, bbox_inches="tight")
    print("Saved: data/backtest/catches_vs_habitat.png")
    plt.close()


# ── Run all ──
if __name__ == "__main__":
    plot_early_season_signal()
    plot_season_overlay()
    plot_season_tracker()
    plot_catches_vs_habitat()
    print("\nAll charts generated in data/backtest/")
