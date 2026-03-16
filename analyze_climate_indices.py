"""Correlate ENSO/SOI/IOD climate indices with Perth marlin catches."""
import json, numpy as np
from scipy import stats

# Load catch data
pmc = json.load(open("data/perth_marlin_catches.json"))
catch_by_year = {y["year"]: y["total"] for y in pmc["years"]}


def parse_index(text):
    """Parse NOAA PSL format: year val1 val2 ... val12"""
    data = {}
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) < 13:
            continue
        year = int(parts[0])
        vals = [float(v) for v in parts[1:13]]
        data[year] = vals
    return data


# Nino 3.4 SST anomaly (NOAA PSL)
NINO34_RAW = """
1990 0.01 0.21 0.04 0.08 0.08 -0.08 0.09 0.22 0.22 0.21 0.10 0.35
1991 0.49 0.32 0.03 0.19 0.26 0.52 0.76 0.68 0.42 0.86 1.20 1.69
1992 1.84 1.78 1.38 1.20 1.04 0.58 0.22 0.05 -0.06 -0.30 -0.28 -0.15
1993 0.14 0.41 0.39 0.59 0.78 0.35 0.23 0.13 0.35 0.06 0.00 0.16
1994 0.10 0.06 0.11 0.26 0.31 0.32 0.24 0.53 0.48 0.75 1.11 1.25
1995 1.02 0.73 0.46 0.28 -0.11 -0.14 -0.21 -0.63 -0.84 -0.88 -1.10 -0.95
1996 -0.86 -0.86 -0.61 -0.47 -0.38 -0.44 -0.44 -0.22 -0.45 -0.44 -0.41 -0.64
1997 -0.53 -0.37 -0.25 0.16 0.64 1.09 1.56 1.89 2.13 2.36 2.41 2.29
1998 2.38 2.03 1.34 0.78 0.57 -0.39 -0.99 -1.28 -1.26 -1.46 -1.46 -1.69
1999 -1.69 -1.32 -0.95 -1.11 -1.15 -1.19 -1.17 -1.22 -1.09 -1.23 -1.58 -1.74
2000 -1.77 -1.55 -0.98 -0.87 -0.86 -0.79 -0.67 -0.49 -0.52 -0.70 -0.79 -0.92
2001 -0.73 -0.63 -0.48 -0.49 -0.34 -0.19 -0.04 -0.05 -0.20 -0.14 -0.37 -0.41
2002 -0.15 -0.04 0.01 0.02 0.31 0.72 0.74 0.87 1.09 1.25 1.47 1.37
2003 0.60 0.64 0.36 -0.14 -0.61 -0.29 0.21 0.26 0.27 0.42 0.33 0.43
2004 0.27 0.23 0.12 0.07 0.06 0.13 0.49 0.76 0.81 0.73 0.66 0.74
2005 0.66 0.36 0.45 0.26 0.30 0.04 -0.23 -0.05 -0.04 -0.06 -0.59 -0.92
2006 -0.91 -0.67 -0.71 -0.32 -0.09 0.00 0.01 0.31 0.60 0.70 0.99 1.14
2007 0.70 0.13 -0.18 -0.32 -0.47 -0.35 -0.59 -0.72 -1.11 -1.39 -1.54 -1.58
2008 -1.68 -1.67 -1.21 -0.99 -0.84 -0.68 -0.30 -0.13 -0.25 -0.35 -0.46 -0.86
2009 -0.89 -0.79 -0.69 -0.35 0.06 0.31 0.48 0.56 0.68 0.89 1.46 1.74
2010 1.52 1.25 0.90 0.38 -0.22 -0.69 -1.07 -1.39 -1.60 -1.69 -1.64 -1.60
2011 -1.54 -1.11 -0.93 -0.77 -0.52 -0.38 -0.43 -0.65 -0.80 -1.05 -1.19 -1.06
2012 -0.87 -0.67 -0.61 -0.50 -0.32 0.02 0.25 0.47 0.38 0.26 0.16 -0.25
2013 -0.53 -0.52 -0.25 -0.25 -0.40 -0.42 -0.39 -0.38 -0.18 -0.20 -0.14 -0.17
2014 -0.49 -0.62 -0.28 0.08 0.32 0.23 -0.06 -0.03 0.29 0.44 0.75 0.71
2015 0.51 0.42 0.47 0.70 0.92 1.18 1.46 1.93 2.21 2.36 2.72 2.66
2016 2.57 2.26 1.62 0.91 0.30 -0.03 -0.48 -0.58 -0.58 -0.74 -0.76 -0.50
2017 -0.43 -0.08 0.03 0.22 0.37 0.34 0.25 -0.16 -0.43 -0.56 -0.97 -0.98
2018 -0.98 -0.78 -0.80 -0.51 -0.20 0.04 0.12 0.09 0.47 0.90 0.90 0.89
2019 0.65 0.71 0.81 0.62 0.55 0.45 0.35 0.04 0.03 0.48 0.52 0.52
2020 0.60 0.37 0.48 0.36 -0.27 -0.34 -0.30 -0.59 -0.83 -1.26 -1.42 -1.15
2021 -1.00 -1.00 -0.80 -0.72 -0.46 -0.28 -0.39 -0.53 -0.55 -0.94 -0.94 -1.06
2022 -0.94 -0.89 -0.97 -1.11 -1.11 -0.75 -0.69 -0.97 -1.07 -0.99 -0.90 -0.85
2023 -0.72 -0.46 -0.11 0.14 0.46 0.84 1.02 1.35 1.60 1.72 2.02 2.03
2024 1.81 1.52 1.13 0.77 0.23 0.17 0.04 -0.12 -0.26 -0.27 -0.25 -0.60
2025 -0.74 -0.43 0.01 -0.14 -0.13 -0.07 -0.14 -0.36 -0.47 -0.50 -0.70 -0.67
"""

# SOI (NOAA PSL)
SOI_RAW = """
1990 -0.10 -3.00 -0.70 0.30 2.00 0.50 0.90 -0.30 -1.20 0.40 -0.80 -0.40
1991 1.00 0.40 -1.10 -1.00 -1.70 -0.20 0.00 -0.70 -2.50 -1.70 -1.10 -2.90
1992 -4.70 -1.50 -3.30 -1.70 0.40 -1.00 -1.00 0.60 0.10 -2.30 -1.10 -0.90
1993 -1.50 -1.20 -0.80 -1.90 -0.60 -1.40 -1.30 -1.60 -1.20 -1.80 -0.10 0.30
1994 -0.20 0.40 -1.10 -2.10 -1.10 -0.70 -2.20 -2.00 -2.60 -1.90 -0.90 -2.00
1995 -0.70 -0.20 1.20 -1.10 -0.60 0.20 0.70 0.50 0.40 0.00 0.10 -0.80
1996 1.60 0.40 1.90 1.30 0.50 1.90 1.10 1.20 1.00 1.00 -0.10 1.50
1997 0.80 2.90 -0.70 -1.00 -2.20 -2.30 -1.20 -2.40 -2.40 -2.40 -2.00 -1.60
1998 -4.40 -3.40 -4.00 -2.40 0.40 1.60 2.00 1.90 1.70 1.80 1.70 2.30
1999 3.00 1.60 2.10 2.30 0.40 0.40 0.90 0.60 -0.10 1.60 1.70 2.40
2000 1.10 2.70 2.20 2.00 0.60 -0.30 -0.30 1.20 1.40 1.80 3.00 1.30
2001 1.60 2.80 1.50 0.30 -0.80 0.50 -0.30 -0.70 0.30 -0.10 1.10 -1.40
2002 0.70 1.80 -0.40 -0.10 -1.40 -0.40 -0.80 -1.60 -1.00 -0.60 -0.70 -1.80
2003 -0.30 -1.10 -0.50 -0.20 -0.50 -1.00 0.50 0.20 -0.20 0.00 -0.50 1.80
2004 -2.20 2.00 0.70 -1.50 1.70 -1.40 -0.80 -0.50 -0.60 -0.10 -1.10 -1.30
2005 0.60 -5.20 0.50 -1.00 -1.30 0.70 0.30 -0.60 0.60 2.00 -0.30 -0.00
2006 2.70 0.20 2.90 1.80 -0.80 -0.40 -1.00 -1.70 -1.00 -2.10 0.10 -0.50
2007 -1.30 -0.10 0.30 -0.20 -0.20 0.90 -0.50 0.70 0.30 1.20 1.40 2.70
2008 2.90 4.40 2.40 1.10 -0.10 1.00 0.50 1.70 2.00 2.10 2.20 2.40
2009 1.80 3.10 0.70 1.30 -0.20 0.20 0.40 -0.30 0.50 -2.00 -1.00 -1.20
2010 -1.80 -2.40 -1.10 2.00 1.50 0.60 3.00 3.00 3.70 2.90 2.10 4.80
2011 3.80 4.50 4.20 3.10 0.60 0.40 1.60 0.70 1.70 1.20 1.80 4.10
2012 1.80 0.80 1.20 -0.40 0.10 -0.70 -0.10 -0.30 0.40 0.50 0.40 -1.00
2013 -0.10 -0.40 2.50 0.40 1.30 2.00 1.30 0.30 0.50 -0.10 1.20 0.10
2014 2.40 0.10 -1.50 1.30 0.90 0.30 -0.30 -1.20 -1.20 -1.00 -1.50 -0.90
2015 -1.40 0.40 -1.20 -0.10 -1.20 -0.90 -1.90 -2.40 -2.70 -2.80 -0.80 -0.90
2016 -3.60 -3.20 -0.10 -2.00 0.70 1.10 0.70 1.20 2.00 -0.40 -0.20 0.50
2017 0.30 -0.10 1.50 -0.30 0.40 -0.70 1.30 0.90 1.00 1.50 1.50 -0.20
2018 1.80 -0.80 2.40 0.80 0.60 -0.20 0.40 -0.50 -1.50 0.60 -0.10 1.70
2019 -0.10 -2.30 -0.50 0.20 -0.70 -0.70 -0.60 -0.20 -1.90 -0.60 -1.40 -0.90
2020 0.30 -0.10 -0.20 0.30 0.70 -0.60 0.70 1.80 1.50 0.80 1.10 3.00
2021 3.20 2.50 0.60 0.60 0.80 0.70 2.30 1.00 1.30 1.20 1.60 2.50
2022 0.80 1.80 2.90 2.80 2.40 2.80 1.30 1.70 2.70 2.80 0.50 3.50
2023 2.30 2.30 0.30 0.40 -1.70 0.40 -0.40 -1.40 -2.10 -0.80 -1.30 -0.40
2024 0.80 -2.30 0.60 -0.30 0.80 0.10 -1.20 1.50 -0.20 0.80 0.80 1.90
2025 0.30 0.90 2.80 0.90 0.70 0.50 1.00 0.70 0.10 1.90 1.80 -0.00
"""

nino34 = parse_index(NINO34_RAW)
soi_data = parse_index(SOI_RAW)

print("CLIMATE INDEX vs PERTH MARLIN CATCHES")
print("=" * 75)

# Test multiple windows
configs = [
    # (label, data, month_indices, negate, lag_description)
    ("Jun-Oct Nino34", nino34, [5,6,7,8,9], True, "pre-season -> next year"),
    ("Aug-Oct Nino34", nino34, [7,8,9], True, "pre-season -> next year"),
    ("Jan-May Nino34", nino34, [0,1,2,3,4], True, "winter -> next year"),
    ("Jun-Oct SOI", soi_data, [5,6,7,8,9], False, "pre-season -> next year"),
    ("Aug-Oct SOI", soi_data, [7,8,9], False, "pre-season -> next year"),
]

print(f"\n{'Window':<20} {'rho':>8} {'p':>10} {'n':>4} {'Signal':>8}  Notes")
print("-" * 75)

for label, data, months, negate, note in configs:
    vals, catches = [], []
    for year in sorted(data.keys()):
        if len(data[year]) < max(months) + 1:
            continue
        next_year = year + 1
        if next_year not in catch_by_year:
            continue
        window_vals = [data[year][m] for m in months]
        if any(v < -90 for v in window_vals):
            continue
        mean_val = np.mean(window_vals)
        if negate:
            mean_val = -mean_val
        vals.append(mean_val)
        catches.append(catch_by_year[next_year])

    if len(vals) >= 5:
        rho, p = stats.spearmanr(vals, catches)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{label:<20} {rho:>+8.3f} {p:>10.4f} {len(vals):>4} {sig:>8}  {note}")

# Same-year correlations
print(f"\n{'Window':<20} {'rho':>8} {'p':>10} {'n':>4} {'Signal':>8}  Notes")
print("-" * 75)
for label, data, months, negate in [
    ("Jan-Apr Nino34", nino34, [0,1,2,3], True),
    ("Jan-Apr SOI", soi_data, [0,1,2,3], False),
]:
    vals, catches = [], []
    for year in sorted(data.keys()):
        if year not in catch_by_year:
            continue
        if len(data[year]) < max(months) + 1:
            continue
        window_vals = [data[year][m] for m in months]
        if any(v < -90 for v in window_vals):
            continue
        mean_val = np.mean(window_vals)
        if negate:
            mean_val = -mean_val
        vals.append(mean_val)
        catches.append(catch_by_year[year])

    if len(vals) >= 5:
        rho, p = stats.spearmanr(vals, catches)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{label:<20} {rho:>+8.3f} {p:>10.4f} {len(vals):>4} {sig:>8}  same year (concurrent)")

# Detail table for best predictor
print("\n\nDETAIL: Pre-season ENSO state vs next year catches")
print("=" * 65)
print(f"{'Year':>6} {'Nino34 JunOct':>14} {'ENSO State':>12} {'Next Catches':>13}")
print("-" * 50)

for year in sorted(nino34.keys()):
    if len(nino34[year]) < 10:
        continue
    next_year = year + 1
    if next_year not in catch_by_year:
        continue
    vals = nino34[year][5:10]
    if any(v < -90 for v in vals):
        continue
    mean_val = np.mean(vals)
    if mean_val < -0.5:
        state = "La Nina"
    elif mean_val > 0.5:
        state = "El Nino"
    else:
        state = "Neutral"
    c = catch_by_year[next_year]
    marker = " <--" if c >= 10 else ""
    print(f"{year:>6} {mean_val:>+14.2f} {state:>12} {c:>12}{marker}")

# Summary by ENSO state
print("\nSummary by ENSO state:")
la_nina, neutral, el_nino = [], [], []
for year in sorted(nino34.keys()):
    if len(nino34[year]) < 10:
        continue
    next_year = year + 1
    if next_year not in catch_by_year:
        continue
    vals = nino34[year][5:10]
    if any(v < -90 for v in vals):
        continue
    mean_val = np.mean(vals)
    c = catch_by_year[next_year]
    if mean_val < -0.5:
        la_nina.append(c)
    elif mean_val > 0.5:
        el_nino.append(c)
    else:
        neutral.append(c)

print(f"  La Nina years  (n={len(la_nina):>2}): mean={np.mean(la_nina):>5.1f} catches, median={np.median(la_nina):>4.0f}")
print(f"  Neutral years  (n={len(neutral):>2}): mean={np.mean(neutral):>5.1f} catches, median={np.median(neutral):>4.0f}")
print(f"  El Nino years  (n={len(el_nino):>2}): mean={np.mean(el_nino):>5.1f} catches, median={np.median(el_nino):>4.0f}")
