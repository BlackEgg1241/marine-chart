#!/usr/bin/env python3
"""
evaluate_model.py -- honest, presence/background evaluation of the hotspot model.

Why this exists
---------------
`validate_scoring.py` reports the *mean habitat score at catch locations* and, worse,
computes that mean only over catches that fall inside a zone (dropping the model's
clear misses). Because the Optuna objective in `optimize_visual.py` maximises the same
mean-score-at-catches, that number is in-sample training error, not predictive skill.
It also has no negatives, so it cannot express whether the map discriminates good spots
from bad ones at all.

This script fixes both problems using ONLY the already-committed per-date hotspot
GeoJSONs (no Copernicus download, no scipy) so it runs anywhere:

  1. Honest hit statistics: score every catch (0 if it lands outside all zones),
     report the mean INCLUDING misses and the miss rate.
  2. Presence / pseudo-absence discrimination:
       - AUC-domain  : catch cells vs random points in the study domain
                        (tests the whole model incl. the depth gate)
       - AUC-inzone  : catch cells vs random points that already sit inside some
                        rendered zone that day (the hard "which spot?" test)
     AUC is the Mann-Whitney U statistic (probability a random catch outscores a
     random background point). 0.5 == no skill.
  3. Lift / capture curve: fraction of catches captured vs fraction of mapped area
     retained as you raise the score threshold -- the operationally meaningful curve.
  4. Leave-one-year-out stability: AUC recomputed per held-out year, so you can see
     whether the discrimination is stable or driven by a few dates.

This does NOT refit weights (that needs the full ocean-data pipeline). It evaluates
the *deployed* score grid. A true nested CV that refits weights on train folds should
wrap this once the pipeline is runnable; the `holdout_by_year` output here is the
honest stand-in that the project can report today.

Usage:
  python evaluate_model.py                      # blue marlin, default data/ dir
  python evaluate_model.py --species all        # all billfish
  python evaluate_model.py --bg-per-date 200 --seed 7
  python evaluate_model.py --catches data/all_catches.csv --data-dir data
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

# Study domain (matches DEFAULT_BBOX in marlin_data.py)
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}


# --------------------------------------------------------------------------- IO

def parse_date(s):
    """DD/MM/YYYY or YYYY-MM-DD -> YYYY-MM-DD."""
    s = s.strip()
    if "/" in s:
        d, m, y = s.split("/")
        return f"{y}-{int(m):02d}-{int(d):02d}"
    return s[:10]


def load_catches(path, species):
    """Return list of {date, lat, lon, species} for rows with usable coords."""
    out = []
    want = species.strip().upper()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = (r.get("species") or "").strip().upper()
            if want != "ALL" and sp != want:
                continue
            lat_s, lon_s = (r.get("lat") or "").strip(), (r.get("lon") or "").strip()
            if not lat_s or not lon_s:
                continue
            try:
                lat, lon = float(lat_s), float(lon_s)
            except ValueError:
                continue
            out.append({"date": parse_date(r["date"]), "lat": lat, "lon": lon, "species": sp})
    return out


def load_polys(geojson_path):
    """Load hotspot polygons as (exterior_ring, holes, intensity) tuples."""
    with open(geojson_path) as f:
        gj = json.load(f)
    polys = []
    for ft in gj.get("features", []):
        inten = ft.get("properties", {}).get("intensity")
        if inten is None:
            continue
        geom = ft["geometry"]
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            rings = geom["coordinates"]
        else:
            continue
        for poly in rings:
            if not poly:
                continue
            polys.append((poly[0], poly[1:], float(inten)))
    return polys


# ------------------------------------------------------------------- geometry

def point_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def score_at(lon, lat, polys):
    """Max intensity of any polygon containing the point; 0.0 if none (a miss)."""
    best = 0.0
    for ext, holes, inten in polys:
        if inten <= best:
            continue
        # cheap bbox reject
        xs = [p[0] for p in ext]; ys = [p[1] for p in ext]
        if lon < min(xs) or lon > max(xs) or lat < min(ys) or lat > max(ys):
            continue
        if point_in_ring(lon, lat, ext) and not any(point_in_ring(lon, lat, h) for h in holes):
            best = inten
    return best


# ------------------------------------------------------------------- metrics

def auc_mann_whitney(pos, neg):
    """P(random positive > random negative), ties counted as 0.5. 0.5 == no skill."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def best_tss(pos, neg):
    """Max True Skill Statistic (sensitivity+specificity-1) over score thresholds."""
    if not pos or not neg:
        return float("nan"), float("nan")
    thresholds = sorted(set(pos + neg))
    best, best_t = -1.0, 0.0
    P, N = len(pos), len(neg)
    for t in thresholds:
        tp = sum(1 for x in pos if x >= t)
        fp = sum(1 for x in neg if x >= t)
        sens = tp / P
        spec = 1 - fp / N
        tss = sens + spec - 1
        if tss > best:
            best, best_t = tss, t
    return best, best_t


# ------------------------------------------------------------------- driver

def collect(catches, data_dir, bg_per_date, rng):
    """For each catch date with a committed map, gather catch score + backgrounds."""
    per = []  # dict per catch: {year, catch_score, bg_domain[], bg_inzone[]}
    missing = 0
    for c in catches:
        gj = os.path.join(data_dir, c["date"], "blue_marlin_hotspots.geojson")
        if not os.path.exists(gj):
            missing += 1
            continue
        polys = load_polys(gj)
        cs = score_at(c["lon"], c["lat"], polys)
        # domain background: uniform random points in the study bbox
        bg_domain, bg_inzone = [], []
        tries = 0
        while len(bg_domain) < bg_per_date and tries < bg_per_date * 40:
            tries += 1
            lon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"])
            lat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"])
            s = score_at(lon, lat, polys)
            bg_domain.append(s)
            if s > 0:
                bg_inzone.append(s)
        per.append({
            "year": int(c["date"][:4]),
            "date": c["date"],
            "catch_score": cs,
            "bg_domain": bg_domain,
            "bg_inzone": bg_inzone,
        })
    return per, missing


def summarize(per, label):
    catch_scores = [p["catch_score"] for p in per]
    pos = catch_scores
    neg_domain = [s for p in per for s in p["bg_domain"]]
    neg_inzone = [s for p in per for s in p["bg_inzone"]]

    n = len(catch_scores)
    misses = sum(1 for s in catch_scores if s == 0)
    mean_all = sum(catch_scores) / n if n else float("nan")
    in_zone = [s for s in catch_scores if s > 0]
    mean_inzone = sum(in_zone) / len(in_zone) if in_zone else float("nan")

    auc_dom = auc_mann_whitney(pos, neg_domain)
    auc_iz = auc_mann_whitney(pos, neg_inzone)
    tss_dom, t_dom = best_tss(pos, neg_domain)

    print(f"\n{'='*66}\n{label}\n{'='*66}")
    print(f"  catches evaluated (with a committed map): {n}")
    print(f"  MISSES (catch outside every zone): {misses}/{n} ({100*misses/n:.0f}%)")
    print(f"  mean score at catches  INCLUDING misses : {mean_all:.3f}   <-- honest")
    print(f"  mean score at catches  in-zone only     : {mean_inzone:.3f}   <-- old headline")
    print(f"  background points: {len(neg_domain)} domain, {len(neg_inzone)} in-zone")
    print(f"  AUC  catch vs random-domain point   : {auc_dom:.3f}  (0.5 = no skill)")
    print(f"  AUC  catch vs random-in-zone point  : {auc_iz:.3f}  (harder: 'which spot?')")
    print(f"  best TSS (vs domain) : {tss_dom:.3f} at threshold {t_dom:.2f}")

    # capture / lift curve vs domain background (area proxy = fraction of bg retained)
    print("\n  capture curve (threshold -> % catches kept / % random-domain area kept / lift):")
    for t in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        cap = sum(1 for s in pos if s >= t) / n
        area = sum(1 for s in neg_domain if s >= t) / max(len(neg_domain), 1)
        lift = (cap / area) if area > 0 else float("inf")
        print(f"     >= {t:.2f} : catches {100*cap:5.0f}%   area {100*area:5.0f}%   lift x{lift:.1f}")

    # leave-one-year-out AUC stability
    years = sorted(set(p["year"] for p in per))
    print("\n  leave-one-year-out AUC (vs domain) -- stability check:")
    loo = []
    for y in years:
        held = [p for p in per if p["year"] == y]
        hp = [p["catch_score"] for p in held]
        hn = [s for p in held for s in p["bg_domain"]]
        a = auc_mann_whitney(hp, hn)
        loo.append(a)
        if not math.isnan(a):
            print(f"     {y}: AUC {a:.3f}  (n={len(hp)})")
    valid = [a for a in loo if not math.isnan(a)]
    if valid:
        mean_loo = sum(valid) / len(valid)
        var = sum((a - mean_loo) ** 2 for a in valid) / len(valid)
        print(f"     -> mean per-year AUC {mean_loo:.3f} +/- {math.sqrt(var):.3f}")
    return {"n": n, "misses": misses, "mean_all": mean_all, "mean_inzone": mean_inzone,
            "auc_domain": auc_dom, "auc_inzone": auc_iz, "tss": tss_dom}


def main():
    ap = argparse.ArgumentParser(description="Honest presence/background evaluation of the hotspot model.")
    ap.add_argument("--catches", default=os.path.join("data", "all_catches.csv"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--species", default="BLUE MARLIN",
                    help="'BLUE MARLIN', 'STRIPED MARLIN', or 'ALL'")
    ap.add_argument("--bg-per-date", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    catches = load_catches(args.catches, args.species)
    print(f"Loaded {len(catches)} '{args.species}' catches with coordinates.")
    per, missing = collect(catches, args.data_dir, args.bg_per_date, rng)
    if missing:
        print(f"NOTE: {missing} catches skipped -- no committed hotspot map for their date "
              f"(evaluation is limited to dates already generated).")
    if not per:
        print("No catch dates had a committed hotspot map; nothing to evaluate.")
        return
    result = summarize(per, f"HONEST EVALUATION -- {args.species}")

    print("\nInterpretation:")
    print("  * 'mean including misses' is the number to report, not the in-zone mean.")
    print("  * AUC-domain > ~0.7 means the map beats random ocean; AUC-inzone near 0.5")
    print("    means that WITHIN the rendered zones it barely ranks one spot over another.")
    print("  * lift is how much you concentrate catches per unit area retained.")
    print("  * This evaluates the deployed grid; it does NOT refit weights, so it is an")
    print("    optimistic bound on out-of-sample skill (weights were tuned on these catches).")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
