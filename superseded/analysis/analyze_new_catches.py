"""Analyze new Perth marlin catch/absence data from Perth_Marlin_Data.xlsx.

Scores each date using the habitat algorithm (zone average over Accessible Trench Zone)
and compares catch dates vs absence/low-effort dates.

No GPS coordinates — all records are scored as zone averages across the fishable area.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import marlin_data
from marlin_data import generate_blue_marlin_hotspots

# Accessible Trench Zone bounds (same as generate_forecast_summary.py)
ZONE_W, ZONE_E = 114.98, 115.3333
ZONE_S, ZONE_N = -32.1667, -31.7287

BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}

# New data from Perth_Marlin_Data.xlsx
CAPTURES = [
    {"date": "2017-02-05", "boat": "Rockall", "raised": 4, "hooked": 3, "tagged": 1,
     "species": "blue marlin", "type": "catch"},
    {"date": "2024-01-06", "boat": "Depth Charger", "raised": 1, "hooked": 1, "tagged": 1,
     "species": "blue marlin", "type": "catch"},
    {"date": "2025-02-14", "boat": "Aftermath", "raised": 2, "hooked": 2, "tagged": 2,
     "species": "marlin", "type": "catch"},
    {"date": "2025-02-14", "boat": "Hood Oos", "raised": 1, "hooked": 1, "tagged": 1,
     "species": "marlin", "type": "catch"},
    {"date": "2025-02-14", "boat": "Naturaliste", "raised": 1, "hooked": 1, "tagged": 1,
     "species": "marlin", "type": "catch"},
    {"date": "2025-02-24", "boat": "Hit n Run", "raised": 1, "hooked": 1, "tagged": 1,
     "species": "black marlin", "type": "catch"},
    {"date": "2025-03-23", "boat": "Send N Bend", "raised": 4, "hooked": 2, "tagged": 2,
     "species": "marlin", "type": "catch"},
    {"date": "2026-02-02", "boat": "Running Bear", "raised": 1, "hooked": 1, "tagged": 1,
     "species": "blue marlin", "type": "catch"},
]

ABSENCES = [
    {"date": "2025-12-31", "boat": "4 PGFC boats", "raised": 0, "hooked": 0, "tagged": 0,
     "species": "none", "type": "absence", "notes": "Rottnest Trench, no marlin"},
]

# Tournament fleet-level data (not per-boat, but gives effort context)
TOURNAMENTS = [
    {"name": "Blue Marlin Classic 2025", "dates": ["2025-03-01", "2025-03-02", "2025-03-03"],
     "fleet_raised": 62, "fleet_hooked": 45, "fleet_tagged": 15, "type": "tournament_catch"},
    {"name": "Marlin Cup 2025", "dates": ["2025-02-15", "2025-02-16"],
     "fleet_raised": 66, "fleet_hooked": 39, "fleet_tagged": 13, "type": "tournament_catch"},
    {"name": "Marlin Cup 2024", "dates": ["2024-02-17", "2024-02-18"],
     "fleet_raised": 8, "fleet_hooked": 7, "fleet_tagged": 2, "type": "tournament_low"},
]


def score_date(date_str):
    """Run habitat scoring for a date, return zone stats."""
    dated_dir = os.path.join(SCRIPT_DIR, "data", date_str)
    sst_path = os.path.join(dated_dir, "sst_raw.nc")
    if not os.path.exists(sst_path):
        return None

    # Point marlin_data at the date-specific directory (same pattern as fetch_prediction.py)
    original_output_dir = marlin_data.OUTPUT_DIR
    marlin_data.OUTPUT_DIR = dated_dir

    try:
        tif_path = os.path.join(dated_dir, "bathy_gmrt.tif")
        if not os.path.exists(tif_path):
            src_tif = os.path.join(SCRIPT_DIR, "bathy_gmrt.tif")
            if os.path.exists(src_tif):
                tif_path = src_tif
            else:
                tif_path = None
        result = generate_blue_marlin_hotspots(BBOX, tif_path=tif_path)
    except Exception as e:
        print(f"  ERROR scoring {date_str}: {e}")
        return None
    finally:
        marlin_data.OUTPUT_DIR = original_output_dir

    if not result or not isinstance(result, dict):
        return None

    grid = result["grid"]
    lats = result["lats"]
    lons = result["lons"]

    # Extract zone scores
    lat_mask = (lats >= ZONE_S) & (lats <= ZONE_N)
    lon_mask = (lons >= ZONE_W) & (lons <= ZONE_E)
    zone_grid = grid[np.ix_(lat_mask, lon_mask)]
    valid = zone_grid[~np.isnan(zone_grid)]

    if len(valid) == 0:
        return {"zone_max": 0, "zone_mean": 0, "zone_median": 0, "zone_cells": 0}

    return {
        "zone_max": round(float(np.max(valid)) * 100, 1),
        "zone_mean": round(float(np.mean(valid)) * 100, 1),
        "zone_median": round(float(np.median(valid)) * 100, 1),
        "zone_p75": round(float(np.percentile(valid, 75)) * 100, 1),
        "zone_cells": int(np.sum(valid > 0)),
        "zone_above_50": round(float(np.sum(valid >= 0.5) / len(valid)) * 100, 1),
    }


def main():
    print("=" * 70)
    print("PERTH MARLIN DATA ANALYSIS — New Catch & Absence Records")
    print("=" * 70)
    print("Scoring Accessible Trench Zone averages (no GPS — zone-wide analysis)")
    print()

    results = []

    # Score individual captures
    print("--- INDIVIDUAL CAPTURES ---")
    for c in CAPTURES:
        scores = score_date(c["date"])
        if scores:
            c.update(scores)
            results.append(c)
            print(f"  {c['date']} {c['boat']:15s} {c['species']:12s} "
                  f"zone_max={scores['zone_max']:5.1f}% mean={scores['zone_mean']:5.1f}% "
                  f"med={scores['zone_median']:5.1f}%")
        else:
            print(f"  {c['date']} {c['boat']:15s} -- NO DATA --")

    # Score absences
    print("\n--- ABSENCE DATES ---")
    for a in ABSENCES:
        scores = score_date(a["date"])
        if scores:
            a.update(scores)
            results.append(a)
            print(f"  {a['date']} {a['boat']:15s} "
                  f"zone_max={scores['zone_max']:5.1f}% mean={scores['zone_mean']:5.1f}% "
                  f"med={scores['zone_median']:5.1f}%  <-- NO MARLIN")
        else:
            print(f"  {a['date']} {a['boat']:15s} -- NO DATA --")

    # Score tournament dates
    print("\n--- TOURNAMENT DATES ---")
    for t in TOURNAMENTS:
        print(f"  {t['name']} ({t['fleet_raised']}R/{t['fleet_hooked']}H/{t['fleet_tagged']}T):")
        for d in t["dates"]:
            scores = score_date(d)
            if scores:
                rec = {"date": d, "tournament": t["name"], "type": t["type"]}
                rec.update(scores)
                results.append(rec)
                tag_rate = t["fleet_tagged"] / max(t["fleet_raised"], 1) * 100
                print(f"    {d}: zone_max={scores['zone_max']:5.1f}% "
                      f"mean={scores['zone_mean']:5.1f}% med={scores['zone_median']:5.1f}%")
            else:
                print(f"    {d}: -- NO DATA --")

    # Summary analysis
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    catch_scores = [r for r in results if r.get("type") in ("catch", "tournament_catch") and "zone_max" in r]
    absence_scores = [r for r in results if r.get("type") in ("absence",) and "zone_max" in r]
    low_scores = [r for r in results if r.get("type") in ("tournament_low",) and "zone_max" in r]

    if catch_scores:
        maxes = [r["zone_max"] for r in catch_scores]
        means = [r["zone_mean"] for r in catch_scores]
        print(f"\nCatch dates ({len(catch_scores)}):")
        print(f"  Zone max:  mean={np.mean(maxes):.1f}%, range={min(maxes):.1f}-{max(maxes):.1f}%")
        print(f"  Zone mean: mean={np.mean(means):.1f}%, range={min(means):.1f}-{max(means):.1f}%")

    if absence_scores:
        maxes = [r["zone_max"] for r in absence_scores]
        means = [r["zone_mean"] for r in absence_scores]
        print(f"\nAbsence dates ({len(absence_scores)}):")
        print(f"  Zone max:  mean={np.mean(maxes):.1f}%, range={min(maxes):.1f}-{max(maxes):.1f}%")
        print(f"  Zone mean: mean={np.mean(means):.1f}%, range={min(means):.1f}-{max(means):.1f}%")

    if low_scores:
        maxes = [r["zone_max"] for r in low_scores]
        means = [r["zone_mean"] for r in low_scores]
        print(f"\nLow-effort tournament dates ({len(low_scores)}):")
        print(f"  Zone max:  mean={np.mean(maxes):.1f}%, range={min(maxes):.1f}-{max(maxes):.1f}%")
        print(f"  Zone mean: mean={np.mean(means):.1f}%, range={min(means):.1f}-{max(means):.1f}%")

    if catch_scores and (absence_scores or low_scores):
        catch_mean = np.mean([r["zone_mean"] for r in catch_scores])
        neg = absence_scores + low_scores
        neg_mean = np.mean([r["zone_mean"] for r in neg])
        diff = catch_mean - neg_mean
        print(f"\nDiscrimination: catch mean {catch_mean:.1f}% vs negative mean {neg_mean:.1f}% "
              f"(delta = {diff:+.1f}%)")

    # Save results
    out_path = os.path.join(SCRIPT_DIR, "data", "new_catch_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
