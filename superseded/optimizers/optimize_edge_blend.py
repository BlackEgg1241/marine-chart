#!/usr/bin/env python3
"""
Optimize the blend between peak-based and edge-based scoring.

Sweeps key parameters that control how much we reward feature EDGES vs PEAKS,
running validation for each combination to find the optimal mix.

Parameters optimized:
  1. mld_edge_blend: 0.0 (pure value) to 1.0 (pure edge) — currently 0.4
  2. current_edge_blend: 0.0 to 1.0 — currently 0.4
  3. shelf_widen_sigma: 0.0 (no widen) to 4.0 — currently 2.0
  4. front_widen_sigma: 0.0 to 4.0 — currently 2.0
  5. conv_edge_blend: 0.0 to 1.0 — currently 0.7
  6. upwell_widen_sigma: 0.0 to 4.0 — currently 2.5
  7. zero_band_mult: 0.50 to 0.90 — currently 0.75
  8. one_band_mult: 0.75 to 1.00 — currently 0.90
  9. edge_boost_strength: 0.0 to 0.10 per feature — currently 0.05
"""

import csv
import os
import sys
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches():
    catches = []
    with open(CSV_PATH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if 'BLUE MARLIN' not in r.get('Species_Name', '').upper():
                continue
            lat_raw = r.get('Latitude', '').strip()
            lon_raw = r.get('Longitude', '').strip()
            if not lat_raw or not lon_raw:
                continue
            lat_val = lat_raw.rstrip('SsNn')
            lon_val = lon_raw.rstrip('EeWw')
            lat = ddm_to_dd(lat_val, negative='S' in lat_raw.upper())
            lon = ddm_to_dd(lon_val, negative='W' in lon_raw.upper())
            d = r.get('Release_Date', '').strip()
            if 'T' in d:
                d = d.split('T')[0]
            for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                try:
                    dt = datetime.strptime(d, fmt)
                    catches.append({'date': dt.strftime('%Y-%m-%d'), 'lat': lat, 'lon': lon,
                                    'tag': r.get('Tag_Number', '')})
                    break
                except ValueError:
                    continue
    return catches


def run_trial(catches, params):
    """Run scoring with given params, return validation metrics."""
    import importlib
    import marlin_data
    importlib.reload(marlin_data)

    # Set parameters via module attributes
    marlin_data._edge_mld_blend = params['mld_edge_blend']
    marlin_data._edge_current_blend = params['current_edge_blend']
    marlin_data._edge_shelf_sigma = params['shelf_widen_sigma']
    marlin_data._edge_front_sigma = params['front_widen_sigma']
    marlin_data._edge_conv_blend = params['conv_edge_blend']
    marlin_data._edge_upwell_sigma = params['upwell_widen_sigma']
    marlin_data._opt_zero_band_mult = params['zero_band_mult']
    marlin_data._opt_one_band_mult = params['one_band_mult']
    marlin_data._edge_boost_strength = params['edge_boost_strength']

    from collections import defaultdict
    by_date = defaultdict(list)
    for c in catches:
        by_date[c['date']].append(c)

    scores = []
    for date_str in sorted(by_date.keys()):
        ddir = os.path.join(SCRIPT_DIR, 'data', date_str)
        if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
            continue

        marlin_data.OUTPUT_DIR = ddir
        tif = os.path.join(ddir, 'bathy_gmrt.tif')
        try:
            result = marlin_data.generate_blue_marlin_hotspots(
                BBOX, tif_path=tif if os.path.exists(tif) else None,
                date_str=date_str)
        except Exception:
            continue

        if result is None:
            continue

        grid = result['grid']
        lats, lons = result['lats'], result['lons']

        for c in by_date[date_str]:
            li = np.argmin(np.abs(lats - c['lat']))
            lo = np.argmin(np.abs(lons - c['lon']))
            val = grid[li, lo]
            if not np.isnan(val):
                scores.append(float(val) * 100)

    if not scores:
        return None

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    pct_70 = sum(1 for s in scores if s >= 70) / len(scores) * 100
    min_score = min(scores)
    max_score = max(scores)

    # Composite objective: weighted combination favoring catch coverage
    # 40% mean + 30% >=70% coverage + 20% min score + 10% max score
    objective = 0.40 * mean_score + 0.30 * pct_70 + 0.20 * min_score + 0.10 * max_score

    return {
        'mean': mean_score, 'median': median_score, 'pct_70': pct_70,
        'min': min_score, 'max': max_score, 'n': len(scores),
        'objective': objective,
    }


def main():
    catches = load_catches()
    print(f"Loaded {len(catches)} catches")

    # Current settings (baseline)
    current = {
        'mld_edge_blend': 0.4,
        'current_edge_blend': 0.4,
        'shelf_widen_sigma': 2.0,
        'front_widen_sigma': 2.0,
        'conv_edge_blend': 0.7,
        'upwell_widen_sigma': 2.5,
        'zero_band_mult': 0.75,
        'one_band_mult': 0.90,
        'edge_boost_strength': 0.05,
    }

    # Old settings (pure peak-based)
    old = {
        'mld_edge_blend': 0.0,
        'current_edge_blend': 0.0,
        'shelf_widen_sigma': 0.0,
        'front_widen_sigma': 0.0,
        'conv_edge_blend': 0.0,
        'upwell_widen_sigma': 0.0,
        'zero_band_mult': 0.65,
        'one_band_mult': 0.85,
        'edge_boost_strength': 0.0,
    }

    # Sweep: blend from old (0.0) to current (1.0) and beyond (1.2)
    blends = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3]

    print(f"\n{'='*100}")
    print(f"PHASE 1: GLOBAL BLEND SWEEP (old peak-based <-> new edge-based)")
    print(f"{'='*100}")
    print(f"{'Blend':>6s} {'Mean':>5s} {'Med':>5s} {'>=70%':>6s} {'Min':>5s} {'Max':>5s} {'Obj':>6s} {'N':>3s}")
    print('-' * 50)

    best_blend = 0
    best_obj = 0
    results = []

    for blend in blends:
        params = {}
        for key in current:
            params[key] = old[key] + blend * (current[key] - old[key])

        print(f"  Blend {blend:.2f} ...", end=" ", flush=True)
        r = run_trial(catches, params)
        if r is None:
            print("FAILED")
            continue

        results.append((blend, r))
        marker = " <-- BEST" if r['objective'] > best_obj else ""
        if r['objective'] > best_obj:
            best_obj = r['objective']
            best_blend = blend
        print(f"mean={r['mean']:.1f}% >=70%={r['pct_70']:.0f}% min={r['min']:.0f}% obj={r['objective']:.1f}{marker}")

    print(f"\nBest global blend: {best_blend:.2f} (objective={best_obj:.1f})")

    # Phase 2: Per-parameter sweep around the best blend
    print(f"\n{'='*100}")
    print(f"PHASE 2: PER-PARAMETER SWEEP (around best global blend {best_blend:.2f})")
    print(f"{'='*100}")

    # Start from the best global blend
    best_params = {}
    for key in current:
        best_params[key] = old[key] + best_blend * (current[key] - old[key])

    param_ranges = {
        'mld_edge_blend':      [0.0, 0.2, 0.4, 0.6, 0.8],
        'current_edge_blend':  [0.0, 0.2, 0.4, 0.6, 0.8],
        'shelf_widen_sigma':   [0.0, 1.0, 2.0, 3.0, 4.0],
        'front_widen_sigma':   [0.0, 1.0, 2.0, 3.0, 4.0],
        'conv_edge_blend':     [0.0, 0.3, 0.5, 0.7, 0.9],
        'upwell_widen_sigma':  [0.0, 1.0, 2.0, 3.0, 4.0],
        'zero_band_mult':      [0.55, 0.65, 0.75, 0.85],
        'one_band_mult':       [0.80, 0.85, 0.90, 0.95],
        'edge_boost_strength': [0.0, 0.03, 0.05, 0.07, 0.10],
    }

    for param_name, values in param_ranges.items():
        print(f"\n  Sweeping {param_name}:")
        param_best_val = best_params[param_name]
        param_best_obj = 0

        for val in values:
            trial_params = dict(best_params)
            trial_params[param_name] = val
            r = run_trial(catches, trial_params)
            if r is None:
                continue
            marker = ""
            if r['objective'] > param_best_obj:
                param_best_obj = r['objective']
                param_best_val = val
                marker = " <-- BEST"
            print(f"    {param_name}={val:.2f}: mean={r['mean']:.1f}% >=70%={r['pct_70']:.0f}% "
                  f"min={r['min']:.0f}% obj={r['objective']:.1f}{marker}", flush=True)

        best_params[param_name] = param_best_val
        print(f"    -> Best: {param_name}={param_best_val:.2f}")

    # Final validation with optimal params
    print(f"\n{'='*100}")
    print(f"FINAL OPTIMAL PARAMETERS")
    print(f"{'='*100}")
    for key, val in best_params.items():
        old_val = old[key]
        cur_val = current[key]
        print(f"  {key:<25s} = {val:.3f}  (old={old_val:.3f}, current={cur_val:.3f})")

    print(f"\nFinal validation with optimal params:")
    r = run_trial(catches, best_params)
    if r:
        print(f"  Mean: {r['mean']:.1f}%")
        print(f"  Median: {r['median']:.1f}%")
        print(f"  >= 70%: {r['pct_70']:.0f}%")
        print(f"  Min: {r['min']:.0f}%")
        print(f"  Max: {r['max']:.0f}%")
        print(f"  Objective: {r['objective']:.1f}")
        print(f"  N: {r['n']}")

    # Save results
    import json
    out = {
        'optimal_params': best_params,
        'global_blend_results': [(b, r) for b, r in results],
        'final_metrics': r,
    }
    with open('data/edge_blend_optimization.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to data/edge_blend_optimization.json")


if __name__ == "__main__":
    main()
