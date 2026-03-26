#!/usr/bin/env python3
"""
Automated optimizer for marlin hotspot scoring parameters.

Parallelized: each date scored in a separate process (16 workers).

Objective components:
  A) Accuracy — catches score high (mean, >=70%, min)
  B) Edge quality — gradient at catch + catch-to-peak ratio near 85-90%
  C) Discrimination — catch lift above ocean + percentile rank
  D) Selectivity — anti-inflation guards

14 weights + 8 edge + 28 geometry = 50 tunable params, 2000 trials.

Usage:
    python optimize_visual.py
"""

import csv
import io
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}

# Evaluation zone — tight box around catch cluster for discrimination checks.
EVAL_BBOX = {"lon_min": 114.82, "lon_max": 115.38, "lat_min": -32.31, "lat_max": -31.67}

# Number of parallel workers (leave headroom for OS + memory)
N_WORKERS = 16


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches(unique_only=False):
    """Load catches. If unique_only=True, exclude ALL catches at locations
    that appear on multiple dates (removes potential DDM-rounded duplicates)."""
    catches = []
    seen = set()

    if os.path.exists(CSV_PATH):
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
                        date_str = dt.strftime('%Y-%m-%d')
                        key = (date_str, round(lat, 4), round(lon, 4))
                        if key not in seen:
                            seen.add(key)
                            catches.append({'date': date_str, 'lat': lat, 'lon': lon,
                                            'tag': r.get('Tag_Number', '')})
                        break
                    except ValueError:
                        continue

    all_catches_path = os.path.join(SCRIPT_DIR, 'data', 'all_catches.csv')
    if os.path.exists(all_catches_path):
        with open(all_catches_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                species = r.get('species', '').upper()
                if 'BLUE MARLIN' not in species:
                    continue
                lat_raw = r.get('lat', '').strip()
                lon_raw = r.get('lon', '').strip()
                if not lat_raw or not lon_raw:
                    continue
                try:
                    lat = float(lat_raw)
                    lon = float(lon_raw)
                except ValueError:
                    continue
                d = r.get('date', '').strip()
                if 'T' in d:
                    d = d.split('T')[0]
                for fmt in ['%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(d, fmt)
                        date_str = dt.strftime('%Y-%m-%d')
                        key = (date_str, round(lat, 4), round(lon, 4))
                        if key not in seen:
                            seen.add(key)
                            catches.append({'date': date_str, 'lat': lat, 'lon': lon,
                                            'tag': r.get('tag', '')})
                        break
                    except ValueError:
                        continue

    if unique_only:
        # Count dates per location
        from collections import Counter
        loc_dates = {}
        for c in catches:
            loc_key = (round(c['lat'], 4), round(c['lon'], 4))
            if loc_key not in loc_dates:
                loc_dates[loc_key] = set()
            loc_dates[loc_key].add(c['date'])
        # Keep only locations that appear on exactly one date
        catches = [c for c in catches
                   if len(loc_dates[(round(c['lat'], 4), round(c['lon'], 4))]) == 1]

    return catches


def _score_one_date(args):
    """Worker function: score a single date with given params. Runs in separate process."""
    date_str, date_catches, params, script_dir, bbox, eval_bbox = args

    # Suppress all output in worker
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import marlin_data

        # Override ALL weights — normalized to 1.0
        marlin_data.BLUE_MARLIN_WEIGHTS['sst'] = params['w_sst']
        marlin_data.BLUE_MARLIN_WEIGHTS['sst_front'] = params['w_sst_front']
        marlin_data.BLUE_MARLIN_WEIGHTS['front_corridor'] = params['w_front_corridor']
        marlin_data.BLUE_MARLIN_WEIGHTS['chl'] = params['w_chl']
        marlin_data.BLUE_MARLIN_WEIGHTS['chl_curvature'] = params['w_chl_curvature']
        marlin_data.BLUE_MARLIN_WEIGHTS['ssh'] = params['w_ssh']
        marlin_data.BLUE_MARLIN_WEIGHTS['shelf_break'] = params['w_shelf']
        marlin_data.BLUE_MARLIN_WEIGHTS['current'] = params['w_current']
        marlin_data.BLUE_MARLIN_WEIGHTS['current_shear'] = params['w_shear']
        marlin_data.BLUE_MARLIN_WEIGHTS['upwelling_edge'] = params['w_upwell']
        marlin_data.BLUE_MARLIN_WEIGHTS['ftle'] = params['w_ftle']
        marlin_data.BLUE_MARLIN_WEIGHTS['vertical_velocity'] = params['w_vert_vel']
        marlin_data.BLUE_MARLIN_WEIGHTS['salinity_front'] = params['w_salinity_front']
        marlin_data.BLUE_MARLIN_WEIGHTS['okubo_weiss'] = params['w_okubo_weiss']
        # Dropped features — processing bugs or redundancy (see scoring_antipatterns.md)
        marlin_data.BLUE_MARLIN_WEIGHTS['mld'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['rugosity'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['sst_chl_bivariate'] = params.get('w_bivariate', 0)
        marlin_data.BLUE_MARLIN_WEIGHTS['thermocline_lift'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['stratification'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['convergence'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['sst_intrusion'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['ssta'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['clarity'] = 0
        marlin_data.BLUE_MARLIN_WEIGHTS['o2'] = 0

        # Override geometry params
        marlin_data._opt_shelf_boost = params.get('shelf_boost', 0.08)
        marlin_data._shelf_prox_blend = params['shelf_prox_blend']
        marlin_data._shelf_prox_sigma = params['shelf_prox_sigma']
        marlin_data._depth_shallow_full = params['depth_shallow_full']
        marlin_data._opt_pool_percentile = params['pool_pct']
        marlin_data._depth_zero_cut = params.get('depth_zero_cut', 50)
        marlin_data._depth_taper_start = params.get('depth_taper_start', 300)
        marlin_data._depth_taper_mid = params.get('depth_taper_mid', 800)
        marlin_data._depth_floor = params.get('depth_floor', 0.25)
        marlin_data._edge_shelf_sigma = params.get('edge_shelf_sigma', 2.5)
        marlin_data._shelf_prox_depth = params.get('shelf_prox_depth', 210)
        marlin_data._depth_shallow_floor = params.get('depth_shallow_floor', 0.75)
        marlin_data._opt_band_shore_ratio = params.get('band_shore_ratio', 0.20)
        marlin_data._opt_band_deep_ratio = params.get('band_deep_ratio', 0.20)
        marlin_data._opt_shallow_cut = params.get('shallow_cut', 0.55)
        marlin_data._opt_shear_depth_thresh = params.get('shear_depth_thresh', 70)
        marlin_data._opt_shear_depth_full = params.get('shear_depth_full', 125)
        marlin_data._opt_sst_optimal = params.get('sst_optimal', 22.5)
        marlin_data._opt_sst_sigma = params.get('sst_sigma', 2.25)
        marlin_data._opt_sst_sigma_above = params.get('sst_sigma_above', 2.0)
        marlin_data._opt_chl_threshold = params.get('chl_threshold', 0.14)
        marlin_data._opt_chl_sigma = params.get('chl_sigma', 0.20)
        marlin_data._opt_band_width_nm = params.get('band_width_nm', 2.5)
        marlin_data._opt_band_boost = params.get('band_boost', 0.45)
        marlin_data._opt_band_decay = params.get('band_decay', 0.60)
        marlin_data._opt_lunar_boost = params.get('lunar_boost', 0.02)
        marlin_data._opt_bathy_w_200 = params.get('bathy_w_200', 0.6)
        marlin_data._opt_bathy_w_500 = params.get('bathy_w_500', 0.1)
        marlin_data._opt_corridor_pct = params.get('corridor_pct', 75)
        marlin_data._edge_front_sigma = params.get('front_sigma', 2.0)

        # Per-feature edge scoring params (value-space Gaussian, 4 active features)
        for feat in ['okubo_weiss', 'upwelling_edge', 'current_shear', 'chl_curvature']:
            ec_key = f'{feat}_edge_center'
            ew_key = f'{feat}_edge_width'
            if ec_key in params:
                setattr(marlin_data, f'_opt_{feat}_edge_center', params[ec_key])
            if ew_key in params:
                setattr(marlin_data, f'_opt_{feat}_edge_width', params[ew_key])

        ddir = os.path.join(script_dir, 'data', date_str)
        if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
            return None

        marlin_data.OUTPUT_DIR = ddir
        tif = os.path.join(ddir, 'bathy_gmrt.tif')
        result = marlin_data.generate_blue_marlin_hotspots(
            bbox, tif_path=tif if os.path.exists(tif) else None,
            date_str=date_str)

        if result is None:
            return None

        grid = result['grid']
        lats, lons = result['lats'], result['lons']

        # Eval zone metrics
        lat_mask = (lats >= eval_bbox['lat_min']) & (lats <= eval_bbox['lat_max'])
        lon_mask = (lons >= eval_bbox['lon_min']) & (lons <= eval_bbox['lon_max'])
        eval_grid = grid[np.ix_(lat_mask, lon_mask)]
        eval_vals = eval_grid[~np.isnan(eval_grid)] * 100

        date_result = {
            'ocean_mean': float(np.mean(eval_vals)) if len(eval_vals) > 0 else None,
            'ocean_pct70': float(np.sum(eval_vals >= 70) / len(eval_vals) * 100) if len(eval_vals) > 0 else None,
            'top10_area': float(np.sum(eval_vals >= np.percentile(eval_vals, 90)) / len(eval_vals) * 100) if len(eval_vals) > 0 else None,
            'catches': [],
        }

        gy, gx = np.gradient(grid)

        for c in date_catches:
            li = np.argmin(np.abs(lats - c['lat']))
            lo = np.argmin(np.abs(lons - c['lon']))
            val = grid[li, lo]
            if np.isnan(val):
                continue

            catch_score = float(val) * 100
            catch_data = {'score': catch_score, 'lat': c['lat'], 'lon': c['lon']}

            # Percentile rank
            if len(eval_vals) > 0:
                catch_data['percentile'] = float(np.sum(eval_vals < catch_score) / len(eval_vals) * 100)

            # Gradient
            grad_mag = float(np.sqrt(gx[li, lo]**2 + gy[li, lo]**2)) * 100
            catch_data['gradient'] = grad_mag

            # Peak ratio (10nm radius)
            dlat_10nm = 10.0 / 60.0
            dlon_10nm = 10.0 / (60.0 * np.cos(np.radians(c['lat'])))
            pk_lat_mask = (lats >= c['lat'] - dlat_10nm) & (lats <= c['lat'] + dlat_10nm)
            pk_lon_mask = (lons >= c['lon'] - dlon_10nm) & (lons <= c['lon'] + dlon_10nm)
            pk_lat_idx = np.where(pk_lat_mask)[0]
            pk_lon_idx = np.where(pk_lon_mask)[0]
            if len(pk_lat_idx) > 0 and len(pk_lon_idx) > 0:
                local_box = grid[np.ix_(pk_lat_idx, pk_lon_idx)]
                local_peak = float(np.nanmax(local_box)) * 100
                if local_peak > 0:
                    catch_data['peak_ratio'] = catch_score / local_peak

            date_result['catches'].append(catch_data)

        return date_result

    except Exception as e:
        # Log error to file for debugging (stdout is /dev/null)
        try:
            import traceback
            with open(os.path.join(script_dir, "optuna_worker_error.log"), "a") as ef:
                ef.write(f"{date_str}: {e}\n")
                traceback.print_exc(file=ef)
                ef.write("\n")
        except Exception:
            pass
        return None


def run_trial(catches, params):
    """Run scoring with given params using parallel workers, return metrics."""
    from collections import defaultdict
    by_date = defaultdict(list)
    for c in catches:
        by_date[c['date']].append(c)

    # Build work items
    work_items = []
    for date_str in sorted(by_date.keys()):
        work_items.append((date_str, by_date[date_str], params,
                           SCRIPT_DIR, BBOX, EVAL_BBOX))

    # Run in parallel
    results = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_score_one_date, item): item[0] for item in work_items}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    if not results:
        return None

    # Aggregate — deduplicate by location so clusters don't dominate.
    # Each unique GPS coordinate contributes ONE score (mean across dates).
    # This prevents Optuna from getting 8x reward for scoring PGFC well.
    catch_scores = []
    catch_percentiles = []
    catch_gradients = []
    catch_peak_ratios = []
    ocean_means = []
    ocean_pct70 = []
    ocean_top10_areas = []

    # Collect per-catch data keyed by location
    from collections import defaultdict
    loc_scores = defaultdict(list)
    loc_percentiles = defaultdict(list)
    loc_gradients = defaultdict(list)
    loc_peak_ratios = defaultdict(list)

    for dr in results:
        if dr['ocean_mean'] is not None:
            ocean_means.append(dr['ocean_mean'])
            ocean_pct70.append(dr['ocean_pct70'])
            ocean_top10_areas.append(dr['top10_area'])

        for cd in dr['catches']:
            # Use rounded coords as location key (matches DDM precision)
            loc_key = (round(cd.get('lat', 0), 4), round(cd.get('lon', 0), 4))
            loc_scores[loc_key].append(cd['score'])
            if 'percentile' in cd:
                loc_percentiles[loc_key].append(cd['percentile'])
            if 'gradient' in cd:
                loc_gradients[loc_key].append(cd['gradient'])
            if 'peak_ratio' in cd:
                loc_peak_ratios[loc_key].append(cd['peak_ratio'])

    # Average each location's scores across dates, then aggregate
    for loc_key, scores in loc_scores.items():
        catch_scores.append(np.mean(scores))
        if loc_key in loc_percentiles:
            catch_percentiles.append(np.mean(loc_percentiles[loc_key]))
        if loc_key in loc_gradients:
            catch_gradients.append(np.mean(loc_gradients[loc_key]))
        if loc_key in loc_peak_ratios:
            catch_peak_ratios.append(np.mean(loc_peak_ratios[loc_key]))

    if not catch_scores:
        return None

    mean_score = np.mean(catch_scores)
    median_score = np.median(catch_scores)
    pct_70 = sum(1 for s in catch_scores if s >= 70) / len(catch_scores) * 100
    pct_80 = sum(1 for s in catch_scores if s >= 80) / len(catch_scores) * 100
    min_score = min(catch_scores)
    max_score = max(catch_scores)

    avg_ocean_mean = np.mean(ocean_means) if ocean_means else 50
    avg_ocean_pct70 = np.mean(ocean_pct70) if ocean_pct70 else 30
    catch_lift = mean_score - avg_ocean_mean
    avg_gradient = np.mean(catch_gradients) if catch_gradients else 0
    avg_percentile = np.mean(catch_percentiles) if catch_percentiles else 50
    avg_top10_area = np.mean(ocean_top10_areas) if ocean_top10_areas else 10
    avg_peak_ratio = np.mean(catch_peak_ratios) if catch_peak_ratios else 0.85

    # ===================================================================
    # COMPOSITE OBJECTIVE — edge-aligned scoring
    # ===================================================================

    # A) ACCURACY
    accuracy = 0.10 * mean_score + 0.05 * pct_70 + 0.03 * min_score

    # B) EDGE QUALITY
    gradient_component = 0.80 * min(avg_gradient, 10)
    if avg_peak_ratio < 0.82:
        ratio_penalty = 15.0 * (0.82 - avg_peak_ratio)
    elif avg_peak_ratio > 0.92:
        ratio_penalty = 10.0 * (avg_peak_ratio - 0.92)
    else:
        ratio_penalty = 0
    edge_quality = gradient_component - ratio_penalty

    # C) DISCRIMINATION
    lift_component = 0.30 * min(catch_lift, 30)
    percentile_component = 0.05 * avg_percentile
    discrimination = lift_component + percentile_component

    # D) SELECTIVITY — penalize inflated ocean scoring / large zones
    selectivity_penalty = 0
    if avg_ocean_mean > 55:
        selectivity_penalty += 2.0 * (avg_ocean_mean - 55)
    if avg_ocean_pct70 > 40:
        selectivity_penalty += 1.5 * (avg_ocean_pct70 - 40)
    if avg_top10_area > 12:
        selectivity_penalty += 1.0 * (avg_top10_area - 12)

    objective = accuracy + edge_quality + discrimination - selectivity_penalty

    return {
        'mean': mean_score, 'median': median_score, 'pct_70': pct_70,
        'pct_80': pct_80, 'min': min_score, 'max': max_score,
        'n': len(catch_scores), 'objective': objective,
        'ocean_mean': avg_ocean_mean, 'ocean_pct70': avg_ocean_pct70,
        'catch_lift': catch_lift,
        'avg_gradient': avg_gradient,
        'avg_percentile': avg_percentile, 'avg_top10_area': avg_top10_area,
        'avg_peak_ratio': avg_peak_ratio,
    }


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    catches = load_catches(unique_only=True)
    print(f"Loaded {len(catches)} catches (unique locations only, duplicates excluded)")
    print(f"Using {N_WORKERS} parallel workers on {os.cpu_count()} cores")

    best_result = {'objective': -999}
    trial_count = [0]
    t0 = time.time()

    def objective(trial):
        # Feature weights (normalized to sum=1.0 after sampling)
        w_sst = trial.suggest_float('w_sst', 0.04, 0.35, step=0.01)
        w_sst_front = trial.suggest_float('w_sst_front', 0.00, 0.12, step=0.01)
        w_front_corridor = trial.suggest_float('w_front_corridor', 0.00, 0.15, step=0.01)
        w_chl = trial.suggest_float('w_chl', 0.02, 0.20, step=0.01)
        w_chl_curvature = trial.suggest_float('w_chl_curvature', 0.00, 0.25, step=0.01)
        w_ssh = trial.suggest_float('w_ssh', 0.00, 0.25, step=0.01)
        w_shelf = trial.suggest_float('w_shelf', 0.00, 0.18, step=0.01)
        w_current = trial.suggest_float('w_current', 0.00, 0.08, step=0.01)
        w_shear = trial.suggest_float('w_shear', 0.00, 0.18, step=0.01)
        w_upwell = trial.suggest_float('w_upwell', 0.00, 0.08, step=0.01)
        w_ftle = trial.suggest_float('w_ftle', 0.00, 0.15, step=0.01)
        w_vert_vel = trial.suggest_float('w_vert_vel', 0.00, 0.08, step=0.01)
        w_salinity_front = trial.suggest_float('w_salinity_front', 0.00, 0.20, step=0.01)
        w_okubo_weiss = trial.suggest_float('w_okubo_weiss', 0.00, 0.12, step=0.01)
        w_bivariate = trial.suggest_float('w_bivariate', 0.00, 0.20, step=0.01)

        # Normalize to sum to 1.0
        all_weights = [w_sst, w_sst_front, w_front_corridor, w_chl, w_chl_curvature,
                       w_ssh, w_shelf, w_current, w_shear, w_upwell,
                       w_ftle, w_vert_vel, w_salinity_front, w_okubo_weiss, w_bivariate]
        total_w = sum(all_weights)
        if total_w == 0:
            return -999
        scale = 1.0 / total_w
        w_sst *= scale; w_sst_front *= scale; w_front_corridor *= scale
        w_chl *= scale; w_chl_curvature *= scale; w_ssh *= scale
        w_shelf *= scale; w_current *= scale; w_shear *= scale
        w_upwell *= scale; w_ftle *= scale; w_vert_vel *= scale
        w_salinity_front *= scale; w_okubo_weiss *= scale; w_bivariate *= scale

        # Geometry params
        shelf_prox_blend = trial.suggest_float('shelf_prox_blend', 0.0, 0.8, step=0.1)
        shelf_prox_sigma = trial.suggest_float('shelf_prox_sigma', 30, 150, step=10)
        depth_shallow_full = trial.suggest_int('depth_shallow_full', 80, 200, step=20)
        pool_pct = trial.suggest_int('pool_pct', 65, 100, step=5)

        # Depth gate params
        depth_zero_cut = trial.suggest_int('depth_zero_cut', 30, 80, step=10)
        depth_shallow_floor = trial.suggest_float('depth_shallow_floor', 0.50, 0.95, step=0.05)
        depth_taper_start = trial.suggest_int('depth_taper_start', 500, 3000, step=250)
        depth_taper_mid = trial.suggest_int('depth_taper_mid', 1500, 5000, step=500)
        depth_floor = trial.suggest_float('depth_floor', 0.60, 0.95, step=0.05)

        # Shelf proximity
        shelf_prox_depth = trial.suggest_int('shelf_prox_depth', 100, 300, step=10)
        edge_shelf_sigma = trial.suggest_float('edge_shelf_sigma', 1.0, 8.0, step=0.5)

        # SST / CHL core params
        sst_optimal = trial.suggest_float('sst_optimal', 21.0, 25.0, step=0.25)
        sst_sigma = trial.suggest_float('sst_sigma', 1.0, 4.0, step=0.25)
        sst_sigma_above = trial.suggest_float('sst_sigma_above', 1.5, 5.0, step=0.5)
        chl_threshold = trial.suggest_float('chl_threshold', 0.06, 0.30, step=0.02)
        chl_sigma = trial.suggest_float('chl_sigma', 0.05, 0.50, step=0.05)

        # Band system
        band_width_nm = trial.suggest_float('band_width_nm', 1.0, 6.0, step=0.5)
        band_boost = trial.suggest_float('band_boost', 0.05, 0.60, step=0.05)
        band_decay = trial.suggest_float('band_decay', 0.20, 0.90, step=0.05)

        # Bathy band tolerances
        band_shore_ratio = trial.suggest_float('band_shore_ratio', 0.05, 0.40, step=0.05)
        band_deep_ratio = trial.suggest_float('band_deep_ratio', 0.05, 0.40, step=0.05)
        shallow_cut = trial.suggest_float('shallow_cut', 0.25, 0.70, step=0.05)

        # Shear depth ramp (thresh must be < full for correct ramp direction)
        shear_depth_thresh = trial.suggest_int('shear_depth_thresh', 40, 120, step=10)
        shear_depth_full = trial.suggest_int('shear_depth_full', 150, 350, step=25)
        if shear_depth_thresh >= shear_depth_full:
            shear_depth_full = shear_depth_thresh + 50

        # Lunar & bathy weights
        lunar_boost = trial.suggest_float('lunar_boost', 0.0, 0.15, step=0.02)
        bathy_w_200 = trial.suggest_float('bathy_w_200', 0.2, 1.0, step=0.1)
        bathy_w_500 = trial.suggest_float('bathy_w_500', 0.0, 0.8, step=0.1)

        # Front corridor percentile threshold + SST front sigma
        corridor_pct = trial.suggest_int('corridor_pct', 50, 95, step=5)
        front_sigma = trial.suggest_float('front_sigma', 0.5, 4.0, step=0.5)

        # Per-feature edge scoring (catch raw medians: shear=0.55, chl_curv=0.88, OW=1.00, upwell=0.76)
        edge_params = {}
        for feat, (c_lo, c_hi, w_lo, w_hi) in {
            'okubo_weiss':   (0.30, 1.00, 0.15, 0.80),
            'upwelling_edge':(0.15, 0.90, 0.10, 0.60),
            'current_shear': (0.15, 0.80, 0.10, 0.60),
            'chl_curvature': (0.30, 1.00, 0.15, 0.80),
        }.items():
            ec = trial.suggest_float(f'{feat}_edge_center', c_lo, c_hi, step=0.05)
            ew = trial.suggest_float(f'{feat}_edge_width', w_lo, w_hi, step=0.05)
            edge_params[f'{feat}_edge_center'] = ec
            edge_params[f'{feat}_edge_width'] = ew

        params = {
            'w_sst': w_sst, 'w_sst_front': w_sst_front,
            'w_front_corridor': w_front_corridor,
            'w_chl': w_chl, 'w_chl_curvature': w_chl_curvature,
            'w_ssh': w_ssh, 'w_shelf': w_shelf, 'w_current': w_current,
            'w_shear': w_shear, 'w_upwell': w_upwell,
            'w_ftle': w_ftle, 'w_vert_vel': w_vert_vel,
            'w_salinity_front': w_salinity_front,
            'w_okubo_weiss': w_okubo_weiss,
            'w_bivariate': w_bivariate,
            'shelf_prox_blend': shelf_prox_blend,
            'shelf_prox_sigma': shelf_prox_sigma,
            'depth_shallow_full': depth_shallow_full,
            'pool_pct': pool_pct,
            # Depth gate
            'depth_zero_cut': depth_zero_cut,
            'depth_shallow_floor': depth_shallow_floor,
            'depth_taper_start': depth_taper_start,
            'depth_taper_mid': depth_taper_mid,
            'depth_floor': depth_floor,
            # Shelf proximity
            'shelf_prox_depth': shelf_prox_depth,
            'edge_shelf_sigma': edge_shelf_sigma,
            # SST / CHL
            'sst_optimal': sst_optimal,
            'sst_sigma': sst_sigma,
            'sst_sigma_above': sst_sigma_above,
            'chl_threshold': chl_threshold,
            'chl_sigma': chl_sigma,
            # Band system
            'band_width_nm': band_width_nm,
            'band_boost': band_boost,
            'band_decay': band_decay,
            # Bathy bands
            'band_shore_ratio': band_shore_ratio,
            'band_deep_ratio': band_deep_ratio,
            'shallow_cut': shallow_cut,
            # Shear
            'shear_depth_thresh': shear_depth_thresh,
            'shear_depth_full': shear_depth_full,
            # Lunar & bathy
            'lunar_boost': lunar_boost,
            'bathy_w_200': bathy_w_200,
            'bathy_w_500': bathy_w_500,
            # Front corridor + sigma
            'corridor_pct': corridor_pct,
            'front_sigma': front_sigma,
            # Edge params
            **edge_params,
        }

        try:
            r = run_trial(catches, params)
        except Exception as e:
            trial_count[0] += 1
            print(f"  Trial {trial_count[0]}: EXCEPTION in run_trial: {e}")
            import traceback
            traceback.print_exc()
            return -999

        trial_count[0] += 1
        if r is None:
            print(f"  Trial {trial_count[0]}: FAILED (run_trial returned None)")
            return -999

        elapsed = time.time() - t0
        is_best = r['objective'] > best_result.get('objective', -999)
        marker = " ** BEST **" if is_best else ""
        if is_best:
            best_result.update(r)
            best_result['params'] = dict(params)

        print(f"  T{trial_count[0]:3d} [{elapsed:5.0f}s] "
              f"mean={r['mean']:.1f} >=70={r['pct_70']:.0f}% min={r['min']:.0f} "
              f"| ocean={r['ocean_mean']:.0f}% lift={r['catch_lift']:.0f} "
              f"grad={r['avg_gradient']:.1f} ratio={r['avg_peak_ratio']:.3f} "
              f"pctl={r['avg_percentile']:.0f} obj={r['objective']:.1f}{marker}")

        return r['objective']

    # --- Baseline ---
    print("\n--- Baseline (current settings) ---")
    baseline_params = {
        # Baseline: v18 best params
        'w_sst': 0.225, 'w_sst_front': 0.00, 'w_front_corridor': 0.023,
        'w_chl': 0.116, 'w_chl_curvature': 0.109,
        'w_ssh': 0.163, 'w_shelf': 0.00, 'w_current': 0.00,
        'w_shear': 0.124, 'w_upwell': 0.116,
        'w_ftle': 0.03, 'w_vert_vel': 0.00,
        'w_salinity_front': 0.047, 'w_okubo_weiss': 0.078, 'w_bivariate': 0.00,
        # Geometry
        'shelf_prox_blend': 0.70, 'shelf_prox_sigma': 90,
        'depth_shallow_full': 100, 'pool_pct': 65,
        # Depth gate
        'depth_zero_cut': 50, 'depth_shallow_floor': 0.50, 'depth_taper_start': 2000,
        'depth_taper_mid': 4000, 'depth_floor': 0.85,
        # Shelf proximity
        'shelf_prox_depth': 140, 'edge_shelf_sigma': 3.5,
        # SST / CHL
        'sst_optimal': 22.0, 'sst_sigma': 1.0,
        'sst_sigma_above': 4.0,
        'chl_threshold': 0.18, 'chl_sigma': 0.50,
        # Band system
        'band_width_nm': 3.0, 'band_boost': 0.30,
        'band_decay': 0.60,
        # Bathy bands
        'band_shore_ratio': 0.10, 'band_deep_ratio': 0.05,
        'shallow_cut': 0.70,
        # Shear
        'shear_depth_thresh': 40, 'shear_depth_full': 280,
        # Lunar & bathy
        'lunar_boost': 0.08,
        'bathy_w_200': 0.2, 'bathy_w_500': 0.4,
        # Front corridor + sigma
        'corridor_pct': 95, 'front_sigma': 4.0,
        # Edge scoring
        'okubo_weiss_edge_center': 0.95, 'okubo_weiss_edge_width': 0.60,
        'upwelling_edge_edge_center': 0.45, 'upwelling_edge_edge_width': 0.55,
        'current_shear_edge_center': 0.65, 'current_shear_edge_width': 0.50,
        'chl_curvature_edge_center': 0.50, 'chl_curvature_edge_width': 0.80,
    }
    baseline = run_trial(catches, baseline_params)

    if baseline:
        print(f"  Baseline: mean={baseline['mean']:.1f}% >=70%={baseline['pct_70']:.0f}% "
              f"min={baseline['min']:.0f}% | ocean={baseline['ocean_mean']:.0f}% "
              f"lift={baseline['catch_lift']:.0f} grad={baseline['avg_gradient']:.1f} "
              f"ratio={baseline['avg_peak_ratio']:.3f} "
              f"pctl={baseline['avg_percentile']:.0f} obj={baseline['objective']:.1f}")
        best_result.update(baseline)
        best_result['params'] = dict(baseline_params)

    n_trials = 400
    print(f"\n{'='*90}")
    print(f"OPTUNA -- {n_trials} trials ({N_WORKERS} workers), 50 params, location-deduplicated")
    print(f"{'='*90}")

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))

    study.enqueue_trial(dict(baseline_params))

    study.optimize(objective, n_trials=n_trials)

    # --- Results ---
    print(f"\n{'='*90}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*90}")
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  Objective: {study.best_value:.1f}")

    print(f"\nBest parameters:")
    bp = best_result.get('params', {})
    for k, v in sorted(bp.items()):
        if isinstance(v, float):
            print(f"  {k:<25s} = {v:.4f}")
        else:
            print(f"  {k:<25s} = {v}")

    print(f"\nBest metrics:")
    metric_keys = ['mean', 'median', 'pct_70', 'pct_80', 'min', 'max', 'n',
                   'ocean_mean', 'ocean_pct70', 'catch_lift',
                   'avg_gradient', 'avg_percentile',
                   'avg_top10_area', 'avg_peak_ratio', 'objective']
    for k in metric_keys:
        if k in best_result:
            v = best_result[k]
            print(f"  {k:<18s} = {v:.3f}" if isinstance(v, float) else f"  {k:<18s} = {v}")

    if baseline:
        print(f"\n--- Improvement over baseline ---")
        def _cmp(key, fmt=".1f"):
            bv = baseline.get(key, 0)
            nv = best_result.get(key, 0)
            delta = nv - bv
            return f"  {key:<18s}: {bv:{fmt}} -> {nv:{fmt}}  ({delta:+{fmt}})"
        for k in ['mean', 'pct_70', 'min', 'ocean_mean', 'catch_lift',
                   'avg_percentile', 'avg_gradient',
                   'avg_peak_ratio', 'objective']:
            print(_cmp(k, '.3f' if k == 'avg_peak_ratio' else '.1f'))

    # Save
    out = {
        'best_params': bp,
        'best_metrics': {k: best_result[k] for k in metric_keys if k in best_result},
        'baseline_metrics': {k: baseline[k] for k in metric_keys if k in baseline} if baseline else None,
        'n_trials': n_trials,
        'n_workers': N_WORKERS,
        'objective_components': [
            'A) accuracy: mean(10%) + pct_70(5%) + min(3%)',
            'B) edge_quality: gradient(0.8x, cap 10) + peak_ratio_penalty([0.82-0.92] sweet spot)',
            'C) discrimination: catch_lift(0.3x, cap 30) + percentile(0.05x)',
            'D) selectivity: -penalty(ocean_mean>55@2x, ocean_pct70>40@1.5x, top10>12@1x)',
        ],
    }
    out_path = os.path.join(SCRIPT_DIR, 'data', 'visual_optimization_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
