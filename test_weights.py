"""Test weight configurations: catch vs control discrimination using cached data."""
import json, csv, numpy as np, os
import marlin_data
from marlin_data import generate_blue_marlin_hotspots, DEFAULT_BBOX

# ── Load catch locations from marlin_catches.geojson (71 GFAA tagged catches) ──
geojson = json.load(open("data/marlin_catches.geojson"))
catches = []
for f in geojson["features"]:
    p = f["properties"]
    coords = f["geometry"]["coordinates"]
    catches.append({"date": p["date"], "lat": coords[1], "lon": coords[0],
                    "species": p["species"], "tag": p.get("tag", "")})

catch_dates_set = set(c["date"] for c in catches)

# ── Build list of control dates from backtest (non-catch, peak season Jan-Apr) ──
backtest_dirs = []
for d in sorted(os.listdir("data/backtest")):
    path = f"data/backtest/{d}"
    if os.path.isdir(path) and os.path.exists(f"{path}/sst_raw.nc"):
        month = int(d[5:7])
        if 1 <= month <= 4:  # Peak season only for fair comparison
            backtest_dirs.append({"date": d, "dir": path})

# Remove any backtest dates that overlap with catch dates (within 3 days)
from datetime import datetime, timedelta
catch_dts = {datetime.strptime(d, "%Y-%m-%d") for d in catch_dates_set}

control_dates = []
for bd in backtest_dirs:
    bd_dt = datetime.strptime(bd["date"], "%Y-%m-%d")
    too_close = any(abs((bd_dt - cd).days) <= 3 for cd in catch_dts)
    if not too_close:
        control_dates.append(bd)

print(f"Catch dates with data: {len(set(c['date'] for c in catches))}")
print(f"Control dates (peak season, no catch): {len(control_dates)}")

# ── Zone sampling points (grid within Accessible Trench Zone) ──
# Zone: lon [114.98, 115.3333], lat [-32.1667, -31.7287]
ZONE_LAT_MIN, ZONE_LAT_MAX = -32.1667, -31.7287
ZONE_LON_MIN, ZONE_LON_MAX = 114.98, 115.3333

WEIGHT_CONFIGS = {
    "current": {
        "sst": 0.31, "ssh": 0.19, "current": 0.12, "sst_front": 0.09,
        "chl": 0.08, "mld": 0.08, "convergence": 0.04, "o2": 0.04,
        "clarity": 0.04, "sst_intrusion": 0.03
    },
    "v2_presence": {
        "sst": 0.25, "ssh": 0.15, "current": 0.12, "sst_front": 0.16,
        "chl": 0.04, "mld": 0.06, "convergence": 0.10, "o2": 0.00,
        "clarity": 0.02, "sst_intrusion": 0.10
    },
    "v3_front_heavy": {
        "sst": 0.22, "ssh": 0.15, "current": 0.10, "sst_front": 0.20,
        "chl": 0.03, "mld": 0.05, "convergence": 0.12, "o2": 0.00,
        "clarity": 0.01, "sst_intrusion": 0.12
    },
}


def zone_max_score(grid, lats, lons):
    """Get max score within the Accessible Trench Zone."""
    lat_mask = (lats >= ZONE_LAT_MIN) & (lats <= ZONE_LAT_MAX)
    lon_mask = (lons >= ZONE_LON_MIN) & (lons <= ZONE_LON_MAX)
    zone = grid[np.ix_(lat_mask, lon_mask)]
    valid = zone[~np.isnan(zone)]
    if len(valid) == 0:
        return np.nan
    return np.max(valid) * 100


def zone_mean_score(grid, lats, lons):
    """Get mean score within the Accessible Trench Zone."""
    lat_mask = (lats >= ZONE_LAT_MIN) & (lats <= ZONE_LAT_MAX)
    lon_mask = (lons >= ZONE_LON_MIN) & (lons <= ZONE_LON_MAX)
    zone = grid[np.ix_(lat_mask, lon_mask)]
    valid = zone[~np.isnan(zone)]
    if len(valid) == 0:
        return np.nan
    return np.mean(valid) * 100


def run_hotspots(out_dir):
    """Run hotspot scoring for a given data directory."""
    marlin_data.OUTPUT_DIR = out_dir
    tif_path = os.path.join(out_dir, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        tif_path = "data/bathy_gmrt.tif"
    if not os.path.exists(tif_path):
        tif_path = None
    return generate_blue_marlin_hotspots(DEFAULT_BBOX, tif_path=tif_path)


# ── Run each config ──
print("\n" + "=" * 80)

for config_name, weights in WEIGHT_CONFIGS.items():
    marlin_data.BLUE_MARLIN_WEIGHTS = weights

    # Score at catch GPS locations
    catch_point_scores = []
    catch_zone_maxes = []
    catch_zone_means = []
    catch_dates_processed = set()

    unique_catch_dates = sorted(set(c["date"] for c in catches))
    for date in unique_catch_dates:
        out_dir = f"data/{date}"
        if not os.path.exists(f"{out_dir}/sst_raw.nc"):
            continue
        try:
            result = run_hotspots(out_dir)
            if result is None:
                continue
            grid, lats, lons = result["grid"], result["lats"], result["lons"]
            catch_dates_processed.add(date)

            # Zone scores
            zm = zone_max_score(grid, lats, lons)
            zmn = zone_mean_score(grid, lats, lons)
            if not np.isnan(zm):
                catch_zone_maxes.append(zm)
            if not np.isnan(zmn):
                catch_zone_means.append(zmn)

            # Point scores at GPS coords
            for c in catches:
                if c["date"] != date:
                    continue
                lat_idx = np.argmin(np.abs(lats - c["lat"]))
                lon_idx = np.argmin(np.abs(lons - c["lon"]))
                score = grid[lat_idx, lon_idx]
                if not np.isnan(score):
                    catch_point_scores.append(score * 100)
        except Exception as e:
            print(f"  ERROR catch {date}: {e}")

    # Score control dates (zone-level only, no GPS points)
    control_zone_maxes = []
    control_zone_means = []
    for cd in control_dates:
        try:
            result = run_hotspots(cd["dir"])
            if result is None:
                continue
            grid, lats, lons = result["grid"], result["lats"], result["lons"]

            zm = zone_max_score(grid, lats, lons)
            zmn = zone_mean_score(grid, lats, lons)
            if not np.isnan(zm):
                control_zone_maxes.append(zm)
            if not np.isnan(zmn):
                control_zone_means.append(zmn)
        except Exception as e:
            print(f"  ERROR control {cd['date']}: {e}")

    # ── Report ──
    print(f"\n{'=' * 80}")
    print(f"CONFIG: {config_name}")
    print(f"{'=' * 80}")

    if catch_point_scores:
        arr = np.array(catch_point_scores)
        print(f"\n  CATCH POINT scores (n={len(arr)}, {len(catch_dates_processed)} dates):")
        print(f"    Mean: {np.mean(arr):.1f}%  Median: {np.median(arr):.1f}%  Min: {np.min(arr):.1f}%")
        print(f"    >=70%: {np.sum(arr>=70)/len(arr)*100:.0f}%  >=50%: {np.sum(arr>=50)/len(arr)*100:.0f}%")

    if catch_zone_maxes:
        czm = np.array(catch_zone_maxes)
        czmn = np.array(catch_zone_means)
        print(f"\n  CATCH ZONE scores ({len(czm)} dates):")
        print(f"    Zone Max:  mean={np.mean(czm):.1f}%  median={np.median(czm):.1f}%  min={np.min(czm):.1f}%")
        print(f"    Zone Mean: mean={np.mean(czmn):.1f}%  median={np.median(czmn):.1f}%  min={np.min(czmn):.1f}%")

    if control_zone_maxes:
        ctm = np.array(control_zone_maxes)
        ctmn = np.array(control_zone_means)
        print(f"\n  CONTROL ZONE scores ({len(ctm)} dates, peak season non-catch):")
        print(f"    Zone Max:  mean={np.mean(ctm):.1f}%  median={np.median(ctm):.1f}%  min={np.min(ctm):.1f}%")
        print(f"    Zone Mean: mean={np.mean(ctmn):.1f}%  median={np.median(ctmn):.1f}%  min={np.min(ctmn):.1f}%")

    # Discrimination metrics
    if catch_zone_maxes and control_zone_maxes:
        from scipy import stats
        diff_max = np.mean(czm) - np.mean(ctm)
        diff_mean = np.mean(czmn) - np.mean(ctmn)
        u_max, p_max = stats.mannwhitneyu(czm, ctm, alternative="greater")
        u_mean, p_mean = stats.mannwhitneyu(czmn, ctmn, alternative="greater")

        # Effect size (Cohen's d)
        pooled_std_max = np.sqrt((np.std(czm)**2 + np.std(ctm)**2) / 2)
        pooled_std_mean = np.sqrt((np.std(czmn)**2 + np.std(ctmn)**2) / 2)
        d_max = diff_max / pooled_std_max if pooled_std_max > 0 else 0
        d_mean = diff_mean / pooled_std_mean if pooled_std_mean > 0 else 0

        # AUC-ROC (probability catch > control)
        auc_max = u_max / (len(czm) * len(ctm))
        auc_mean = u_mean / (len(czmn) * len(ctmn))

        print(f"\n  DISCRIMINATION (catch vs control):")
        print(f"    Zone Max:  diff={diff_max:+.1f}%  p={p_max:.4f}  Cohen's d={d_max:.2f}  AUC={auc_max:.3f}")
        print(f"    Zone Mean: diff={diff_mean:+.1f}%  p={p_mean:.4f}  Cohen's d={d_mean:.2f}  AUC={auc_mean:.3f}")

    print(f"  Weights: {weights}")

# Reset OUTPUT_DIR
marlin_data.OUTPUT_DIR = "data"
