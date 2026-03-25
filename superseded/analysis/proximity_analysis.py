#!/usr/bin/env python3
"""
proximity_analysis.py — Analyze proximity of historical marlin catches
to oceanographic features on their catch dates.

Computes minimum distances (nm) from each catch to SST fronts, isotherms,
CHL edges, MLD contours, SSH eddies, water clarity boundaries, and samples
hotspot intensity at catch locations. Compares against random control points.
"""

import json
import math
import os
import random
import sys
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats
from shapely.geometry import Point, LineString, shape, MultiLineString

# ── Constants ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Fishable/accessible trench zone
ZONE_LON_MIN, ZONE_LON_MAX = 114.98, 115.3333
ZONE_LAT_MIN, ZONE_LAT_MAX = -32.1667, -31.7287

# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065

# Number of control points per catch date
N_CONTROLS = 10

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ── Haversine distance ────────────────────────────────────────────────
def haversine_nm(lon1, lat1, lon2, lat2):
    """Haversine distance in nautical miles."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))


def point_to_linestring_min_dist_nm(px, py, coords):
    """Minimum distance from point (px,py) to a polyline given as list of [lon,lat]."""
    min_d = float("inf")
    for i in range(len(coords) - 1):
        d = point_to_segment_nm(px, py, coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        if d < min_d:
            min_d = d
    if len(coords) == 1:
        min_d = haversine_nm(px, py, coords[0][0], coords[0][1])
    return min_d


def point_to_segment_nm(px, py, ax, ay, bx, by):
    """Approximate minimum distance from point to line segment using projection.
    Uses flat-earth approximation for the projection parameter, then haversine for distance."""
    # Scale lon by cos(lat) for local flat approximation
    cos_lat = math.cos(math.radians(py))
    dx = (bx - ax) * cos_lat
    dy = by - ay
    if dx == 0 and dy == 0:
        return haversine_nm(px, py, ax, ay)
    t = ((px - ax) * cos_lat * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    nx = ax + t * (bx - ax)
    ny = ay + t * (by - ay)
    return haversine_nm(px, py, nx, ny)


def min_dist_to_features(lon, lat, features):
    """Minimum distance in nm from (lon,lat) to any feature geometry (LineString/MultiLineString)."""
    min_d = float("inf")
    for feat in features:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            d = point_to_linestring_min_dist_nm(lon, lat, geom["coordinates"])
            if d < min_d:
                min_d = d
        elif geom["type"] == "MultiLineString":
            for line in geom["coordinates"]:
                d = point_to_linestring_min_dist_nm(lon, lat, line)
                if d < min_d:
                    min_d = d
    return min_d if min_d < float("inf") else None


def sample_hotspot_intensity(lon, lat, hotspot_features):
    """Sample hotspot intensity at point using shapely point-in-polygon."""
    pt = Point(lon, lat)
    best_intensity = 0.0
    for feat in hotspot_features:
        try:
            poly = shape(feat["geometry"])
            if poly.contains(pt):
                intensity = feat["properties"].get("intensity", 0)
                if intensity > best_intensity:
                    best_intensity = intensity
        except Exception:
            continue
    return best_intensity


def load_geojson(filepath):
    """Load a GeoJSON file, return features list or empty list."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("features", [])
    except Exception:
        return []


def filter_features(features, **criteria):
    """Filter features by property criteria. None values in criteria are skipped."""
    result = []
    for feat in features:
        props = feat.get("properties", {})
        match = True
        for key, vals in criteria.items():
            if vals is None:
                continue
            if not isinstance(vals, (list, tuple, set)):
                vals = [vals]
            if props.get(key) not in vals:
                match = False
                break
        if match:
            result.append(feat)
    return result


def generate_control_points(n, existing_catches_on_date=None):
    """Generate n random control points within the trench zone."""
    points = []
    for _ in range(n):
        lon = random.uniform(ZONE_LON_MIN, ZONE_LON_MAX)
        lat = random.uniform(ZONE_LAT_MIN, ZONE_LAT_MAX)
        points.append((lon, lat))
    return points


def compute_distances(lon, lat, overlays):
    """Compute all feature distances for a single point. Returns dict of feature->distance_nm."""
    result = {}

    # SST fronts
    sst_fronts = filter_features(overlays.get("sst_fronts", []), type="sst_front")
    result["sst_front"] = min_dist_to_features(lon, lat, sst_fronts)

    # Blue marlin isotherm (prime first, then good)
    isotherms_prime = filter_features(overlays.get("sst_fronts", []), type="isotherm", species="blue", tier="prime")
    isotherms_good = filter_features(overlays.get("sst_fronts", []), type="isotherm", species="blue", tier="good")
    isotherms = isotherms_prime if isotherms_prime else isotherms_good
    result["blue_isotherm"] = min_dist_to_features(lon, lat, isotherms) if isotherms else None

    # CHL edges
    chl_feats = overlays.get("chl_edges", [])
    result["chl_edge"] = min_dist_to_features(lon, lat, chl_feats) if chl_feats else None

    # MLD contours by label
    mld_feats = overlays.get("mld_contours", [])
    for label in ["very_shallow", "shallow", "moderate", "deep"]:
        subset = filter_features(mld_feats, label=label)
        result[f"mld_{label}"] = min_dist_to_features(lon, lat, subset) if subset else None

    # Warm SSH eddy
    ssh_feats = overlays.get("ssh_eddies", [])
    warm = filter_features(ssh_feats, label=["warm_eddy", "warm_core"])
    result["warm_eddy"] = min_dist_to_features(lon, lat, warm) if warm else None

    # Water clarity clean boundary
    clarity_feats = overlays.get("water_clarity", [])
    clean = filter_features(clarity_feats, label="clean")
    result["clarity_clean"] = min_dist_to_features(lon, lat, clean) if clean else None

    # Hotspot intensity
    hotspots = overlays.get("hotspots", [])
    result["hotspot_intensity"] = sample_hotspot_intensity(lon, lat, hotspots)

    return result


def load_overlays_for_date(date_str):
    """Load all overlay GeoJSONs for a given date directory."""
    date_dir = os.path.join(DATA_DIR, date_str)
    if not os.path.isdir(date_dir):
        return None

    overlays = {}
    overlays["sst_fronts"] = load_geojson(os.path.join(date_dir, "sst_fronts.geojson"))
    overlays["chl_edges"] = load_geojson(os.path.join(date_dir, "chl_edges.geojson"))
    overlays["mld_contours"] = load_geojson(os.path.join(date_dir, "mld_contours.geojson"))
    overlays["ssh_eddies"] = load_geojson(os.path.join(date_dir, "ssh_eddies.geojson"))
    overlays["water_clarity"] = load_geojson(os.path.join(date_dir, "water_clarity.geojson"))
    overlays["hotspots"] = load_geojson(os.path.join(date_dir, "blue_marlin_hotspots.geojson"))

    return overlays


def hotspot_percentile(catch_intensity, hotspot_features, zone_bounds):
    """What percentile is the catch intensity among all zone cells?"""
    lon_min, lon_max, lat_min, lat_max = zone_bounds
    zone_intensities = []
    for feat in hotspot_features:
        try:
            poly = shape(feat["geometry"])
            centroid = poly.centroid
            if lon_min <= centroid.x <= lon_max and lat_min <= centroid.y <= lat_max:
                zone_intensities.append(feat["properties"].get("intensity", 0))
        except Exception:
            continue
    if not zone_intensities:
        return None, 0
    zone_intensities.sort()
    # Percentile: fraction of zone cells with intensity <= catch_intensity
    n_below = sum(1 for v in zone_intensities if v <= catch_intensity)
    percentile = n_below / len(zone_intensities) * 100
    return percentile, len(zone_intensities)


def cohens_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    pooled_std = math.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    # Load catches
    catches_path = os.path.join(DATA_DIR, "marlin_catches.geojson")
    with open(catches_path) as f:
        catches_data = json.load(f)
    catches = catches_data["features"]
    print(f"Loaded {len(catches)} catches")

    # Group catches by date
    catches_by_date = defaultdict(list)
    for c in catches:
        date = c["properties"]["date"]
        lon, lat = c["geometry"]["coordinates"]
        species = c["properties"]["species"]
        tag = c["properties"]["tag"]
        catches_by_date[date].append({"lon": lon, "lat": lat, "species": species, "tag": tag, "date": date})

    print(f"Unique catch dates: {len(catches_by_date)}")

    # Process each date
    catch_results = []
    control_results = []
    dates_processed = 0
    dates_missing = 0

    for date_str in sorted(catches_by_date.keys()):
        overlays = load_overlays_for_date(date_str)
        if overlays is None:
            dates_missing += 1
            continue

        dates_processed += 1
        date_catches = catches_by_date[date_str]

        # Process catches
        for c in date_catches:
            dists = compute_distances(c["lon"], c["lat"], overlays)
            dists["date"] = date_str
            dists["species"] = c["species"]
            dists["tag"] = c["tag"]
            dists["lon"] = c["lon"]
            dists["lat"] = c["lat"]

            # Hotspot percentile
            pct, n_cells = hotspot_percentile(
                dists["hotspot_intensity"],
                overlays["hotspots"],
                (ZONE_LON_MIN, ZONE_LON_MAX, ZONE_LAT_MIN, ZONE_LAT_MAX),
            )
            dists["hotspot_percentile"] = pct
            dists["zone_cells"] = n_cells

            catch_results.append(dists)

        # Generate control points for this date
        controls = generate_control_points(N_CONTROLS)
        for lon, lat in controls:
            dists = compute_distances(lon, lat, overlays)
            dists["date"] = date_str
            dists["lon"] = lon
            dists["lat"] = lat

            pct, n_cells = hotspot_percentile(
                dists["hotspot_intensity"],
                overlays["hotspots"],
                (ZONE_LON_MIN, ZONE_LON_MAX, ZONE_LAT_MIN, ZONE_LAT_MAX),
            )
            dists["hotspot_percentile"] = pct

            control_results.append(dists)

    print(f"Dates processed: {dates_processed}, missing: {dates_missing}")
    print(f"Catch measurements: {len(catch_results)}, Control measurements: {len(control_results)}")
    print()

    # ── Statistical Analysis ──────────────────────────────────────────
    feature_keys = [
        "sst_front",
        "blue_isotherm",
        "chl_edge",
        "mld_very_shallow",
        "mld_shallow",
        "mld_moderate",
        "mld_deep",
        "warm_eddy",
        "clarity_clean",
    ]

    summary = {}
    print("=" * 90)
    print("PROXIMITY ANALYSIS: Historical Marlin Catches vs Oceanographic Features")
    print("=" * 90)
    print()

    # ── Distance comparisons ──────────────────────────────────────────
    print("─" * 90)
    print(f"{'Feature':<20} {'Catch Med':>10} {'Catch Mean':>11} {'Ctrl Med':>10} {'Ctrl Mean':>11} {'M-W p':>10} {'Cohen d':>8}")
    print("─" * 90)

    for key in feature_keys:
        catch_vals = [r[key] for r in catch_results if r[key] is not None]
        ctrl_vals = [r[key] for r in control_results if r[key] is not None]

        feat_summary = {
            "catch_n": len(catch_vals),
            "control_n": len(ctrl_vals),
        }

        if len(catch_vals) >= 3 and len(ctrl_vals) >= 3:
            catch_arr = np.array(catch_vals)
            ctrl_arr = np.array(ctrl_vals)

            feat_summary["catch_median"] = round(float(np.median(catch_arr)), 2)
            feat_summary["catch_mean"] = round(float(np.mean(catch_arr)), 2)
            feat_summary["catch_std"] = round(float(np.std(catch_arr, ddof=1)), 2)
            feat_summary["control_median"] = round(float(np.median(ctrl_arr)), 2)
            feat_summary["control_mean"] = round(float(np.mean(ctrl_arr)), 2)
            feat_summary["control_std"] = round(float(np.std(ctrl_arr, ddof=1)), 2)

            # Mann-Whitney U
            try:
                u_stat, u_p = stats.mannwhitneyu(catch_arr, ctrl_arr, alternative="two-sided")
                feat_summary["mannwhitney_U"] = round(float(u_stat), 1)
                feat_summary["mannwhitney_p"] = round(float(u_p), 4)
            except Exception:
                u_p = None

            # Cohen's d
            d = cohens_d(catch_vals, ctrl_vals)
            feat_summary["cohens_d"] = round(d, 3) if d is not None else None

            sig = "*" if u_p is not None and u_p < 0.05 else " "
            print(
                f"{key:<20} {feat_summary['catch_median']:>9.1f}nm {feat_summary['catch_mean']:>9.1f}nm "
                f"{feat_summary['control_median']:>9.1f}nm {feat_summary['control_mean']:>9.1f}nm "
                f"{u_p:>9.4f}{sig} {d:>7.3f}" if u_p is not None and d is not None else
                f"{key:<20} {feat_summary['catch_median']:>9.1f}nm {feat_summary['catch_mean']:>9.1f}nm "
                f"{feat_summary['control_median']:>9.1f}nm {feat_summary['control_mean']:>9.1f}nm "
                f"{'N/A':>10} {'N/A':>8}"
            )
        else:
            print(f"{key:<20} {'insufficient data (n_catch=' + str(len(catch_vals)) + ', n_ctrl=' + str(len(ctrl_vals)) + ')'}")

        summary[key] = feat_summary

    print("─" * 90)
    print("  * p < 0.05")
    print()

    # ── Paired Wilcoxon (catch median vs control median per date) ──────
    print("─" * 90)
    print("PAIRED ANALYSIS: Wilcoxon signed-rank (catch median vs control median per date)")
    print("─" * 90)

    # Group by date
    catch_by_date = defaultdict(list)
    ctrl_by_date = defaultdict(list)
    for r in catch_results:
        catch_by_date[r["date"]].append(r)
    for r in control_results:
        ctrl_by_date[r["date"]].append(r)

    paired_summary = {}
    print(f"{'Feature':<20} {'N pairs':>8} {'Catch>Ctrl':>11} {'Wilcox p':>10} {'Sig':>4}")
    print("─" * 90)

    for key in feature_keys:
        catch_medians = []
        ctrl_medians = []
        for date in sorted(catch_by_date.keys()):
            c_vals = [r[key] for r in catch_by_date[date] if r[key] is not None]
            t_vals = [r[key] for r in ctrl_by_date[date] if r[key] is not None]
            if c_vals and t_vals:
                catch_medians.append(np.median(c_vals))
                ctrl_medians.append(np.median(t_vals))

        n_pairs = len(catch_medians)
        if n_pairs >= 6:
            catch_arr = np.array(catch_medians)
            ctrl_arr = np.array(ctrl_medians)
            n_closer = int(np.sum(catch_arr < ctrl_arr))

            try:
                w_stat, w_p = stats.wilcoxon(catch_arr, ctrl_arr, alternative="two-sided")
                sig = "*" if w_p < 0.05 else " "
                print(f"{key:<20} {n_pairs:>8} {n_closer:>8}/{n_pairs:<3} {w_p:>9.4f} {sig:>4}")
                paired_summary[key] = {
                    "n_pairs": n_pairs,
                    "catch_closer_count": n_closer,
                    "wilcoxon_p": round(float(w_p), 4),
                    "significant": w_p < 0.05,
                }
            except Exception as e:
                print(f"{key:<20} {n_pairs:>8} {n_closer:>8}/{n_pairs:<3} {'error':>10}")
                paired_summary[key] = {"n_pairs": n_pairs, "error": str(e)}
        else:
            print(f"{key:<20} {n_pairs:>8} {'insufficient pairs'}")
            paired_summary[key] = {"n_pairs": n_pairs, "insufficient": True}

    print("─" * 90)
    print()

    # ── Hotspot Intensity Analysis ────────────────────────────────────
    print("─" * 90)
    print("HOTSPOT INTENSITY ANALYSIS")
    print("─" * 90)

    catch_intensities = [r["hotspot_intensity"] for r in catch_results]
    ctrl_intensities = [r["hotspot_intensity"] for r in control_results]
    catch_percentiles = [r["hotspot_percentile"] for r in catch_results if r["hotspot_percentile"] is not None]

    c_int = np.array(catch_intensities)
    t_int = np.array(ctrl_intensities)

    print(f"  Catch hotspot intensity:   median={np.median(c_int):.2f}, mean={np.mean(c_int):.2f}, std={np.std(c_int, ddof=1):.2f}")
    print(f"  Control hotspot intensity: median={np.median(t_int):.2f}, mean={np.mean(t_int):.2f}, std={np.std(t_int, ddof=1):.2f}")

    if len(c_int) >= 3 and len(t_int) >= 3:
        u_stat, u_p = stats.mannwhitneyu(c_int, t_int, alternative="greater")
        d = cohens_d(catch_intensities, ctrl_intensities)
        print(f"  Mann-Whitney U (catch > ctrl): U={u_stat:.0f}, p={u_p:.4f} {'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else ''}")
        print(f"  Cohen's d: {d:.3f}")
    print()

    if catch_percentiles:
        pct_arr = np.array(catch_percentiles)
        print(f"  Catch hotspot percentile among zone cells:")
        print(f"    Median: {np.median(pct_arr):.1f}%")
        print(f"    Mean:   {np.mean(pct_arr):.1f}%")
        print(f"    Min:    {np.min(pct_arr):.1f}%")
        print(f"    Max:    {np.max(pct_arr):.1f}%")
        print(f"    >= 75th percentile: {np.sum(pct_arr >= 75)}/{len(pct_arr)} ({np.sum(pct_arr >= 75) / len(pct_arr) * 100:.0f}%)")
        print(f"    >= 50th percentile: {np.sum(pct_arr >= 50)}/{len(pct_arr)} ({np.sum(pct_arr >= 50) / len(pct_arr) * 100:.0f}%)")
    print("─" * 90)
    print()

    # ── Per-species breakdown ─────────────────────────────────────────
    print("─" * 90)
    print("PER-SPECIES SUMMARY (median distance in nm)")
    print("─" * 90)
    species_groups = defaultdict(list)
    for r in catch_results:
        species_groups[r["species"]].append(r)

    header_keys = ["sst_front", "blue_isotherm", "chl_edge", "warm_eddy", "clarity_clean"]
    print(f"{'Species':<20}", end="")
    for k in header_keys:
        print(f" {k:>14}", end="")
    print(f" {'hotspot_int':>12}")
    print("─" * 90)

    species_summary = {}
    for sp in sorted(species_groups.keys()):
        recs = species_groups[sp]
        print(f"{sp:<20}", end="")
        sp_data = {"n": len(recs)}
        for k in header_keys:
            vals = [r[k] for r in recs if r[k] is not None]
            if vals:
                med = np.median(vals)
                sp_data[k] = round(float(med), 1)
                print(f" {med:>12.1f}nm", end="")
            else:
                print(f" {'N/A':>14}", end="")
        intensities = [r["hotspot_intensity"] for r in recs]
        sp_data["hotspot_intensity_median"] = round(float(np.median(intensities)), 2)
        print(f" {np.median(intensities):>11.2f}")
        species_summary[sp] = sp_data

    print("─" * 90)
    print()

    # ── Top proximity catches ─────────────────────────────────────────
    print("─" * 90)
    print("CLOSEST CATCHES TO SST FRONTS (top 10)")
    print("─" * 90)
    sorted_by_front = sorted(
        [r for r in catch_results if r["sst_front"] is not None],
        key=lambda x: x["sst_front"],
    )
    for r in sorted_by_front[:10]:
        print(
            f"  {r['date']} {r['species']:<16} {r['sst_front']:>5.1f}nm to front, "
            f"isotherm={'%.1f' % r['blue_isotherm'] + 'nm' if r['blue_isotherm'] is not None else 'N/A':>8}, "
            f"hotspot={r['hotspot_intensity']:.0%}"
        )
    print()

    # ── Summary narrative ─────────────────────────────────────────────
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Analyzed {len(catch_results)} catches across {dates_processed} dates")
    print(f"  ({dates_missing} catch dates had no overlay data)")
    print(f"  Compared against {N_CONTROLS} random control points per date ({len(control_results)} total)")
    print()

    # Find significant features
    sig_features = []
    for key in feature_keys:
        if key in summary and "mannwhitney_p" in summary[key]:
            if summary[key]["mannwhitney_p"] < 0.05:
                direction = "CLOSER" if summary[key]["catch_median"] < summary[key]["control_median"] else "FARTHER"
                sig_features.append((key, summary[key]["mannwhitney_p"], direction, summary[key].get("cohens_d", 0)))

    if sig_features:
        print("  Significant proximity differences (Mann-Whitney, p<0.05):")
        for feat, p, direction, d in sorted(sig_features, key=lambda x: x[1]):
            print(f"    - {feat}: catches are {direction} (p={p:.4f}, d={d:.3f})")
    else:
        print("  No significant proximity differences found (Mann-Whitney, p<0.05)")
    print()

    sig_paired = []
    for key in feature_keys:
        if key in paired_summary and paired_summary[key].get("significant"):
            sig_paired.append((key, paired_summary[key]["wilcoxon_p"], paired_summary[key]["catch_closer_count"], paired_summary[key]["n_pairs"]))

    if sig_paired:
        print("  Significant paired differences (Wilcoxon, p<0.05):")
        for feat, p, closer, total in sorted(sig_paired, key=lambda x: x[1]):
            print(f"    - {feat}: catches closer in {closer}/{total} dates (p={p:.4f})")
    else:
        print("  No significant paired differences found (Wilcoxon, p<0.05)")
    print()

    if catch_intensities:
        print(f"  Hotspot intensity: catches median={np.median(catch_intensities):.0%} vs controls median={np.median(ctrl_intensities):.0%}")
    if catch_percentiles:
        print(f"  Catch hotspot percentile: median={np.median(catch_percentiles):.0f}th, mean={np.mean(catch_percentiles):.0f}th")
    print()

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_catches": len(catch_results),
            "n_controls_per_date": N_CONTROLS,
            "n_control_total": len(control_results),
            "dates_processed": dates_processed,
            "dates_missing": dates_missing,
            "zone": {
                "lon_min": ZONE_LON_MIN, "lon_max": ZONE_LON_MAX,
                "lat_min": ZONE_LAT_MIN, "lat_max": ZONE_LAT_MAX,
            },
        },
        "distance_summary": summary,
        "paired_summary": paired_summary,
        "hotspot_analysis": {
            "catch_median": round(float(np.median(catch_intensities)), 3) if catch_intensities else None,
            "catch_mean": round(float(np.mean(catch_intensities)), 3) if catch_intensities else None,
            "control_median": round(float(np.median(ctrl_intensities)), 3) if ctrl_intensities else None,
            "control_mean": round(float(np.mean(ctrl_intensities)), 3) if ctrl_intensities else None,
            "catch_percentile_median": round(float(np.median(catch_percentiles)), 1) if catch_percentiles else None,
            "catch_percentile_mean": round(float(np.mean(catch_percentiles)), 1) if catch_percentiles else None,
        },
        "species_summary": species_summary,
        "catch_details": [
            {
                "date": r["date"], "species": r["species"], "tag": r["tag"],
                "lon": r["lon"], "lat": r["lat"],
                "sst_front_nm": r["sst_front"],
                "blue_isotherm_nm": r["blue_isotherm"],
                "chl_edge_nm": r["chl_edge"],
                "warm_eddy_nm": r["warm_eddy"],
                "clarity_clean_nm": r["clarity_clean"],
                "hotspot_intensity": r["hotspot_intensity"],
                "hotspot_percentile": r["hotspot_percentile"],
            }
            for r in catch_results
        ],
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path = os.path.join(DATA_DIR, "proximity_analysis_report.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
