"""
Analyze proximity of blue marlin catches to SSTA contour lines.
Computes minimum Haversine distance from each catch to nearest SSTA contour segment.
"""
import json
import os
import math
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def haversine_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def point_to_segment_min_dist(plat, plon, coords):
    """Min distance from point to a polyline (list of [lon, lat] coords) in nm.
    Projects point onto each segment for accurate min distance."""
    min_d = float('inf')
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i+1]
        # Project point onto segment using flat-earth approx for projection ratio
        cos_lat = math.cos(math.radians(plat))
        dx = (lon2 - lon1) * cos_lat
        dy = lat2 - lat1
        seg_len_sq = dx*dx + dy*dy
        if seg_len_sq < 1e-14:
            d = haversine_nm(plat, plon, lat1, lon1)
        else:
            t = ((plon - lon1)*cos_lat*dx + (plat - lat1)*dy) / seg_len_sq
            t = max(0, min(1, t))
            proj_lon = lon1 + t * (lon2 - lon1)
            proj_lat = lat1 + t * (lat2 - lat1)
            d = haversine_nm(plat, plon, proj_lat, proj_lon)
        if d < min_d:
            min_d = d
    return min_d

def main():
    # Load catches
    with open(os.path.join(DATA_DIR, "catch_environment.json")) as f:
        data = json.load(f)
    catches = data["catches"]
    print(f"Total catches: {len(catches)}")

    # Process each catch
    results = []  # (date, lat, lon, min_dist_any, {level: min_dist})
    no_contour = []

    for c in catches:
        date = c["date"]
        lat, lon = c["lat"], c["lon"]
        contour_path = os.path.join(DATA_DIR, date, "ssta_contours.geojson")

        if not os.path.exists(contour_path):
            no_contour.append(date)
            continue

        with open(contour_path) as f:
            gj = json.load(f)

        level_dists = defaultdict(lambda: float('inf'))
        min_any = float('inf')

        for feat in gj["features"]:
            anomaly = feat["properties"]["anomaly"]
            geom = feat["geometry"]

            if geom["type"] == "LineString":
                coord_lists = [geom["coordinates"]]
            elif geom["type"] == "MultiLineString":
                coord_lists = geom["coordinates"]
            else:
                continue

            for coords in coord_lists:
                if len(coords) < 2:
                    continue
                d = point_to_segment_min_dist(lat, lon, coords)
                if d < level_dists[anomaly]:
                    level_dists[anomaly] = d
                if d < min_any:
                    min_any = d

        results.append({
            "date": date,
            "lat": lat,
            "lon": lon,
            "min_dist_any": min_any,
            "level_dists": dict(level_dists),
        })

    # Report
    print(f"\nCatches with SSTA contour data: {len(results)} / {len(catches)}")
    if no_contour:
        print(f"Missing contour data for: {', '.join(no_contour)}")

    # Overall min distances
    all_min = sorted([r["min_dist_any"] for r in results])
    print(f"\n{'='*60}")
    print("DISTANCE TO NEAREST SSTA CONTOUR (any level)")
    print(f"{'='*60}")
    arr = np.array(all_min)
    print(f"  N     = {len(arr)}")
    print(f"  Mean  = {arr.mean():.2f} nm")
    print(f"  Median= {np.median(arr):.2f} nm")
    print(f"  Std   = {arr.std():.2f} nm")
    print(f"  Min   = {arr.min():.2f} nm")
    print(f"  Max   = {arr.max():.2f} nm")
    print(f"  P10   = {np.percentile(arr, 10):.2f} nm")
    print(f"  P25   = {np.percentile(arr, 25):.2f} nm")
    print(f"  P75   = {np.percentile(arr, 75):.2f} nm")
    print(f"  P90   = {np.percentile(arr, 90):.2f} nm")

    # Proximity thresholds
    print(f"\n  Within  2 nm: {sum(1 for d in all_min if d <= 2):2d} / {len(all_min)} ({100*sum(1 for d in all_min if d <= 2)/len(all_min):.1f}%)")
    print(f"  Within  5 nm: {sum(1 for d in all_min if d <= 5):2d} / {len(all_min)} ({100*sum(1 for d in all_min if d <= 5)/len(all_min):.1f}%)")
    print(f"  Within 10 nm: {sum(1 for d in all_min if d <= 10):2d} / {len(all_min)} ({100*sum(1 for d in all_min if d <= 10)/len(all_min):.1f}%)")
    print(f"  Within 15 nm: {sum(1 for d in all_min if d <= 15):2d} / {len(all_min)} ({100*sum(1 for d in all_min if d <= 15)/len(all_min):.1f}%)")

    # By contour level
    all_levels = set()
    for r in results:
        all_levels.update(r["level_dists"].keys())
    all_levels = sorted(all_levels)

    print(f"\n{'='*60}")
    print("DISTANCE BY CONTOUR LEVEL")
    print(f"{'='*60}")

    for level in all_levels:
        dists = sorted([r["level_dists"][level] for r in results if level in r["level_dists"] and r["level_dists"][level] < float('inf')])
        if not dists:
            print(f"\n  Anomaly {level:+.1f}C: no contours found near any catch")
            continue
        arr = np.array(dists)
        sign = "+" if level >= 0 else ""
        print(f"\n  Anomaly {sign}{level:.1f}C  (N={len(dists)} catches with this contour)")
        print(f"    Mean={arr.mean():.2f} nm, Median={np.median(arr):.2f} nm, Std={arr.std():.2f} nm")
        print(f"    Min={arr.min():.2f} nm, Max={arr.max():.2f} nm")
        print(f"    Within  2nm: {sum(1 for d in dists if d <= 2):2d} / {len(dists)} ({100*sum(1 for d in dists if d <= 2)/len(dists):.1f}%)")
        print(f"    Within  5nm: {sum(1 for d in dists if d <= 5):2d} / {len(dists)} ({100*sum(1 for d in dists if d <= 5)/len(dists):.1f}%)")
        print(f"    Within 10nm: {sum(1 for d in dists if d <= 10):2d} / {len(dists)} ({100*sum(1 for d in dists if d <= 10)/len(dists):.1f}%)")
        print(f"    Within 15nm: {sum(1 for d in dists if d <= 15):2d} / {len(dists)} ({100*sum(1 for d in dists if d <= 15)/len(dists):.1f}%)")

    # Individual catch details sorted by distance
    print(f"\n{'='*60}")
    print("INDIVIDUAL CATCHES (sorted by distance to nearest SSTA contour)")
    print(f"{'='*60}")
    results_sorted = sorted(results, key=lambda r: r["min_dist_any"])
    for r in results_sorted:
        nearest_level = min(r["level_dists"].items(), key=lambda x: x[1])
        print(f"  {r['date']}  ({r['lat']:.4f}, {r['lon']:.4f})  "
              f"min={r['min_dist_any']:.2f} nm  "
              f"nearest_level={nearest_level[0]:+.1f}C")

    # Summary table: which levels are catches closest to?
    print(f"\n{'='*60}")
    print("NEAREST CONTOUR LEVEL DISTRIBUTION")
    print(f"{'='*60}")
    nearest_counts = defaultdict(int)
    for r in results:
        nearest_level = min(r["level_dists"].items(), key=lambda x: x[1])
        nearest_counts[nearest_level[0]] += 1
    for level in sorted(nearest_counts.keys()):
        sign = "+" if level >= 0 else ""
        print(f"  {sign}{level:.1f}C: {nearest_counts[level]} catches ({100*nearest_counts[level]/len(results):.1f}%)")

if __name__ == "__main__":
    main()
