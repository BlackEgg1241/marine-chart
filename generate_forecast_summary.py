"""Generate forecast summary with Accessible Trench Zone max scores and eddy proximity."""
import json, os
import numpy as np

# Accessible Trench Zone: bounding rectangle defined by key marks
# North Metro 04: [115.1754, -31.7287]  (north)
# Rottnest Trench: [114.98, -32.01]     (west)
# Fibrelite Boats: [115.1667, -32.1667] (south)
# Club Marine: [115.3333, -32.05]       (east)
ZONE_W = 114.98
ZONE_E = 115.3333
ZONE_S = -32.1667
ZONE_N = -31.7287

# Sub-zones based on GPS catch clustering analysis (45 blue marlin catches):
# - 47% of catches in SE quadrant (PGFC/Club Marine area)
# - Early season (Jan-Feb) catches 2.7nm further north/west (Canyon Head area)
# - Late season (Mar-Apr) catches shift SE toward PGFC
# - 90% of catches within 14nm of Canyon Head
SUB_ZONES = {
    "canyon": {
        "name": "Canyon Head",
        "bounds": [114.95, -32.02, 115.15, -31.85],  # W, S, E, N
        "desc": "Perth Canyon head & Rottnest Trench - early season hotspot",
        "color": "#38bdf8",
    },
    "pgfc": {
        "name": "PGFC",
        "bounds": [115.15, -32.12, 115.35, -31.92],  # W, S, E, N
        "desc": "PGFC, FURUNO, Club Marine - peak season hotspot (47% of catches)",
        "color": "#c084fc",
    },
    "north": {
        "name": "North",
        "bounds": [114.98, -31.90, 115.25, -31.73],  # W, S, E, N
        "desc": "North Metro marks - occasional catches, esp. early/late",
        "color": "#22c55e",
    },
    "south": {
        "name": "South",
        "bounds": [115.05, -32.17, 115.25, -32.05],  # W, S, E, N
        "desc": "Fibrelite, Woodman Pt 5 - deep canyon edge",
        "color": "#fbbf24",
    },
}

def make_rect(w=ZONE_W, e=ZONE_E, s=ZONE_S, n=ZONE_N):
    return [[w, n], [e, n], [e, s], [w, s], [w, n]]

def poly_intersects_box(ring, w, e, s, n):
    """Check if polygon's bounding box overlaps the given box."""
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    if max(lons) < w or min(lons) > e:
        return False
    if max(lats) < s or min(lats) > n:
        return False
    return True

def poly_intersects_zone(ring):
    """Check if polygon bbox overlaps the main Accessible Trench Zone."""
    return poly_intersects_box(ring, ZONE_W, ZONE_E, ZONE_S, ZONE_N)

def score_subzones(features):
    """Score each sub-zone from hotspot GeoJSON features.

    Returns dict of {zone_key: {max, mean, cells}} for each sub-zone.
    """
    results = {}
    for key, sz in SUB_ZONES.items():
        w, s, e, n = sz["bounds"]
        vals = []
        for feat in features:
            rings = feat["geometry"]["coordinates"]
            if poly_intersects_box(rings[0], w, e, s, n):
                vals.append(feat["properties"]["intensity"])
        results[key] = {
            "max": round(max(vals) * 100, 1) if vals else 0,
            "mean": round(sum(vals) / len(vals) * 100, 1) if vals else 0,
            "cells": len(vals),
        }
    return results

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees."""
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def bearing_label(deg):
    """Convert bearing to compass label."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) / 22.5) % 16]


def analyze_eddy_proximity(date_str):
    """Analyze warm-core eddy proximity to the Accessible Trench Zone.

    Reads SSH (SLA or zos) NetCDF for the given date, identifies warm-core
    regions (SLA > threshold), and computes distance/bearing of nearest
    warm water to the zone center.

    Returns dict with eddy proximity metrics, or None if no SSH data.
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter, label

    # Try prediction dir first, then observation dir
    for base in [f"data/prediction/{date_str}", f"data/{date_str}"]:
        ssh_path = os.path.join(base, "ssh_raw.nc")
        if os.path.exists(ssh_path):
            break
    else:
        return None

    try:
        ds = xr.open_dataset(ssh_path)
        for var in ["sla", "zos", "adt"]:
            if var in ds:
                sla_da = ds[var].squeeze()
                break
        else:
            ds.close()
            return None

        lons = sla_da.longitude.values if "longitude" in sla_da.dims else sla_da.lon.values
        lats = sla_da.latitude.values if "latitude" in sla_da.dims else sla_da.lat.values
        sla = sla_da.values.astype(float)
        ds.close()

        # For zos (absolute SSH), convert to anomaly by subtracting spatial mean
        if var == "zos":
            sla = sla - np.nanmean(sla)

        # Zone center for distance calculations
        zone_center_lat = (ZONE_S + ZONE_N) / 2  # ~-31.95
        zone_center_lon = (ZONE_W + ZONE_E) / 2  # ~115.16

        # Warm-core threshold: SLA > 0.05m (moderate warm anomaly)
        WARM_THRESHOLD = 0.05
        STRONG_THRESHOLD = 0.10

        # Find warm-core pixels
        warm_mask = sla > WARM_THRESHOLD
        strong_mask = sla > STRONG_THRESHOLD

        # Check if warm water is IN the zone
        lat_in = (lats >= ZONE_S) & (lats <= ZONE_N)
        lon_in = (lons >= ZONE_W) & (lons <= ZONE_E)
        zone_sla = sla[np.ix_(lat_in, lon_in)]
        zone_warm = np.any(zone_sla > WARM_THRESHOLD)
        zone_strong = np.any(zone_sla > STRONG_THRESHOLD)
        zone_max_sla = float(np.nanmax(zone_sla)) if zone_sla.size > 0 else None

        result = {
            "zone_max_sla_m": round(zone_max_sla, 3) if zone_max_sla is not None else None,
            "warm_in_zone": bool(zone_warm),
            "strong_in_zone": bool(zone_strong),
        }

        if zone_warm:
            result["status"] = "WARM EDDY IN ZONE"
            result["distance_nm"] = 0
            result["bearing"] = None
        else:
            # Find nearest warm-core pixel to zone center
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            warm_lats = lat_grid[warm_mask]
            warm_lons = lon_grid[warm_mask]

            if len(warm_lats) > 0:
                dists = np.array([haversine_nm(zone_center_lat, zone_center_lon, la, lo)
                                  for la, lo in zip(warm_lats, warm_lons)])
                nearest_idx = np.argmin(dists)
                nearest_dist = dists[nearest_idx]
                nearest_lat = warm_lats[nearest_idx]
                nearest_lon = warm_lons[nearest_idx]
                brg = bearing_deg(zone_center_lat, zone_center_lon, nearest_lat, nearest_lon)

                result["distance_nm"] = round(float(nearest_dist), 1)
                result["bearing"] = round(float(brg), 0)
                result["bearing_label"] = bearing_label(brg)
                result["nearest_lat"] = round(float(nearest_lat), 3)
                result["nearest_lon"] = round(float(nearest_lon), 3)
                result["nearest_sla_m"] = round(float(sla[warm_mask][nearest_idx]), 3)

                if nearest_dist < 20:
                    result["status"] = "WARM EDDY APPROACHING"
                elif nearest_dist < 50:
                    result["status"] = f"WARM WATER {nearest_dist:.0f}nm {bearing_label(brg)}"
                else:
                    result["status"] = f"NEAREST WARM {nearest_dist:.0f}nm {bearing_label(brg)}"
            else:
                result["status"] = "NO WARM EDDIES"
                result["distance_nm"] = None
                result["bearing"] = None

        # Label distinct warm-core features (connected components)
        warm_filled = warm_mask.copy()
        labeled, n_eddies = label(warm_filled)
        result["n_warm_features"] = int(n_eddies)

        return result

    except Exception as e:
        print(f"  [Eddy] Error for {date_str}: {e}")
        return None


# Zone boundary GeoJSON (main zone + sub-zones)
zone_geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [make_rect()]},
        "properties": {"name": "Accessible Trench Zone", "type": "main"}
    }]
}
for sz_key, sz in SUB_ZONES.items():
    w, s, e, n = sz["bounds"]
    zone_geojson["features"].append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [make_rect(w, e, s, n)]},
        "properties": {
            "name": sz["name"],
            "type": "subzone",
            "key": sz_key,
            "desc": sz["desc"],
            "color": sz["color"],
        }
    })

# Load prediction results
pred_path = "data/prediction/prediction_results.json"
with open(pred_path) as f:
    pred = json.load(f)

summary = {
    "generated": pred["generated"],
    "zone_bounds": [ZONE_W, ZONE_S, ZONE_E, ZONE_N],
    "days": []
}

for p in pred["predictions"]:
    date = p["date"]
    hs_path = f"data/prediction/{date}/blue_marlin_hotspots.geojson"
    day = {
        "date": date,
        "model_score": p["score"],
        "label": p["label"],
        "top_params": p.get("top_params", {}),
        "bottom_params": p.get("bottom_params", {})
    }
    if os.path.exists(hs_path):
        with open(hs_path) as f:
            hs = json.load(f)
        zone_max = 0
        zone_vals = []
        for feat in hs["features"]:
            rings = feat["geometry"]["coordinates"]
            if poly_intersects_zone(rings[0]):
                v = feat["properties"]["intensity"]
                zone_vals.append(v)
                zone_max = max(zone_max, v)
        day["zone_max"] = round(zone_max * 100, 1)
        day["zone_mean"] = round((sum(zone_vals) / len(zone_vals) * 100), 1) if zone_vals else 0
        day["zone_cells"] = len(zone_vals)
        # Sub-zone scoring: prefer raw grid scores (fine-grained), fallback to polygon
        sz_path = f"data/prediction/{date}/subzone_scores.json"
        if os.path.exists(sz_path):
            with open(sz_path) as f:
                day["subzones"] = json.load(f)
        else:
            day["subzones"] = score_subzones(hs["features"])
    else:
        day["zone_max"] = None
        day["zone_mean"] = None
        day["zone_cells"] = 0
        day["subzones"] = {}
    # Eddy proximity analysis
    eddy = analyze_eddy_proximity(date)
    if eddy:
        day["eddy"] = eddy
        eddy_str = f"  eddy: {eddy['status']}"
        if eddy.get("distance_nm") and eddy["distance_nm"] > 0:
            eddy_str += f" ({eddy['distance_nm']}nm)"
    else:
        eddy_str = ""

    summary["days"].append(day)
    sz_str = ""
    if day.get("subzones"):
        sz_parts = [f"{k}={v['max']}%" for k, v in day["subzones"].items() if v["cells"] > 0]
        if sz_parts:
            sz_str = f"  sub: {', '.join(sz_parts)}"
    print(f"{date}: model={p['score']}%  zone_max={day['zone_max']}%  zone_mean={day.get('zone_mean')}%  cells={day['zone_cells']}{sz_str}{eddy_str}")

# Compute eddy movement trends between consecutive days
prev_eddy = None
for day in summary["days"]:
    eddy = day.get("eddy")
    if eddy and prev_eddy:
        d_now = eddy.get("distance_nm")
        d_prev = prev_eddy.get("distance_nm")
        if d_now is not None and d_prev is not None and d_prev > 0:
            delta = d_now - d_prev
            eddy["distance_change_nm"] = round(delta, 1)
            if delta < -5:
                eddy["movement"] = "APPROACHING"
            elif delta > 5:
                eddy["movement"] = "DEPARTING"
            else:
                eddy["movement"] = "STATIONARY"
    prev_eddy = eddy

# Print eddy movement summary
eddy_days = [d for d in summary["days"] if d.get("eddy")]
if eddy_days:
    print(f"\nEddy Proximity Forecast:")
    for d in eddy_days:
        e = d["eddy"]
        dist_str = f"{e['distance_nm']}nm" if e.get("distance_nm") and e["distance_nm"] > 0 else "IN ZONE"
        move_str = f" [{e['movement']}]" if e.get("movement") else ""
        delta_str = f" ({e['distance_change_nm']:+.0f}nm/day)" if e.get("distance_change_nm") is not None else ""
        print(f"  {d['date']}: {e['status']}  dist={dist_str}{delta_str}{move_str}")

# Write outputs
os.makedirs("data/prediction", exist_ok=True)
with open("data/prediction/forecast_zone.geojson", "w") as f:
    json.dump(zone_geojson, f)
with open("data/prediction/forecast_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nWrote forecast_zone.geojson and forecast_summary.json")
