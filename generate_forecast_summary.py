"""Generate forecast summary with Accessible Trench Zone max scores."""
import json, os

# Accessible Trench Zone: bounding rectangle defined by key marks
# North Metro 04: [115.1754, -31.7287]  (north)
# Rottnest Trench: [114.98, -32.01]     (west)
# Fibrelite Boats: [115.1667, -32.1667] (south)
# Club Marine: [115.3333, -32.05]       (east)
ZONE_W = 114.98
ZONE_E = 115.3333
ZONE_S = -32.1667
ZONE_N = -31.7287

def make_rect():
    return [[ZONE_W, ZONE_N], [ZONE_E, ZONE_N], [ZONE_E, ZONE_S], [ZONE_W, ZONE_S], [ZONE_W, ZONE_N]]

def in_zone(lon, lat):
    return ZONE_W <= lon <= ZONE_E and ZONE_S <= lat <= ZONE_N

def centroid(coords):
    n = len(coords)
    return sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n

# Zone boundary GeoJSON
zone_geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [make_rect()]},
        "properties": {"name": "Accessible Trench Zone"}
    }]
}

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
            cx, cy = centroid(rings[0])
            if in_zone(cx, cy):
                v = feat["properties"]["intensity"]
                zone_vals.append(v)
                zone_max = max(zone_max, v)
        day["zone_max"] = round(zone_max * 100, 1)
        day["zone_mean"] = round((sum(zone_vals) / len(zone_vals) * 100), 1) if zone_vals else 0
        day["zone_cells"] = len(zone_vals)
    else:
        day["zone_max"] = None
        day["zone_mean"] = None
        day["zone_cells"] = 0
    summary["days"].append(day)
    print(f"{date}: model={p['score']}%  zone_max={day['zone_max']}%  zone_mean={day.get('zone_mean')}%  cells={day['zone_cells']}")

# Write outputs
os.makedirs("data/prediction", exist_ok=True)
with open("data/prediction/forecast_zone.geojson", "w") as f:
    json.dump(zone_geojson, f)
with open("data/prediction/forecast_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nWrote forecast_zone.geojson and forecast_summary.json")
