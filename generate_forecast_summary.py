"""Generate forecast summary with Accessible Trench Zone max scores."""
import json, math, os

# Accessible Trench Zone: circle around Perth Canyon fishing grounds
CENTER = [115.05, -31.95]
RADIUS_NM = 30

def make_circle(center, radius_nm, n=64):
    coords = []
    for i in range(n + 1):
        angle = (i / n) * 2 * math.pi
        dlat = (radius_nm / 60.0) * math.sin(angle)
        dlon = (radius_nm / (60.0 * math.cos(math.radians(center[1])))) * math.cos(angle)
        coords.append([round(center[0] + dlon, 6), round(center[1] + dlat, 6)])
    return coords

def in_zone(lon, lat):
    dlat = (lat - CENTER[1]) * 60
    dlon = (lon - CENTER[0]) * 60 * math.cos(math.radians(CENTER[1]))
    return math.sqrt(dlat**2 + dlon**2) <= RADIUS_NM

def centroid(coords):
    n = len(coords)
    return sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n

# Zone boundary GeoJSON
zone_geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [make_circle(CENTER, RADIUS_NM)]},
        "properties": {"name": "Accessible Trench Zone", "radius_nm": RADIUS_NM}
    }]
}

# Load prediction results
pred_path = "data/prediction/prediction_results.json"
with open(pred_path) as f:
    pred = json.load(f)

summary = {
    "generated": pred["generated"],
    "zone_center": CENTER,
    "zone_radius_nm": RADIUS_NM,
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
