"""Fetch marine weather data from Open-Meteo APIs for Rottnest and Hillarys."""
import json, os, urllib.request
from datetime import datetime

LOCATIONS = {
    "rottnest": {"lat": -32.00, "lon": 115.50, "label": "Rottnest Island"},
    "hillarys": {"lat": -31.82, "lon": 115.74, "label": "Hillarys"},
}

MARINE_PARAMS = [
    "wave_height", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period", "secondary_swell_wave_direction",
    "wind_wave_height", "wind_wave_period", "wind_wave_direction",
    "sea_surface_temperature",
]
WEATHER_PARAMS = [
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "temperature_2m",
]

FORECAST_DAYS = 8


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_location(name, loc):
    lat, lon = loc["lat"], loc["lon"]

    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(MARINE_PARAMS)}"
        f"&forecast_days={FORECAST_DAYS}&timezone=Australia%2FPerth"
    )
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(WEATHER_PARAMS)}"
        f"&forecast_days={FORECAST_DAYS}&timezone=Australia%2FPerth"
        f"&wind_speed_unit=kn"
    )

    print(f"Fetching marine data for {name}...")
    marine = fetch_json(marine_url)
    print(f"Fetching weather data for {name}...")
    weather = fetch_json(weather_url)

    # Merge into single hourly array
    times = marine["hourly"]["time"]
    hourly = []
    for i, t in enumerate(times):
        row = {"time": t}
        for p in MARINE_PARAMS:
            row[p] = marine["hourly"][p][i]
        # Weather times should align
        for p in WEATHER_PARAMS:
            row[p] = weather["hourly"][p][i] if i < len(weather["hourly"][p]) else None

        # Compute dominant swell using wave energy proxy (H^2 * T)
        # This resolves the period bias vs BOM/Seabreeze/Willyweather:
        # a smaller-height but longer-period groundswell carries more energy
        ph = row.get("swell_wave_height") or 0
        pp = row.get("swell_wave_period") or 0
        pd = row.get("swell_wave_direction")
        sh = row.get("secondary_swell_wave_height") or 0
        sp = row.get("secondary_swell_wave_period") or 0
        sd = row.get("secondary_swell_wave_direction")
        pe = ph * ph * pp  # primary energy proxy
        se = sh * sh * sp  # secondary energy proxy
        if se > pe and sp > 0:
            row["dominant_swell_height"] = sh
            row["dominant_swell_period"] = sp
            row["dominant_swell_direction"] = sd
        else:
            row["dominant_swell_height"] = ph
            row["dominant_swell_period"] = pp
            row["dominant_swell_direction"] = pd
        # Combined swell height (RSS of components)
        row["total_swell_height"] = round((ph**2 + sh**2) ** 0.5, 2)
        # Max period across all components for groundswell detection
        periods = [p for p in [pp, sp] if p and p > 0]
        row["max_swell_period"] = max(periods) if periods else None

        hourly.append(row)

    return {
        "label": loc["label"],
        "lat": lat,
        "lon": lon,
        "hourly": hourly,
    }


def main():
    result = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "locations": {},
    }
    for name, loc in LOCATIONS.items():
        result["locations"][name] = fetch_location(name, loc)

    os.makedirs("data", exist_ok=True)
    out_path = "data/marine_weather.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    # Quick summary
    for name, data in result["locations"].items():
        hrs = data["hourly"]
        winds = [h["wind_speed_10m"] for h in hrs if h["wind_speed_10m"] is not None]
        swells = [h["swell_wave_height"] for h in hrs if h["swell_wave_height"] is not None]
        print(f"  {name}: {len(hrs)} hours, wind {min(winds):.0f}-{max(winds):.0f} kn, swell {min(swells):.1f}-{max(swells):.1f} m")


if __name__ == "__main__":
    main()
