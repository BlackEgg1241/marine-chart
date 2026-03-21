"""Refresh BOM wind observations for Rottnest and Hillarys in marine_weather.json."""
import json, os, ssl, urllib.request, urllib.error, time
from datetime import datetime


BOM_STATIONS = {
    "rottnest": ("Rottnest", 94602),
    "hillarys": ("Hillarys", 95605),
}

DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5, "CALM": None,
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WEATHER_PATH = os.path.join(DATA_DIR, "marine_weather.json")


def fetch_bom_station(label, station_id):
    """Fetch BOM observations for a single station with SSL retry."""
    url = f"https://reg.bom.gov.au/fwo/IDW60801/IDW60801.{station_id}.json"
    today_str = datetime.now().strftime("%Y%m%d")

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            break
        except (ssl.SSLError, urllib.error.URLError) as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/3 for BOM {label}: {e}")
                time.sleep(wait)
            else:
                print(f"  BOM fetch failed for {label} ({station_id}): {e}")
                return []

    obs_list = data.get("observations", {}).get("data", [])
    if not obs_list:
        print(f"  BOM {label}: no observations")
        return []

    # Latest observation summary
    o = obs_list[0]
    print(f"  BOM {label}: {o.get('wind_dir')} {o.get('wind_spd_kt')}kn, gusts {o.get('gust_kt')}kn, {o.get('air_temp')}C")

    # Collect today's observations for wind chart
    today_obs = []
    for ob in obs_list:
        ts = ob.get("local_date_time_full", "")
        if not ts.startswith(today_str):
            continue
        wind_kt = ob.get("wind_spd_kt")
        if wind_kt is None:
            continue
        deg = DIR_MAP.get(ob.get("wind_dir", ""), None)
        today_obs.append({
            "time": ts,
            "wind_spd_kt": wind_kt,
            "gust_kt": ob.get("gust_kt"),
            "wind_dir": ob.get("wind_dir", ""),
            "wind_dir_deg": deg,
        })
    today_obs.sort(key=lambda x: x["time"])
    print(f"  BOM {label}: {len(today_obs)} observations since midnight")
    return today_obs


def main():
    # Read existing marine_weather.json
    if not os.path.exists(WEATHER_PATH):
        print(f"Error: {WEATHER_PATH} not found")
        return

    with open(WEATHER_PATH, "r") as f:
        weather = json.load(f)

    # Fetch fresh BOM wind data
    print("Fetching BOM observations...")
    for key, (label, station_id) in BOM_STATIONS.items():
        obs = fetch_bom_station(label, station_id)
        weather[f"bom_{key}_wind"] = obs

    # Write back
    with open(WEATHER_PATH, "w") as f:
        json.dump(weather, f, indent=2)

    now = datetime.now().strftime("%H:%M:%S")
    rot_n = len(weather.get("bom_rottnest_wind", []))
    hil_n = len(weather.get("bom_hillarys_wind", []))
    print(f"\nBOM wind updated at {now}: Rottnest {rot_n} obs, Hillarys {hil_n} obs -> {WEATHER_PATH}")


if __name__ == "__main__":
    main()
