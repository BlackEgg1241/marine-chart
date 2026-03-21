"""Fetch marine weather data from Open-Meteo APIs for Rottnest and Hillarys."""
import json, math, os, urllib.request
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
    "precipitation", "weather_code", "visibility",
]

FORECAST_DAYS = 8


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _offset_time(iso_str, minutes):
    """Offset an ISO time string by N minutes. e.g. '2026-03-13T06:16' + (-27)."""
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M")
    from datetime import timedelta
    dt += timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M")


def rate_comfort(h):
    """Rate hourly boating comfort 0-100 for a 5.2m boat off Perth (FADs run).

    Scoring (weights sum to 1.0):
      Wind speed  (0.30) — <10kn ideal, 15kn marginal, >20kn poor
      Wind gusts  (0.10) — gusts matter more on small boats
      Swell height(0.25) — <1m ideal, 1.5m marginal, >2m poor
      Wave height (0.10) — total sea state including wind chop
      Wind dir    (0.10) — E/NE/SE give lee from coast; N/NW dangerous offshore
      Rain/storms (0.10) — precipitation and thunderstorm penalty
      Visibility  (0.05) — fog/haze reduces safety offshore
    """
    ws = h.get("wind_speed_10m") or 0
    wg = h.get("wind_gusts_10m") or 0
    wd = h.get("wind_direction_10m")
    sw = h.get("total_swell_height") or h.get("swell_wave_height") or 0
    wh = h.get("wave_height") or 0
    rain = h.get("precipitation") or 0
    wmo = h.get("weather_code") or 0
    vis = h.get("visibility")  # metres, None if unavailable

    def clamp(v):
        return max(0, min(100, v))

    # Wind speed: 100 at 0kn, 80 at 10kn, 40 at 15kn, 0 at 25kn
    if ws <= 10:
        s_ws = 100 - (ws / 10) * 20
    elif ws <= 15:
        s_ws = 80 - ((ws - 10) / 5) * 40
    elif ws <= 25:
        s_ws = 40 - ((ws - 15) / 10) * 40
    else:
        s_ws = 0

    # Gusts: 100 at 0kn, 70 at 15kn, 0 at 30kn
    if wg <= 15:
        s_wg = 100 - (wg / 15) * 30
    elif wg <= 30:
        s_wg = 70 - ((wg - 15) / 15) * 70
    else:
        s_wg = 0

    # Swell: 100 at 0m, 80 at 0.5m, 50 at 1m, 20 at 1.5m, 0 at 2.5m
    if sw <= 0.5:
        s_sw = 100 - (sw / 0.5) * 20
    elif sw <= 1.0:
        s_sw = 80 - ((sw - 0.5) / 0.5) * 30
    elif sw <= 1.5:
        s_sw = 50 - ((sw - 1.0) / 0.5) * 30
    elif sw <= 2.5:
        s_sw = 20 - ((sw - 1.5) / 1.0) * 20
    else:
        s_sw = 0

    # Wave height (total sea state): 100 at 0m, 60 at 1m, 0 at 2m
    if wh <= 1.0:
        s_wh = 100 - (wh / 1.0) * 40
    elif wh <= 2.0:
        s_wh = 60 - ((wh - 1.0) / 1.0) * 60
    else:
        s_wh = 0

    # Wind direction: E/SE best (sheltered), N/NW dangerous (front pattern),
    # W/SW bad (open ocean). Perth coast runs N-S.
    # Northerlies (330-030) are dangerous: often precede fronts, push you offshore
    if wd is not None:
        dir_cos = math.cos(math.radians(wd - 90))  # 1.0 at E, -1.0 at W
        s_wd = clamp(50 + dir_cos * 50)
        # Extra penalty for northerlies (330-030) — front warning pattern
        if wd >= 330 or wd <= 30:
            s_wd = max(0, s_wd - 30)
        elif wd >= 300 or wd <= 60:  # NW/NE — mild concern
            s_wd = max(0, s_wd - 15)
    else:
        s_wd = 50

    # Rain/storms: WMO weather codes
    # 0-3: clear/overcast (fine), 45-48: fog, 51-65: drizzle/rain,
    # 71-77: snow, 80-82: showers, 95-99: thunderstorms
    if wmo >= 95:  # thunderstorm
        s_rain = 0
    elif wmo >= 80:  # heavy showers
        s_rain = 20
    elif wmo >= 61:  # moderate+ rain
        s_rain = 30
    elif wmo >= 51:  # light drizzle/rain
        s_rain = 60
    elif wmo >= 45:  # fog
        s_rain = 40
    elif rain > 2.0:  # >2mm/hr
        s_rain = 30
    elif rain > 0.5:
        s_rain = 60
    elif rain > 0:
        s_rain = 80
    else:
        s_rain = 100

    # Visibility: 100 at >10km, 50 at 5km, 0 at <1km
    if vis is not None:
        vis_km = vis / 1000.0
        if vis_km >= 10:
            s_vis = 100
        elif vis_km >= 5:
            s_vis = 50 + (vis_km - 5) / 5 * 50
        elif vis_km >= 1:
            s_vis = (vis_km - 1) / 4 * 50
        else:
            s_vis = 0
    else:
        s_vis = 80  # assume OK if not available

    score = (
        0.30 * clamp(s_ws) +
        0.10 * clamp(s_wg) +
        0.25 * clamp(s_sw) +
        0.10 * clamp(s_wh) +
        0.10 * clamp(s_wd) +
        0.10 * clamp(s_rain) +
        0.05 * clamp(s_vis)
    )
    return round(score)


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
        f"&daily=sunrise,sunset"
        f"&forecast_days={FORECAST_DAYS}&timezone=Australia%2FPerth"
        f"&wind_speed_unit=kn"
        f"&models=ecmwf_ifs025"
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

        # Compute combined swell from primary + secondary components.
        # Open-Meteo splits swell into two spectral partitions; BOM/Seabreeze/
        # Willyweather report a single combined swell. We recombine using:
        #   Height = RSS(H1, H2)
        #   Period = energy-weighted average: (H1^2*T1 + H2^2*T2) / (H1^2 + H2^2)
        #   Direction = energy-weighted circular mean
        # Verified against Willyweather hourly data — matches within ~0.5s.
        ph = row.get("swell_wave_height") or 0
        pp = row.get("swell_wave_period") or 0
        pd = row.get("swell_wave_direction")
        sh = row.get("secondary_swell_wave_height") or 0
        sp = row.get("secondary_swell_wave_period") or 0
        sd = row.get("secondary_swell_wave_direction")

        # Combined height (root sum of squares)
        denom = ph**2 + sh**2
        row["total_swell_height"] = round(denom ** 0.5, 2)

        # Energy-weighted average period: T = (H1^2*T1 + H2^2*T2) / (H1^2 + H2^2)
        # Matches Willyweather within ~1s during daytime (verified 12 Mar 2026).
        # Residual 3-5s gap vs BOM on long-period groundswell events is due to
        # ECMWF WAM vs AUSWAVE model differences, not calculation method.
        if denom > 0:
            row["dominant_swell_period"] = round((ph**2 * pp + sh**2 * sp) / denom, 1)
        else:
            row["dominant_swell_period"] = pp if pp else None

        # Energy-weighted circular mean direction
        if denom > 0 and pd is not None:
            sx = ph**2 * math.sin(math.radians(pd or 0)) + sh**2 * math.sin(math.radians(sd or 0))
            sy = ph**2 * math.cos(math.radians(pd or 0)) + sh**2 * math.cos(math.radians(sd or 0))
            row["dominant_swell_direction"] = round(math.degrees(math.atan2(sx, sy)) % 360, 1)
        else:
            row["dominant_swell_direction"] = pd

        # Max period across components (for groundswell alerts)
        periods = [p for p in [pp, sp] if p and p > 0]
        row["max_swell_period"] = max(periods) if periods else None

        # Boating comfort rating for 5.2m boat
        row["comfort"] = rate_comfort(row)

        hourly.append(row)

    # Build sun data with civil twilight (~27 min offset for Perth latitude)
    sun = []
    daily = weather.get("daily", {})
    for i, d in enumerate(daily.get("time", [])):
        sr = daily["sunrise"][i]  # e.g. "2026-03-13T06:16"
        ss = daily["sunset"][i]
        sun.append({
            "date": d,
            "first_light": _offset_time(sr, -27),
            "sunrise": sr,
            "sunset": ss,
            "last_light": _offset_time(ss, 27),
        })

    return {
        "label": loc["label"],
        "lat": lat,
        "lon": lon,
        "hourly": hourly,
        "sun": sun,
    }


BOM_STATIONS = [
    ("Rottnest", 94602),
    ("Hillarys", 95605),
    ("Swanbourne", 94614),
]


DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5, "CALM": None,
}


def fetch_bom_observations():
    """Fetch BOM observations for each station. Latest for summary, all today for Rottnest chart."""
    stations = []
    rottnest_today = []
    today_str = datetime.now().strftime("%Y%m%d")
    for label, station_id in BOM_STATIONS:
        url = f"https://reg.bom.gov.au/fwo/IDW60801/IDW60801.{station_id}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            obs_list = data.get("observations", {}).get("data", [])
            if obs_list:
                o = obs_list[0]
                stations.append({
                    "station": label,
                    "station_id": station_id,
                    "time": o.get("local_date_time_full", ""),
                    "wind_spd_kt": o.get("wind_spd_kt"),
                    "wind_dir": o.get("wind_dir"),
                    "gust_kt": o.get("gust_kt"),
                    "air_temp": o.get("air_temp"),
                    "press_msl": o.get("press_msl"),
                    "rain_trace": o.get("rain_trace"),
                })
                print(f"  BOM {label}: {o.get('wind_dir')} {o.get('wind_spd_kt')}kn, gusts {o.get('gust_kt')}kn, {o.get('air_temp')}C")
                # Save all Rottnest observations since midnight for the live wind chart
                if station_id == 94602:
                    for ob in obs_list:
                        ts = ob.get("local_date_time_full", "")
                        if not ts.startswith(today_str):
                            continue
                        wind_kt = ob.get("wind_spd_kt")
                        if wind_kt is None:
                            continue
                        deg = DIR_MAP.get(ob.get("wind_dir", ""), None)
                        rottnest_today.append({
                            "time": ts,
                            "wind_spd_kt": wind_kt,
                            "gust_kt": ob.get("gust_kt"),
                            "wind_dir": ob.get("wind_dir", ""),
                            "wind_dir_deg": deg,
                        })
                    rottnest_today.sort(key=lambda x: x["time"])
                    print(f"  BOM Rottnest: {len(rottnest_today)} observations since midnight")
        except Exception as e:
            print(f"  BOM fetch failed for {label} ({station_id}): {e}")
    return stations, rottnest_today


def main():
    result = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "locations": {},
    }
    for name, loc in LOCATIONS.items():
        result["locations"][name] = fetch_location(name, loc)

    # Fetch BOM live observations
    print("Fetching BOM observations...")
    bom_stations, rottnest_today = fetch_bom_observations()
    result["bom_observations"] = bom_stations
    result["bom_rottnest_wind"] = rottnest_today

    # Wave buoy chart URLs (Transport WA, Rottnest Island RDW47)
    result["wave_buoy_charts"] = {
        "wave_height": "https://www.transport.wa.gov.au/getmedia/b3dfc548-cd46-4c32-aada-0261a3a66fc1/RDW_WAVE.GIF",
        "wave_direction": "https://www.transport.wa.gov.au/getmedia/70e0fe88-19fa-49fb-9094-d6c1bf148a01/RDW_POLD.GIF",
    }

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
