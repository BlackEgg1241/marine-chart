"""Verify Open-Meteo forecast model against BOM and wave buoy observations.

Fetches ~92 days of Open-Meteo model output (ECMWF IFS wind, ECMWF WAM marine)
for Rottnest and compares against:
  - BOM Rottnest Island weather station (009193): 9am/3pm wind observations
  - Transport WA Rottnest wave buoy (56005): hourly wave observations

Outputs correction factors and error metrics to data/forecast_verification.json.
"""
import argparse, csv, io, json, math, os, re, urllib.request
from datetime import datetime, timedelta


# Rottnest coordinates (same as fetch_marine_weather.py)
LAT, LON = -32.00, 115.50

# BOM Rottnest Island station monthly CSV URL pattern
# Station 009193, product IDCJDW6118
BOM_URL = "http://www.bom.gov.au/climate/dwo/{ym}/text/IDCJDW6118.{ym}.csv"

# Open-Meteo endpoints
OM_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m"
    "&start_date={start}&end_date={end}"
    "&timezone=Australia%2FPerth&wind_speed_unit=kn"
)
OM_MARINE_URL = (
    "https://marine-api.open-meteo.com/v1/marine"
    "?latitude={lat}&longitude={lon}"
    "&hourly=wave_height,wave_period,wave_direction,"
    "swell_wave_height,swell_wave_period,swell_wave_direction,"
    "secondary_swell_wave_height,secondary_swell_wave_period,secondary_swell_wave_direction"
    "&start_date={start}&end_date={end}"
    "&timezone=Australia%2FPerth"
)


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# BOM parsing
# ---------------------------------------------------------------------------
def fetch_bom_observations(start_date, end_date):
    """Fetch BOM Rottnest daily weather observations for the date range.

    Returns list of dicts with keys:
      date, max_gust_dir, max_gust_kmh,
      wind_9am_dir, wind_9am_kmh, wind_3pm_dir, wind_3pm_kmh
    """
    # Determine months to fetch
    months = set()
    d = start_date.replace(day=1)
    while d <= end_date:
        months.add(d.strftime("%Y%m"))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)

    rows = []
    for ym in sorted(months):
        url = BOM_URL.format(ym=ym)
        print(f"  Fetching BOM observations for {ym}...")
        try:
            text = fetch_text(url)
        except Exception as e:
            print(f"    Warning: could not fetch {ym}: {e}")
            continue
        rows.extend(_parse_bom_csv(text, start_date, end_date))
    return rows


def _parse_bom_csv(text, start_date, end_date):
    """Parse BOM daily weather observations CSV.

    BOM CSV fixed column layout (leading empty col 0):
      1: Date, 2: Min temp, 3: Max temp, 4: Rainfall, 5: Evaporation,
      6: Sunshine, 7: Max gust dir, 8: Max gust speed (km/h), 9: Max gust time,
      10: 9am temp, 11: 9am RH, 12: 9am cloud, 13: 9am wind dir,
      14: 9am wind speed (km/h), 15: 9am pressure,
      16: 3pm temp, 17: 3pm RH, 18: 3pm cloud, 19: 3pm wind dir,
      20: 3pm wind speed (km/h), 21: 3pm pressure
    """
    rows = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        reader = csv.reader(io.StringIO(line))
        vals = next(reader, [])
        if len(vals) < 21:
            continue

        # Extract date from col 1 — BOM uses "2026-03-1" format (no zero padding)
        date_str = vals[1].strip()
        try:
            # Handle both "2026-03-01" and "2026-03-1"
            parts = date_str.split("-")
            if len(parts) != 3:
                continue
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()
        except (ValueError, IndexError):
            continue

        if dt < start_date.date() or dt > end_date.date():
            continue

        def safe_float(idx):
            if idx >= len(vals):
                return None
            v = vals[idx].strip()
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        def safe_str(idx):
            if idx >= len(vals):
                return None
            v = vals[idx].strip()
            return v if v and v != "Calm" else None

        rows.append({
            "date": dt.isoformat(),
            "max_gust_dir": safe_str(7),
            "max_gust_kmh": safe_float(8),
            "wind_9am_dir": safe_str(13),
            "wind_9am_kmh": safe_float(14),
            "wind_3pm_dir": safe_str(19),
            "wind_3pm_kmh": safe_float(20),
        })

    return rows


# ---------------------------------------------------------------------------
# Wave buoy
# ---------------------------------------------------------------------------
def fetch_wave_buoy(start_date, end_date):
    """Fetch Rottnest wave buoy data from Transport WA / EPA WA.

    Tries multiple endpoint patterns. Returns list of dicts with:
      time, hsig, tp, dp
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Try EPA WA API
    urls_to_try = [
        f"https://wavebuoys.epa.wa.gov.au/api/data/plot?buoyId=56005&startDate={start_str}&endDate={end_str}&frequency=hourly",
        f"https://wavebuoys.epa.wa.gov.au/api/data/csv?buoyId=56005&startDate={start_str}&endDate={end_str}",
    ]

    for url in urls_to_try:
        print(f"  Trying wave buoy: {url[:80]}...")
        try:
            text = fetch_text(url)
            # Try JSON first
            try:
                data = json.loads(text)
                return _parse_wave_json(data)
            except json.JSONDecodeError:
                # Try CSV
                return _parse_wave_csv(text, start_date, end_date)
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    print("  Warning: Could not fetch wave buoy data. Skipping wave verification.")
    return []


def _parse_wave_json(data):
    """Parse wave buoy JSON response."""
    rows = []
    # Handle various possible JSON structures
    records = data if isinstance(data, list) else data.get("data", data.get("records", []))
    if not isinstance(records, list):
        return rows

    for r in records:
        row = {
            "time": r.get("time") or r.get("dateTime") or r.get("timestamp", ""),
            "hsig": _try_float(r, "hsig", "Hsig", "significantWaveHeight", "hs"),
            "tp": _try_float(r, "tp", "Tp", "peakPeriod", "peak_period"),
            "dp": _try_float(r, "dp", "Dp", "peakDirection", "peak_direction"),
        }
        if row["time"] and row["hsig"] is not None:
            rows.append(row)
    return rows


def _try_float(d, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def _parse_wave_csv(text, start_date, end_date):
    """Parse wave buoy CSV data."""
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        time_str = r.get("Time") or r.get("DateTime") or r.get("time", "")
        hsig = None
        for k in ["Hsig", "hsig", "Hs", "hs", "SignificantWaveHeight"]:
            if k in r:
                try:
                    hsig = float(r[k])
                except (ValueError, TypeError):
                    pass
                break
        tp = None
        for k in ["Tp", "tp", "PeakPeriod", "peak_period"]:
            if k in r:
                try:
                    tp = float(r[k])
                except (ValueError, TypeError):
                    pass
                break
        dp = None
        for k in ["Dp", "dp", "PeakDirection", "peak_direction"]:
            if k in r:
                try:
                    dp = float(r[k])
                except (ValueError, TypeError):
                    pass
                break

        if time_str and hsig is not None:
            rows.append({"time": time_str, "hsig": hsig, "tp": tp, "dp": dp})
    return rows


# ---------------------------------------------------------------------------
# Open-Meteo forecast model data
# ---------------------------------------------------------------------------
def fetch_om_weather(start_date, end_date):
    """Fetch Open-Meteo historical weather model data."""
    url = OM_WEATHER_URL.format(
        lat=LAT, lon=LON,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    print(f"  Fetching Open-Meteo weather model data...")
    data = fetch_json(url)
    hourly = data["hourly"]
    rows = {}
    for i, t in enumerate(hourly["time"]):
        rows[t] = {
            "wind_speed_kn": hourly["wind_speed_10m"][i],
            "wind_dir": hourly["wind_direction_10m"][i],
            "wind_gust_kn": hourly["wind_gusts_10m"][i],
        }
    return rows


def fetch_om_marine(start_date, end_date):
    """Fetch Open-Meteo historical marine model data."""
    url = OM_MARINE_URL.format(
        lat=LAT, lon=LON,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    print(f"  Fetching Open-Meteo marine model data...")
    data = fetch_json(url)
    hourly = data["hourly"]
    rows = {}
    for i, t in enumerate(hourly["time"]):
        # Combine primary + secondary swell (same logic as fetch_marine_weather.py)
        ph = hourly["swell_wave_height"][i] or 0
        pp = hourly["swell_wave_period"][i] or 0
        sh = hourly.get("secondary_swell_wave_height", [0] * len(hourly["time"]))[i] or 0
        sp = hourly.get("secondary_swell_wave_period", [0] * len(hourly["time"]))[i] or 0
        denom = ph**2 + sh**2
        total_swell = round(denom**0.5, 2)
        if denom > 0:
            dom_period = round((ph**2 * pp + sh**2 * sp) / denom, 1)
        else:
            dom_period = pp if pp else None

        rows[t] = {
            "wave_height": hourly["wave_height"][i],
            "wave_period": hourly["wave_period"][i],
            "wave_dir": hourly["wave_direction"][i],
            "swell_height": total_swell,
            "swell_period": dom_period,
        }
    return rows


# ---------------------------------------------------------------------------
# Compass direction -> degrees
# ---------------------------------------------------------------------------
COMPASS_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def compass_to_deg(s):
    if s is None:
        return None
    return COMPASS_MAP.get(s.strip().upper())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(forecast, observed, circular=False):
    """Compute verification metrics for paired forecast/observed values."""
    pairs = [(f, o) for f, o in zip(forecast, observed) if f is not None and o is not None]
    if len(pairs) < 5:
        return None

    f_vals = [p[0] for p in pairs]
    o_vals = [p[1] for p in pairs]
    n = len(pairs)

    if circular:
        # Circular statistics for direction
        diffs = []
        for fv, ov in pairs:
            d = fv - ov
            # Wrap to [-180, 180]
            while d > 180:
                d -= 360
            while d < -180:
                d += 360
            diffs.append(d)
        bias = sum(diffs) / n
        mae = sum(abs(d) for d in diffs) / n
        rmse = (sum(d**2 for d in diffs) / n) ** 0.5
        # Circular correlation
        sx = sum(math.sin(math.radians(d)) for d in diffs)
        sy = sum(math.cos(math.radians(d)) for d in diffs)
        r = (sx**2 + sy**2)**0.5 / n  # mean resultant length
        return {
            "bias": round(bias, 1),
            "mae": round(mae, 1),
            "rmse": round(rmse, 1),
            "r": round(r, 3),
            "correction_deg": round(-bias, 1),
            "n": n,
        }
    else:
        diffs = [f - o for f, o in pairs]
        bias = sum(diffs) / n
        mae = sum(abs(d) for d in diffs) / n
        rmse = (sum(d**2 for d in diffs) / n) ** 0.5

        # Pearson correlation
        f_mean = sum(f_vals) / n
        o_mean = sum(o_vals) / n
        cov = sum((f - f_mean) * (o - o_mean) for f, o in pairs) / n
        f_std = (sum((f - f_mean)**2 for f in f_vals) / n) ** 0.5
        o_std = (sum((o - o_mean)**2 for o in o_vals) / n) ** 0.5
        r = cov / (f_std * o_std) if f_std > 0 and o_std > 0 else 0

        # Multiplicative correction: mean(observed / forecast)
        ratios = [o / f for f, o in pairs if f > 0]
        correction = sum(ratios) / len(ratios) if ratios else 1.0

        return {
            "bias": round(bias, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r": round(r, 3),
            "correction_factor": round(correction, 3),
            "n": n,
        }


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify forecast against observations")
    parser.add_argument("--days", type=int, default=92, help="Days to look back (default 92)")
    args = parser.parse_args()

    end_date = datetime.now() - timedelta(days=1)  # yesterday
    start_date = end_date - timedelta(days=args.days)
    print(f"Verification period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Location: Rottnest Island ({LAT}, {LON})")

    # 1. Fetch Open-Meteo model data
    print("\n=== Fetching forecast model data ===")
    om_weather = fetch_om_weather(start_date, end_date)
    om_marine = fetch_om_marine(start_date, end_date)
    print(f"  Got {len(om_weather)} hourly weather records, {len(om_marine)} hourly marine records")

    # 2. Fetch BOM observations
    print("\n=== Fetching BOM wind observations ===")
    bom_obs = fetch_bom_observations(start_date, end_date)
    print(f"  Got {len(bom_obs)} daily BOM records")

    # 3. Fetch wave buoy observations
    print("\n=== Fetching wave buoy observations ===")
    buoy_obs = fetch_wave_buoy(start_date, end_date)
    print(f"  Got {len(buoy_obs)} wave buoy records")

    # 4. Match and compute wind metrics
    print("\n=== Computing wind verification ===")
    wind_results = verify_wind(om_weather, bom_obs)

    # 5. Match and compute wave metrics
    print("\n=== Computing wave verification ===")
    wave_results = verify_waves(om_marine, buoy_obs)

    # 6. Build report
    report = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "period": {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "days": args.days,
        },
        "location": {"name": "Rottnest Island", "lat": LAT, "lon": LON},
        "wind": wind_results,
        "waves": wave_results,
    }

    # Write outputs
    os.makedirs("data", exist_ok=True)
    json_path = "data/forecast_verification.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {json_path}")

    txt_path = "data/forecast_verification.txt"
    write_text_report(report, txt_path)
    print(f"Wrote {txt_path}")


def verify_wind(om_weather, bom_obs):
    """Match BOM 9am/3pm observations against Open-Meteo hourly model data."""
    results = {}

    # BOM gives 9am and 3pm observations
    # Match against Open-Meteo at those hours
    f_speed_9am, o_speed_9am = [], []
    f_speed_3pm, o_speed_3pm = [], []
    f_gust_max, o_gust_max = [], []
    f_dir_9am, o_dir_9am = [], []
    f_dir_3pm, o_dir_3pm = [], []

    kmh_to_kn = 1.0 / 1.852

    for obs in bom_obs:
        date = obs["date"]

        # 9am AWST
        key_9am = f"{date}T09:00"
        om = om_weather.get(key_9am)
        if om:
            if obs["wind_9am_kmh"] is not None:
                f_speed_9am.append(om["wind_speed_kn"])
                o_speed_9am.append(obs["wind_9am_kmh"] * kmh_to_kn)
            dir_obs = compass_to_deg(obs["wind_9am_dir"])
            if dir_obs is not None and om["wind_dir"] is not None:
                f_dir_9am.append(om["wind_dir"])
                o_dir_9am.append(dir_obs)

        # 3pm AWST
        key_3pm = f"{date}T15:00"
        om = om_weather.get(key_3pm)
        if om:
            if obs["wind_3pm_kmh"] is not None:
                f_speed_3pm.append(om["wind_speed_kn"])
                o_speed_3pm.append(obs["wind_3pm_kmh"] * kmh_to_kn)
            dir_obs = compass_to_deg(obs["wind_3pm_dir"])
            if dir_obs is not None and om["wind_dir"] is not None:
                f_dir_3pm.append(om["wind_dir"])
                o_dir_3pm.append(dir_obs)

        # Daily max gust: compare against max OM gust for that day
        if obs["max_gust_kmh"] is not None:
            day_gusts = []
            for h in range(24):
                key = f"{date}T{h:02d}:00"
                om = om_weather.get(key)
                if om and om["wind_gust_kn"] is not None:
                    day_gusts.append(om["wind_gust_kn"])
            if day_gusts:
                f_gust_max.append(max(day_gusts))
                o_gust_max.append(obs["max_gust_kmh"] * kmh_to_kn)

    # Combine 9am + 3pm for overall wind speed/direction metrics
    f_speed_all = f_speed_9am + f_speed_3pm
    o_speed_all = o_speed_9am + o_speed_3pm
    f_dir_all = f_dir_9am + f_dir_3pm
    o_dir_all = o_dir_9am + o_dir_3pm

    results["speed_kn"] = compute_metrics(f_speed_all, o_speed_all)
    results["speed_9am_kn"] = compute_metrics(f_speed_9am, o_speed_9am)
    results["speed_3pm_kn"] = compute_metrics(f_speed_3pm, o_speed_3pm)
    results["direction_deg"] = compute_metrics(f_dir_all, o_dir_all, circular=True)
    results["max_gust_kn"] = compute_metrics(f_gust_max, o_gust_max)

    for k, v in results.items():
        if v:
            print(f"  {k}: bias={v.get('bias')}, rmse={v.get('rmse')}, r={v.get('r')}, n={v.get('n')}")
        else:
            print(f"  {k}: insufficient data")

    return results


def verify_waves(om_marine, buoy_obs):
    """Match wave buoy observations against Open-Meteo hourly marine model data."""
    results = {}

    if not buoy_obs:
        print("  No wave buoy data available")
        return results

    f_height, o_height = [], []
    f_period, o_period = [], []
    f_dir, o_dir = [], []

    for obs in buoy_obs:
        # Try to match buoy time to Open-Meteo time
        t = obs["time"]
        # Normalize time format to match Open-Meteo (YYYY-MM-DDTHH:00)
        try:
            if "T" in t:
                dt = datetime.fromisoformat(t.replace("Z", ""))
            else:
                dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            key = dt.strftime("%Y-%m-%dT%H:00")
        except (ValueError, TypeError):
            continue

        om = om_marine.get(key)
        if not om:
            continue

        # Wave height
        if obs["hsig"] is not None and om["wave_height"] is not None:
            f_height.append(om["wave_height"])
            o_height.append(obs["hsig"])

        # Peak period vs our dominant swell period
        if obs["tp"] is not None and om["swell_period"] is not None:
            f_period.append(om["swell_period"])
            o_period.append(obs["tp"])

        # Direction
        if obs["dp"] is not None and om["wave_dir"] is not None:
            f_dir.append(om["wave_dir"])
            o_dir.append(obs["dp"])

    results["height_m"] = compute_metrics(f_height, o_height)
    results["period_s"] = compute_metrics(f_period, o_period)
    results["direction_deg"] = compute_metrics(f_dir, o_dir, circular=True)

    for k, v in results.items():
        if v:
            print(f"  {k}: bias={v.get('bias')}, rmse={v.get('rmse')}, r={v.get('r')}, n={v.get('n')}")
        else:
            print(f"  {k}: insufficient data")

    return results


def write_text_report(report, path):
    """Write human-readable verification report."""
    lines = []
    lines.append("=" * 60)
    lines.append("FORECAST VERIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {report['generated']}")
    lines.append(f"Period: {report['period']['start']} to {report['period']['end']} ({report['period']['days']} days)")
    lines.append(f"Location: {report['location']['name']} ({report['location']['lat']}, {report['location']['lon']})")
    lines.append("")

    lines.append("-" * 60)
    lines.append("WIND VERIFICATION (BOM Rottnest 009193)")
    lines.append("-" * 60)
    lines.append("Forecast model: Open-Meteo ECMWF IFS")
    lines.append("Observations: BOM 9am/3pm synoptic + daily max gust")
    lines.append("")

    wind = report.get("wind", {})
    for key, label in [
        ("speed_kn", "Wind Speed (all)"),
        ("speed_9am_kn", "Wind Speed (9am)"),
        ("speed_3pm_kn", "Wind Speed (3pm)"),
        ("max_gust_kn", "Max Gust"),
        ("direction_deg", "Wind Direction"),
    ]:
        m = wind.get(key)
        if m:
            lines.append(f"  {label}:")
            lines.append(f"    Bias:  {m.get('bias', 'N/A'):>8}  (forecast - observed)")
            lines.append(f"    RMSE:  {m.get('rmse', 'N/A'):>8}")
            lines.append(f"    MAE:   {m.get('mae', 'N/A'):>8}")
            lines.append(f"    Corr:  {m.get('r', 'N/A'):>8}")
            if "correction_factor" in m:
                lines.append(f"    Correction factor: {m['correction_factor']} (multiply forecast by this)")
            if "correction_deg" in m:
                lines.append(f"    Correction offset: {m['correction_deg']} deg (add to forecast)")
            lines.append(f"    N:     {m.get('n', 0):>8} paired observations")
            lines.append("")
        else:
            lines.append(f"  {label}: insufficient data")
            lines.append("")

    lines.append("-" * 60)
    lines.append("WAVE VERIFICATION (Rottnest wave buoy 56005)")
    lines.append("-" * 60)
    lines.append("Forecast model: Open-Meteo ECMWF WAM")
    lines.append("Observations: Transport WA / EPA WA wave buoy")
    lines.append("")

    waves = report.get("waves", {})
    for key, label in [
        ("height_m", "Wave Height"),
        ("period_s", "Swell Period"),
        ("direction_deg", "Wave Direction"),
    ]:
        m = waves.get(key)
        if m:
            lines.append(f"  {label}:")
            lines.append(f"    Bias:  {m.get('bias', 'N/A'):>8}  (forecast - observed)")
            lines.append(f"    RMSE:  {m.get('rmse', 'N/A'):>8}")
            lines.append(f"    MAE:   {m.get('mae', 'N/A'):>8}")
            lines.append(f"    Corr:  {m.get('r', 'N/A'):>8}")
            if "correction_factor" in m:
                lines.append(f"    Correction factor: {m['correction_factor']} (multiply forecast by this)")
            if "correction_deg" in m:
                lines.append(f"    Correction offset: {m['correction_deg']} deg (add to forecast)")
            lines.append(f"    N:     {m.get('n', 0):>8} paired observations")
            lines.append("")
        else:
            lines.append(f"  {label}: insufficient data")
            lines.append("")

    lines.append("=" * 60)
    lines.append("INTERPRETATION")
    lines.append("=" * 60)
    lines.append("Bias > 0: forecast over-predicts; < 0: under-predicts")
    lines.append("RMSE: overall error magnitude (lower is better)")
    lines.append("MAE:  average absolute error (lower is better)")
    lines.append("Corr: 1.0 = perfect correlation (higher is better)")
    lines.append("Correction factor: multiply raw forecast by this value")
    lines.append("Correction offset: add this value to raw forecast direction")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
