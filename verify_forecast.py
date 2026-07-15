"""Forecast verification -- close MarLEEn's accuracy loop.

Every morning archive_forecast.py saves that day's 8-day Open-Meteo forecast to
    data/forecast_archive/<archive_date>/forecast_raw.json
(one hourly record per hour, per location, exactly as fetch_marine_weather.py
produces it -- wind_speed_10m in knots, total_swell_height, wave_height, etc.).

Nothing has ever read those archives back to see whether the forecasts came true.
This script does. For a given "as-of" date it grades the forecasts that were made
1..7 days earlier for that date against what actually happened.

For each lead time L (1..7 days ahead):
  - open the archive written on (as_of - L days)
  - pull its hourly forecast for the as-of date  (that is the "+L day" forecast)
  - compare it, hour for hour, against the OBSERVED conditions for the as-of date

Observations come from Open-Meteo's historical/archive endpoints when the network
is reachable (ERA5 archive / recent-forecast history for wind, marine history for
waves).  If the requests/network path is unavailable it falls back to the locally
archived lead-0 forecast (the nowcast written on the target day itself).  Every
network call is wrapped so the script degrades gracefully and NEVER raises -- it is
safe to drop straight into the daily pipeline.

It writes:
  data/forecast_verification/<date>.json   full per-lead MAE / bias metrics + rating hit
  data/forecast_verification/history.csv   one appended summary line per run

Run it daily AFTER archive_forecast.py.  Over time data/forecast_verification/
accumulates real skill numbers, and the documented skill loss past day 3 becomes
plainly visible:
    python verify_forecast.py --summary

Usage:
    python verify_forecast.py                     # verify today
    python verify_forecast.py --date 2026-07-10   # verify a specific past day
    python verify_forecast.py --lookback-days 7   # max lead time to score (default 7)
    python verify_forecast.py --summary           # rolling MAE-by-lead-time table
"""
import argparse, csv, json, math, os, urllib.request
from datetime import datetime, timedelta

# --- optional / guarded imports (stdlib is enough; these are just nice-to-haves) --
try:
    import requests  # faster, but not required -- urllib is the fallback
except Exception:
    requests = None

try:
    # Keep coordinates in sync with the rest of the app when importable.
    from fetch_marine_weather import LOCATIONS as _FM_LOCATIONS
except Exception:
    _FM_LOCATIONS = None

LOCATIONS = _FM_LOCATIONS or {
    "rottnest": {"lat": -32.00, "lon": 115.50, "label": "Rottnest Island"},
    "hillarys": {"lat": -31.82, "lon": 115.74, "label": "Hillarys"},
}

DEFAULT_LOCATION = "rottnest"
MAX_LEAD = 7  # archive holds today + 7 days (FORECAST_DAYS = 8), so leads 1..7

ARCHIVE_DIR = os.path.join("data", "forecast_archive")
OUT_DIR = os.path.join("data", "forecast_verification")
HISTORY_CSV = os.path.join(OUT_DIR, "history.csv")

HEADERS = {"User-Agent": "MarLEEn/1.0"}

# Forecast variables to grade: (output label, hourly key, circular?)
VARIABLES = [
    ("wind_speed_kn", "wind_speed_10m", False),
    ("wind_gust_kn", "wind_gusts_10m", False),
    ("wind_dir_deg", "wind_direction_10m", True),
    ("swell_height_m", "total_swell_height", False),
    ("wave_height_m", "wave_height", False),
]

# CSV schema: fixed columns + wind & swell MAE per lead (1..MAX_LEAD).
CSV_FIELDS = (
    ["date", "generated", "obs_source", "n_pairs_total",
     "rating_hits", "rating_total", "rating_hit_rate"]
    + [f"wind_mae_l{L}" for L in range(1, MAX_LEAD + 1)]
    + [f"swell_mae_l{L}" for L in range(1, MAX_LEAD + 1)]
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[Verify] {msg}")


def _http_get_json(url, timeout=30):
    """Fetch JSON, preferring requests but falling back to urllib. Never raises."""
    try:
        if requests is not None:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            log(f"HTTP {resp.status_code} for {url.split('?')[0]}")
            return None
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log(f"fetch failed ({e}) for {url.split('?')[0]}")
        return None


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _num(v):
    """Coerce to float or return None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _combine_swell(primary_h, secondary_h):
    """Root-sum-of-squares combined swell height (matches fetch_marine_weather.py)."""
    ph = _num(primary_h) or 0.0
    sh = _num(secondary_h) or 0.0
    return round((ph ** 2 + sh ** 2) ** 0.5, 2)


# ---------------------------------------------------------------------------
# Observations (the "truth" for the as-of date)
# ---------------------------------------------------------------------------
def _get_weather_obs(date_str, lat, lon):
    """Hourly wind observations for one date from Open-Meteo.

    Tries the ERA5 archive first, then the forecast endpoint's retained recent
    history (which covers the last ~3 months -- important for verifying days that
    ERA5 has not ingested yet). Returns the 'hourly' dict or None.
    """
    hourly_params = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code,precipitation"
    endpoints = [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://api.open-meteo.com/v1/forecast",
    ]
    for base in endpoints:
        url = (
            f"{base}?latitude={lat}&longitude={lon}"
            f"&hourly={hourly_params}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=Australia%2FPerth&wind_speed_unit=kn"
        )
        data = _http_get_json(url)
        hourly = (data or {}).get("hourly") or {}
        speeds = hourly.get("wind_speed_10m") or []
        if hourly.get("time") and any(v is not None for v in speeds):
            return hourly
    return None


def _get_marine_obs(date_str, lat, lon):
    """Hourly wave observations for one date from Open-Meteo marine history."""
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
        f"&hourly=wave_height,swell_wave_height,secondary_swell_wave_height"
        f"&start_date={date_str}&end_date={date_str}"
        f"&timezone=Australia%2FPerth"
    )
    data = _http_get_json(url)
    hourly = (data or {}).get("hourly") or {}
    heights = hourly.get("wave_height") or []
    if hourly.get("time") and any(v is not None for v in heights):
        return hourly
    return None


def _fetch_obs_network(date_str, lat, lon):
    """Assemble time-keyed observation rows from Open-Meteo. Returns {} if nothing."""
    rows = {}
    weather = _get_weather_obs(date_str, lat, lon)
    if weather:
        times = weather.get("time", [])
        for i, t in enumerate(times):
            row = rows.setdefault(t, {})
            row["wind_speed_10m"] = _num(weather.get("wind_speed_10m", [None] * len(times))[i])
            row["wind_direction_10m"] = _num(weather.get("wind_direction_10m", [None] * len(times))[i])
            row["wind_gusts_10m"] = _num(weather.get("wind_gusts_10m", [None] * len(times))[i])
            wcode = weather.get("weather_code", [None] * len(times))[i]
            row["weather_code"] = _num(wcode)
            row["precipitation"] = _num(weather.get("precipitation", [None] * len(times))[i])

    marine = _get_marine_obs(date_str, lat, lon)
    if marine:
        times = marine.get("time", [])
        prim = marine.get("swell_wave_height", [None] * len(times))
        sec = marine.get("secondary_swell_wave_height", [None] * len(times))
        for i, t in enumerate(times):
            row = rows.setdefault(t, {})
            row["wave_height"] = _num(marine.get("wave_height", [None] * len(times))[i])
            row["total_swell_height"] = _combine_swell(prim[i], sec[i])
    return rows


def _fetch_obs_local(date_str, location):
    """Fallback: use locally archived JSON as observations when offline.

    Prefers an explicit observations file if one exists, otherwise falls back to
    the lead-0 archive (the nowcast forecast written on the target day itself).
    """
    day_dir = os.path.join(ARCHIVE_DIR, date_str)
    for fname in ("observations.json", "forecast_raw.json"):
        data = _load_json(os.path.join(day_dir, fname))
        if not data:
            continue
        rows = _hourly_for_date(data, location, date_str)
        if rows:
            return rows
    return {}


def get_observations(date_str, location):
    """Return (time->row dict, source label) of observed conditions for the date."""
    loc = LOCATIONS.get(location, LOCATIONS[DEFAULT_LOCATION])
    lat, lon = loc["lat"], loc["lon"]

    rows = _fetch_obs_network(date_str, lat, lon)
    if rows:
        return rows, "open-meteo-archive"

    log("network observations unavailable -- trying local archived JSON")
    rows = _fetch_obs_local(date_str, location)
    if rows:
        return rows, "local-archive-lead0"

    return {}, "none"


# ---------------------------------------------------------------------------
# Archived forecast access
# ---------------------------------------------------------------------------
def _hourly_for_date(archive, location, date_str):
    """Return {time_str: hourly_row} from an archive, limited to one target date."""
    loc = (archive.get("locations") or {}).get(location) or {}
    rows = {}
    for h in loc.get("hourly", []):
        t = h.get("time", "")
        if t.startswith(date_str):
            rows[t] = h
    return rows


def load_archived_forecast(archive_date, location, target_date):
    """Load the forecast for target_date out of the archive made on archive_date."""
    path = os.path.join(ARCHIVE_DIR, archive_date, "forecast_raw.json")
    archive = _load_json(path)
    if archive is None:
        return {}
    return _hourly_for_date(archive, location, target_date)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pair_metrics(forecast, observed, circular=False, min_n=3):
    """MAE / bias / RMSE for paired forecast/observed values (None-safe)."""
    pairs = [(f, o) for f, o in zip(forecast, observed) if f is not None and o is not None]
    if len(pairs) < min_n:
        return None
    n = len(pairs)

    if circular:
        diffs = []
        for f, o in pairs:
            d = f - o
            while d > 180:
                d -= 360
            while d < -180:
                d += 360
            diffs.append(d)
    else:
        diffs = [f - o for f, o in pairs]

    bias = sum(diffs) / n
    mae = sum(abs(d) for d in diffs) / n
    rmse = (sum(d * d for d in diffs) / n) ** 0.5
    return {"mae": round(mae, 2), "bias": round(bias, 2), "rmse": round(rmse, 2), "n": n}


def compute_lead_metrics(fc_rows, obs_rows):
    """Compute per-variable metrics for one lead time. Returns (metrics, n_pairs)."""
    common_times = sorted(set(fc_rows) & set(obs_rows))
    metrics = {}
    for label, key, circular in VARIABLES:
        fvals, ovals = [], []
        for t in common_times:
            fvals.append(_num(fc_rows[t].get(key)))
            ovals.append(_num(obs_rows[t].get(key)))
        m = pair_metrics(fvals, ovals, circular=circular)
        if m:
            metrics[label] = m
    return metrics, len(common_times)


# ---------------------------------------------------------------------------
# GO / NO-GO rating (replicates run_daily.py's FADs day-call from hourly data)
# ---------------------------------------------------------------------------
def fads_rating(day_rows):
    """GO / MARGINAL / NO GO for the daylight hours of one day, or None if no data.

    Mirrors _build_fads_section() in run_daily.py so a forecast rating and an
    observed rating are computed the same way and can be compared.
    """
    hours = []
    for t, h in day_rows.items():
        try:
            hr = int(t.split("T")[1][:2])
        except (IndexError, ValueError):
            continue
        if 5 <= hr <= 18:  # daylight boating window
            hours.append(h)
    if not hours:
        return None

    def mx(key, alt=None):
        vals = []
        for h in hours:
            v = h.get(key)
            if v is None and alt is not None:
                v = h.get(alt)
            vals.append(_num(v) or 0.0)
        return max(vals) if vals else 0.0

    max_swell = mx("total_swell_height", "swell_wave_height")
    max_wind = mx("wind_speed_10m")
    max_gust = mx("wind_gusts_10m")
    max_wmo = mx("weather_code")
    northerly = 0
    for h in hours:
        wd = _num(h.get("wind_direction_10m"))
        if wd is not None and (wd >= 330 or wd <= 30):
            northerly += 1

    go = True
    if max_swell >= 1.5:
        go = False
    if max_wind >= 15:
        go = False
    if max_gust >= 25:
        go = False
    if northerly >= 3:
        go = False
    if max_wmo >= 80:
        go = False

    marginal = (
        go and (max_swell >= 1.2 or max_wind >= 10 or max_gust >= 18
                or northerly >= 1 or max_wmo >= 51)
    )
    if go and not marginal:
        return "GO"
    if go and marginal:
        return "MARGINAL"
    return "NO GO"


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------
def verify(as_of_date, lookback_days=MAX_LEAD, location=DEFAULT_LOCATION):
    """Verify forecasts of lead 1..lookback_days against observations for as_of_date.

    Returns the report dict (also written to disk). Never raises.
    """
    lookback_days = max(1, min(int(lookback_days or MAX_LEAD), MAX_LEAD))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    log(f"Verifying {location} for {as_of_date} (leads +1..+{lookback_days})")
    obs_rows, obs_source = get_observations(as_of_date, location)
    log(f"Observations: {len(obs_rows)} hourly rows (source: {obs_source})")

    obs_rating = fads_rating(obs_rows) if obs_rows else None

    leads = {}
    rating_hits = 0
    rating_total = 0
    n_pairs_total = 0

    for L in range(1, lookback_days + 1):
        archive_date = (as_of - timedelta(days=L)).strftime("%Y-%m-%d")
        fc_rows = load_archived_forecast(archive_date, location, as_of_date)
        if not fc_rows:
            log(f"lead +{L}: no archive at {archive_date} -- skipped")
            continue

        metrics, n_pairs = ({}, 0)
        if obs_rows:
            metrics, n_pairs = compute_lead_metrics(fc_rows, obs_rows)
        n_pairs_total = max(n_pairs_total, n_pairs)

        fc_rating = fads_rating(fc_rows)
        hit = None
        if fc_rating is not None and obs_rating is not None:
            hit = (fc_rating == obs_rating)
            rating_total += 1
            if hit:
                rating_hits += 1

        leads[str(L)] = {
            "archive_date": archive_date,
            "n_pairs": n_pairs,
            "metrics": metrics,
            "rating": {"forecast": fc_rating, "observed": obs_rating, "hit": hit},
        }

        ws = metrics.get("wind_speed_kn", {})
        sw = metrics.get("swell_height_m", {})
        dr = metrics.get("wind_dir_deg", {})
        hit_str = "HIT" if hit else ("MISS" if hit is False else "n/a")
        log(
            f"lead +{L} (archive {archive_date}): "
            f"wind MAE {ws.get('mae', '-')}kn bias {ws.get('bias', '-')}, "
            f"swell MAE {sw.get('mae', '-')}m, dir MAE {dr.get('mae', '-')}deg, "
            f"rating fc={fc_rating} obs={obs_rating} {hit_str} (n={n_pairs})"
        )

    rating_hit_rate = round(rating_hits / rating_total, 3) if rating_total else None

    # Compact "skill vs lead" view so day-3 fall-off is easy to read.
    skill_by_lead = []
    for L in range(1, lookback_days + 1):
        info = leads.get(str(L))
        if not info:
            continue
        row = {"lead": L}
        for label, _, _ in VARIABLES:
            row[label + "_mae"] = info["metrics"].get(label, {}).get("mae")
        skill_by_lead.append(row)

    report = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "as_of_date": as_of_date,
        "location": location,
        "observation_source": obs_source,
        "n_pairs_total": n_pairs_total,
        "lookback_days": lookback_days,
        "leads": leads,
        "skill_by_lead": skill_by_lead,
        "rating": {
            "observed": obs_rating,
            "hits": rating_hits,
            "total": rating_total,
            "hit_rate": rating_hit_rate,
        },
    }

    _write_report(report)
    _append_history(report)
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_report(report):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{report['as_of_date']}.json")
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log(f"Wrote {path}")
    except Exception as e:
        log(f"could not write {path}: {e}")


def _append_history(report):
    os.makedirs(OUT_DIR, exist_ok=True)
    leads = report.get("leads", {})

    def mae(L, label):
        v = leads.get(str(L), {}).get("metrics", {}).get(label, {}).get("mae")
        return "" if v is None else f"{v:.3f}"

    row = {
        "date": report["as_of_date"],
        "generated": report["generated"],
        "obs_source": report["observation_source"],
        "n_pairs_total": report["n_pairs_total"],
        "rating_hits": report["rating"]["hits"],
        "rating_total": report["rating"]["total"],
        "rating_hit_rate": ("" if report["rating"]["hit_rate"] is None
                            else report["rating"]["hit_rate"]),
    }
    for L in range(1, MAX_LEAD + 1):
        row[f"wind_mae_l{L}"] = mae(L, "wind_speed_kn")
        row[f"swell_mae_l{L}"] = mae(L, "swell_height_m")

    try:
        new_file = not os.path.exists(HISTORY_CSV)
        with open(HISTORY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        log(f"Appended summary to {HISTORY_CSV}")
    except Exception as e:
        log(f"could not append history: {e}")


# ---------------------------------------------------------------------------
# Rolling summary from history.csv
# ---------------------------------------------------------------------------
def print_summary():
    """Print a rolling MAE-by-lead-time table averaged over all history rows."""
    if not os.path.exists(HISTORY_CSV):
        log(f"no history yet at {HISTORY_CSV} -- run a verification first")
        return

    rows = _load_csv(HISTORY_CSV)
    if not rows:
        log("history is empty")
        return

    print(f"[Verify] Rolling forecast skill over {len(rows)} verified day(s)")
    print("  Lead   Wind MAE (kn)   Swell MAE (m)   Days")
    print("  ----   -------------   -------------   ----")
    for L in range(1, MAX_LEAD + 1):
        wind_vals = _col_floats(rows, f"wind_mae_l{L}")
        swell_vals = _col_floats(rows, f"swell_mae_l{L}")
        n = max(len(wind_vals), len(swell_vals))
        wind_str = f"{sum(wind_vals) / len(wind_vals):>10.2f}" if wind_vals else "         -"
        swell_str = f"{sum(swell_vals) / len(swell_vals):>10.2f}" if swell_vals else "         -"
        print(f"  +{L:<4}  {wind_str}     {swell_str}     {n:>4}")

    hit_rates = _col_floats(rows, "rating_hit_rate")
    if hit_rates:
        print(f"\n  GO/NO-GO rating hit rate: {sum(hit_rates) / len(hit_rates):.1%} "
              f"(mean over {len(hit_rates)} day(s))")
    print("\n  (MAE rising with lead time = the documented skill loss past day 3.)")


def _load_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        log(f"could not read {path}: {e}")
        return []


def _col_floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in (None, ""):
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify archived MarLEEn forecasts against observed conditions."
    )
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="As-of date to verify, YYYY-MM-DD (default: today)")
    parser.add_argument("--lookback-days", type=int, default=MAX_LEAD,
                        help=f"Max lead time to score, 1..{MAX_LEAD} (default: {MAX_LEAD})")
    parser.add_argument("--location", default=DEFAULT_LOCATION,
                        choices=sorted(LOCATIONS.keys()),
                        help=f"Location to verify (default: {DEFAULT_LOCATION})")
    parser.add_argument("--summary", action="store_true",
                        help="Print rolling MAE-by-lead-time table from history.csv and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return 0

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        log(f"invalid --date '{args.date}' (expected YYYY-MM-DD)")
        return 0  # never fail the pipeline over a bad arg

    try:
        verify(args.date, lookback_days=args.lookback_days, location=args.location)
    except Exception as e:
        # Last-resort guard: verification must never crash the daily pipeline.
        log(f"verification aborted with error (non-fatal): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
