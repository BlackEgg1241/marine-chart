"""Daily automation: run all MarLEEn data pipelines.

Runs in sequence:
  1. fetch_marine_weather.py   - Weather panel data (wind/swell/comfort)
  2. fetch_prediction.py       - 7-day marlin hotspot maps
  3. generate_forecast_summary.py - Zone scores for UI
  4. archive_forecast.py       - Archive today's forecast for verification

Usage:
    python run_daily.py              # run all steps
    python run_daily.py --install    # install Windows scheduled task (05:00 AWST daily)
    python run_daily.py --uninstall  # remove scheduled task
    python run_daily.py --no-email   # run without sending email

Email setup:
    Set MARLEEN_GMAIL_APP_PASSWORD environment variable with your Gmail app password.
    Generate one at: https://myaccount.google.com/apppasswords

Windows Task Scheduler:
    The --install flag creates a task "MarLEEn Daily Update" that runs
    daily at 05:00 AWST (21:00 UTC previous day), after ECMWF model runs complete.
"""
import json, os, smtplib, subprocess, sys, time, urllib.request
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase


STEPS = [
    ("Weather data", "fetch_marine_weather.py"),
    ("Marlin prediction", "fetch_prediction.py"),
    ("Forecast summary", "generate_forecast_summary.py"),
    ("Archive forecast", "archive_forecast.py"),
]

TASK_NAME = "MarLEEn Daily Update"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Email config
EMAIL_TO = "leerferguson@gmail.com"
EMAIL_FROM = "leerferguson@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def run_pipeline():
    today = datetime.now()
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{today:%Y-%m-%d}.log")

    log_lines = []
    log_lines.append(f"MarLEEn Daily Pipeline - {today:%Y-%m-%d %H:%M:%S}")
    log_lines.append("=" * 60)

    results = []
    total_start = time.time()

    for name, script in STEPS:
        print(f"\n{'=' * 60}")
        print(f"  {name} ({script})")
        print(f"{'=' * 60}")

        script_path = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(script_path):
            msg = f"SKIP - {script} not found"
            print(f"  {msg}")
            results.append((name, "SKIP", 0, msg))
            log_lines.append(f"\n[SKIP] {name}: {script} not found")
            continue

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True,
                timeout=1800,  # 30 min timeout
                cwd=SCRIPT_DIR,
            )
            elapsed = time.time() - start
            status = "OK" if proc.returncode == 0 else "FAIL"

            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)

            results.append((name, status, elapsed, proc.returncode))
            log_lines.append(f"\n[{status}] {name} ({elapsed:.1f}s, exit={proc.returncode})")
            if proc.stdout:
                log_lines.append(proc.stdout)
            if proc.stderr:
                log_lines.append(f"STDERR: {proc.stderr}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  TIMEOUT after {elapsed:.0f}s")
            results.append((name, "TIMEOUT", elapsed, -1))
            log_lines.append(f"\n[TIMEOUT] {name} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            results.append((name, "ERROR", elapsed, str(e)))
            log_lines.append(f"\n[ERROR] {name}: {e}")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for name, status, elapsed, _ in results:
        indicator = "+" if status == "OK" else "-"
        print(f"  [{indicator}] {name}: {status} ({elapsed:.1f}s)")
        if status != "OK":
            all_ok = False
    print(f"\n  Total: {total_elapsed:.1f}s")
    print(f"  Status: {'ALL OK' if all_ok else 'SOME FAILURES'}")

    log_lines.append(f"\n{'=' * 60}")
    log_lines.append(f"Total: {total_elapsed:.1f}s - {'ALL OK' if all_ok else 'SOME FAILURES'}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Log: {log_path}")

    # Push updated data to GitHub so the online app refreshes
    if all_ok:
        push_to_github(today)

    # Send email summary
    if "--no-email" not in sys.argv:
        send_email_summary(today, results, total_elapsed, all_ok)

    return 0 if all_ok else 1


def push_to_github(today):
    """Commit and push updated data files so the online app auto-refreshes."""
    print(f"\n{'=' * 60}")
    print(f"  Pushing to GitHub")
    print(f"{'=' * 60}")

    data_files = [
        "data/marine_weather.json",
        "data/prediction/prediction_results.json",
        "data/prediction/forecast_summary.json",
        "data/prediction/forecast_zone.geojson",
    ]
    # Add per-day hotspot GeoJSONs
    pred_dir = os.path.join(SCRIPT_DIR, "data", "prediction")
    for name in os.listdir(pred_dir):
        day_dir = os.path.join(pred_dir, name)
        if os.path.isdir(day_dir) and len(name) == 10 and name[4] == "-":  # date dirs
            hs = os.path.join(day_dir, "blue_marlin_hotspots.geojson")
            if os.path.exists(hs):
                data_files.append(f"data/prediction/{name}/blue_marlin_hotspots.geojson")

    try:
        # Stage data files
        subprocess.run(
            ["git", "add"] + data_files,
            cwd=SCRIPT_DIR, capture_output=True, text=True,
        )
        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=SCRIPT_DIR, capture_output=True,
        )
        if status.returncode == 0:
            print("  No data changes to push.")
            return

        # Commit
        msg = f"Daily update {today:%Y-%m-%d} - weather & marlin forecast"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=SCRIPT_DIR, capture_output=True, text=True,
        )
        # Push
        result = subprocess.run(
            ["git", "push"],
            cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print("  Pushed to GitHub successfully.")
        else:
            print(f"  Push failed: {result.stderr}")
    except Exception as e:
        print(f"  Git push error: {e}")


def send_email_summary(today, results, total_elapsed, all_ok):
    """Send pipeline summary email via Gmail SMTP."""
    app_password = os.environ.get("MARLEEN_GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("\n  Email skipped: MARLEEN_GMAIL_APP_PASSWORD not set")
        print("  To enable: set MARLEEN_GMAIL_APP_PASSWORD=<your-gmail-app-password>")
        print("  Generate at: https://myaccount.google.com/apppasswords")
        return

    status_icon = "OK" if all_ok else "FAILURES"

    # Load marlin forecast, FADs assessment, BOM observations, and seasonal outlook
    marlin_html, marlin_plain = _build_marlin_section()
    fads_html, fads_plain = _build_fads_section()
    bom_html, bom_plain, wave_charts = _build_bom_section()
    season_html, season_plain = _build_seasonal_outlook()

    subject = f"MarLEEn Daily Update - {today:%d %b %Y} - {status_icon}"

    # Build HTML body
    rows_html = ""
    for name, status, elapsed, _ in results:
        color = "#22c55e" if status == "OK" else "#ef4444"
        icon = "+" if status == "OK" else "x"
        rows_html += (
            f'<tr><td style="padding:4px 12px">[{icon}] {name}</td>'
            f'<td style="padding:4px 12px;color:{color};font-weight:bold">{status}</td>'
            f'<td style="padding:4px 12px;text-align:right">{elapsed:.1f}s</td></tr>\n'
        )

    html = f"""\
<html><body style="font-family:monospace;font-size:14px;background:#0f172a;color:#e2e8f0;padding:20px">
<h2 style="color:#38bdf8">MarLEEn Daily Pipeline</h2>
<p>{today:%A %d %B %Y} at {today:%H:%M:%S} AWST</p>
<table style="border-collapse:collapse;margin:16px 0">
<tr style="border-bottom:1px solid #334155">
  <th style="padding:4px 12px;text-align:left">Step</th>
  <th style="padding:4px 12px;text-align:left">Status</th>
  <th style="padding:4px 12px;text-align:right">Time</th>
</tr>
{rows_html}
</table>
<p style="font-size:16px;font-weight:bold;color:{'#22c55e' if all_ok else '#ef4444'}">
  {'ALL STEPS COMPLETED SUCCESSFULLY' if all_ok else 'SOME STEPS FAILED - CHECK LOGS'}
</p>
{marlin_html}
{season_html}
{fads_html}
{bom_html}
<p style="color:#94a3b8">Total runtime: {total_elapsed:.1f}s</p>
<hr style="border-color:#334155">
<p style="color:#64748b;font-size:12px">MarLEEn Tracker - Automated Pipeline Report</p>
</body></html>"""

    # Generate charts
    chart_cids = {}
    try:
        chart_cids = _generate_email_charts()
    except Exception as e:
        print(f"  Chart generation failed: {e}")

    # Add chart images to HTML
    charts_html = ""
    if chart_cids:
        charts_html = """
<hr style="border-color:#334155;margin:16px 0">
<h3 style="color:#38bdf8">7-Day Forecast Charts</h3>
"""
        if "wind" in chart_cids:
            charts_html += '<p style="margin:8px 0"><img src="cid:wind_chart" style="width:100%;max-width:700px;border-radius:4px"></p>\n'
        if "swell" in chart_cids:
            charts_html += '<p style="margin:8px 0"><img src="cid:swell_chart" style="width:100%;max-width:700px;border-radius:4px"></p>\n'
        if "trench" in chart_cids:
            charts_html += '<p style="margin:8px 0"><img src="cid:trench_chart" style="width:100%;max-width:700px;border-radius:4px"></p>\n'

    # Insert charts before the footer
    html = html.replace(
        '<p style="color:#94a3b8">Total runtime:',
        charts_html + '<p style="color:#94a3b8">Total runtime:'
    )

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    # Alternative part (plain + HTML)
    msg_alt = MIMEMultipart("alternative")

    # Plain text fallback
    plain = f"MarLEEn Daily Pipeline - {today:%Y-%m-%d %H:%M:%S}\n\n"
    for name, status, elapsed, _ in results:
        indicator = "+" if status == "OK" else "x"
        plain += f"  [{indicator}] {name}: {status} ({elapsed:.1f}s)\n"
    plain += f"\nTotal: {total_elapsed:.1f}s\nStatus: {'ALL OK' if all_ok else 'SOME FAILURES'}\n"
    plain += marlin_plain
    plain += season_plain
    plain += fads_plain
    plain += bom_plain

    msg_alt.attach(MIMEText(plain, "plain"))
    msg_alt.attach(MIMEText(html, "html"))
    msg.attach(msg_alt)

    # Attach chart images as inline
    for name, img_data in chart_cids.items():
        img = MIMEImage(img_data, _subtype="png")
        img.add_header("Content-ID", f"<{name}_chart>")
        img.add_header("Content-Disposition", "inline", filename=f"{name}_chart.png")
        msg.attach(img)

    # Attach wave buoy + BOM wind charts as inline
    for name, img_data in wave_charts.items():
        if name == "bom_wind":
            img = MIMEImage(img_data, _subtype="png")
            img.add_header("Content-ID", f"<{name}_chart>")
            img.add_header("Content-Disposition", "inline", filename=f"{name}_chart.png")
        else:
            img = MIMEImage(img_data, _subtype="gif")
            img.add_header("Content-ID", f"<{name}_chart>")
            img.add_header("Content-Disposition", "inline", filename=f"{name}_chart.gif")
        msg.attach(img)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, app_password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"\n  Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"\n  Email failed: {e}")


def _score_color(score, comfort=None):
    """Return HTML color for a marlin score."""
    if score is None:
        return "#64748b"
    # Low comfort caps color to match capped label
    if comfort is not None and comfort < 40:
        if score >= 45:
            return "#fbbf24"  # fair (capped)
        return "#ef4444"  # poor
    if score >= 75:
        return "#22c55e"  # great
    if score >= 60:
        return "#86efac"  # good
    if score >= 45:
        return "#fbbf24"  # fair
    return "#ef4444"  # poor


def _score_label(score, comfort=None):
    if score is None:
        return "N/A"
    # Low comfort caps rating — unfishable conditions can't be "GOOD"
    if comfort is not None and comfort < 40:
        if score >= 45:
            return "FAIR"
        return "POOR"
    if score >= 75:
        return "GREAT"
    if score >= 60:
        return "GOOD"
    if score >= 45:
        return "FAIR"
    return "POOR"


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fetch_bom_json(station_id):
    """Fetch latest BOM observations JSON for a station."""
    url = f"https://reg.bom.gov.au/fwo/IDW60801/IDW60801.{station_id}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        obs_list = data.get("observations", {}).get("data", [])
        return obs_list
    except Exception as e:
        print(f"  BOM fetch failed for {station_id}: {e}")
        return []


# BOM observation stations for the email
BOM_STATIONS = [
    ("Rottnest", 94602),
    ("Hillarys", 95605),
    ("Swanbourne", 94614),
]

# Transport WA wave buoy charts (Rottnest Island RDW47, auto-updated)
WAVE_CHARTS = {
    "wave_height": "https://www.transport.wa.gov.au/getmedia/b3dfc548-cd46-4c32-aada-0261a3a66fc1/RDW_WAVE.GIF",
    "wave_direction": "https://www.transport.wa.gov.au/getmedia/70e0fe88-19fa-49fb-9094-d6c1bf148a01/RDW_POLD.GIF",
}


def _fetch_wave_charts():
    """Fetch Rottnest wave buoy chart GIFs from Transport WA."""
    charts = {}
    for name, url in WAVE_CHARTS.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MarLEEn/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                charts[name] = resp.read()
        except Exception as e:
            print(f"  Wave chart fetch failed ({name}): {e}")
    return charts


def _generate_bom_wind_chart(obs_list):
    """Generate a wind speed/gust/direction chart from BOM observations since midnight.

    Returns PNG bytes or None on failure.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # BOM obs are newest-first; filter to today since midnight
    today_str = datetime.now().strftime("%Y%m%d")
    points = []
    for ob in obs_list:
        ts = ob.get("local_date_time_full", "")
        if not ts.startswith(today_str):
            continue
        try:
            dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        wind_kt = ob.get("wind_spd_kt")
        gust_kt = ob.get("gust_kt")
        wind_dir = ob.get("wind_dir", "")
        if wind_kt is None:
            continue
        points.append((dt, wind_kt, gust_kt, wind_dir))

    if len(points) < 2:
        return None

    points.sort(key=lambda x: x[0])
    times = [p[0] for p in points]
    speeds = [p[1] for p in points]
    gusts = [p[2] if p[2] is not None else p[1] for p in points]
    dirs_ = [p[3] for p in points]

    # Direction to degrees for arrow plotting
    DIR_MAP = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
        "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5, "CALM": None,
    }

    import math

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f172a")

    # Wind speed + gusts with layered fills
    ax1.set_facecolor("#1e293b")
    # Blue fill: 0 to wind speed
    ax1.fill_between(times, 0, speeds, alpha=0.25, color="#38bdf8")
    # Red fill: wind speed to gusts
    ax1.fill_between(times, speeds, gusts, alpha=0.3, color="#ef4444")
    ax1.plot(times, gusts, color="#ef4444", linewidth=1.2, alpha=0.8, label="Gusts")
    ax1.plot(times, speeds, color="#38bdf8", linewidth=2, label="Wind")
    ax1.set_ylabel("Knots", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1e293b", edgecolor="#334155",
               labelcolor="#e2e8f0")
    ax1.set_title("Rottnest Island - BOM Live Wind Today", color="#38bdf8", fontsize=11, pad=8)
    ax1.grid(axis="y", color="#334155", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#334155")
    ax1.spines["bottom"].set_color("#334155")
    y_max = max(max(gusts), max(speeds)) + 3
    ax1.set_ylim(0, y_max)

    # Direction strip — arrows show where wind is GOING (BOM gives FROM direction)
    ax2.set_facecolor("#1e293b")

    # Use quiver for properly scaled arrows, colored by wind speed
    def _arrow_color(kn):
        if kn <= 10: return "#22c55e"
        if kn <= 15: return "#fbbf24"
        if kn <= 20: return "#fb923c"
        if kn <= 25: return "#ef4444"
        return "#dc2626"

    arrow_x = []
    arrow_y = []
    arrow_u = []  # screen-right component
    arrow_v = []  # screen-up component
    arrow_colors = []
    prev_label = None
    last_label_i = -99
    for i, (t, d) in enumerate(zip(times, dirs_)):
        deg = DIR_MAP.get(d)
        if deg is not None:
            # Wind comes FROM deg, so air moves TOWARD deg+180
            # On screen: right=E, up=N in compass terms
            to_rad = math.radians(deg + 180)
            arrow_x.append(mdates.date2num(t))
            arrow_y.append(0.5)
            arrow_u.append(math.sin(to_rad))   # east/right component
            arrow_v.append(math.cos(to_rad))   # north/up component
            arrow_colors.append(_arrow_color(speeds[i]))

        # Direction text labels every ~2hrs or on change
        show_label = (d != prev_label) or (i - last_label_i >= 4)
        if show_label and i % 2 == 0:
            ax2.text(mdates.date2num(t), -0.3, d, ha="center", va="top",
                     fontsize=7, color="#e2e8f0", fontweight="bold")
            prev_label = d
            last_label_i = i

    if arrow_x:
        ax2.quiver(arrow_x, arrow_y, arrow_u, arrow_v,
                   angles="uv", scale_units="height", scale=3,
                   width=0.005, headwidth=4, headlength=3, headaxislength=2.5,
                   pivot="mid", color=arrow_colors, alpha=0.9, zorder=4)

    ax2.set_ylim(-0.6, 1.3)
    ax2.set_ylabel("Dir", color="#94a3b8", fontsize=9)
    ax2.set_yticks([])
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#334155")
    ax2.spines["bottom"].set_color("#334155")

    fig.autofmt_xdate()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _build_bom_section():
    """Build real-time BOM observations + wave buoy section for the email.

    Returns (html, plain, wave_charts_dict).
    wave_charts_dict maps chart name -> bytes for inline email attachment.
    """
    all_obs = {}
    for label, station_id in BOM_STATIONS:
        obs = _fetch_bom_json(station_id)
        if obs:
            all_obs[label] = obs[0]  # most recent observation

    if not all_obs:
        return "", "", {}

    rows_html = ""
    rows_plain = ""
    for label, ob in all_obs.items():
        time_str = ob.get("local_date_time", "?")
        wind_kt = ob.get("wind_spd_kt")
        gust_kt = ob.get("gust_kt")
        wind_dir = ob.get("wind_dir", "-")
        temp = ob.get("air_temp")
        press = ob.get("press_msl") or ob.get("press")

        wind_str = f"{wind_kt}kn" if wind_kt is not None else "-"
        gust_str = f"{gust_kt}kn" if gust_kt is not None else "-"
        temp_str = f"{temp:.1f}C" if temp is not None else "-"
        press_str = f"{press:.1f}" if press is not None else "-"

        rows_html += (
            f'<tr>'
            f'<td style="padding:3px 10px;font-weight:bold">{label}</td>'
            f'<td style="padding:3px 10px;text-align:center">{wind_dir}</td>'
            f'<td style="padding:3px 10px;text-align:center">{wind_str}</td>'
            f'<td style="padding:3px 10px;text-align:center">{gust_str}</td>'
            f'<td style="padding:3px 10px;text-align:center">{temp_str}</td>'
            f'<td style="padding:3px 10px;text-align:center">{press_str}</td>'
            f'</tr>\n'
        )
        rows_plain += (
            f"  {label:12s}  {wind_dir:>4s}  {wind_str:>5s}  "
            f"G {gust_str:>5s}  {temp_str:>6s}  {press_str:>7s}\n"
        )

    # Get timestamp from first observation
    first_ob = list(all_obs.values())[0]
    obs_time = first_ob.get("local_date_time", "")

    # Fetch wave buoy charts
    wave_charts = _fetch_wave_charts()

    # Generate BOM wind chart from Rottnest observations since midnight
    rottnest_obs = _fetch_bom_json(94602)
    bom_wind_chart = _generate_bom_wind_chart(rottnest_obs) if rottnest_obs else None

    # Build inline chart refs: BOM wind first, then wave buoy
    charts_html = ""
    if bom_wind_chart:
        charts_html += '<p style="margin:8px 0"><img src="cid:bom_wind_chart" style="width:100%;max-width:700px;border-radius:4px"></p>\n'
    if wave_charts:
        charts_html += '<p style="color:#94a3b8;font-size:12px;margin:12px 0 4px">Rottnest Wave Buoy (RDW47, Transport WA)</p>\n'
        if "wave_height" in wave_charts:
            charts_html += '<p style="margin:4px 0"><img src="cid:wave_height_chart" style="width:100%;max-width:540px;border-radius:4px;background:#ffffff;padding:4px"></p>\n'
        if "wave_direction" in wave_charts:
            charts_html += '<p style="margin:4px 0"><img src="cid:wave_direction_chart" style="width:100%;max-width:540px;border-radius:4px;background:#ffffff;padding:4px"></p>\n'

    # Merge bom_wind_chart into wave_charts dict for attachment
    if bom_wind_chart:
        wave_charts["bom_wind"] = bom_wind_chart

    html = f"""
<hr style="border-color:#334155;margin:16px 0">
<h3 style="color:#38bdf8">Current Conditions</h3>
<p style="color:#94a3b8;font-size:12px">BOM Observations at {obs_time} AWST</p>
<table style="border-collapse:collapse;margin:12px 0">
<tr style="border-bottom:1px solid #334155">
  <th style="padding:3px 10px;text-align:left">Station</th>
  <th style="padding:3px 10px;text-align:center">Dir</th>
  <th style="padding:3px 10px;text-align:center">Wind</th>
  <th style="padding:3px 10px;text-align:center">Gust</th>
  <th style="padding:3px 10px;text-align:center">Temp</th>
  <th style="padding:3px 10px;text-align:center">Press</th>
</tr>
{rows_html}
</table>
{charts_html}
"""

    plain = f"\n\nCURRENT CONDITIONS (BOM at {obs_time} AWST)\n"
    plain += "-" * 60 + "\n"
    plain += f"  {'Station':12s}  {'Dir':>4s}  {'Wind':>5s}  {'Gust':>6s}  {'Temp':>6s}  {'Press':>7s}\n"
    plain += rows_plain
    if wave_charts:
        plain += "  Rottnest wind + wave buoy charts attached\n"

    return html, plain, wave_charts


def _build_subzone_html(days):
    """Build HTML showing today's sub-zone breakdown."""
    # Use today's data, or first available day
    today_str = datetime.now().strftime("%Y-%m-%d")
    day = next((d for d in days if d["date"] == today_str), days[0] if days else None)
    if not day or not day.get("subzones"):
        return ""

    # Sub-zone display order and styling
    sz_config = {
        "canyon": {"name": "Canyon Head", "color": "#38bdf8", "emoji": ""},
        "pgfc": {"name": "PGFC", "color": "#c084fc", "emoji": ""},
        "north": {"name": "North", "color": "#22c55e", "emoji": ""},
        "south": {"name": "South", "color": "#fbbf24", "emoji": ""},
    }

    parts = []
    for key in ["canyon", "pgfc", "north", "south"]:
        sz = day["subzones"].get(key, {})
        cfg = sz_config[key]
        mx = sz.get("max", 0)
        mn = sz.get("mean", 0)
        cells = sz.get("cells", 0)
        if cells > 0:
            parts.append(
                f'<span style="color:{cfg["color"]}">{cfg["name"]}</span>'
                f'<span style="color:#94a3b8"> {mx:.0f}%</span>'
            )

    if not parts:
        return ""
    return (
        f'<p style="margin:6px 0;font-size:12px">'
        f'<span style="color:#64748b">Sub-zones: </span>'
        f'{" &middot; ".join(parts)}'
        f'</p>\n'
    )


def _build_marlin_section():
    """Load forecast summary and build HTML + plain text marlin forecast section."""
    summary_path = os.path.join(SCRIPT_DIR, "data", "prediction", "forecast_summary.json")
    if not os.path.exists(summary_path):
        return "", ""

    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception:
        return "", ""

    days = summary.get("days", [])
    if not days:
        return "", ""

    # Filter to 7 days: today + 6 forward
    today_str = datetime.now().strftime("%Y-%m-%d")
    max_date = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")
    days = [d for d in days if today_str <= d["date"] <= max_date]
    if not days:
        return "", ""

    # Load comfort data from weather
    comfort_map = {}
    wx_path = os.path.join(SCRIPT_DIR, "data", "marine_weather.json")
    if os.path.exists(wx_path):
        try:
            with open(wx_path) as f:
                wx = json.load(f)
            rot_hours = wx.get("locations", {}).get("rottnest", {}).get("hourly", [])
            from collections import defaultdict
            day_comforts = defaultdict(list)
            for h in rot_hours:
                date = h["time"].split("T")[0]
                hour = int(h["time"].split("T")[1].split(":")[0])
                if 5 <= hour <= 18:  # daylight hours only
                    day_comforts[date].append(h.get("comfort", 50))
            for date, vals in day_comforts.items():
                comfort_map[date] = sum(vals) / len(vals)
        except Exception:
            pass

    # Blended rating: 60% habitat (spatial scoring) + 40% comfort (boating)
    # The habitat score is the primary signal — it incorporates SST, SSH, CHL,
    # currents, feature bands, FAD proximity, and bathy.
    # Model (trend) score is shown for reference but not used in the rating.
    def _blended(d):
        z = d.get("zone_max") or 0
        c = comfort_map.get(d["date"], 50)
        return 0.60 * z + 0.40 * c

    # Find best day by blended score
    best_day = max(days, key=_blended)
    best_date = datetime.strptime(best_day["date"], "%Y-%m-%d")
    best_name = WEEKDAYS[best_date.weekday()]

    # Trend: compare first half vs second half of blended scores
    scores = [_blended(d) for d in days]
    first_half = sum(scores[:len(scores)//2]) / max(len(scores)//2, 1)
    second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)
    if second_half > first_half + 3:
        trend_text = "IMPROVING"
        trend_color = "#22c55e"
    elif first_half > second_half + 3:
        trend_text = "DECLINING"
        trend_color = "#fb923c"
    else:
        trend_text = "STABLE"
        trend_color = "#38bdf8"

    # Build forecast rows
    forecast_rows = ""
    for d in days:
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        day_name = WEEKDAYS[dt.weekday()]
        date_str = dt.strftime("%d %b")
        zone = d.get("zone_max")
        model = d.get("model_score")
        comfort = comfort_map.get(d["date"])
        blend = _blended(d)
        zone_str = f"{zone:.0f}%" if zone is not None else "N/A"
        model_str = f"{model:.0f}%" if model is not None else "N/A"
        comfort_str = f"{comfort:.0f}%" if comfort is not None else "N/A"
        zc = _score_color(zone)
        mc = _score_color(model)
        cc = _score_color(comfort)
        rc = _score_color(blend, comfort)
        is_today = d["date"] == datetime.now().strftime("%Y-%m-%d")
        row_bg = "background:#1e293b;" if is_today else ""
        today_tag = " *" if is_today else ""
        forecast_rows += (
            f'<tr style="{row_bg}">'
            f'<td style="padding:3px 10px">{day_name} {date_str}{today_tag}</td>'
            f'<td style="padding:3px 10px;color:{zc};text-align:center">{zone_str}</td>'
            f'<td style="padding:3px 10px;color:{mc};text-align:center;font-weight:bold">{model_str}</td>'
            f'<td style="padding:3px 10px;color:{cc};text-align:center">{comfort_str}</td>'
            f'<td style="padding:3px 10px;color:{rc};font-weight:bold">{_score_label(blend, comfort)}</td>'
            f'</tr>\n'
        )

    # Build eddy proximity row
    eddy_html = ""
    eddy_plain = ""
    eddy_days_with_data = [d for d in days if d.get("eddy")]
    if eddy_days_with_data:
        today_eddy = next((d["eddy"] for d in eddy_days_with_data
                          if d["date"] == datetime.now().strftime("%Y-%m-%d")), None)
        if today_eddy is None:
            today_eddy = eddy_days_with_data[0].get("eddy", {})
        status = today_eddy.get("status", "N/A")
        sla = today_eddy.get("zone_max_sla_m")
        sla_str = f"{sla*100:.0f}cm" if sla is not None else ""

        # Determine eddy movement trend across forecast
        dists = [d["eddy"].get("distance_nm") for d in eddy_days_with_data
                 if d["eddy"].get("distance_nm") is not None]
        if len(dists) >= 2:
            if dists[-1] < dists[0] - 5:
                eddy_trend = "APPROACHING"
                eddy_trend_color = "#22c55e"
            elif dists[-1] > dists[0] + 5:
                eddy_trend = "DEPARTING"
                eddy_trend_color = "#fb923c"
            else:
                eddy_trend = "STATIONARY"
                eddy_trend_color = "#38bdf8"
        else:
            eddy_trend = ""
            eddy_trend_color = "#94a3b8"

        eddy_html = f"""
<p style="margin:8px 0">
  <span style="color:#94a3b8">Warm Eddy:</span>
  <span style="color:#c084fc;font-weight:bold"> {status}</span>
  <span style="color:#94a3b8">{f' (SLA {sla_str})' if sla_str else ''}</span>
  {f'<span style="color:{eddy_trend_color};font-weight:bold"> [{eddy_trend}]</span>' if eddy_trend else ''}
</p>
"""
        eddy_plain = f"Warm Eddy: {status}{f' (SLA {sla_str})' if sla_str else ''}{f' [{eddy_trend}]' if eddy_trend else ''}\n"

    html = f"""
<hr style="border-color:#334155;margin:16px 0">
<h3 style="color:#a78bfa">Blue Marlin 7-Day Forecast</h3>
<p style="color:#94a3b8">Accessible Trench Zone - Perth Canyon</p>
<table style="border-collapse:collapse;margin:12px 0">
<tr style="border-bottom:1px solid #334155">
  <th style="padding:3px 10px;text-align:left">Day</th>
  <th style="padding:3px 10px;text-align:center">Habitat</th>
  <th style="padding:3px 10px;text-align:center">Trend</th>
  <th style="padding:3px 10px;text-align:center">Comfort</th>
  <th style="padding:3px 10px;text-align:left">Rating</th>
</tr>
{forecast_rows}
</table>
{eddy_html}
{_build_subzone_html(days)}
<p style="margin:8px 0">
  <span style="color:#94a3b8">Best day:</span>
  <span style="color:#22c55e;font-weight:bold"> {best_name} {best_date:%d %b}</span>
  <span style="color:#94a3b8"> (Habitat {best_day.get('zone_max', 0):.0f}% / Comfort {comfort_map.get(best_day['date'], 0):.0f}%)</span>
</p>
<p style="margin:4px 0">
  <span style="color:#94a3b8">7-Day Trend:</span>
  <span style="color:{trend_color};font-weight:bold"> {trend_text}</span>
</p>
<p style="color:#64748b;font-size:11px;margin:4px 0">
  Rating = 60% Habitat (spatial scoring) + 40% Comfort (boating conditions)
</p>
"""

    # Plain text version
    plain = "\n\nBLUE MARLIN 7-DAY FORECAST\n"
    plain += "Accessible Trench Zone - Perth Canyon\n"
    plain += "-" * 65 + "\n"
    for d in days:
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        day_name = WEEKDAYS[dt.weekday()]
        zone = d.get("zone_max")
        model = d.get("model_score")
        comfort = comfort_map.get(d["date"])
        blend = _blended(d)
        zone_str = f"{zone:5.1f}%" if zone is not None else "  N/A "
        model_str = f"{model:5.1f}%" if model is not None else "  N/A "
        comfort_str = f"{comfort:4.0f}%" if comfort is not None else " N/A"
        is_today = d["date"] == datetime.now().strftime("%Y-%m-%d")
        marker = " <-- TODAY" if is_today else ""
        plain += f"  {day_name} {dt:%d %b}  Habitat:{zone_str}  Trend:{model_str}  Comfort:{comfort_str}  {_score_label(blend, comfort)}{marker}\n"
    if eddy_plain:
        plain += eddy_plain
    # Sub-zone plain text
    today_day = next((d for d in days if d["date"] == datetime.now().strftime("%Y-%m-%d")), days[0] if days else None)
    if today_day and today_day.get("subzones"):
        sz_parts = []
        for key, name in [("canyon", "Canyon"), ("pgfc", "PGFC"), ("north", "North"), ("south", "South")]:
            sz = today_day["subzones"].get(key, {})
            if sz.get("cells", 0) > 0:
                sz_parts.append(f"{name} {sz['max']:.0f}%")
        if sz_parts:
            plain += f"Sub-zones: {' | '.join(sz_parts)}\n"
    plain += f"\nBest day: {best_name} {best_date:%d %b} (Habitat {best_day.get('zone_max', 0):.0f}% / Comfort {comfort_map.get(best_day['date'], 0):.0f}%)\n"
    plain += f"7-Day Trend: {trend_text}\n"
    plain += "Rating = 60% Habitat + 40% Comfort\n"

    return html, plain


def _build_seasonal_outlook():
    """Build seasonal outlook from backtest CHL + zone_spread composite.

    Composite predictor (rho=+0.604, p=0.0005, 83% LOO accuracy):
    - CHL sub-score: food chain priming (lower score = higher actual chlorophyll = better)
    - Zone spread (zone_max - zone_mean): warm water penetration heterogeneity

    Also predicts first-catch timing (rho=-0.483, p=0.036):
    - Higher composite -> earlier first catches (potentially January)

    Only shown during pre-season (Oct-Dec) and early season (Jan-Feb).
    """
    from collections import defaultdict
    import csv

    now = datetime.now()
    month = now.month

    # Only show during relevant months (Oct-Feb)
    if month not in [1, 2, 10, 11, 12]:
        return "", ""

    bt_path = os.path.join(SCRIPT_DIR, "data", "backtest", "backtest_results.json")
    if not os.path.exists(bt_path):
        return "", ""

    try:
        import numpy as np
        from scipy.stats import rankdata

        with open(bt_path) as f:
            bt = json.load(f)

        # Load catches from all sources
        catch_by_season = defaultdict(int)
        first_catch = {}  # year -> earliest catch date string
        geo_path = os.path.join(SCRIPT_DIR, "data", "marlin_catches.geojson")
        csv_path = os.path.join(SCRIPT_DIR, "data", "all_catches.csv")
        if os.path.exists(geo_path):
            with open(geo_path) as f:
                geo = json.load(f)
            for feat in geo["features"]:
                p = feat["properties"]
                if p["species"] == "BLUE MARLIN":
                    dt = datetime.strptime(p["date"], "%Y-%m-%d")
                    if dt.month <= 4:
                        catch_by_season[dt.year] += 1
                        if dt.year not in first_catch or p["date"] < first_catch[dt.year]:
                            first_catch[dt.year] = p["date"]
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    if r["species"] == "BLUE MARLIN":
                        try:
                            dt = datetime.strptime(r["date"], "%d/%m/%Y")
                            if dt.month <= 4:
                                catch_by_season[dt.year] += 1
                                d = dt.strftime("%Y-%m-%d")
                                if dt.year not in first_catch or d < first_catch[dt.year]:
                                    first_catch[dt.year] = d
                        except ValueError:
                            pass

        # Group backtest metrics by year and month
        monthly_chl = defaultdict(lambda: defaultdict(list))
        monthly_spread = defaultdict(lambda: defaultdict(list))
        for e in bt["dates"]:
            year = int(e["date"][:4])
            m = int(e["date"][5:7])
            if e.get("s_chl") is not None:
                monthly_chl[year][m].append(e["s_chl"])
            if e.get("zone_max") is not None and e.get("zone_mean") is not None:
                monthly_spread[year][m].append(e["zone_max"] - e["zone_mean"])

        current_year = now.year
        # For Jan-Feb, show previous year's pre-season signal
        outlook_year = current_year if month >= 10 else current_year - 1

        # Compute current Oct-Nov metrics (or partial if still in progress)
        target_months = [10, 11]
        current_chl_vals = []
        current_spread_vals = []
        for m in target_months:
            if m in monthly_chl.get(outlook_year, {}):
                current_chl_vals.extend(monthly_chl[outlook_year][m])
            if m in monthly_spread.get(outlook_year, {}):
                current_spread_vals.extend(monthly_spread[outlook_year][m])

        if not current_chl_vals:
            return "", ""

        current_chl = np.mean(current_chl_vals)
        current_spread = np.mean(current_spread_vals) if current_spread_vals else None

        # Historical Oct-Nov metrics for all years
        # CHL: LOWER score = HIGHER chlorophyll = BETTER season
        # Spread: HIGHER = more warm water heterogeneity = BETTER season
        hist_chl = {}
        hist_spread = {}
        for year in sorted(monthly_chl.keys()):
            chl_vals = []
            spread_vals = []
            for m in target_months:
                chl_vals.extend(monthly_chl[year].get(m, []))
                spread_vals.extend(monthly_spread[year].get(m, []))
            if len(chl_vals) >= 2:
                hist_chl[year] = np.mean(chl_vals)
            if len(spread_vals) >= 2:
                hist_spread[year] = np.mean(spread_vals)

        if len(hist_chl) < 5:
            return "", ""

        # Build composite using rank-average method
        # Both years must have both metrics for composite
        common_years = sorted(set(hist_chl.keys()) & set(hist_spread.keys()))
        has_composite = current_spread is not None and len(common_years) >= 5

        if has_composite:
            # Rank CHL inverted (lower score = better = higher rank)
            chl_values = [hist_chl[y] for y in common_years]
            spread_values = [hist_spread[y] for y in common_years]

            # Add current year for ranking
            all_chl = chl_values + [current_chl]
            all_spread = spread_values + [current_spread]

            # CHL: rank inverted (lowest score gets highest rank)
            chl_ranks = rankdata([-v for v in all_chl])
            # Spread: rank normal (highest spread gets highest rank)
            spread_ranks = rankdata(all_spread)
            # Composite = average of ranks
            composite_ranks = (chl_ranks + spread_ranks) / 2.0

            # Current year is last element
            current_composite_rank = composite_ranks[-1]
            n_total = len(composite_ranks)
            composite_percentile = current_composite_rank / n_total * 100

            # Historical composite percentiles for similar-year lookup
            hist_composite = {}
            for i, y in enumerate(common_years):
                hist_composite[y] = composite_ranks[i]
        else:
            # Fallback to CHL-only percentile
            all_vals = sorted(hist_chl.values())
            composite_percentile = sum(1 for v in all_vals if v >= current_chl) / len(all_vals) * 100

        # Season strength assessment
        if composite_percentile >= 75:
            outlook = "STRONG"
            outlook_color = "#22c55e"
            outlook_desc = "high productivity + warm water penetration"
        elif composite_percentile >= 50:
            outlook = "ABOVE AVERAGE"
            outlook_color = "#38bdf8"
            outlook_desc = "above average pre-season conditions"
        elif composite_percentile >= 25:
            outlook = "AVERAGE"
            outlook_color = "#fbbf24"
            outlook_desc = "near average pre-season conditions"
        else:
            outlook = "BELOW AVERAGE"
            outlook_color = "#fb923c"
            outlook_desc = "low pre-season productivity"

        # First-catch timing prediction
        # Higher composite -> earlier first catch
        timing_text = ""
        if composite_percentile >= 75:
            timing_text = "First fish likely January"
        elif composite_percentile >= 50:
            timing_text = "First fish likely early-mid February"
        elif composite_percentile >= 25:
            timing_text = "First fish likely late February-March"
        else:
            timing_text = "Late start expected if fish arrive"

        # Find similar historical years (by composite rank proximity if available)
        similar_catches = []
        if has_composite:
            similar = sorted(hist_composite.items(),
                             key=lambda x: abs(x[1] - current_composite_rank))[:3]
            for y, _ in similar:
                next_year = y + 1
                c = catch_by_season.get(next_year, 0)
                fc = first_catch.get(next_year)
                fc_str = fc[5:] if fc else "none"
                similar_catches.append((next_year, c, fc_str))
        else:
            similar = sorted(hist_chl.items(), key=lambda x: abs(x[1] - current_chl))[:3]
            for y, _ in similar:
                next_year = y + 1
                c = catch_by_season.get(next_year, 0)
                fc = first_catch.get(next_year)
                fc_str = fc[5:] if fc else "none"
                similar_catches.append((next_year, c, fc_str))

        # Months included label
        months_done = [m for m in target_months if m in monthly_chl.get(outlook_year, {})]
        month_names = {10: "Oct", 11: "Nov"}
        months_label = "-".join(month_names[m] for m in months_done)
        completeness = f"{len(current_chl_vals)} weeks" if len(months_done) < 2 else "complete"

        # Similar years text
        sim_text = ""
        if similar_catches:
            sim_parts = [f"{y} ({c} fish, 1st {fc})" for y, c, fc in similar_catches]
            sim_text = f"Similar years: {', '.join(sim_parts)}"

        next_season = outlook_year + 1
        season_label = f"{next_season} Season" if month >= 10 else f"{current_year} Season"

        # Detail line
        detail_parts = [f"{months_label} CHL: {current_chl:.4f}"]
        if current_spread is not None:
            detail_parts.append(f"spread: {current_spread:.1f}")
        detail_parts.append(f"composite: {composite_percentile:.0f}th pct")
        detail_parts.append(completeness)
        detail_line = " | ".join(detail_parts)

        html = f"""
<hr style="border-color:#334155;margin:16px 0">
<h3 style="color:#c084fc">Season Outlook</h3>
<p style="margin:4px 0">
  <span style="color:#94a3b8">{season_label}:</span>
  <span style="color:{outlook_color};font-weight:bold"> {outlook}</span>
  <span style="color:#94a3b8"> ({outlook_desc})</span>
</p>
<p style="margin:4px 0;color:#e2e8f0;font-size:13px">
  {timing_text}
</p>
<p style="margin:4px 0;color:#94a3b8;font-size:12px">
  {detail_line}
  {f'<br>{sim_text}' if sim_text else ''}
</p>
"""

        plain = f"\nSEASON OUTLOOK: {season_label}\n"
        plain += f"  {outlook} - {outlook_desc}\n"
        plain += f"  {timing_text}\n"
        plain += f"  {detail_line}\n"
        if sim_text:
            plain += f"  {sim_text}\n"

        return html, plain

    except Exception as e:
        print(f"  Seasonal outlook error: {e}")
        return "", ""


def _build_fads_section():
    """Build FADs Go/No-Go assessment from marine weather data.

    Go criteria for 5.2m boat:
      - Swell < 1.5m all daylight hours
      - Wind < 10kn sustained (gusts < 15kn)
      - No northerly pattern (330-030 deg) — precedes fronts
      - No rain/storms (WMO code < 61)
      - Visibility > 5km
    """
    wx_path = os.path.join(SCRIPT_DIR, "data", "marine_weather.json")
    if not os.path.exists(wx_path):
        return "", ""

    try:
        with open(wx_path) as f:
            wx = json.load(f)
    except Exception:
        return "", ""

    rottnest = wx.get("locations", {}).get("rottnest", {})
    hours = rottnest.get("hourly", [])
    sun_data = rottnest.get("sun", [])
    if not hours:
        return "", ""

    # Build sun lookup: date -> (first_light, last_light)
    sun_map = {}
    for s in sun_data:
        sun_map[s["date"]] = s

    # Group hours by date, daylight only (06:00-18:00 as fallback)
    from collections import defaultdict
    days = defaultdict(list)
    for h in hours:
        t = h["time"]
        date = t.split("T")[0]
        hour = int(t.split("T")[1].split(":")[0])
        # Only daylight hours matter for offshore fishing
        if 5 <= hour <= 18:
            days[date].append(h)

    fads_rows_html = ""
    fads_rows_plain = ""

    # Filter to 7 days: today + 6 forward
    today_str = datetime.now().strftime("%Y-%m-%d")
    max_date_str = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")
    valid_dates = [d for d in sorted(days.keys()) if today_str <= d <= max_date_str]

    for date in valid_dates:
        day_hours = days[date]
        dt = datetime.strptime(date, "%Y-%m-%d")
        day_name = WEEKDAYS[dt.weekday()]
        date_str = dt.strftime("%d %b")

        # Check each criterion
        max_swell = max((h.get("total_swell_height") or h.get("swell_wave_height") or 0) for h in day_hours)
        max_wind = max((h.get("wind_speed_10m") or 0) for h in day_hours)
        max_gust = max((h.get("wind_gusts_10m") or 0) for h in day_hours)
        avg_comfort = sum(h.get("comfort", 50) for h in day_hours) / len(day_hours)

        # Check for northerlies (330-030)
        northerly_hours = 0
        for h in day_hours:
            wd = h.get("wind_direction_10m")
            if wd is not None and (wd >= 330 or wd <= 30):
                northerly_hours += 1

        # Check for rain/storms
        max_wmo = max((h.get("weather_code") or 0) for h in day_hours)
        total_rain = sum((h.get("precipitation") or 0) for h in day_hours)

        # Build warnings
        warnings = []
        if max_swell >= 2.0:
            warnings.append(f"SWELL {max_swell:.1f}m")
        elif max_swell >= 1.5:
            warnings.append(f"Swell {max_swell:.1f}m")
        if max_wind >= 15:
            warnings.append(f"WIND {max_wind:.0f}kn")
        elif max_wind >= 10:
            warnings.append(f"Wind {max_wind:.0f}kn")
        if max_gust >= 20:
            warnings.append(f"Gusts {max_gust:.0f}kn")
        if northerly_hours >= 3:
            warnings.append("N'ly pattern")
        elif northerly_hours >= 1:
            warnings.append("N'ly shift")
        if max_wmo >= 95:
            warnings.append("STORM")
        elif max_wmo >= 61:
            warnings.append("Rain")
        elif total_rain > 2:
            warnings.append("Showers")

        # Go/No-Go decision
        go = True
        if max_swell >= 1.5:
            go = False
        if max_wind >= 15:
            go = False
        if max_gust >= 25:
            go = False
        if northerly_hours >= 3:
            go = False
        if max_wmo >= 80:
            go = False

        # Marginal
        marginal = False
        if go and (max_swell >= 1.2 or max_wind >= 10 or max_gust >= 18
                   or northerly_hours >= 1 or max_wmo >= 51):
            marginal = True

        if go and not marginal:
            status = "GO"
            status_color = "#22c55e"
            status_icon = ">>>"
        elif go and marginal:
            status = "MARGINAL"
            status_color = "#fbbf24"
            status_icon = "~"
        else:
            status = "NO GO"
            status_color = "#ef4444"
            status_icon = "X"

        warning_str = ", ".join(warnings) if warnings else "Clear"
        is_today = date == datetime.now().strftime("%Y-%m-%d")
        row_bg = "background:#1e293b;" if is_today else ""
        today_tag = " *" if is_today else ""

        # Sun times for this date
        sd = sun_map.get(date)
        if sd:
            def _fmt_sun(iso):
                return iso.split("T")[1][:5]
            fl = _fmt_sun(sd["first_light"])
            sr = _fmt_sun(sd["sunrise"])
            ss = _fmt_sun(sd["sunset"])
            ll = _fmt_sun(sd["last_light"])
            sun_html = f'{fl} / {sr} / {ss} / {ll}'
            sun_plain = f"  FL:{fl} SR:{sr} SS:{ss} LL:{ll}"
        else:
            sun_html = "N/A"
            sun_plain = ""

        fads_rows_html += (
            f'<tr style="{row_bg}">'
            f'<td style="padding:3px 10px">{day_name} {date_str}{today_tag}</td>'
            f'<td style="padding:3px 10px;color:{status_color};font-weight:bold;text-align:center">{status}</td>'
            f'<td style="padding:3px 10px;text-align:center">{max_wind:.0f}kn</td>'
            f'<td style="padding:3px 10px;text-align:center">{max_swell:.1f}m</td>'
            f'<td style="padding:3px 10px;text-align:center">{avg_comfort:.0f}%</td>'
            f'<td style="padding:3px 10px;color:#fb923c;font-size:11px;text-align:center;white-space:nowrap">{sun_html}</td>'
            f'<td style="padding:3px 10px;color:#94a3b8;font-size:12px">{warning_str}</td>'
            f'</tr>\n'
        )
        fads_rows_plain += (
            f"  {status_icon:>3} {day_name} {date_str}  {status:>8}  "
            f"Wind:{max_wind:4.0f}kn  Swell:{max_swell:4.1f}m  "
            f"Comfort:{avg_comfort:3.0f}%{sun_plain}  {warning_str}{' <-- TODAY' if is_today else ''}\n"
        )

    html = f"""
<hr style="border-color:#334155;margin:16px 0">
<h3 style="color:#fb923c">FADs Go/No-Go (5.2m boat)</h3>
<p style="color:#94a3b8;font-size:12px">
  GO: swell &lt;1.5m, wind &lt;15kn, no storms, no persistent northerlies
</p>
<table style="border-collapse:collapse;margin:12px 0">
<tr style="border-bottom:1px solid #334155">
  <th style="padding:3px 10px;text-align:left">Day</th>
  <th style="padding:3px 10px;text-align:center">Status</th>
  <th style="padding:3px 10px;text-align:center">Wind</th>
  <th style="padding:3px 10px;text-align:center">Swell</th>
  <th style="padding:3px 10px;text-align:center">Comfort</th>
  <th style="padding:3px 10px;text-align:center">FL / Rise / Set / LL</th>
  <th style="padding:3px 10px;text-align:left">Warnings</th>
</tr>
{fads_rows_html}
</table>
"""

    plain = "\n\nFADs GO/NO-GO (5.2m boat)\n"
    plain += "GO: swell <1.5m, wind <15kn, no storms, no persistent northerlies\n"
    plain += "-" * 70 + "\n"
    plain += fads_rows_plain

    return html, plain


def _generate_email_charts():
    """Generate wind, swell, and accessible trench zone charts as PNG bytes."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import to_rgba
    import numpy as np

    # Dark theme matching the app
    BG = "#0f172a"
    FG = "#e2e8f0"
    GRID = "#1e293b"
    MUTED = "#64748b"

    # Sunlight strip colors (matching index.html wxDrawSunStrip)
    SUN_NIGHT = "#0f172a"
    SUN_DAWN = "#c2410c"
    SUN_GLOW = "#fb923c"
    SUN_DAY = "#38bdf8"

    wx_path = os.path.join(SCRIPT_DIR, "data", "marine_weather.json")
    if not os.path.exists(wx_path):
        return {}
    with open(wx_path) as f:
        wx = json.load(f)

    rottnest = wx.get("locations", {}).get("rottnest", {})
    hours = rottnest.get("hourly", [])
    sun_data = rottnest.get("sun", [])
    if not hours:
        return {}

    # Filter to 7 days from today
    today_str = datetime.now().strftime("%Y-%m-%d")
    max_date = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")
    hours = [h for h in hours if today_str <= h["time"].split("T")[0] <= max_date]
    if not hours:
        return {}

    times = [datetime.strptime(h["time"], "%Y-%m-%dT%H:%M") for h in hours]

    # Build sun lookup: date -> {first_light, sunrise, sunset, last_light} as minutes
    sun_map = {}
    for s in (sun_data or []):
        def _to_min(iso):
            p = iso.split("T")[1].split(":")
            return int(p[0]) * 60 + int(p[1])
        sun_map[s["date"]] = {
            "fl": _to_min(s["first_light"]),
            "sr": _to_min(s["sunrise"]),
            "ss": _to_min(s["sunset"]),
            "ll": _to_min(s["last_light"]),
        }

    def _sun_color(iso_time):
        """Return sunlight color for a given ISO time, matching the app."""
        date = iso_time.split("T")[0]
        p = iso_time.split("T")[1].split(":")
        t_min = int(p[0]) * 60 + int(p[1])
        sd = sun_map.get(date)
        if not sd:
            return SUN_NIGHT
        if t_min < sd["fl"] or t_min >= sd["ll"]:
            return SUN_NIGHT
        if t_min < sd["sr"]:
            return SUN_DAWN
        if t_min < sd["sr"] + 30:
            return SUN_GLOW
        if t_min >= sd["ss"] - 30 and t_min < sd["ss"]:
            return SUN_GLOW
        if t_min >= sd["ss"]:
            return SUN_DAWN
        return SUN_DAY

    def _draw_sun_strip(ax, hours_list, times_list):
        """Draw sunlight strip at bottom of chart, matching the app."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        strip_h = (ylim[1] - ylim[0]) * 0.04
        strip_y = ylim[0]
        for i in range(len(hours_list)):
            color = _sun_color(hours_list[i]["time"])
            t0 = mdates.date2num(times_list[i])
            t1 = mdates.date2num(times_list[i + 1]) if i + 1 < len(times_list) else t0 + 1/24
            ax.barh(strip_y + strip_h / 2, t1 - t0, left=t0, height=strip_h,
                    color=color, zorder=5, clip_on=True)
        # Adjust ylim to include strip
        ax.set_ylim(ylim[0], ylim[1])

    def style_ax(ax, ylabel):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_ylabel(ylabel, color=FG, fontsize=8)
        # Centre day labels at midday (12:00) instead of midnight
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        # Draw midnight gridlines
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.grid(True, which="minor", alpha=0.12, color="#fff", linewidth=0.5)
        ax.grid(False, which="major")
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    charts = {}

    # --- Wind Chart ---
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=BG)
    winds = [h.get("wind_speed_10m") or 0 for h in hours]
    gusts = [h.get("wind_gusts_10m") or 0 for h in hours]

    def wind_color(kn):
        if kn <= 10: return "#22c55e"
        if kn <= 15: return "#fbbf24"
        if kn <= 20: return "#fb923c"
        if kn <= 25: return "#ef4444"
        return "#dc2626"

    # Line chart style (like Seabreeze/Willy Weather)
    ax.plot(times, gusts, color="#fb923c", linewidth=1, alpha=0.3, label="Gusts", zorder=3)
    ax.plot(times, winds, color="#38bdf8", linewidth=0.5, label="Wind", zorder=3)

    # Faint horizontal gridlines every 5 knots
    for kn in range(5, int(max(max(gusts), 25)) + 5, 5):
        ax.axhline(kn, color="#ffffff", linewidth=0.3, alpha=0.15, zorder=1)
    ax.axhline(15, color="#ef4444", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(times[0], 15.5, "15kn limit", color="#ef4444", fontsize=6, alpha=0.7)

    style_ax(ax, "Wind (kn)")
    ax.set_title("Wind Speed & Gusts \u2014 Rottnest", color=FG, fontsize=9, pad=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ymax = max(max(gusts), 25) + 5
    ax.set_ylim(0, ymax)

    # Wind direction arrows every 3 hours ON the wind line
    # Wind comes FROM wd degrees; arrow shows air movement direction (wd+180)
    import math
    arrow_times = []
    arrow_ys = []
    arrow_u = []
    arrow_v = []
    arrow_colors = []
    for i in range(0, len(hours), 3):
        wd = hours[i].get("wind_direction_10m")
        if wd is None:
            continue
        arrow_times.append(mdates.date2num(times[i]))
        arrow_ys.append(winds[i])
        to_rad = math.radians(wd + 180)
        arrow_u.append(math.sin(to_rad))   # screen-right (east)
        arrow_v.append(math.cos(to_rad))   # screen-up (north)
        arrow_colors.append(wind_color(winds[i]))
    if arrow_times:
        ax.quiver(arrow_times, arrow_ys, arrow_u, arrow_v,
                  angles="uv", scale_units="height", scale=12,
                  width=0.004, headwidth=4, headlength=3, headaxislength=2.5,
                  pivot="mid", color=arrow_colors, alpha=0.9, zorder=5)

    _draw_sun_strip(ax, hours, times)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=BG)
    plt.close(fig)
    charts["wind"] = buf.getvalue()

    # --- Swell Chart ---
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=BG)
    swells = [h.get("total_swell_height") or h.get("swell_wave_height") or 0 for h in hours]
    waves = [h.get("wave_height") or 0 for h in hours]

    ax.fill_between(times, waves, alpha=0.1, color="#38bdf8", zorder=1)
    ax.plot(times, swells, color="#38bdf8", linewidth=1.5, label="Swell", zorder=3)
    ax.plot(times, waves, color="#38bdf8", linewidth=0.8, linestyle="--", alpha=0.4, label="Wave", zorder=2)

    ax.axhline(1.5, color="#ef4444", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(times[0], 1.55, "1.5m limit", color="#ef4444", fontsize=6, alpha=0.7)

    for i in range(0, len(hours), 6):
        sp = hours[i].get("dominant_swell_period") or hours[i].get("swell_wave_period")
        if sp:
            ax.annotate(f"{sp:.0f}s", (times[i], swells[i]),
                        textcoords="offset points", xytext=(0, 8),
                        color=MUTED, fontsize=5.5, ha="center")

    style_ax(ax, "Height (m)")
    ax.set_title("Swell & Wave Height \u2014 Rottnest", color=FG, fontsize=9, pad=8)
    ymax = max(max(swells), max(waves), 2) + 0.5
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.3, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    _draw_sun_strip(ax, hours, times)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=BG)
    plt.close(fig)
    charts["swell"] = buf.getvalue()

    # --- Accessible Trench Zone Chart ---
    summary_path = os.path.join(SCRIPT_DIR, "data", "prediction", "forecast_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        fdays = summary.get("days", [])
        fdays = [d for d in fdays if today_str <= d["date"] <= max_date]

        if fdays:
            fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=BG)
            # Centre bars at midday
            dates = [datetime.strptime(d["date"], "%Y-%m-%d") + timedelta(hours=12) for d in fdays]
            zone_scores = [d.get("zone_max") or 0 for d in fdays]
            model_scores = [d.get("model_score") or 0 for d in fdays]

            bar_colors = [_score_color(s) for s in zone_scores]
            ax.bar(dates, zone_scores, width=0.6, color=bar_colors, alpha=0.8, label="Habitat Score", zorder=2)

            ax.plot(dates, model_scores, color="#fb923c", linewidth=1.5, linestyle="--",
                    marker="o", markersize=4, alpha=0.8, label="Trend Score", zorder=3)

            for i, (d, s) in enumerate(zip(dates, zone_scores)):
                ax.text(d, s + 1.5, f"{s:.0f}%", ha="center", va="bottom",
                        color=FG, fontsize=7, fontweight="bold")

            style_ax(ax, "Score (%)")
            ax.set_title("Accessible Trench Zone \u2014 Habitat & Marlin Scores", color=FG, fontsize=9, pad=8)
            ax.set_ylim(0, max(max(zone_scores), max(model_scores), 50) + 10)
            ax.legend(loc="upper left", fontsize=6, framealpha=0.3, facecolor=BG, edgecolor=GRID, labelcolor=FG)
            fig.tight_layout(pad=0.5)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, facecolor=BG)
            plt.close(fig)
            charts["trench"] = buf.getvalue()

    return charts


def install_task():
    """Install Windows scheduled task to run daily at 05:00."""
    python_path = sys.executable
    script_path = os.path.join(SCRIPT_DIR, "run_daily.py")
    cmd = (
        f'schtasks /Create /TN "{TASK_NAME}" '
        f'/TR "\"{python_path}\" \"{script_path}\"" '
        f'/SC DAILY /ST 05:00 '
        f'/F'
    )
    print(f"Installing scheduled task: {TASK_NAME}")
    print(f"  Schedule: Daily at 05:00 AWST")
    print(f"  Command: {python_path} {script_path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Task installed successfully.")
        print(f"  To verify: schtasks /Query /TN \"{TASK_NAME}\"")
    else:
        print(f"  Failed: {result.stderr}")
        if "Access is denied" in result.stderr:
            print("  Try running as Administrator.")
    return result.returncode


def uninstall_task():
    """Remove the scheduled task."""
    cmd = f'schtasks /Delete /TN "{TASK_NAME}" /F'
    print(f"Removing scheduled task: {TASK_NAME}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("  Task removed.")
    else:
        print(f"  Failed: {result.stderr}")
    return result.returncode


def main():
    if "--install" in sys.argv:
        sys.exit(install_task())
    elif "--uninstall" in sys.argv:
        sys.exit(uninstall_task())
    else:
        sys.exit(run_pipeline())


if __name__ == "__main__":
    main()
