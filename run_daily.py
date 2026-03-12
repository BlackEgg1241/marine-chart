"""Daily automation: run all MarLEEn data pipelines.

Runs in sequence:
  1. fetch_marine_weather.py   - Weather panel data (wind/swell/comfort)
  2. fetch_prediction.py       - 7-day marlin hotspot maps
  3. generate_forecast_summary.py - Zone scores for UI
  4. archive_forecast.py       - Archive today's forecast for verification

Usage:
    python run_daily.py              # run all steps
    python run_daily.py --install    # install Windows scheduled task (14:00 AWST daily)
    python run_daily.py --uninstall  # remove scheduled task
    python run_daily.py --no-email   # run without sending email

Email setup:
    Set MARLEEN_GMAIL_APP_PASSWORD environment variable with your Gmail app password.
    Generate one at: https://myaccount.google.com/apppasswords

Windows Task Scheduler:
    The --install flag creates a task "MarLEEn Daily Update" that runs
    daily at 14:00 AWST (06:00 UTC), after ECMWF model runs complete.
"""
import os, smtplib, subprocess, sys, time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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

    # Send email summary
    if "--no-email" not in sys.argv:
        send_email_summary(today, results, total_elapsed, all_ok)

    return 0 if all_ok else 1


def send_email_summary(today, results, total_elapsed, all_ok):
    """Send pipeline summary email via Gmail SMTP."""
    app_password = os.environ.get("MARLEEN_GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("\n  Email skipped: MARLEEN_GMAIL_APP_PASSWORD not set")
        print("  To enable: set MARLEEN_GMAIL_APP_PASSWORD=<your-gmail-app-password>")
        print("  Generate at: https://myaccount.google.com/apppasswords")
        return

    status_icon = "OK" if all_ok else "FAILURES"
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
<p style="color:#94a3b8">Total runtime: {total_elapsed:.1f}s</p>
<hr style="border-color:#334155">
<p style="color:#64748b;font-size:12px">MarLEEn Tracker - Automated Pipeline Report</p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    # Plain text fallback
    plain = f"MarLEEn Daily Pipeline - {today:%Y-%m-%d %H:%M:%S}\n\n"
    for name, status, elapsed, _ in results:
        indicator = "+" if status == "OK" else "x"
        plain += f"  [{indicator}] {name}: {status} ({elapsed:.1f}s)\n"
    plain += f"\nTotal: {total_elapsed:.1f}s\nStatus: {'ALL OK' if all_ok else 'SOME FAILURES'}\n"

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, app_password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"\n  Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"\n  Email failed: {e}")


def install_task():
    """Install Windows scheduled task to run daily at 14:00."""
    python_path = sys.executable
    script_path = os.path.join(SCRIPT_DIR, "run_daily.py")
    cmd = (
        f'schtasks /Create /TN "{TASK_NAME}" '
        f'/TR "\"{python_path}\" \"{script_path}\"" '
        f'/SC DAILY /ST 14:00 '
        f'/F'
    )
    print(f"Installing scheduled task: {TASK_NAME}")
    print(f"  Schedule: Daily at 14:00 AWST")
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
