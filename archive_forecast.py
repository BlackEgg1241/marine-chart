"""Archive today's Open-Meteo forecast for future verification."""
import json, os, sys
from datetime import datetime

# Import from our existing fetch module
from fetch_marine_weather import LOCATIONS, fetch_location


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    archive_dir = os.path.join("data", "forecast_archive", today)
    out_path = os.path.join(archive_dir, "forecast_raw.json")

    if os.path.exists(out_path):
        print(f"Already archived for {today}, skipping.")
        return

    os.makedirs(archive_dir, exist_ok=True)

    result = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "archive_date": today,
        "locations": {},
    }
    for name, loc in LOCATIONS.items():
        result["locations"][name] = fetch_location(name, loc)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    for name, data in result["locations"].items():
        n = len(data["hourly"])
        t0 = data["hourly"][0]["time"] if n else "?"
        t1 = data["hourly"][-1]["time"] if n else "?"
        print(f"  {name}: {n} hours ({t0} -> {t1})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
