"""Batch-fetch subsurface temperature (250m) for all catch dates.
This data is needed for stratification index and thermocline_lift scoring."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from optimize_visual import load_catches
from marlin_data import fetch_copernicus_subsurface_temp, DEFAULT_BBOX as BBOX

dates = sorted(set(c['date'] for c in load_catches()))
print(f"Fetching subsurface temp for {len(dates)} catch dates...")

success = 0
for i, d in enumerate(dates, 1):
    out_dir = os.path.join("data", d)
    out_file = os.path.join(out_dir, "subsurface_temp_raw.nc")
    if os.path.exists(out_file):
        print(f"[{i}/{len(dates)}] {d} — already exists, skipping")
        success += 1
        continue

    # Temporarily set OUTPUT_DIR so fetch writes to the right place
    import marlin_data
    marlin_data.OUTPUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)

    try:
        result = fetch_copernicus_subsurface_temp(d, BBOX)
        if result:
            success += 1
            print(f"[{i}/{len(dates)}] {d} — OK")
        else:
            print(f"[{i}/{len(dates)}] {d} — NO DATA")
    except Exception as e:
        print(f"[{i}/{len(dates)}] {d} — FAILED: {e}")

print(f"\nDone: {success}/{len(dates)} dates have subsurface temp data")
