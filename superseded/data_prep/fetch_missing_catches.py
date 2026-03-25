"""Fetch observation data for catch dates missing data directories."""
import sys, os, shutil
sys.path.insert(0, '.')
import marlin_data

# Missing observation dates for catch records
MISSING_DATES = [
    "2022-03-12", "2023-02-25", "2023-02-26", "2023-03-18", "2023-03-21",
    "2024-01-07", "2024-01-20", "2024-02-01", "2024-02-16", "2024-03-02",
    "2024-03-03", "2025-01-02", "2025-01-11", "2025-01-15", "2025-02-08",
    "2026-02-01", "2026-02-14", "2026-02-15",
]

bbox = dict(marlin_data.DEFAULT_BBOX)

for i, date_str in enumerate(MISSING_DATES):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(MISSING_DATES)}] {date_str}")
    print(f"{'='*60}")

    out_dir = os.path.join("data", date_str)
    os.makedirs(out_dir, exist_ok=True)
    marlin_data.OUTPUT_DIR = out_dir

    fetchers = [
        ("SST", marlin_data.fetch_copernicus_sst),
        ("Currents", marlin_data.fetch_copernicus_currents),
        ("CHL", marlin_data.fetch_copernicus_chlorophyll),
        ("KD490", marlin_data.fetch_copernicus_kd490),
        ("SSH", marlin_data.fetch_copernicus_ssh),
        ("MLD", marlin_data.fetch_copernicus_mld),
        ("Oxygen", marlin_data.fetch_copernicus_oxygen),
    ]
    for name, fn in fetchers:
        try:
            fn(date_str, bbox)
        except Exception as e:
            print(f"  [{name}] Error: {e}")

    try:
        tif = os.path.join(out_dir, "bathy_gmrt.tif")
        if not os.path.exists(tif):
            src = "data/bathy_gmrt.tif"
            if os.path.exists(src):
                shutil.copy2(src, tif)
        tif_path = tif if os.path.exists(tif) else None
        result = marlin_data.generate_blue_marlin_hotspots(bbox, tif_path=tif_path)
        if result:
            print(f"  -> Hotspots generated")
        else:
            print(f"  -> Scoring returned None")
    except Exception as e:
        print(f"  -> Scoring error: {e}")

print("\nDone!")
