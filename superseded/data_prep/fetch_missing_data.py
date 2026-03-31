"""Batch-fetch all missing data for catch dates:
1. Salinity (surface so)
2. Subsurface temperature (thetao 200-250m)
3. FTLE neighbor currents (day-1 and day+1 for each catch date)
"""
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from optimize_visual import load_catches
from marlin_data import (
    fetch_copernicus_salinity,
    fetch_copernicus_currents,
    fetch_copernicus_subsurface_temp,
    DEFAULT_BBOX as BBOX,
)
import marlin_data

dates = sorted(set(c['date'] for c in load_catches()))
print(f"=== Fetching missing data for {len(dates)} catch dates ===\n")

# --- 1. Salinity ---
print("--- SALINITY ---")
sal_ok = 0
for i, d in enumerate(dates, 1):
    out_dir = os.path.join("data", d)
    out_file = os.path.join(out_dir, "salinity_raw.nc")
    if os.path.exists(out_file):
        sal_ok += 1
        continue
    marlin_data.OUTPUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = fetch_copernicus_salinity(d, BBOX)
        if result:
            sal_ok += 1
            print(f"[{i}/{len(dates)}] {d} salinity OK", flush=True)
        else:
            print(f"[{i}/{len(dates)}] {d} salinity NO DATA", flush=True)
    except Exception as e:
        print(f"[{i}/{len(dates)}] {d} salinity FAILED: {e}", flush=True)
print(f"Salinity: {sal_ok}/{len(dates)} dates\n")

# --- 2. Subsurface Temperature ---
print("--- SUBSURFACE TEMP (200-250m) ---")
sub_ok = 0
for i, d in enumerate(dates, 1):
    out_dir = os.path.join("data", d)
    out_file = os.path.join(out_dir, "subsurface_temp_raw.nc")
    if os.path.exists(out_file):
        sub_ok += 1
        continue
    marlin_data.OUTPUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = fetch_copernicus_subsurface_temp(d, BBOX)
        if result:
            sub_ok += 1
            print(f"[{i}/{len(dates)}] {d} subsurface_temp OK", flush=True)
        else:
            print(f"[{i}/{len(dates)}] {d} subsurface_temp NO DATA", flush=True)
    except Exception as e:
        print(f"[{i}/{len(dates)}] {d} subsurface_temp FAILED: {e}", flush=True)
print(f"Subsurface temp: {sub_ok}/{len(dates)} dates\n")

# --- 3. FTLE neighbor currents ---
print("--- FTLE NEIGHBOR CURRENTS ---")
neighbor_dates = set()
for d in dates:
    dt = datetime.strptime(d, "%Y-%m-%d")
    for delta in [-1, 1]:
        nd = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        neighbor_dates.add(nd)
neighbor_dates = sorted(nd for nd in neighbor_dates
                        if not os.path.exists(os.path.join("data", nd, "currents_raw.nc")))
print(f"Need to fetch currents for {len(neighbor_dates)} neighbor dates")

cur_ok = 0
for i, d in enumerate(neighbor_dates, 1):
    out_dir = os.path.join("data", d)
    marlin_data.OUTPUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = fetch_copernicus_currents(d, BBOX)
        if result:
            cur_ok += 1
            print(f"[{i}/{len(neighbor_dates)}] {d} currents OK", flush=True)
        else:
            print(f"[{i}/{len(neighbor_dates)}] {d} currents NO DATA", flush=True)
    except Exception as e:
        print(f"[{i}/{len(neighbor_dates)}] {d} currents FAILED: {e}", flush=True)
print(f"Neighbor currents: {cur_ok}/{len(neighbor_dates)} dates\n")

# --- Summary ---
print("=== SUMMARY ===")
sal_final = sum(1 for d in dates if os.path.exists(os.path.join("data", d, "salinity_raw.nc")))
sub_final = sum(1 for d in dates if os.path.exists(os.path.join("data", d, "subsurface_temp_raw.nc")))
ftle_final = 0
for d in dates:
    dt = datetime.strptime(d, "%Y-%m-%d")
    prev = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    if (os.path.exists(os.path.join("data", prev, "currents_raw.nc")) and
        os.path.exists(os.path.join("data", nxt, "currents_raw.nc"))):
        ftle_final += 1
print(f"Salinity:        {sal_final}/{len(dates)} catch dates")
print(f"Subsurface temp: {sub_final}/{len(dates)} catch dates")
print(f"FTLE neighbors:  {ftle_final}/{len(dates)} catch dates with day-1 + day+1 currents")
