#!/usr/bin/env python3
"""
fetch_lookback.py — Fetch ocean data for 7 days prior to each blue marlin catch.

Downloads SST, currents, SSH, CHL, MLD from Copernicus Marine for the week
before each catch date. Data is stored in a SEPARATE directory tree:

    data/lookback/YYYY-MM-DD/   <-- trend/prediction analysis data
    data/YYYY-MM-DD/            <-- catch date data (maps, scoring)

This separation prevents lookback data from being confused with catch data.

Usage:
    python fetch_lookback.py              # fetch all missing dates
    python fetch_lookback.py --days 3     # only 3 days prior (not 7)
    python fetch_lookback.py --dry-run    # show what would be fetched
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime, timedelta

import copernicusmarine

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
BASE_DIR = "data"
LOOKBACK_DIR = os.path.join(BASE_DIR, "lookback")  # separate from catch data
BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}


def ddm_to_dd(raw_str, negative=False):
    """Convert degrees.minutes string (e.g. '31.49') to decimal degrees."""
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_blue_marlin_dates():
    """Return sorted list of unique blue marlin catch dates."""
    dates = set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Species_Name"] == "BLUE MARLIN":
                dates.add(r["Release_Date"][:10])
    return sorted(dates)


def _subset(out_path, dataset_id, variables, date_str, depth_range=None):
    """Single copernicusmarine.subset call."""
    kwargs = dict(
        dataset_id=dataset_id, variables=variables,
        minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
        minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
        start_datetime=f"{date_str}T00:00:00", end_datetime=f"{date_str}T23:59:59",
        output_filename=out_path, output_directory=".", overwrite=True,
    )
    if depth_range:
        kwargs["minimum_depth"] = depth_range[0]
        kwargs["maximum_depth"] = depth_range[1]
    copernicusmarine.subset(**kwargs)


def fetch_date(date_str, out_dir):
    """Fetch all ocean variables for a single date into lookback directory."""
    os.makedirs(out_dir, exist_ok=True)
    # Mark as lookback data (not catch data)
    marker = os.path.join(out_dir, "_LOOKBACK_DATA")
    if not os.path.exists(marker):
        with open(marker, "w") as f:
            f.write(f"Lookback trend data for prediction analysis.\n"
                    f"Date: {date_str}\n"
                    f"NOT a catch date — do not use for map generation.\n")
    results = {}

    # --- SST ---
    sst_path = os.path.join(out_dir, "sst_raw.nc")
    if not os.path.exists(sst_path):
        for ds_id, var in [
            ("METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "analysed_sst"),
            ("cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao"),
        ]:
            try:
                depth = (0, 1) if var == "thetao" else None
                _subset(sst_path, ds_id, [var], date_str, depth)
                results["sst"] = True
                break
            except Exception:
                continue
        if "sst" not in results:
            print(f"    [SST] FAILED")
    else:
        results["sst"] = True

    # --- Currents ---
    cur_path = os.path.join(out_dir, "currents_raw.nc")
    if not os.path.exists(cur_path):
        for ds_id in [
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        ]:
            try:
                _subset(cur_path, ds_id, ["uo", "vo"], date_str, (0, 1))
                results["currents"] = True
                break
            except Exception:
                continue
        if "currents" not in results:
            print(f"    [Currents] FAILED")
    else:
        results["currents"] = True

    # --- SSH ---
    ssh_path = os.path.join(out_dir, "ssh_raw.nc")
    if not os.path.exists(ssh_path):
        for ds_id in [
            "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
            "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        ]:
            try:
                _subset(ssh_path, ds_id, ["sla"], date_str)
                results["ssh"] = True
                break
            except Exception:
                continue
        if "ssh" not in results:
            print(f"    [SSH] FAILED")
    else:
        results["ssh"] = True

    # --- Chlorophyll ---
    chl_path = os.path.join(out_dir, "chl_raw.nc")
    if not os.path.exists(chl_path):
        for ds_id, var in [
            ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
            ("cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", "CHL"),
            ("cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m", "chl"),
        ]:
            try:
                depth = (0, 1) if var == "chl" else None
                _subset(chl_path, ds_id, [var], date_str, depth)
                results["chl"] = True
                break
            except Exception:
                continue
        if "chl" not in results:
            print(f"    [CHL] FAILED")
    else:
        results["chl"] = True

    # --- MLD ---
    mld_path = os.path.join(out_dir, "mld_raw.nc")
    if not os.path.exists(mld_path):
        for ds_id in [
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        ]:
            try:
                _subset(mld_path, ds_id, ["mlotst"], date_str)
                results["mld"] = True
                break
            except Exception:
                continue
        if "mld" not in results:
            print(f"    [MLD] FAILED")
    else:
        results["mld"] = True

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days prior to fetch (default: 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without downloading")
    args = parser.parse_args()

    os.makedirs(LOOKBACK_DIR, exist_ok=True)

    catch_dates = load_blue_marlin_dates()
    print(f"Blue marlin catch dates: {len(catch_dates)}")
    print(f"Lookback storage: {LOOKBACK_DIR}/YYYY-MM-DD/")

    # Build list of lookback dates, tracking which catch they relate to
    lookback_dates = set()
    for d in catch_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        for offset in range(1, args.days + 1):
            lookback_dates.add((dt - timedelta(days=offset)).strftime("%Y-%m-%d"))

    lookback_dates = sorted(lookback_dates)

    # Check what's already cached (in lookback dir OR catch data dir)
    need_fetch = []
    already_cached = 0
    for d in lookback_dates:
        lookback_path = os.path.join(LOOKBACK_DIR, d, "sst_raw.nc")
        catch_path = os.path.join(BASE_DIR, d, "sst_raw.nc")
        if os.path.exists(lookback_path) or os.path.exists(catch_path):
            already_cached += 1
        else:
            need_fetch.append(d)

    print(f"Lookback dates ({args.days} days prior): {len(lookback_dates)}")
    print(f"Already cached: {already_cached}")
    print(f"Need to fetch: {len(need_fetch)}")

    if args.dry_run:
        print("\nDates to fetch:")
        for d in need_fetch:
            print(f"  {d}")
        return

    if len(need_fetch) == 0:
        print("\nAll lookback data already cached!")
        return

    print(f"\nFetching {len(need_fetch)} dates into {LOOKBACK_DIR}/...\n")

    success = 0
    failed = 0
    for i, date_str in enumerate(need_fetch):
        out_dir = os.path.join(LOOKBACK_DIR, date_str)
        print(f"[{i+1}/{len(need_fetch)}] {date_str}")
        try:
            results = fetch_date(date_str, out_dir)
            n_ok = sum(1 for v in results.values() if v)
            print(f"    -> {n_ok}/5 variables OK")
            if n_ok >= 3:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    -> ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"COMPLETE: {success} dates OK, {failed} dates with issues")
    print(f"Total cached dates: {already_cached + success}")
    print(f"\nLookback data stored in: {LOOKBACK_DIR}/")
    print(f"Catch date data remains in: {BASE_DIR}/YYYY-MM-DD/")


if __name__ == "__main__":
    main()
