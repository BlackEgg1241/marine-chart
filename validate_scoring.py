#!/usr/bin/env python3
"""
validate_scoring.py — Fetch ocean data for all historical marlin catch dates
and run our scoring algorithm to validate accuracy.

Reads catch records from Export.csv, fetches SST/currents/SSH/CHL/MLD for
each unique catch date, runs the hotspot scoring, and samples the score
at each catch location.

Output: data/validation_results.csv
"""

import csv
import json
import os
import shutil
from datetime import datetime

import numpy as np
import copernicusmarine

import marlin_data
from marlin_data import generate_blue_marlin_hotspots

BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}

CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")
BASE_DIR = "data"


def ddm_to_dd(raw_str, negative=False):
    """Convert degrees.minutes string (e.g. '31.49') to decimal degrees."""
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def _parse_date(date_str):
    """Convert DD/MM/YYYY or YYYY-MM-DD to YYYY-MM-DD."""
    if "/" in date_str:
        parts = date_str.strip().split("/")
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    return date_str[:10]


def load_catches():
    catches = []
    seen = set()

    # GFAA Export.csv (DDM format coordinates)
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            wt = float(r["Weight"]) if r["Weight"] else None
            ln = float(r["Length"]) if r["Length"] and r["Length"] != "0" else None
            date = r["Release_Date"][:10]
            key = (date, round(lat, 4), round(lon, 4))
            if key not in seen:
                seen.add(key)
                catches.append({
                    "date": date, "lat": lat, "lon": lon,
                    "species": r["Species_Name"], "weight": wt, "length": ln,
                    "tag": r["Tag_Number"],
                })

    # all_catches.csv (decimal degrees, only rows with GPS coords)
    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lat_str = r.get("lat", "").strip()
                lon_str = r.get("lon", "").strip()
                if not lat_str or not lon_str:
                    continue
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    continue
                date = _parse_date(r["date"])
                key = (date, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({
                        "date": date, "lat": lat, "lon": lon,
                        "species": r.get("species", ""),
                        "weight": None, "length": None,
                        "tag": r.get("tag", ""),
                    })

    return [c for c in catches if c["species"].strip().upper() == "BLUE MARLIN"]


def _subset(out_path, dataset_id, variables, date_str, depth_range=None):
    """Single copernicusmarine.subset call with consistent params."""
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


def fetch_all(date_str, out_dir):
    """Fetch SST, currents, SSH, CHL, MLD for a date. Uses reanalysis for older dates."""
    year = int(date_str[:4])
    results = {}

    # --- SST ---
    sst_path = os.path.join(out_dir, "sst_raw.nc")
    if not os.path.exists(sst_path):
        # Observation L4 goes back to ~2007; reanalysis for older
        for ds_id, var in [
            ("METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "analysed_sst"),
            ("cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao"),
        ]:
            try:
                depth = (0, 1) if var == "thetao" else None
                _subset(sst_path, ds_id, [var], date_str, depth)
                print(f"  [SST] OK ({ds_id.split('_')[0][:10]})")
                results["sst"] = True
                break
            except Exception as e:
                continue
        if "sst" not in results:
            print(f"  [SST] FAILED")
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
                print(f"  [Currents] OK ({ds_id.split('_')[-1][:10]})")
                results["currents"] = True
                break
            except Exception:
                continue
        if "currents" not in results:
            print(f"  [Currents] FAILED")
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
                print(f"  [SSH] OK")
                results["ssh"] = True
                break
            except Exception:
                continue
        if "ssh" not in results:
            print(f"  [SSH] FAILED")
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
                print(f"  [CHL] OK")
                results["chl"] = True
                break
            except Exception:
                continue
        if "chl" not in results:
            print(f"  [CHL] FAILED")
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
                print(f"  [MLD] OK")
                results["mld"] = True
                break
            except Exception:
                continue
        if "mld" not in results:
            print(f"  [MLD] FAILED")
    else:
        results["mld"] = True

    return results


def sample_hotspot_at_locations(hotspot_path, locations):
    """Point-in-polygon sampling of hotspot scores at catch locations."""
    if not os.path.exists(hotspot_path):
        return [None] * len(locations)

    with open(hotspot_path) as f:
        gj = json.load(f)

    results = []
    for lat, lon in locations:
        best = None
        best_intensity = -1
        for feat in gj["features"]:
            coords = feat["geometry"]["coordinates"]
            if feat["geometry"]["type"] == "Polygon":
                ring = coords[0]
            elif feat["geometry"]["type"] == "MultiPolygon":
                ring = coords[0][0]
            else:
                continue
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            if lon < min(xs) or lon > max(xs) or lat < min(ys) or lat > max(ys):
                continue
            # Ray casting point-in-polygon
            inside = False
            n = len(ring)
            j = n - 1
            for i in range(n):
                xi, yi = ring[i]
                xj, yj = ring[j]
                if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            if inside and feat["properties"]["intensity"] > best_intensity:
                best = feat["properties"]
                best_intensity = feat["properties"]["intensity"]
        results.append(best)
    return results


def _is_duplicate_location(catch, all_catches):
    """Check if this catch shares coordinates with another catch on a different date."""
    key = (round(catch["lat"], 4), round(catch["lon"], 4))
    dates_at_loc = set()
    for c in all_catches:
        if (round(c["lat"], 4), round(c["lon"], 4)) == key:
            dates_at_loc.add(c["date"])
    return len(dates_at_loc) > 1


def _proximity_max(grid, lats, lons, lat, lon, radius_deg=0.017):
    """Find max score within radius_deg (~1nm) of the target point.

    Returns (max_score, best_yi, best_xi).
    """
    lat_mask = np.abs(lats - lat) <= radius_deg
    lon_mask = np.abs(lons - lon) <= radius_deg
    yi_indices = np.where(lat_mask)[0]
    xi_indices = np.where(lon_mask)[0]

    best_score = 0.0
    best_yi = np.argmin(np.abs(lats - lat))
    best_xi = np.argmin(np.abs(lons - lon))

    for yi in yi_indices:
        for xi in xi_indices:
            val = grid[yi, xi]
            if not np.isnan(val) and val > best_score:
                best_score = val
                best_yi = yi
                best_xi = xi

    return float(best_score), best_yi, best_xi


def main():
    catches = load_catches()
    dates = sorted(set(c["date"] for c in catches))
    print(f"Loaded {len(catches)} catches across {len(dates)} unique dates\n")

    # Ensure bathymetry tif exists
    tif_path = os.path.join(BASE_DIR, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        # Copy from any existing dated dir
        for d in os.listdir(BASE_DIR):
            candidate = os.path.join(BASE_DIR, d, "bathy_gmrt.tif")
            if os.path.exists(candidate):
                shutil.copy2(candidate, tif_path)
                break

    all_results = []
    for di, date_str in enumerate(dates):
        date_catches = [c for c in catches if c["date"] == date_str]
        print(f"[{di+1}/{len(dates)}] {date_str} — {len(date_catches)} catches")

        dated_dir = os.path.join(BASE_DIR, date_str)
        os.makedirs(dated_dir, exist_ok=True)
        marlin_data.OUTPUT_DIR = dated_dir

        # Copy bathy tif
        dated_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
        if not os.path.exists(dated_tif) and os.path.exists(tif_path):
            shutil.copy2(tif_path, dated_tif)

        hotspot_path = os.path.join(dated_dir, "blue_marlin_hotspots.geojson")

        grid_result = None
        if os.path.exists(hotspot_path):
            print(f"  Cached — re-running scoring for pixel-level sampling")
        # Always run scoring to get pixel-level grid data
        success = fetch_all(date_str, dated_dir)
        if "sst" not in success and not os.path.exists(hotspot_path):
            print(f"  SKIP — no SST")
            for c in date_catches:
                all_results.append({**c, "hotspot_score": None, "data_available": False})
            continue

        try:
            grid_result = generate_blue_marlin_hotspots(BBOX, tif_path=dated_tif)
        except Exception as e:
            print(f"  Scoring failed: {e}")
            for c in date_catches:
                all_results.append({**c, "hotspot_score": None, "data_available": False})
            continue

        # Pixel-level sampling from the grid
        # Two-tier scoring: exact pixel for unique GPS locations,
        # max within ~1nm radius for duplicate (DDM-rounded) locations
        for c in date_catches:
            result = {**c, "data_available": True}
            is_dupe = _is_duplicate_location(c, catches)
            result["scoring_mode"] = "proximity" if is_dupe else "exact"
            if grid_result and isinstance(grid_result, dict):
                grid = grid_result["grid"]
                glats = grid_result["lats"]
                glons = grid_result["lons"]
                sscores = grid_result["sub_scores"]
                weights = grid_result["weights"]
                # Find nearest pixel
                yi = np.argmin(np.abs(glats - c["lat"]))
                xi = np.argmin(np.abs(glons - c["lon"]))

                if is_dupe:
                    # Proximity mode: max score within ~1nm (~0.017 deg) radius
                    # This accounts for DDM rounding error at shared coordinates
                    pixel_score, yi_best, xi_best = _proximity_max(
                        grid, glats, glons, c["lat"], c["lon"], radius_deg=0.017
                    )
                    result["exact_score"] = round(
                        float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0, 2
                    )
                    # Sample sub-scores at best pixel location
                    for name, arr in sscores.items():
                        val = float(arr[yi_best, xi_best]) if not np.isnan(arr[yi_best, xi_best]) else None
                        result[f"s_{name}"] = round(val, 2) if val is not None else None
                else:
                    # Exact mode: score at nearest pixel (unique GPS mark)
                    pixel_score = float(grid[yi, xi]) if not np.isnan(grid[yi, xi]) else 0
                    for name, arr in sscores.items():
                        val = float(arr[yi, xi]) if not np.isnan(arr[yi, xi]) else None
                        result[f"s_{name}"] = round(val, 2) if val is not None else None

                result["hotspot_score"] = round(pixel_score, 2)
                for name, w in weights.items():
                    result[f"w_{name}"] = w
            else:
                result["hotspot_score"] = 0
            all_results.append(result)
            sc = result["hotspot_score"]
            mode_tag = " [prox]" if is_dupe else ""
            print(f"  {c['species'][:4]} ({c['lon']:.2f}E, {abs(c['lat']):.2f}S): {sc:.0%}{mode_tag}" if sc else
                  f"  {c['species'][:4]} ({c['lon']:.2f}E, {abs(c['lat']):.2f}S): outside zones")

    # Save CSV
    output_csv = os.path.join(BASE_DIR, "validation_results.csv")
    # Build fieldnames dynamically from all sub-scores and weights present
    base_fields = [
        "date", "lat", "lon", "species", "weight", "length", "tag",
        "data_available", "hotspot_score", "scoring_mode", "exact_score",
    ]
    s_keys = sorted(set(k for r in all_results for k in r if k.startswith("s_")))
    w_keys = sorted(set(k for r in all_results for k in r if k.startswith("w_")))
    fieldnames = base_fields + s_keys + w_keys
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE — {len(all_results)} catch records")
    scored = [r for r in all_results if r.get("data_available")]
    in_zone = [r for r in scored if r["hotspot_score"] and r["hotspot_score"] > 0]
    no_zone = [r for r in scored if not r["hotspot_score"] or r["hotspot_score"] == 0]
    print(f"  Data available: {len(scored)}/{len(all_results)}")
    print(f"  In a hotspot zone: {len(in_zone)}/{len(scored)} ({len(in_zone)/len(scored)*100:.0f}%)")
    print(f"  Outside all zones: {len(no_zone)}/{len(scored)}")
    if in_zone:
        scores = [r["hotspot_score"] for r in in_zone]
        print(f"  Score range: {min(scores):.0%} - {max(scores):.0%}")
        print(f"  Mean score at catch locations: {np.mean(scores):.0%}")
        print(f"  Median score: {np.median(scores):.0%}")
        for thresh in [0.3, 0.5, 0.7]:
            n = sum(1 for s in scores if s >= thresh)
            print(f"  Score >= {thresh:.0%}: {n}/{len(scored)} ({n/len(scored)*100:.0f}%)")

    # Two-tier breakdown
    exact_catches = [r for r in scored if r.get("scoring_mode") == "exact"]
    prox_catches = [r for r in scored if r.get("scoring_mode") == "proximity"]
    if exact_catches and prox_catches:
        print(f"\n  Two-tier scoring breakdown:")
        ex_scores = [r["hotspot_score"] for r in exact_catches if r["hotspot_score"] and r["hotspot_score"] > 0]
        pr_scores = [r["hotspot_score"] for r in prox_catches if r["hotspot_score"] and r["hotspot_score"] > 0]
        if ex_scores:
            print(f"    Unique GPS (exact):    n={len(exact_catches)}, mean={np.mean(ex_scores):.0%}, median={np.median(ex_scores):.0%}")
        if pr_scores:
            print(f"    Duplicate loc (prox):  n={len(prox_catches)}, mean={np.mean(pr_scores):.0%}, median={np.median(pr_scores):.0%}")
            # Show how much proximity scoring helped
            exact_at_prox = [r.get("exact_score", r["hotspot_score"]) for r in prox_catches if r["hotspot_score"] and r["hotspot_score"] > 0]
            if exact_at_prox:
                lift = np.mean(pr_scores) - np.mean(exact_at_prox)
                print(f"    Proximity lift:        {lift:+.0%} (exact mean {np.mean(exact_at_prox):.0%} -> prox mean {np.mean(pr_scores):.0%})")

    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    main()
