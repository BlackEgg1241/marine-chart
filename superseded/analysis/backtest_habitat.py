"""
Historical backtest of blue marlin habitat scores over 12 months.

Fetches ocean data for weekly samples, runs the scoring pipeline,
and extracts zone-max habitat scores for the Accessible Trench Zone.

Usage:
    python backtest_habitat.py                    # 52 weekly samples, past 12 months
    python backtest_habitat.py --weeks 26         # 26 weeks
    python backtest_habitat.py --start 2025-06-01 # custom start date
    python backtest_habitat.py --skip-fetch       # re-score existing data only
"""

import argparse
import json
import os
import shutil
import sys
import numpy as np
from datetime import datetime, timedelta

# Zone bounds (same as generate_forecast_summary.py)
ZONE_W = 114.98
ZONE_E = 115.3333
ZONE_S = -32.1667
ZONE_N = -31.7287


def compute_zone_stats(grid, lats, lons):
    """Extract zone stats from the scoring grid."""
    lat_mask = (lats >= ZONE_S) & (lats <= ZONE_N)
    lon_mask = (lons >= ZONE_W) & (lons <= ZONE_E)

    zone = grid[np.ix_(lat_mask, lon_mask)]
    if zone.size == 0:
        return None, None, None, 0
    valid = zone[~np.isnan(zone)]
    if valid.size == 0:
        return None, None, None, 0
    return (
        round(float(np.nanmax(valid)) * 100, 1),
        round(float(np.nanmean(valid)) * 100, 1),
        round(float(np.nanmedian(valid)) * 100, 1),
        int(valid.size),
    )


def compute_zone_subscores(sub_scores, weights, lats, lons):
    """Extract per-variable zone-mean sub-scores from the scoring result."""
    lat_mask = (lats >= ZONE_S) & (lats <= ZONE_N)
    lon_mask = (lons >= ZONE_W) & (lons <= ZONE_E)

    result = {}
    for var_name, grid in sub_scores.items():
        zone = grid[np.ix_(lat_mask, lon_mask)]
        valid = zone[~np.isnan(zone)] if zone.size > 0 else np.array([])
        if valid.size > 0:
            result[f"s_{var_name}"] = round(float(np.nanmean(valid)), 4)
        else:
            result[f"s_{var_name}"] = None
    for var_name, w in weights.items():
        result[f"w_{var_name}"] = w
    return result


# Sub-zone definitions for spatial breakdown
SUB_ZONES = {
    "canyon": [114.95, -32.02, 115.15, -31.85],
    "pgfc": [115.15, -32.12, 115.35, -31.92],
    "north": [114.98, -31.90, 115.25, -31.73],
    "south": [115.05, -32.17, 115.25, -32.05],
}


def compute_subzone_stats(grid, lats, lons):
    """Extract per-subzone max scores from the scoring grid."""
    result = {}
    for key, (w, s, e, n) in SUB_ZONES.items():
        lat_mask = (lats >= s) & (lats <= n)
        lon_mask = (lons >= w) & (lons <= e)
        zone = grid[np.ix_(lat_mask, lon_mask)]
        valid = zone[~np.isnan(zone)] if zone.size > 0 else np.array([])
        if valid.size > 0:
            result[f"sz_{key}_max"] = round(float(np.nanmax(valid)) * 100, 1)
            result[f"sz_{key}_mean"] = round(float(np.nanmean(valid)) * 100, 1)
        else:
            result[f"sz_{key}_max"] = None
            result[f"sz_{key}_mean"] = None
    return result


NRT_CUTOFF = "2024-02-01"  # Before this date, use reanalysis datasets


def _fetch_reanalysis(date_str, bbox, output_dir):
    """Fetch all variables from CMEMS reanalysis (covers 1993-present)."""
    import copernicusmarine

    def _subset(dataset_id, variables, output_name, **kwargs):
        out = os.path.join(output_dir, output_name)
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=bbox["lon_min"],
            maximum_longitude=bbox["lon_max"],
            minimum_latitude=bbox["lat_min"],
            maximum_latitude=bbox["lat_max"],
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T23:59:59",
            output_filename=out,
            output_directory=".",
            overwrite=True,
            **kwargs,
        )

    fetchers = [
        ("SST", "cmems_mod_glo_phy_my_0.083deg_P1D-m", ["thetao"], "sst_raw.nc",
         {"minimum_depth": 0, "maximum_depth": 1}),
        ("Currents", "cmems_mod_glo_phy_my_0.083deg_P1D-m", ["uo", "vo"], "currents_raw.nc",
         {"minimum_depth": 0, "maximum_depth": 1}),
        ("MLD", "cmems_mod_glo_phy_my_0.083deg_P1D-m", ["mlotst"], "mld_raw.nc", {}),
        ("CHL", "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", ["CHL"], "chl_raw.nc", {}),
        ("SSH", "cmems_mod_glo_phy_my_0.083deg_P1D-m", ["zos"], "ssh_raw.nc", {}),
        ("O2", "cmems_mod_glo_bgc_my_0.25deg_P1D-m", ["o2"], "oxygen_raw.nc",
         {"minimum_depth": 95, "maximum_depth": 105}),
        ("KD490", "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D", ["KD490"], "kd490_raw.nc", {}),
    ]

    for name, dataset_id, variables, output_name, extra_kwargs in fetchers:
        try:
            print(f"  [{name}] Fetching reanalysis for {date_str}...")
            _subset(dataset_id, variables, output_name, **extra_kwargs)
        except Exception as e:
            print(f"  [{name}] Error: {e}")


def run_date(date_str, bbox, base_dir, bathy_tif, skip_fetch=False):
    """Fetch data and score one date. Returns (zone_max, zone_mean, zone_median, cells) or Nones."""
    import marlin_data

    dated_dir = os.path.join(base_dir, date_str)
    os.makedirs(dated_dir, exist_ok=True)
    marlin_data.OUTPUT_DIR = dated_dir

    # Reuse cached bathymetry
    local_tif = os.path.join(dated_dir, "bathy_gmrt.tif")
    if bathy_tif and not os.path.exists(local_tif):
        shutil.copy2(bathy_tif, local_tif)

    # Skip fetch if we already have key data files for this date
    key_files = ["sst_raw.nc", "currents_raw.nc", "ssh_raw.nc"]
    has_data = all(os.path.exists(os.path.join(dated_dir, f)) for f in key_files)
    if not skip_fetch and not has_data:
        if date_str < NRT_CUTOFF:
            # Use reanalysis datasets for historical dates
            _fetch_reanalysis(date_str, bbox, dated_dir)
        else:
            # Use NRT/ANFC datasets for recent dates
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
    elif has_data:
        print(f"  [Cache] Using existing data")

    # Run scoring
    tif = local_tif if os.path.exists(local_tif) else None
    result = marlin_data.generate_blue_marlin_hotspots(bbox, tif_path=tif, date_str=date_str)
    if result is None:
        return None, None, None, 0, {}

    stats = compute_zone_stats(result["grid"], result["lats"], result["lons"])
    # Extract per-variable zone averages
    var_scores = {}
    if result.get("sub_scores") and result.get("weights"):
        var_scores = compute_zone_subscores(
            result["sub_scores"], result["weights"],
            result["lats"], result["lons"])
    # Sub-zone spatial breakdown
    sz_scores = compute_subzone_stats(result["grid"], result["lats"], result["lons"])
    var_scores.update(sz_scores)
    return stats[0], stats[1], stats[2], stats[3], var_scores


def main():
    parser = argparse.ArgumentParser(description="Backtest habitat scores over 12 months")
    parser.add_argument("--weeks", type=int, default=52, help="Number of weekly samples (default: 52)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: today minus weeks)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download, score existing files")
    parser.add_argument("--output", default="data/backtest", help="Output directory")
    parser.add_argument("--seasonal", action="store_true",
                        help="Sample 1st and 15th of each month (better for long-range)")
    parser.add_argument("--season-months", default="1,2,3,4,5,6",
                        help="Comma-separated months to sample in seasonal mode (default: 1-6 for marlin season)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dates", default=None,
                        help="Comma-separated explicit dates to process (overrides start/end/weeks)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing results file instead of overwriting")
    args = parser.parse_args()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(weeks=args.weeks)

    # Generate sample dates
    dates = []
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    elif args.seasonal:
        # Sample 1st and 15th of specified months for each year
        season_months = [int(m) for m in args.season_months.split(",")]
        year = start_date.year
        while year <= end_date.year:
            for month in season_months:
                for day in [1, 15]:
                    try:
                        d = datetime(year, month, day)
                        if start_date <= d <= end_date:
                            dates.append(d.strftime("%Y-%m-%d"))
                    except ValueError:
                        pass
            year += 1
    else:
        # Weekly samples
        d = start_date
        while d <= end_date:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(weeks=1)

    # Skip dates already in results file
    existing_dates = set()
    results = []
    output_file = os.path.join(args.output, "backtest_results.json")
    if args.append and os.path.exists(output_file):
        with open(output_file) as f:
            old = json.load(f)
        results = old.get("dates", [])
        existing_dates = {r["date"] for r in results}
        before = len(dates)
        dates = [d for d in dates if d not in existing_dates]
        print(f"Append mode: {len(existing_dates)} existing dates, {before - len(dates)} skipped, {len(dates)} new")

    if not dates:
        print("No new dates to process.")
        return
    print(f"Habitat Backtest: {len(dates)} dates from {dates[0]} to {dates[-1]}")
    print(f"Output: {args.output}/\n")

    import marlin_data
    bbox = dict(marlin_data.DEFAULT_BBOX)
    os.makedirs(args.output, exist_ok=True)

    # Fetch bathymetry once
    bathy_tif = os.path.join(args.output, "bathy_gmrt.tif")
    if not os.path.exists(bathy_tif):
        print("[Bathymetry] Downloading once for all dates...")
        old_dir = marlin_data.OUTPUT_DIR
        marlin_data.OUTPUT_DIR = args.output
        try:
            marlin_data.fetch_bathymetry_gmrt(bbox)
        except Exception as e:
            print(f"[Bathymetry] Error: {e}")
            bathy_tif = None
        marlin_data.OUTPUT_DIR = old_dir

    # output_file already set above; results populated if --append

    for i, date_str in enumerate(dates):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(dates)}] {date_str}")
        print(f"{'='*60}")

        try:
            zone_max, zone_mean, zone_median, cells, var_scores = run_date(
                date_str, bbox, args.output, bathy_tif, args.skip_fetch)
        except Exception as e:
            print(f"  FAILED: {e}")
            zone_max = zone_mean = zone_median = None
            cells = 0
            var_scores = {}

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        entry = {
            "date": date_str,
            "day_name": dt.strftime("%A"),
            "month": dt.strftime("%B"),
            "zone_max": zone_max,
            "zone_mean": zone_mean,
            "zone_median": zone_median,
            "zone_cells": cells,
        }
        if var_scores:
            entry.update(var_scores)
        results.append(entry)

        if zone_mean is not None:
            print(f"  -> Mean: {zone_mean:.1f}%  Median: {zone_median:.1f}%  Max: {zone_max:.1f}%  ({cells} cells)")
        else:
            print(f"  -> FAILED")

        # Save incrementally (sorted by date)
        sorted_results = sorted(results, key=lambda r: r["date"])
        with open(output_file, "w") as f:
            json.dump({"dates": sorted_results}, f, indent=2)

    # Print summary (use all results including appended)
    results = sorted(results, key=lambda r: r["date"])
    valid = [r for r in results if r.get("zone_mean") is not None]
    if valid:
        means = [r["zone_mean"] for r in valid]
        maxes = [r["zone_max"] for r in valid]
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY ({len(valid)}/{len(results)} dates successful)")
        print(f"{'='*60}")
        print(f"  Zone Mean  - avg: {np.mean(means):.1f}%  median: {np.median(means):.1f}%  range: {np.min(means):.1f}-{np.max(means):.1f}%")
        print(f"  Zone Max   - avg: {np.mean(maxes):.1f}%  median: {np.median(maxes):.1f}%  range: {np.min(maxes):.1f}-{np.max(maxes):.1f}%")
        print(f"  Best mean:  {max(valid, key=lambda r: r['zone_mean'])['date']} ({max(means):.1f}%)")
        print(f"  Worst mean: {min(valid, key=lambda r: r['zone_mean'])['date']} ({min(means):.1f}%)")

        # Monthly breakdown
        print(f"\nMonthly averages (zone mean):")
        from collections import defaultdict
        by_month = defaultdict(list)
        for r in valid:
            by_month[r["month"]].append(r["zone_mean"])

        # Order by calendar
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        for m in month_order:
            if m in by_month:
                vals = by_month[m]
                print(f"  {m:>12s}: {np.mean(vals):5.1f}% (n={len(vals)})")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
