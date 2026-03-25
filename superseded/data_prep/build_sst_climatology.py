#!/usr/bin/env python3
"""
Build monthly SST climatology from backtest cache for SSTA computation.

Reads all sst_raw.nc files from data/backtest/YYYY-MM-DD/ directories,
regrids to a common 0.083deg grid, and computes monthly mean SST.

Output: data/sst_climatology.nc  (month x latitude x longitude, variable sst_clim in Celsius)
"""

import os
import sys
import numpy as np
import xarray as xr
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.join(SCRIPT_DIR, "data", "backtest")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "sst_climatology.nc")

# Target grid: 0.083deg over Perth Canyon bbox
LAT_MIN, LAT_MAX = -33.5, -30.5
LON_MIN, LON_MAX = 113.5, 116.5
RESOLUTION = 0.083

# Build target coordinate arrays
target_lat = np.arange(LAT_MIN, LAT_MAX + RESOLUTION / 2, RESOLUTION)
target_lon = np.arange(LON_MIN, LON_MAX + RESOLUTION / 2, RESOLUTION)


def extract_sst(filepath):
    """Extract SST data array from a NetCDF file, returning values in Celsius on the target grid."""
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f"    WARN: Cannot open {filepath}: {e}")
        return None

    # Find SST variable
    sst = None
    for var in ["thetao", "analysed_sst", "sea_surface_temperature", "sst"]:
        if var in ds.data_vars:
            sst = ds[var].squeeze()
            break

    if sst is None:
        ds.close()
        print(f"    WARN: No SST variable found in {filepath}")
        return None

    # Drop any extra dimensions (depth, time, etc.)
    for dim in list(sst.dims):
        if dim not in ("latitude", "longitude", "lat", "lon"):
            if sst.sizes[dim] == 1:
                sst = sst.squeeze(dim, drop=True)
            else:
                # Take first index for non-singleton extra dims
                sst = sst.isel({dim: 0})

    # Convert Kelvin to Celsius if needed
    mean_val = float(np.nanmean(sst.values))
    if mean_val > 100:
        sst = sst - 273.15

    # Normalize coordinate names to latitude/longitude
    rename = {}
    for dim in sst.dims:
        if dim in ("lat",):
            rename[dim] = "latitude"
        elif dim in ("lon",):
            rename[dim] = "longitude"
    if rename:
        sst = sst.rename(rename)

    # Regrid to target grid using linear interpolation
    try:
        sst_regridded = sst.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear",
        )
    except Exception as e:
        ds.close()
        print(f"    WARN: Regrid failed for {filepath}: {e}")
        return None

    ds.close()
    return sst_regridded.values


def main():
    if not os.path.isdir(BACKTEST_DIR):
        print(f"ERROR: Backtest directory not found: {BACKTEST_DIR}")
        sys.exit(1)

    # Collect all backtest directories
    all_dirs = sorted([
        d for d in os.listdir(BACKTEST_DIR)
        if os.path.isdir(os.path.join(BACKTEST_DIR, d)) and len(d) == 10 and d[4] == '-'
    ])
    print(f"Found {len(all_dirs)} backtest directories")

    # Group by month and accumulate SST grids
    monthly_grids = defaultdict(list)  # month -> list of 2D arrays
    skipped = 0
    loaded = 0

    for d in all_dirs:
        sst_path = os.path.join(BACKTEST_DIR, d, "sst_raw.nc")
        if not os.path.exists(sst_path):
            skipped += 1
            continue

        month = int(d[5:7])  # extract month from YYYY-MM-DD
        grid = extract_sst(sst_path)
        if grid is not None:
            monthly_grids[month].append(grid)
            loaded += 1
        else:
            skipped += 1

    print(f"\nLoaded: {loaded}, Skipped: {skipped}")
    print(f"Months with data: {sorted(monthly_grids.keys())}")

    # Compute climatology
    nlat = len(target_lat)
    nlon = len(target_lon)
    clim = np.full((12, nlat, nlon), np.nan, dtype=np.float32)

    print(f"\nTarget grid: {nlat} lat x {nlon} lon ({RESOLUTION}deg)")
    print(f"{'Month':>6} {'Files':>6} {'Mean SST (C)':>14}")
    print("-" * 30)

    for month in range(1, 13):
        grids = monthly_grids.get(month, [])
        if len(grids) == 0:
            print(f"{month:>6} {0:>6} {'N/A':>14}")
            continue

        stacked = np.stack(grids, axis=0)
        clim[month - 1] = np.nanmean(stacked, axis=0)
        mean_sst = np.nanmean(clim[month - 1])
        print(f"{month:>6} {len(grids):>6} {mean_sst:>14.2f}")

    # Save as NetCDF
    months = np.arange(1, 13, dtype=np.int32)
    ds_out = xr.Dataset(
        {
            "sst_clim": xr.DataArray(
                clim,
                dims=["month", "latitude", "longitude"],
                coords={
                    "month": months,
                    "latitude": target_lat,
                    "longitude": target_lon,
                },
                attrs={
                    "units": "degrees_C",
                    "long_name": "Monthly SST climatology (mean across all years)",
                    "source": "Backtest cache sst_raw.nc files (CMEMS reanalysis/NRT)",
                },
            )
        },
        attrs={
            "title": "Monthly SST Climatology for Perth Canyon",
            "description": "Mean SST per calendar month computed from backtest cache",
            "bbox": f"lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]",
            "resolution_deg": RESOLUTION,
            "n_files_total": loaded,
        },
    )

    ds_out.to_netcdf(OUTPUT_FILE)
    print(f"\nSaved climatology to {OUTPUT_FILE}")
    print(f"Dimensions: {dict(ds_out.dims)}")


if __name__ == "__main__":
    main()
