#!/usr/bin/env python3
"""
Analyze current direction & speed vs depth/shelf structure at catch locations.

Questions:
1. What current direction/speed do catches sit at?
2. How does current direction correlate with depth contours?
3. Do catches prefer specific current-over-depth combinations?
4. How does current interact with shelf break structure?
"""

import csv
import os
import sys
import numpy as np
import xarray as xr
from collections import defaultdict

BBOX = {"lon_min": 113.5, "lon_max": 116.5, "lat_min": -33.5, "lat_max": -30.5}
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def load_catches():
    catches = []
    seen = set()
    # Export.csv (DDM format)
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sp = (row.get("Species", "") or row.get("Species_Name", "")).strip().upper()
                if "MARLIN" not in sp and "BLUE" not in sp:
                    continue
                lat_raw = row.get("Latitude", "").strip()
                lon_raw = row.get("Longitude", "").strip()
                date_raw = (row.get("Date", "") or row.get("Release_Date", "")).strip()
                if not lat_raw or not lon_raw or not date_raw:
                    continue
                lat = ddm_to_dd(lat_raw.replace("S", ""), negative=True)
                lon = ddm_to_dd(lon_raw.replace("E", ""))
                if "/" in date_raw:
                    parts = date_raw.split("/")
                    date_str = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                else:
                    date_str = date_raw[:10]  # ISO format: 2016-02-20T00:00:00
                key = (date_str, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date_str, "lat": lat, "lon": lon})
    # all_catches.csv (decimal)
    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sp = row.get("species", "").strip().upper()
                if "MARLIN" not in sp and "BLUE" not in sp:
                    continue
                if not row.get("lat") or not row.get("lon") or not row.get("date"):
                    continue
                lat = float(row["lat"])
                lon = float(row["lon"])
                date_str = row["date"]
                key = (date_str, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date_str, "lat": lat, "lon": lon})
    return catches


def interp_value(grid, lats, lons, lat, lon):
    """Get grid value at lat/lon by nearest-neighbor."""
    li = np.argmin(np.abs(lats - lat))
    lo = np.argmin(np.abs(lons - lon))
    if 0 <= li < len(lats) and 0 <= lo < len(lons):
        return grid[li, lo]
    return np.nan


def _interp_to_grid(data, src_lons, src_lats):
    """Interpolate data onto master grid."""
    from scipy.interpolate import RegularGridInterpolator
    ny, nx = 120, 120
    lons = np.linspace(BBOX["lon_min"], BBOX["lon_max"], nx)
    lats = np.linspace(BBOX["lat_min"], BBOX["lat_max"], ny)
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[::-1]
    interp = RegularGridInterpolator(
        (src_lats, src_lons), data, method="linear",
        bounds_error=False, fill_value=np.nan)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    return interp((lat_grid, lon_grid)), lats, lons


def analyze_date(date_str, date_catches):
    """Extract current direction, speed, depth at catch and surrounding grid."""
    ddir = os.path.join("data", date_str)

    # Load current data
    cur_file = os.path.join(ddir, "currents_raw.nc")
    if not os.path.exists(cur_file):
        return None

    try:
        ds = xr.open_dataset(cur_file)
        uo = ds["uo"].squeeze()
        vo = ds["vo"].squeeze()
        if "depth" in uo.dims:
            uo = uo.isel(depth=0)
            vo = vo.isel(depth=0)

        c_lons = uo.longitude.values if "longitude" in uo.dims else uo.lon.values
        c_lats = uo.latitude.values if "latitude" in uo.dims else uo.lat.values

        u_grid, lats, lons = _interp_to_grid(uo.values.astype(float), c_lons, c_lats)
        v_grid, _, _ = _interp_to_grid(vo.values.astype(float), c_lons, c_lats)

        speed = np.sqrt(u_grid**2 + v_grid**2)
        direction = np.degrees(np.arctan2(u_grid, v_grid)) % 360  # oceanographic: direction current flows TO

        ds.close()
    except Exception as e:
        print(f"  {date_str}: current load failed: {e}")
        return None

    # Load bathymetry
    bathy_tif = os.path.join(ddir, "bathy_gmrt.tif")
    depth_at_catch = []
    if os.path.exists(bathy_tif):
        try:
            import rasterio
            with rasterio.open(bathy_tif) as src:
                bathy = src.read(1).astype(float)
                b_lons = np.linspace(src.bounds.left, src.bounds.right, bathy.shape[1])
                b_lats = np.linspace(src.bounds.top, src.bounds.bottom, bathy.shape[0])
            depth_grid, d_lats, d_lons = _interp_to_grid(
                np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats)
        except Exception as e:
            print(f"  {date_str}: bathy load failed: {e}")
            depth_grid = None
    else:
        depth_grid = None

    results = []
    for c in date_catches:
        spd = interp_value(speed, lats, lons, c["lat"], c["lon"])
        dirn = interp_value(direction, lats, lons, c["lat"], c["lon"])

        depth = np.nan
        if depth_grid is not None:
            depth = interp_value(depth_grid, lats, lons, c["lat"], c["lon"])

        # Also get speed/direction stats in surrounding area
        li = np.argmin(np.abs(lats - c["lat"]))
        lo = np.argmin(np.abs(lons - c["lon"]))
        r = 3  # ~3 grid cells radius
        y0, y1 = max(0, li-r), min(len(lats), li+r+1)
        x0, x1 = max(0, lo-r), min(len(lons), lo+r+1)

        local_speed = speed[y0:y1, x0:x1]
        local_dir = direction[y0:y1, x0:x1]
        local_speed_valid = local_speed[~np.isnan(local_speed)]

        # Speed gradient (how much speed changes near the catch)
        speed_std = np.nanstd(local_speed) if len(local_speed_valid) > 3 else np.nan

        # Direction variability (circular std dev)
        local_dir_valid = local_dir[~np.isnan(local_dir)]
        if len(local_dir_valid) > 3:
            rads = np.radians(local_dir_valid)
            R = np.sqrt(np.mean(np.cos(rads))**2 + np.mean(np.sin(rads))**2)
            dir_std = np.degrees(np.sqrt(-2 * np.log(max(R, 0.001))))
        else:
            dir_std = np.nan

        # Depth gradient at catch
        depth_grad = np.nan
        if depth_grid is not None:
            local_depth = depth_grid[y0:y1, x0:x1]
            if local_depth.size > 3:
                depth_grad = np.nanstd(local_depth)

        # Ocean-wide speed percentile
        ocean_speed = speed[~np.isnan(speed)]
        spd_pct = np.sum(ocean_speed < spd) / len(ocean_speed) * 100 if len(ocean_speed) > 0 and not np.isnan(spd) else np.nan

        results.append({
            "date": date_str,
            "lat": c["lat"], "lon": c["lon"],
            "speed": spd, "speed_pct": spd_pct,
            "direction": dirn,
            "depth": depth,
            "speed_variability": speed_std,
            "direction_variability": dir_std,
            "depth_gradient": depth_grad,
        })

    return results


def direction_label(deg):
    """Convert degrees to 8-point compass."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((deg + 22.5) / 45) % 8
    return labels[idx]


def main():
    catches = load_catches()
    by_date = defaultdict(list)
    for c in catches:
        by_date[c["date"]].append(c)

    all_results = []
    dates = sorted(by_date.keys())
    print(f"Analyzing current + depth for {len(dates)} catch dates ({len(catches)} catches)...\n")

    for d in dates:
        r = analyze_date(d, by_date[d])
        if r:
            all_results.extend(r)

    if not all_results:
        print("No results!")
        return

    # ===== ANALYSIS =====
    speeds = [r["speed"] for r in all_results if not np.isnan(r["speed"])]
    dirs = [r["direction"] for r in all_results if not np.isnan(r["direction"])]
    depths = [r["depth"] for r in all_results if not np.isnan(r["depth"])]
    spd_pcts = [r["speed_pct"] for r in all_results if not np.isnan(r["speed_pct"])]
    spd_vars = [r["speed_variability"] for r in all_results if not np.isnan(r["speed_variability"])]
    dir_vars = [r["direction_variability"] for r in all_results if not np.isnan(r["direction_variability"])]
    depth_grads = [r["depth_gradient"] for r in all_results if not np.isnan(r["depth_gradient"])]

    print(f"{'='*70}")
    print(f"CURRENT DIRECTION + DEPTH ANALYSIS ({len(all_results)} catches)")
    print(f"{'='*70}")

    # 1. Current speed at catches
    print(f"\n--- Current Speed at Catches ---")
    print(f"  Mean:   {np.mean(speeds):.3f} m/s")
    print(f"  Median: {np.median(speeds):.3f} m/s")
    print(f"  Std:    {np.std(speeds):.3f} m/s")
    print(f"  Min:    {np.min(speeds):.3f} m/s")
    print(f"  Max:    {np.max(speeds):.3f} m/s")
    print(f"  Mean percentile (vs domain): {np.mean(spd_pcts):.0f}%")

    # 2. Current direction distribution
    print(f"\n--- Current Direction at Catches ---")
    dir_counts = defaultdict(int)
    for d in dirs:
        dir_counts[direction_label(d)] += 1
    total = len(dirs)
    for label in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        cnt = dir_counts.get(label, 0)
        bar = "#" * int(cnt / total * 40)
        print(f"  {label:>2s}: {cnt:3d} ({cnt/total*100:4.0f}%) {bar}")

    # Mean direction (circular mean)
    rads = np.radians(dirs)
    mean_dir = np.degrees(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))) % 360
    R = np.sqrt(np.mean(np.cos(rads))**2 + np.mean(np.sin(rads))**2)
    print(f"  Mean direction: {mean_dir:.0f} deg ({direction_label(mean_dir)}), concentration R={R:.2f}")

    # 3. Speed variability (catches in shear zones?)
    print(f"\n--- Speed Variability Near Catches ---")
    print(f"  Mean speed std (3-cell radius): {np.mean(spd_vars):.4f} m/s")
    print(f"  Median: {np.median(spd_vars):.4f} m/s")
    print(f"  (High = catch in speed shear zone)")

    # 4. Direction variability (catches at convergence/divergence?)
    print(f"\n--- Direction Variability Near Catches ---")
    print(f"  Mean circular std: {np.mean(dir_vars):.1f} deg")
    print(f"  Median: {np.median(dir_vars):.1f} deg")
    print(f"  (High = catch near direction change / eddy)")

    # 5. Depth at catches
    print(f"\n--- Depth at Catches ---")
    print(f"  Mean:   {np.mean(depths):.0f} m")
    print(f"  Median: {np.median(depths):.0f} m")
    print(f"  P25:    {np.percentile(depths, 25):.0f} m")
    print(f"  P75:    {np.percentile(depths, 75):.0f} m")

    # 6. CROSS-ANALYSIS: Direction vs Depth
    print(f"\n--- Direction vs Depth ---")
    shallow = [r for r in all_results if not np.isnan(r["depth"]) and r["depth"] < 200 and not np.isnan(r["direction"])]
    mid = [r for r in all_results if not np.isnan(r["depth"]) and 200 <= r["depth"] < 500 and not np.isnan(r["direction"])]
    deep = [r for r in all_results if not np.isnan(r["depth"]) and r["depth"] >= 500 and not np.isnan(r["direction"])]

    for label, group in [("< 200m (shelf)", shallow), ("200-500m (slope)", mid), (">= 500m (deep)", deep)]:
        if group:
            g_dirs = [r["direction"] for r in group]
            g_rads = np.radians(g_dirs)
            g_mean = np.degrees(np.arctan2(np.mean(np.sin(g_rads)), np.mean(np.cos(g_rads)))) % 360
            g_spd = np.mean([r["speed"] for r in group])
            g_dir_var = np.mean([r["direction_variability"] for r in group if not np.isnan(r["direction_variability"])])
            print(f"  {label:>20s}: n={len(group):2d}, mean dir={g_mean:5.0f} ({direction_label(g_mean):>2s}), "
                  f"mean speed={g_spd:.3f} m/s, dir variability={g_dir_var:.0f} deg")

    # 7. CROSS-ANALYSIS: Speed vs Depth gradient (shelf edge)
    print(f"\n--- Current Speed vs Depth Gradient (Shelf Edge Proxy) ---")
    paired = [(r["speed"], r["depth_gradient"]) for r in all_results
              if not np.isnan(r["speed"]) and not np.isnan(r["depth_gradient"])]
    if len(paired) > 5:
        spds, dgrads = zip(*paired)
        corr = np.corrcoef(spds, dgrads)[0, 1]
        print(f"  Correlation (speed vs depth_gradient): r = {corr:.3f}")
        print(f"  (Positive = faster current at steeper slopes)")

        # Split by steep vs flat
        dg_med = np.median(dgrads)
        steep = [s for s, d in paired if d > dg_med]
        flat = [s for s, d in paired if d <= dg_med]
        print(f"  Steep slope (depth_grad > {dg_med:.0f}m): mean speed = {np.mean(steep):.3f} m/s")
        print(f"  Flat shelf  (depth_grad <= {dg_med:.0f}m): mean speed = {np.mean(flat):.3f} m/s")

    # 8. CROSS-ANALYSIS: Direction variability vs Depth
    print(f"\n--- Direction Variability vs Depth ---")
    paired2 = [(r["direction_variability"], r["depth"]) for r in all_results
               if not np.isnan(r["direction_variability"]) and not np.isnan(r["depth"])]
    if len(paired2) > 5:
        dvars, ddepths = zip(*paired2)
        corr2 = np.corrcoef(dvars, ddepths)[0, 1]
        print(f"  Correlation (dir_variability vs depth): r = {corr2:.3f}")
        print(f"  (Positive = more directional chaos at deeper catch sites)")

    # 9. CROSS-ANALYSIS: Catches at current-direction boundaries
    print(f"\n--- Current Direction Change at Catch vs Background ---")
    # Compare direction variability at catches vs random ocean points
    bg_dir_vars = []
    for r in all_results:
        if np.isnan(r["direction_variability"]):
            continue
        bg_dir_vars.append(r["direction_variability"])
    if bg_dir_vars:
        print(f"  Mean direction variability at catches: {np.mean(bg_dir_vars):.1f} deg")
        print(f"  (Compare to ocean background ~20-30 deg for uniform flow, >50 for eddies)")

    # 10. Speed percentile distribution
    print(f"\n--- Speed Percentile Distribution ---")
    pct_bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
    for lo, hi in pct_bins:
        cnt = sum(1 for p in spd_pcts if lo <= p < hi)
        bar = "#" * int(cnt / len(spd_pcts) * 40)
        print(f"  p{lo:2d}-p{hi:3d}: {cnt:3d} ({cnt/len(spd_pcts)*100:4.0f}%) {bar}")

    # 11. Does depth predict current direction?
    print(f"\n--- Depth Bands: Current Direction Roses ---")
    bands = [(100, 200, "100-200m"), (200, 300, "200-300m"), (300, 500, "300-500m"), (500, 1000, "500-1000m")]
    for d_lo, d_hi, lbl in bands:
        group = [r for r in all_results if not np.isnan(r["depth"]) and d_lo <= r["depth"] < d_hi and not np.isnan(r["direction"])]
        if len(group) >= 2:
            g_dirs = [r["direction"] for r in group]
            dir_rose = defaultdict(int)
            for d in g_dirs:
                dir_rose[direction_label(d)] += 1
            top = sorted(dir_rose.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v}" for k, v in top)
            g_spd = np.mean([r["speed"] for r in group])
            print(f"  {lbl:>10s}: n={len(group):2d}, speed={g_spd:.3f} m/s, top dirs: {top_str}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
