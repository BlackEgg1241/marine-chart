#!/usr/bin/env python3
"""
Analyze angle between current direction and bathymetry contours at catch locations.

Parallel flow = geostrophic, stable LC jet along isobaths
Perpendicular flow = cross-shelf transport, upwelling/intrusion events

Is there a correlation with catches?
"""

import csv
import os
import sys
import numpy as np
import xarray as xr
from collections import defaultdict
from scipy.ndimage import sobel

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
                    date_str = date_raw[:10]
                key = (date_str, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date_str, "lat": lat, "lon": lon})
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


def _interp_to_grid(data, src_lons, src_lats):
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
    ddir = os.path.join("data", date_str)
    cur_file = os.path.join(ddir, "currents_raw.nc")
    bathy_tif = os.path.join(ddir, "bathy_gmrt.tif")

    if not os.path.exists(cur_file) or not os.path.exists(bathy_tif):
        return None

    try:
        # Load currents
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
        # Current direction vector (u=east, v=north)
        ds.close()

        # Load bathymetry
        import rasterio
        with rasterio.open(bathy_tif) as src:
            bathy = src.read(1).astype(float)
            b_lons = np.linspace(src.bounds.left, src.bounds.right, bathy.shape[1])
            b_lats = np.linspace(src.bounds.top, src.bounds.bottom, bathy.shape[0])
        depth_grid, _, _ = _interp_to_grid(
            np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats)

        # Compute bathymetry gradient direction (Sobel)
        # This gives the direction of steepest descent = perpendicular to contours
        depth_filled = np.where(np.isnan(depth_grid), 0, depth_grid)
        grad_x = sobel(depth_filled, axis=1)  # d(depth)/d(lon) — east gradient
        grad_y = sobel(depth_filled, axis=0)  # d(depth)/d(lat) — north gradient

        # Contour direction = perpendicular to gradient = rotated 90 degrees
        # Gradient points downslope; contour runs along-slope
        # contour_dir = (-grad_y, grad_x) rotated 90 CCW
        contour_u = -grad_y  # along-contour east component
        contour_v = grad_x   # along-contour north component
        contour_mag = np.sqrt(contour_u**2 + contour_v**2)
        contour_mag[contour_mag == 0] = 1  # avoid div by zero

        # Normalize contour direction
        contour_u_n = contour_u / contour_mag
        contour_v_n = contour_v / contour_mag

        # Normalize current direction
        speed_safe = np.where(speed == 0, 1, speed)
        cur_u_n = u_grid / speed_safe
        cur_v_n = v_grid / speed_safe

        # Angle between current and contour: dot product = cos(angle)
        # |dot| = 1 means parallel, |dot| = 0 means perpendicular
        dot = cur_u_n * contour_u_n + cur_v_n * contour_v_n
        alignment = np.abs(dot)  # 0 = perpendicular, 1 = parallel

        # Cross product magnitude = sin(angle) = how perpendicular
        cross = np.abs(cur_u_n * contour_v_n - cur_v_n * contour_u_n)
        # cross: 0 = parallel, 1 = perpendicular

        land = np.isnan(u_grid) | np.isnan(v_grid) | (depth_grid <= 0)

    except Exception as e:
        print(f"  {date_str}: failed: {e}")
        import traceback; traceback.print_exc()
        return None

    results = []
    for c in date_catches:
        li = np.argmin(np.abs(lats - c["lat"]))
        lo = np.argmin(np.abs(lons - c["lon"]))
        if land[li, lo]:
            continue

        catch_alignment = alignment[li, lo]  # 0=perpendicular, 1=parallel
        catch_cross = cross[li, lo]  # 0=parallel, 1=perpendicular
        catch_speed = speed[li, lo]
        catch_depth = depth_grid[li, lo]
        catch_grad_mag = contour_mag[li, lo]

        # Background stats: ocean-wide alignment distribution
        ocean_mask = ~land & (depth_grid > 80)
        ocean_alignment = alignment[ocean_mask]

        # Alignment percentile (is catch more parallel or perpendicular than background?)
        align_pct = np.sum(ocean_alignment < catch_alignment) / len(ocean_alignment) * 100 if len(ocean_alignment) > 0 else np.nan

        # Local neighborhood alignment variability
        r = 3
        y0, y1 = max(0, li-r), min(len(lats), li+r+1)
        x0, x1 = max(0, lo-r), min(len(lons), lo+r+1)
        local_align = alignment[y0:y1, x0:x1]
        local_valid = local_align[~np.isnan(local_align) & ~land[y0:y1, x0:x1]]
        align_std = np.std(local_valid) if len(local_valid) > 3 else np.nan

        results.append({
            "date": date_str,
            "lat": c["lat"], "lon": c["lon"],
            "alignment": catch_alignment,  # 0=perp, 1=parallel
            "cross_flow": catch_cross,     # 0=parallel, 1=perp
            "speed": catch_speed,
            "depth": catch_depth,
            "grad_mag": catch_grad_mag,    # steepness of slope
            "align_pct": align_pct,
            "align_variability": align_std,
        })

    return results


def main():
    catches = load_catches()
    by_date = defaultdict(list)
    for c in catches:
        by_date[c["date"]].append(c)

    all_results = []
    dates = sorted(by_date.keys())
    print(f"Analyzing current-bathymetry alignment for {len(dates)} dates ({len(catches)} catches)...\n")

    for d in dates:
        r = analyze_date(d, by_date[d])
        if r:
            all_results.extend(r)

    if not all_results:
        print("No results!")
        return

    alignments = [r["alignment"] for r in all_results]
    cross_flows = [r["cross_flow"] for r in all_results]
    speeds = [r["speed"] for r in all_results if not np.isnan(r["speed"])]
    depths = [r["depth"] for r in all_results]
    align_pcts = [r["align_pct"] for r in all_results if not np.isnan(r["align_pct"])]
    align_vars = [r["align_variability"] for r in all_results if not np.isnan(r["align_variability"])]

    print(f"{'='*70}")
    print(f"CURRENT-BATHYMETRY ALIGNMENT ANALYSIS ({len(all_results)} catches)")
    print(f"{'='*70}")

    # 1. Overall alignment distribution
    print(f"\n--- Alignment at Catches (0=perpendicular, 1=parallel to contours) ---")
    print(f"  Mean:   {np.mean(alignments):.3f}")
    print(f"  Median: {np.median(alignments):.3f}")
    print(f"  Std:    {np.std(alignments):.3f}")
    print(f"  P25:    {np.percentile(alignments, 25):.3f}")
    print(f"  P75:    {np.percentile(alignments, 75):.3f}")

    # Histogram
    bins = [(0, 0.2, "Perpendicular"), (0.2, 0.4, "Mostly perp"),
            (0.4, 0.6, "Mixed"), (0.6, 0.8, "Mostly parallel"),
            (0.8, 1.01, "Parallel")]
    print(f"\n  Distribution:")
    for lo, hi, label in bins:
        cnt = sum(1 for a in alignments if lo <= a < hi)
        bar = "#" * int(cnt / len(alignments) * 40)
        print(f"    {label:>18s} ({lo:.1f}-{hi:.1f}): {cnt:3d} ({cnt/len(alignments)*100:4.0f}%) {bar}")

    # 2. Alignment percentile vs background
    print(f"\n--- Catch Alignment vs Ocean Background ---")
    print(f"  Mean percentile: {np.mean(align_pcts):.0f}%")
    print(f"  (>50% = catches are MORE parallel than average ocean)")
    print(f"  (<50% = catches are MORE perpendicular than average ocean)")

    # 3. Alignment vs depth
    print(f"\n--- Alignment vs Depth ---")
    depth_bands = [
        (80, 200, "80-200m (shelf)"),
        (200, 400, "200-400m (upper slope)"),
        (400, 800, "400-800m (mid slope)"),
        (800, 2000, "800m+ (deep)"),
    ]
    for d_lo, d_hi, label in depth_bands:
        group = [r for r in all_results if d_lo <= r["depth"] < d_hi]
        if group:
            g_align = np.mean([r["alignment"] for r in group])
            g_cross = np.mean([r["cross_flow"] for r in group])
            g_spd = np.mean([r["speed"] for r in group])
            print(f"  {label:>25s}: n={len(group):2d}, alignment={g_align:.3f}, "
                  f"cross_flow={g_cross:.3f}, speed={g_spd:.3f} m/s")

    # 4. Correlation: alignment vs speed
    print(f"\n--- Alignment vs Speed ---")
    valid = [(r["alignment"], r["speed"]) for r in all_results if not np.isnan(r["speed"])]
    if len(valid) > 5:
        a, s = zip(*valid)
        corr = np.corrcoef(a, s)[0, 1]
        print(f"  Correlation (alignment vs speed): r = {corr:.3f}")
        print(f"  (Positive = faster current when parallel to contours)")

    # 5. Correlation: alignment vs slope steepness
    print(f"\n--- Alignment vs Slope Steepness ---")
    valid2 = [(r["alignment"], r["grad_mag"]) for r in all_results if r["grad_mag"] > 0]
    if len(valid2) > 5:
        a2, g2 = zip(*valid2)
        corr2 = np.corrcoef(a2, g2)[0, 1]
        print(f"  Correlation (alignment vs grad_mag): r = {corr2:.3f}")
        print(f"  (Negative = more perpendicular flow at steeper slopes)")

    # 6. Cross-shelf flow: does high cross-flow predict catches?
    print(f"\n--- Cross-Shelf Flow at Catches ---")
    print(f"  Mean cross-flow: {np.mean(cross_flows):.3f}")
    print(f"  Median: {np.median(cross_flows):.3f}")
    print(f"  (0 = flow parallel to contours, 1 = flow perpendicular/crossing contours)")

    # 7. Alignment variability (transition zones)
    if align_vars:
        print(f"\n--- Alignment Variability Near Catches ---")
        print(f"  Mean alignment std (3-cell radius): {np.mean(align_vars):.3f}")
        print(f"  Median: {np.median(align_vars):.3f}")
        print(f"  (High = catch near transition between parallel and perpendicular flow)")

    # 8. Combined: speed + alignment quadrants
    print(f"\n--- Speed x Alignment Quadrants ---")
    spd_med = np.median(speeds)
    align_med = np.median(alignments)
    quads = {
        "Fast + Parallel":       [r for r in all_results if r["speed"] >= spd_med and r["alignment"] >= align_med],
        "Fast + Perpendicular":  [r for r in all_results if r["speed"] >= spd_med and r["alignment"] < align_med],
        "Slow + Parallel":       [r for r in all_results if r["speed"] < spd_med and r["alignment"] >= align_med],
        "Slow + Perpendicular":  [r for r in all_results if r["speed"] < spd_med and r["alignment"] < align_med],
    }
    for label, group in quads.items():
        pct = len(group) / len(all_results) * 100
        avg_d = np.mean([r["depth"] for r in group]) if group else 0
        print(f"  {label:>25s}: {len(group):2d} ({pct:4.0f}%)  avg depth={avg_d:.0f}m")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
