"""
Analyze oceanographic conditions at all blue marlin catch locations.

Extracts raw SST, CHL, SSH, MLD, currents, O2, SSTA, depth, gradients
at each catch site from the NetCDF data, then computes optimal ranges
and distributions for each variable.

Output: data/catch_environment.json — comprehensive conditions profile
"""

import os
import csv
import json
import numpy as np
from datetime import datetime

DATA_DIR = "data"


def load_netcdf_var(nc_path, var_names):
    """Load a variable from NetCDF, trying multiple names."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        for name in var_names:
            if name in ds:
                data = ds[name].squeeze()
                lats = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
                lons = ds['longitude'].values if 'longitude' in ds else ds['lon'].values
                vals = data.values
                ds.close()
                return vals, lats, lons
        ds.close()
    except Exception:
        pass
    return None, None, None


def sample_at_point(grid, lats, lons, lat, lon):
    """Sample a 2D grid at a lat/lon point using nearest neighbour."""
    if grid is None or lats is None:
        return None
    ri = np.argmin(np.abs(lats - lat))
    ci = np.argmin(np.abs(lons - lon))
    if 0 <= ri < grid.shape[0] and 0 <= ci < grid.shape[1]:
        val = float(grid[ri, ci])
        if np.isnan(val):
            return None
        return val
    return None


def sample_neighbourhood(grid, lats, lons, lat, lon, radius_cells=1):
    """Sample mean of a small neighbourhood around a point."""
    if grid is None or lats is None:
        return None
    ri = np.argmin(np.abs(lats - lat))
    ci = np.argmin(np.abs(lons - lon))
    r0 = max(0, ri - radius_cells)
    r1 = min(grid.shape[0], ri + radius_cells + 1)
    c0 = max(0, ci - radius_cells)
    c1 = min(grid.shape[1], ci + radius_cells + 1)
    region = grid[r0:r1, c0:c1]
    valid = region[~np.isnan(region)]
    if len(valid) == 0:
        return None
    return float(np.mean(valid))


def compute_gradient_at_point(grid, lats, lons, lat, lon):
    """Compute gradient magnitude at a point."""
    if grid is None or lats is None:
        return None
    from scipy.ndimage import sobel
    clean = np.where(np.isnan(grid), np.nanmean(grid), grid)
    gx = sobel(clean, axis=1)
    gy = sobel(clean, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    return sample_at_point(grad, lats, lons, lat, lon)


def get_depth_at_point(lat, lon):
    """Get depth from the bathymetry GeoTIFF."""
    try:
        import rasterio
        tif = os.path.join(DATA_DIR, "bathy_gmrt.tif")
        if not os.path.exists(tif):
            return None
        with rasterio.open(tif) as src:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = float(src.read(1)[row, col])
                if val != src.nodata and not np.isnan(val):
                    return round(-val)  # positive depth in meters
        return None
    except Exception:
        return None


def analyze_catch(date_str, lat, lon, tag):
    """Extract all environmental variables at a catch location."""
    date_dir = os.path.join(DATA_DIR, date_str)
    result = {"date": date_str, "lat": lat, "lon": lon, "tag": tag}

    # SST
    sst_path = os.path.join(date_dir, "sst_raw.nc")
    sst, s_lats, s_lons = load_netcdf_var(sst_path, ["thetao", "analysed_sst"])
    if sst is not None:
        # Convert from Kelvin if needed
        if np.nanmean(sst) > 100:
            sst = sst - 273.15
        result["sst_C"] = round(sample_at_point(sst, s_lats, s_lons, lat, lon) or np.nan, 2)
        result["sst_neighbourhood_C"] = round(sample_neighbourhood(sst, s_lats, s_lons, lat, lon, 2) or np.nan, 2)
        result["sst_gradient"] = round(compute_gradient_at_point(sst, s_lats, s_lons, lat, lon) or np.nan, 4)

    # CHL
    chl_path = os.path.join(date_dir, "chl_raw.nc")
    chl, c_lats, c_lons = load_netcdf_var(chl_path, ["CHL", "chlor_a", "chl"])
    if chl is not None:
        result["chl_mg_m3"] = round(sample_at_point(chl, c_lats, c_lons, lat, lon) or np.nan, 4)
        result["chl_neighbourhood"] = round(sample_neighbourhood(chl, c_lats, c_lons, lat, lon, 2) or np.nan, 4)
        result["chl_gradient"] = round(compute_gradient_at_point(chl, c_lats, c_lons, lat, lon) or np.nan, 6)

    # SSH (Sea Level Anomaly)
    ssh_path = os.path.join(date_dir, "ssh_raw.nc")
    ssh, h_lats, h_lons = load_netcdf_var(ssh_path, ["zos", "sla", "adt"])
    if ssh is not None:
        result["ssh_m"] = round(sample_at_point(ssh, h_lats, h_lons, lat, lon) or np.nan, 4)
        result["ssh_neighbourhood"] = round(sample_neighbourhood(ssh, h_lats, h_lons, lat, lon, 2) or np.nan, 4)

    # MLD
    mld_path = os.path.join(date_dir, "mld_raw.nc")
    mld, m_lats, m_lons = load_netcdf_var(mld_path, ["mlotst", "mld"])
    if mld is not None:
        result["mld_m"] = round(sample_at_point(mld, m_lats, m_lons, lat, lon) or np.nan, 1)

    # Currents
    curr_path = os.path.join(date_dir, "current_raw.nc")
    uo, u_lats, u_lons = load_netcdf_var(curr_path, ["uo"])
    vo, _, _ = load_netcdf_var(curr_path, ["vo"])
    if uo is not None and vo is not None:
        u = sample_at_point(uo, u_lats, u_lons, lat, lon)
        v = sample_at_point(vo, u_lats, u_lons, lat, lon)
        if u is not None and v is not None:
            speed_ms = np.sqrt(u**2 + v**2)
            result["current_speed_ms"] = round(speed_ms, 3)
            result["current_speed_kn"] = round(speed_ms * 1.94384, 2)
            result["current_dir_deg"] = round(np.degrees(np.arctan2(u, v)) % 360, 1)

    # O2
    o2_path = os.path.join(date_dir, "o2_raw.nc")
    o2, o_lats, o_lons = load_netcdf_var(o2_path, ["o2", "dissolved_oxygen"])
    if o2 is not None:
        result["o2_mmol_m3"] = round(sample_at_point(o2, o_lats, o_lons, lat, lon) or np.nan, 1)

    # SSTA
    ssta_path = os.path.join(date_dir, "ssta_raw.nc")
    if os.path.exists(ssta_path):
        ssta, sa_lats, sa_lons = load_netcdf_var(ssta_path, ["thetao", "sst_anomaly", "analysed_sst"])
        if ssta is not None:
            val = sample_at_point(ssta, sa_lats, sa_lons, lat, lon)
            if val is not None:
                result["ssta_C"] = round(val, 2)

    # KD490 (water clarity)
    kd_path = os.path.join(date_dir, "kd490_raw.nc")
    kd, k_lats, k_lons = load_netcdf_var(kd_path, ["KD490", "kd490", "kd_490"])
    if kd is not None:
        result["kd490"] = round(sample_at_point(kd, k_lats, k_lons, lat, lon) or np.nan, 4)

    # Depth from bathymetry
    depth = get_depth_at_point(lat, lon)
    if depth is not None:
        result["depth_m"] = depth

    # Distance from shelf break (200m contour) — approximate from depth
    if depth is not None:
        result["depth_category"] = (
            "inner_shelf" if depth < 100 else
            "outer_shelf" if depth < 200 else
            "shelf_break" if depth < 500 else
            "slope" if depth < 1000 else
            "deep"
        )

    return result


def compute_statistics(values, label):
    """Compute summary statistics for a list of values."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return None
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "optimal_range": f"{round(float(np.percentile(arr, 25)), 2)} - {round(float(np.percentile(arr, 75)), 2)}",
    }


def main():
    # Load catches
    catches = []
    with open(os.path.join(DATA_DIR, "all_catches.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["species"] != "BLUE MARLIN":
                continue
            if not row["lat"] or not row["lon"]:
                continue
            parts = row["date"].split("/")
            date_str = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            if not os.path.isdir(os.path.join(DATA_DIR, date_str)):
                continue
            catches.append({
                "date": date_str,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "tag": row.get("tag", ""),
                "weight_kg": float(row["weight_kg"]) if row.get("weight_kg") else None,
                "length_cm": float(row["length_cm"]) if row.get("length_cm") else None,
            })

    print(f"Analyzing {len(catches)} blue marlin catches...")

    # Extract environment at each catch
    catch_data = []
    for i, c in enumerate(catches):
        print(f"  [{i+1}/{len(catches)}] {c['date']} ({c['lat']:.4f}, {c['lon']:.4f}) {c['tag']}")
        env = analyze_catch(c["date"], c["lat"], c["lon"], c["tag"])
        env["weight_kg"] = c["weight_kg"]
        env["length_cm"] = c["length_cm"]
        catch_data.append(env)

    # Compute summary statistics for each variable
    env_vars = {
        "sst_C": "Sea Surface Temperature (C)",
        "sst_neighbourhood_C": "SST 3x3 neighbourhood (C)",
        "sst_gradient": "SST gradient magnitude",
        "chl_mg_m3": "Chlorophyll-a (mg/m3)",
        "chl_neighbourhood": "CHL 3x3 neighbourhood (mg/m3)",
        "chl_gradient": "CHL gradient magnitude",
        "ssh_m": "Sea Surface Height anomaly (m)",
        "ssh_neighbourhood": "SSH 3x3 neighbourhood (m)",
        "mld_m": "Mixed Layer Depth (m)",
        "current_speed_ms": "Current speed (m/s)",
        "current_speed_kn": "Current speed (knots)",
        "current_dir_deg": "Current direction (deg)",
        "o2_mmol_m3": "Dissolved oxygen (mmol/m3)",
        "ssta_C": "SST Anomaly (C)",
        "kd490": "KD490 diffuse attenuation",
        "depth_m": "Depth (m)",
    }

    summary = {}
    for var, label in env_vars.items():
        values = [c.get(var) for c in catch_data]
        stats = compute_statistics(values, label)
        if stats:
            summary[var] = {"label": label, **stats}

    # Depth category breakdown
    depth_cats = [c.get("depth_category") for c in catch_data if c.get("depth_category")]
    if depth_cats:
        from collections import Counter
        cat_counts = Counter(depth_cats)
        summary["depth_categories"] = {k: v for k, v in sorted(cat_counts.items())}

    # Month distribution
    months = [int(c["date"].split("-")[1]) for c in catch_data]
    from collections import Counter
    month_counts = Counter(months)
    summary["catch_months"] = {str(k): v for k, v in sorted(month_counts.items())}

    # Print results
    print("\n" + "=" * 70)
    print("BLUE MARLIN OPTIMAL ENVIRONMENT PROFILE")
    print(f"Based on {len(catch_data)} catches with oceanographic data")
    print("=" * 70)

    for var, label in env_vars.items():
        if var not in summary:
            continue
        s = summary[var]
        print(f"\n{label}:")
        print(f"  Median: {s['median']}  Mean: {s['mean']}  Std: {s['std']}")
        print(f"  Range: {s['min']} - {s['max']}")
        print(f"  IQR (optimal): {s['optimal_range']}")
        print(f"  P10-P90: {s['p10']} - {s['p90']}")
        print(f"  n={s['n']}")

    if "depth_categories" in summary:
        print(f"\nDepth categories:")
        for cat, count in summary["depth_categories"].items():
            print(f"  {cat}: {count} ({count/len(catch_data)*100:.0f}%)")

    if "catch_months" in summary:
        print(f"\nCatch months:")
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        for m, count in summary["catch_months"].items():
            print(f"  {month_names[int(m)]}: {count}")

    # Save everything
    output = {
        "generated": datetime.now().isoformat(),
        "n_catches": len(catch_data),
        "summary": summary,
        "catches": catch_data,
    }

    out_path = os.path.join(DATA_DIR, "catch_environment.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
