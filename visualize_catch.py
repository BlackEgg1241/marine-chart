#!/usr/bin/env python3
"""
visualize_catch.py — Generate per-feature heatmaps for catch dates
overlaid with catch locations.

Usage:
    python visualize_catch.py                  # uses default date (2022-02-19)
    python visualize_catch.py 2017-02-13       # specific date
    python visualize_catch.py all              # all catch dates

Output: Screenshots/<feature>/<date>.png — filed by feature type so you can
flip through all dates for one variable.
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import marlin_data
from marlin_data import generate_blue_marlin_hotspots

BBOX = {
    "lon_min": 113.5, "lon_max": 116.5,
    "lat_min": -33.5, "lat_max": -30.5,
}
BASE_DIR = "data"
CSV_PATH = r"C:\Users\User\Downloads\Export.csv"
ALL_CATCHES_CSV = os.path.join("data", "all_catches.csv")


def ddm_to_dd(raw_str, negative=False):
    val = float(raw_str)
    degrees = int(val)
    minutes = (val - degrees) * 100
    dd = degrees + minutes / 60.0
    return -dd if negative else dd


def _parse_date(date_str):
    if "/" in date_str:
        parts = date_str.strip().split("/")
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    return date_str[:10]


def load_catches():
    catches = []
    seen = set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat = ddm_to_dd(r["Latitude"].strip().replace("S", ""), negative=True)
            lon = ddm_to_dd(r["Longitude"].strip().replace("E", ""), negative=False)
            date = r["Release_Date"][:10]
            key = (date, round(lat, 4), round(lon, 4))
            if key not in seen:
                seen.add(key)
                catches.append({"date": date, "lat": lat, "lon": lon,
                                "species": r["Species_Name"]})

    if os.path.exists(ALL_CATCHES_CSV):
        with open(ALL_CATCHES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lat_str = r.get("lat", "").strip()
                lon_str = r.get("lon", "").strip()
                if not lat_str or not lon_str:
                    continue
                try:
                    lat, lon = float(lat_str), float(lon_str)
                except ValueError:
                    continue
                date = _parse_date(r["date"])
                key = (date, round(lat, 4), round(lon, 4))
                if key not in seen:
                    seen.add(key)
                    catches.append({"date": date, "lat": lat, "lon": lon,
                                    "species": r.get("species", "")})

    return [c for c in catches if c["species"].strip().upper() == "BLUE MARLIN"]


def plot_feature(ax, lons, lats, grid, title, catch_lons, catch_lats,
                 cmap="YlOrRd", vmin=0, vmax=1):
    """Plot a single feature grid with catch markers."""
    masked = np.ma.masked_invalid(grid)
    im = ax.pcolormesh(lons, lats, masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.scatter(catch_lons, catch_lats, c="cyan", edgecolors="black",
               s=120, zorder=10, linewidths=1.5, marker="*")
    ax.set_xlim(114.6, 115.6)
    ax.set_ylim(-32.4, -31.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    return im


def generate_for_date(date_str, catches):
    """Generate all feature visualizations for one date, filed by feature."""
    date_catches = [c for c in catches if c["date"] == date_str]
    if not date_catches:
        print(f"  No catches for {date_str}")
        return

    catch_lons = [c["lon"] for c in date_catches]
    catch_lats = [c["lat"] for c in date_catches]

    dated_dir = os.path.join(BASE_DIR, date_str)
    if not os.path.exists(dated_dir):
        print(f"  No data dir for {date_str}")
        return

    tif_path = os.path.join(dated_dir, "bathy_gmrt.tif")
    if not os.path.exists(tif_path):
        tif_path = os.path.join(BASE_DIR, "bathy_gmrt.tif")

    marlin_data.OUTPUT_DIR = dated_dir

    print(f"  Scoring {date_str}...")
    try:
        result = generate_blue_marlin_hotspots(BBOX, tif_path=tif_path, date_str=date_str)
    except Exception as e:
        print(f"  Scoring failed: {e}")
        return

    grid = result["grid"]
    lats = result["lats"]
    lons = result["lons"]
    sub_scores = result["sub_scores"]
    weights = result["weights"]

    def _sample(feat_grid):
        scores = []
        for clat, clon in zip(catch_lats, catch_lons):
            yi = np.argmin(np.abs(lats - clat))
            xi = np.argmin(np.abs(lons - clon))
            val = feat_grid[yi, xi]
            scores.append(val if not np.isnan(val) else 0)
        return scores

    def _save(feat_name, feat_grid, title, cmap="YlOrRd", vmin=0, vmax=1):
        feat_dir = os.path.join("Screenshots", feat_name)
        os.makedirs(feat_dir, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = plot_feature(ax, lons, lats, feat_grid, title,
                          catch_lons, catch_lats, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Score")
        # Shelf-break contour (depth gate score ~0.99)
        if "depth" in sub_scores and feat_name not in ("depth", "band_count"):
            try:
                ax.contour(lons, lats, sub_scores["depth"], levels=[0.99],
                           colors="gray", linewidths=0.5, linestyles="--")
            except Exception:
                pass
        fig.tight_layout()
        fig.savefig(os.path.join(feat_dir, f"{date_str}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    def _save_sidebyside(feat_name, raw_grid, scored_grid, raw_title, scored_title,
                         cmap="YlOrRd", vmin=0, vmax=1):
        """Save raw vs edge-scored side by side for comparison."""
        feat_dir = os.path.join("Screenshots", feat_name)
        os.makedirs(feat_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        im1 = plot_feature(ax1, lons, lats, raw_grid, raw_title,
                           catch_lons, catch_lats, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax1, shrink=0.7, label="Score")
        im2 = plot_feature(ax2, lons, lats, scored_grid, scored_title,
                           catch_lons, catch_lats, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=ax2, shrink=0.7, label="Score")
        # Shelf-break contour on both panels
        if "depth" in sub_scores:
            for ax in (ax1, ax2):
                try:
                    ax.contour(lons, lats, sub_scores["depth"], levels=[0.99],
                               colors="gray", linewidths=0.5, linestyles="--")
                except Exception:
                    pass
        fig.tight_layout()
        fig.savefig(os.path.join(feat_dir, f"{date_str}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    n_saved = 0

    # --- Every sub-score (active or not) ---
    for feat_name, feat_grid in sorted(sub_scores.items()):
        # Skip _raw variants (pre-edge-transform) — shown side-by-side with scored version
        if feat_name.endswith("_raw"):
            continue

        w = weights.get(feat_name, 0)
        cs = _sample(feat_grid)
        score_str = ", ".join(f"{s:.0%}" for s in cs)

        # Check if there's a _raw version for side-by-side comparison
        raw_key = f"{feat_name}_raw"
        has_raw = raw_key in sub_scores

        if feat_name == "depth":
            title = f"DEPTH GATE (multiplicative)\n{date_str} | catch: {score_str}"
            _save(feat_name, feat_grid, title, cmap="YlGnBu")
        elif has_raw and w > 0:
            # Side-by-side: raw input vs value-space scored output
            raw_grid = sub_scores[raw_key]
            raw_cs = _sample(raw_grid)
            raw_score_str = ", ".join(f"{s:.0%}" for s in raw_cs)
            raw_title = f"{feat_name.upper()} RAW\n{date_str} | catch: {raw_score_str}"
            scored_title = f"{feat_name.upper()} SCORED (w={w:.3f})\n{date_str} | catch: {score_str}"
            _save_sidebyside(feat_name, raw_grid, feat_grid, raw_title, scored_title)
        elif w > 0:
            title = f"{feat_name.upper()} (w={w:.3f})\n{date_str} | catch: {score_str}"
            _save(feat_name, feat_grid, title)
        else:
            title = f"{feat_name.upper()} (w=0 DISABLED)\n{date_str} | catch: {score_str}"
            _save(feat_name, feat_grid, title, cmap="Greys")
        n_saved += 1

    # --- Composite ---
    cs = _sample(grid)
    score_str = ", ".join(f"{s:.0%}" for s in cs)
    title = f"COMPOSITE SCORE\n{date_str} | catch: {score_str}"
    _save("composite", grid, title)
    n_saved += 1

    # --- Band count ---
    if result.get("band_count") is not None:
        band_grid = result["band_count"]
        cs = _sample(band_grid)
        score_str = ", ".join(f"{s:.1f}" for s in cs)
        title = f"BAND COUNT\n{date_str} | catch: {score_str} bands"
        _save("band_count", band_grid, title, cmap="viridis",
              vmin=0, vmax=max(np.nanmax(band_grid), 1))
        n_saved += 1

    # --- Band mean ---
    if result.get("band_mean") is not None:
        bm_grid = result["band_mean"]
        cs = _sample(bm_grid)
        score_str = ", ".join(f"{s:.2f}" for s in cs)
        title = f"BAND MEAN\n{date_str} | catch: {score_str}"
        _save("band_mean", bm_grid, title, cmap="viridis")
        n_saved += 1

    # --- FTLE fallback: requires multi-day velocity data, may not be in sub_scores ---
    if "ftle" not in sub_scores:
        ftle_cache = os.path.join(dated_dir, "ftle_cache.npz")
        if not os.path.exists(ftle_cache):
            ftle_cache = os.path.join(BASE_DIR, "ftle_cache.npz")
        if os.path.exists(ftle_cache):
            try:
                data = np.load(ftle_cache)
                if date_str in data:
                    ftle_raw = data[date_str]
                    cs_raw = []
                    for clat, clon in zip(catch_lats, catch_lons):
                        yi = np.argmin(np.abs(lats - clat))
                        xi = np.argmin(np.abs(lons - clon))
                        if yi < ftle_raw.shape[0] and xi < ftle_raw.shape[1]:
                            val = ftle_raw[yi, xi]
                            cs_raw.append(val if not np.isnan(val) else 0)
                        else:
                            cs_raw.append(0)
                    score_str = ", ".join(f"{s:.2f}" for s in cs_raw)
                    w = weights.get("ftle", 0)
                    title = f"FTLE (w={w:.3f})\n{date_str} | catch: {score_str}"
                    _save("ftle_raw", ftle_raw, title, cmap="inferno",
                          vmin=0, vmax=max(np.nanmax(ftle_raw), 0.01))
                    n_saved += 1
            except Exception:
                pass

    # --- Current direction + strength visualization ---
    try:
        import xarray as xr
        cur_file = os.path.join(dated_dir, "currents_raw.nc")
        if os.path.exists(cur_file):
            cds = xr.open_dataset(cur_file)
            for vn in ["uo", "uo_cglo", "utotal"]:
                if vn in cds:
                    uo_da = cds[vn]; break
            for vn in ["vo", "vo_cglo", "vtotal"]:
                if vn in cds:
                    vo_da = cds[vn]; break
            if "depth" in uo_da.dims:
                uo_da = uo_da.isel(depth=0)
            if "depth" in vo_da.dims:
                vo_da = vo_da.isel(depth=0)
            uo_da = uo_da.squeeze()
            vo_da = vo_da.squeeze()
            from scipy.interpolate import RegularGridInterpolator
            c_lons = uo_da.coords[uo_da.dims[-1]].values.astype(float)
            c_lats = uo_da.coords[uo_da.dims[-2]].values.astype(float)
            uo_vals = uo_da.values.astype(float)
            vo_vals = vo_da.values.astype(float)
            cds.close()

            # Interpolate to master grid
            uo_interp = RegularGridInterpolator((c_lats, c_lons), uo_vals,
                                                 bounds_error=False, fill_value=np.nan)
            vo_interp = RegularGridInterpolator((c_lats, c_lons), vo_vals,
                                                 bounds_error=False, fill_value=np.nan)
            pts = np.array([[la, lo] for la in lats for lo in lons])
            uo_grid = uo_interp(pts).reshape(len(lats), len(lons))
            vo_grid = vo_interp(pts).reshape(len(lats), len(lons))

            speed = np.sqrt(uo_grid**2 + vo_grid**2)
            angle = np.arctan2(uo_grid, vo_grid)  # oceanographic bearing (0=N, 90=E, CW positive)

            # Cardinal colors: N=Blue, E=Red, S=Yellow, W=Green
            # Map angle to RGBA using cardinal blending
            north = np.array([0.0, 0.3, 1.0])   # blue
            east  = np.array([1.0, 0.0, 0.0])    # red
            south = np.array([1.0, 0.9, 0.0])    # yellow
            west  = np.array([0.0, 0.8, 0.0])    # green

            ny_c, nx_c = speed.shape
            rgba = np.zeros((ny_c, nx_c, 4))
            for yi in range(ny_c):
                for xi in range(nx_c):
                    a = angle[yi, xi]
                    s = speed[yi, xi]
                    if np.isnan(a) or np.isnan(s):
                        rgba[yi, xi] = [1, 1, 1, 0]
                        continue
                    # Normalize angle to 0-360
                    deg = np.degrees(a) % 360
                    # Blend between cardinals (0=N, 90=E, 180=S, 270=W)
                    if deg < 90:
                        t = deg / 90.0
                        rgb = north * (1 - t) + east * t
                    elif deg < 180:
                        t = (deg - 90) / 90.0
                        rgb = east * (1 - t) + south * t
                    elif deg < 270:
                        t = (deg - 180) / 90.0
                        rgb = south * (1 - t) + west * t
                    else:
                        t = (deg - 270) / 90.0
                        rgb = west * (1 - t) + north * t
                    rgba[yi, xi, :3] = rgb
                    rgba[yi, xi, 3] = 1.0  # placeholder, overwritten by speed-based alpha

            # Alpha from speed — percentile-based (p90 = fully opaque)
            ocean_speed = speed[~np.isnan(speed)]
            if len(ocean_speed) > 0:
                sp90 = np.percentile(ocean_speed, 90)
                sp90 = max(sp90, 0.01)
                alpha = np.clip(speed / sp90, 0.05, 1.0)
            else:
                alpha = np.ones_like(speed) * 0.5
            alpha[np.isnan(speed)] = 0
            rgba[:, :, 3] = alpha

            # Plot
            feat_dir = os.path.join("Screenshots", "current_direction")
            os.makedirs(feat_dir, exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(rgba, extent=[lons[0], lons[-1], lats[-1], lats[0]],
                      aspect="equal", origin="upper", interpolation="nearest")
            ax.scatter(catch_lons, catch_lats, c="white", edgecolors="black",
                       s=120, zorder=10, linewidths=1.5, marker="*")
            ax.set_xlim(114.6, 115.6)
            ax.set_ylim(-32.4, -31.5)
            ax.set_title(f"CURRENT DIRECTION + STRENGTH\n{date_str} | "
                         f"N=Blue E=Red S=Yellow W=Green | alpha=speed (p90)",
                         fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=8)
            # Shelf-break contour
            if "depth" in sub_scores:
                try:
                    ax.contour(lons, lats, sub_scores["depth"], levels=[0.99],
                               colors="gray", linewidths=0.5, linestyles="--")
                except Exception:
                    pass
            fig.tight_layout()
            fig.savefig(os.path.join(feat_dir, f"{date_str}.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            n_saved += 1
    except Exception as e:
        print(f"  Current direction viz failed: {e}")

    print(f"  {n_saved} images for {date_str}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "2022-02-19"
    catches = load_catches()

    if target == "all":
        dates = sorted(set(c["date"] for c in catches))
        print(f"Generating visuals for {len(dates)} catch dates...")
        for i, d in enumerate(dates):
            print(f"[{i+1}/{len(dates)}] {d}")
            generate_for_date(d, catches)
    else:
        print(f"Generating visuals for {target}...")
        generate_for_date(target, catches)

    print("\nDone.")


if __name__ == "__main__":
    main()
