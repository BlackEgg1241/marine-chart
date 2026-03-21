"""
Southern Bluefin Tuna (Thunnus maccoyii) habitat scoring.

Similar pelagic model to blue marlin but shifted to:
- Cooler water: optimal 18.5C vs marlin's 22.5C
- Shallower depth gate: 100-500m full vs marlin's 150-800m
- Higher front and CHL weights: SBT are strongly front-associated bait-followers
- Both eddy edges matter (not just warm): score |SLA| instead of positive-only
- No FAD bands: FADs are positioned for marlin, not SBT

Season: April-August (autumn/winter) in Perth/WA.
Bait: Pilchards, anchovies, squid — follow productivity.
"""

import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
SBT_WEIGHTS = {
    "sst":            0.18,  # Primary driver — cooler range than marlin
    "sst_front":      0.14,  # SBT strongly front-associated (marlin 0.10)
    "front_corridor": 0.06,  # Front pinch points funnel bait — same as marlin
    "chl":            0.16,  # Higher than marlin (0.13) — bait-follower
    "chl_curvature":  0.03,  # Bait pockets — slightly higher than marlin
    "ssh":            0.10,  # Both eddy types matter (marlin 0.19)
    "current":        0.06,  # Speed matters, direction less (marlin 0.04)
    "convergence":    0.05,  # Bait traps — slightly higher than marlin (0.04)
    "mld":            0.12,  # Thermocline structure matters (marlin 0.08)
    "clarity":        0.02,  # Low discrimination
    "sst_intrusion":  0.00,  # Not relevant — SBT prefer cool water
    "ssta":           0.04,  # Cool anomaly edges useful
    "o2":             0.04,  # Oxygen at depth matters more for SBT dives
}

# Intensity bands for polygon export — same as marlin (broad range,
# SST/season will naturally discriminate)
HOTSPOT_BANDS = [0.25, 0.32, 0.39, 0.46, 0.53, 0.60, 0.67, 0.74, 0.80, 0.85, 0.90, 0.95]

# Optimal parameters — shifted for SBT biology
_opt_sst_optimal = 18.5       # Peak SST (marlin 22.5)
_opt_sst_sigma = 1.5          # Tighter tolerance (marlin 1.88)
_opt_sst_sigma_above = 3.0    # Asymmetric: tolerate warmer better than cold
_opt_chl_optimal = 0.25       # Higher productivity (marlin 0.14)
_opt_chl_sigma = 0.25         # In log10 space
_opt_front_floor = 0.05       # Min front score in suitable SST water
_opt_corridor_thresh = 0.26   # Front cell mask threshold
_opt_shelf_boost = 0.15       # Higher than marlin (0.10) — more shelf-dependent

# Band system parameters
_opt_band_width_nm = 2.5      # Slightly wider than marlin (2.0)
_opt_band_boost = 0.30
_opt_band_decay = 0.80
_opt_band_front_thresh = 0.25
_opt_band_chl_thresh = 0.45
_opt_band_single = 0.06
_opt_band_overlap = 0.22      # Slightly higher than marlin (0.20)

# Bathy contour weights — emphasis on shallower shelf break
_bathy_band_weights = {
    50:   0.3,   # Inner shelf edge — SBT come shallower
    100:  0.6,   # Key SBT zone (marlin 0.2)
    150:  0.6,   # Same as marlin
    200:  0.7,   # THE shelf break — critical for SBT (marlin 0.6)
    300:  0.5,   # Canyon upper walls
    500:  0.3,   # Getting too deep for SBT (marlin 0.6)
    1000: 0.1,   # Marginal (marlin 0.2)
}

# SBT-specific isotherms
_sbt_isotherms = [17, 18, 19, 20]

# CHL contour for SBT (marlin uses 0.15)
_sbt_chl_contour = 0.25

# SST gradient threshold (same physical quantity as marlin)
SST_GRADIENT_THRESHOLD = 0.5


def generate_sbt_hotspots(bbox, tif_path=None, date_str=None, output_dir=None):
    """
    Build habitat suitability grid for Southern Bluefin Tuna.

    Parameters
    ----------
    bbox : dict with lon_min, lon_max, lat_min, lat_max
    tif_path : str, path to bathymetry GeoTIFF
    date_str : str, YYYY-MM-DD
    output_dir : str, output directory (default: data/)
    """
    import xarray as xr
    from scipy.ndimage import (gaussian_filter, sobel, laplace,
                                distance_transform_edt, convolve, binary_dilation)
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[SBT-Hotspots] Building Southern Bluefin Tuna habitat suitability...")

    # --- Resolve paths ---
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _output_dir = output_dir or os.path.join(script_dir, "data")
    if date_str and not output_dir:
        date_dir = os.path.join(script_dir, "data", date_str)
        if os.path.isdir(date_dir):
            _output_dir = date_dir

    if tif_path is None:
        tif_path = os.path.join(script_dir, "data", "bathymetry.tif")

    # --- Load SST as master grid ---
    sst_file = os.path.join(_output_dir, "sst_raw.nc")
    if not os.path.exists(sst_file):
        print("[SBT-Hotspots] No SST data — skipping")
        return None
    ds = xr.open_dataset(sst_file)
    for var in ["thetao", "analysed_sst", "sst"]:
        if var in ds:
            sst_da = ds[var].squeeze()
            break
    else:
        print("[SBT-Hotspots] No SST variable found")
        return None

    lons = sst_da.longitude.values if "longitude" in sst_da.dims else sst_da.lon.values
    lats = sst_da.latitude.values if "latitude" in sst_da.dims else sst_da.lat.values
    sst = sst_da.values.copy().astype(float)
    ds.close()

    # Convert Kelvin to Celsius
    if np.nanmean(sst) > 100:
        sst -= 273.15

    # Upsample coarse grids to ~0.02 deg
    grid_step = abs(np.diff(lons).mean()) if len(lons) > 1 else 1
    if grid_step > 0.05:
        target_step = 0.02
        fine_lons = np.arange(lons.min(), lons.max() + target_step * 0.5, target_step)
        fine_lats = np.arange(lats.min(), lats.max() + target_step * 0.5, target_step)
        if lats[0] > lats[-1]:
            fine_lats = fine_lats[::-1]
        interp_sst = RegularGridInterpolator(
            (lats, lons), sst, method="linear",
            bounds_error=False, fill_value=np.nan)
        fg_lat, fg_lon = np.meshgrid(fine_lats, fine_lons, indexing="ij")
        sst = interp_sst((fg_lat, fg_lon))
        lons = fine_lons
        lats = fine_lats

    land = np.isnan(sst)
    ny, nx = sst.shape
    score = np.zeros((ny, nx), dtype=float)
    weight_sum = np.zeros((ny, nx), dtype=float)
    sub_scores = {}

    def _interp_to_grid(data, src_lons, src_lats):
        if data.shape == (ny, nx):
            return data
        interp = RegularGridInterpolator(
            (src_lats, src_lons), data,
            method="linear", bounds_error=False, fill_value=np.nan)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        return interp((lat_grid, lon_grid))

    def _maxpool_to_grid(data_hr, hr_lons, hr_lats):
        result = np.zeros((ny, nx))
        dlat = abs(lats[1] - lats[0]) / 2 if ny > 1 else 0.04
        dlon = abs(lons[1] - lons[0]) / 2 if nx > 1 else 0.04
        for yi in range(ny):
            for xi in range(nx):
                b_row = (hr_lats >= lats[yi] - dlat) & (hr_lats <= lats[yi] + dlat)
                b_col = (hr_lons >= lons[xi] - dlon) & (hr_lons <= lons[xi] + dlon)
                if np.any(b_row) and np.any(b_col):
                    result[yi, xi] = np.max(data_hr[np.ix_(b_row, b_col)])
                else:
                    byi = np.argmin(np.abs(hr_lats - lats[yi]))
                    bxi = np.argmin(np.abs(hr_lons - lons[xi]))
                    result[yi, xi] = data_hr[byi, bxi]
        return result

    def _add_score(name, values, mask=None):
        w = SBT_WEIGHTS.get(name, 0)
        if w == 0:
            return
        v = np.clip(values, 0, 1)
        valid_mask = ~np.isnan(v) & ~land
        if mask is not None:
            valid_mask &= ~mask
        score[valid_mask] += w * v[valid_mask]
        weight_sum[valid_mask] += w
        sub_scores[name] = v.copy()

    # ------------------------------------------------------------------
    # 1. SST score — Gaussian centered at 18.5C (SBT optimal)
    # ------------------------------------------------------------------
    sst_filled = sst.copy()
    sst_filled[land] = np.nanmean(sst)
    sst_smooth = gaussian_filter(sst_filled, sigma=0.5)
    sst_smooth[land] = np.nan

    sigma_map = np.where(sst_smooth < _opt_sst_optimal, _opt_sst_sigma, _opt_sst_sigma_above)
    sst_score = np.exp(-0.5 * ((sst_smooth - _opt_sst_optimal) / sigma_map) ** 2)
    _add_score("sst", sst_score)

    mean_sst = float(np.nanmean(sst_smooth[~land]))
    sst_pct = np.sum(sst_score[~land & ~np.isnan(sst_score)] > 0.5) / max(np.sum(~land), 1) * 100
    print(f"[SBT-Hotspots] SST: mean {mean_sst:.1f}C (optimal {_opt_sst_optimal}C), "
          f"{sst_pct:.0f}% cells >50% score")

    # ------------------------------------------------------------------
    # 2. SST front score — Sobel gradient, modulated by SST suitability
    # ------------------------------------------------------------------
    sst_for_grad = sst_filled.copy()
    sst_grad = gaussian_filter(sst_for_grad, sigma=1.5)
    gx = sobel(sst_grad, axis=1)
    gy = sobel(sst_grad, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    coast_buf = binary_dilation(land, iterations=2)
    grad_mag[coast_buf] = 0
    ocean_grad = grad_mag[~coast_buf & ~land]
    g90 = np.nanpercentile(ocean_grad, 90) if len(ocean_grad) > 0 else 0
    if g90 > 0:
        front_score = np.clip(grad_mag / g90, 0, 1)
    else:
        front_score = np.zeros_like(grad_mag)
    front_score = front_score * sst_score
    warm_mask = sst_score > 0.6
    front_score = np.where(warm_mask, np.maximum(front_score, _opt_front_floor), front_score)
    _add_score("sst_front", front_score)

    # ------------------------------------------------------------------
    # 2a. SST Front Corridors — narrow front pinch points
    # ------------------------------------------------------------------
    try:
        front_mask = (front_score > _opt_corridor_thresh).astype(float)
        front_mask[land | coast_buf] = 0
        dist_to_front = distance_transform_edt(1 - front_mask)
        proximity = np.clip(1.0 - dist_to_front / 4.0, 0, 1)
        _gs = abs(lons[1] - lons[0]) if nx > 1 else 0.083
        k_size = max(5, int(round(0.15 / _gs)))
        if k_size % 2 == 0:
            k_size += 1
        half = k_size // 2
        quadrants = [
            (slice(0, half), slice(None)),
            (slice(half + 1, None), slice(None)),
            (slice(None), slice(0, half)),
            (slice(None), slice(half + 1, None)),
        ]
        dir_count = np.zeros((ny, nx))
        for qs in quadrants:
            k = np.zeros((k_size, k_size))
            k[qs] = 1.0
            k /= max(k.sum(), 1)
            d = convolve(front_mask, k, mode='constant', cval=0)
            dir_count += (d > 0.08).astype(float)
        corridor_score = proximity * np.clip((dir_count - 1) / 2.0, 0, 1)
        corridor_score *= sst_score
        corridor_score[land] = np.nan
        _add_score("front_corridor", corridor_score)
    except Exception as e:
        print(f"[SBT-Hotspots] Front corridor scoring failed: {e}")

    # ------------------------------------------------------------------
    # 3. Chlorophyll score — peaks at 0.25 mg/m3 (SBT bait zone)
    # ------------------------------------------------------------------
    chl_grid = None
    chl_file = os.path.join(_output_dir, "chl_raw.nc")
    if os.path.exists(chl_file):
        try:
            cds = xr.open_dataset(chl_file)
            for cv in ["chl", "CHL", "chlor_a"]:
                if cv in cds:
                    chl_da = cds[cv].squeeze()
                    break
            chl_lons = chl_da.longitude.values if "longitude" in chl_da.dims else chl_da.lon.values
            chl_lats = chl_da.latitude.values if "latitude" in chl_da.dims else chl_da.lat.values
            chl_data = _interp_to_grid(chl_da.values.astype(float), chl_lons, chl_lats)
            chl_grid = chl_data
            chl_log = np.log10(np.clip(chl_data, 0.01, 10))
            optimal_chl = np.log10(_opt_chl_optimal)
            chl_score = np.exp(-0.5 * ((chl_log - optimal_chl) / _opt_chl_sigma) ** 2)
            _add_score("chl", chl_score)

            # 3b. CHL Curvature — pockets & peninsulas
            try:
                chl_for_lap = chl_log.copy()
                chl_for_lap[np.isnan(chl_for_lap) | land] = np.nanmean(chl_log[~land])
                chl_lap_smooth = gaussian_filter(chl_for_lap, sigma=2.0)
                chl_laplacian = laplace(chl_lap_smooth)
                chl_laplacian[coast_buf] = 0
                chl_curv = np.abs(chl_laplacian)
                cc90 = np.nanpercentile(chl_curv[~coast_buf & ~land], 90)
                if cc90 > 0:
                    chl_curv_score = np.clip(chl_curv / cc90, 0, 1)
                    chl_curv_score[land] = np.nan
                    _add_score("chl_curvature", chl_curv_score)
            except Exception as e:
                print(f"[SBT-Hotspots] CHL curvature failed: {e}")
            cds.close()
        except Exception as e:
            print(f"[SBT-Hotspots] CHL scoring failed: {e}")

    # ------------------------------------------------------------------
    # 4. Depth score + shelf break — SBT: 30-1200m, peak 100-500m
    # ------------------------------------------------------------------
    _depth_grid = None
    if tif_path and os.path.exists(tif_path):
        try:
            import rasterio
            with rasterio.open(tif_path) as src:
                bathy = src.read(1).astype(float)
                bt = src.transform
                bw, bh = bathy.shape[1], bathy.shape[0]
                b_lons = np.array([bt.c + (j + 0.5) * bt.a for j in range(bw)])
                b_lats = np.array([bt.f + (i + 0.5) * bt.e for i in range(bh)])
                nd = src.nodata
                if nd is not None:
                    bathy[bathy == nd] = np.nan

            # Shelf break gradient
            bathy_filled = bathy.copy()
            bathy_filled[np.isnan(bathy_filled)] = 0
            dgx = sobel(bathy_filled, axis=1)
            dgy = sobel(bathy_filled, axis=0)
            depth_gradient = np.sqrt(dgx**2 + dgy**2)
            depth_gradient[np.isnan(bathy)] = 0
            shelf_break = _interp_to_grid(depth_gradient, b_lons, b_lats)
            shelf_score = np.clip(shelf_break / 100, 0, 1)
            shelf_score[land] = np.nan
            sub_scores["shelf_break"] = shelf_score.copy()

            # Depth score at native resolution, then max-pool
            abs_depth_hr = np.where(np.isnan(bathy), 0, -bathy)
            # SBT depth gate: ramp 30-100m, full 100-500m, taper 500-1000m
            depth_score_hr = np.where(abs_depth_hr < 30, 0,
                             np.where(abs_depth_hr < 100, 0.5 + 0.5 * (abs_depth_hr - 30) / 70,
                             np.where(abs_depth_hr < 500, 1.0,
                             np.where(abs_depth_hr < 1000, 0.6 + 0.4 * (1.0 - (abs_depth_hr - 500) / 500),
                             np.where(abs_depth_hr < 1200, 0.3, 0.0)))))
            depth_score_hr[np.isnan(bathy)] = 0
            depth_score = _maxpool_to_grid(depth_score_hr, b_lons, b_lats)
            depth_score[land] = np.nan
            sub_scores["depth"] = np.clip(depth_score, 0, 1)

            # Depth grid for hover info
            depth_master = _interp_to_grid(
                np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats)
            _depth_grid = depth_master
        except Exception as e:
            print(f"[SBT-Hotspots] Depth/shelf scoring failed: {e}")

    # ------------------------------------------------------------------
    # 5. SSH score — |SLA| (both warm and cold eddy edges matter for SBT)
    # ------------------------------------------------------------------
    ssh_grid = None
    ssh_file = os.path.join(_output_dir, "ssh_raw.nc")
    if os.path.exists(ssh_file):
        try:
            sds = xr.open_dataset(ssh_file)
            sv = "sla" if "sla" in sds else ("zos" if "zos" in sds else "adt")
            sla_da = sds[sv].squeeze()
            s_lons = sla_da.longitude.values if "longitude" in sla_da.dims else sla_da.lon.values
            s_lats = sla_da.latitude.values if "latitude" in sla_da.dims else sla_da.lat.values
            sla_data = _interp_to_grid(sla_da.values.astype(float), s_lons, s_lats)
            ssh_grid = sla_data
            sds.close()

            # SBT: score |SLA| — any eddy activity is good (not just warm)
            abs_score = np.clip(np.abs(sla_data) / 0.10, 0, 1)
            # Relative SLA: eddy edges (same as marlin)
            sla_filled = sla_data.copy()
            sla_filled[np.isnan(sla_filled)] = np.nanmean(sla_data)
            sla_bg = gaussian_filter(sla_filled, sigma=4)
            sla_relative = sla_data - sla_bg
            rel_score = np.clip(np.abs(sla_relative) / 0.04, 0, 1)
            # Blend: 40% absolute + 60% relative (weight edges more for SBT)
            ssh_score = 0.4 * abs_score + 0.6 * rel_score
            ssh_score[land] = np.nan
            _add_score("ssh", ssh_score)
        except Exception as e:
            print(f"[SBT-Hotspots] SSH scoring failed: {e}")

    # ------------------------------------------------------------------
    # 6. MLD score — SBT: best at 30-60m, zero at 120m
    # ------------------------------------------------------------------
    mld_grid = None
    mld_file = os.path.join(_output_dir, "mld_raw.nc")
    if os.path.exists(mld_file):
        try:
            mds = xr.open_dataset(mld_file)
            for mv in ["mlotst", "mld", "MLD"]:
                if mv in mds:
                    mld_da = mds[mv].squeeze()
                    break
            m_lons = mld_da.longitude.values if "longitude" in mld_da.dims else mld_da.lon.values
            m_lats = mld_da.latitude.values if "latitude" in mld_da.dims else mld_da.lat.values
            mld_data = _interp_to_grid(mld_da.values.astype(float), m_lons, m_lats)
            mld_grid = mld_data
            mds.close()
            # SBT: best at 30-60m (defined thermocline), zero at 120m
            mld_score = np.where(mld_data < 30, 0.7 + 0.3 * mld_data / 30,
                        np.where(mld_data < 60, 1.0,
                        np.clip(1.0 - (mld_data - 60) / 60, 0, 1)))
            _add_score("mld", mld_score)
        except Exception as e:
            print(f"[SBT-Hotspots] MLD scoring failed: {e}")

    # ------------------------------------------------------------------
    # 7. SSTA score — SBT use cool anomaly edges
    # ------------------------------------------------------------------
    try:
        from marlin_data import compute_ssta
        ssta = compute_ssta(sst, lats, lons, date_str) if date_str else None
        if ssta is not None and not np.all(np.isnan(ssta)):
            # SBT: negative SSTA (cooler than climatology) is favorable
            # Score peaks at SSTA ~ -0.5C (cool upwelling / Capes Current)
            ssta_optimal = -0.5
            ssta_sigma = 2.0
            ssta_score = np.exp(-0.5 * ((ssta - ssta_optimal) / ssta_sigma) ** 2)
            _add_score("ssta", ssta_score)
    except Exception as e:
        print(f"[SBT-Hotspots] SSTA scoring failed: {e}")

    # ------------------------------------------------------------------
    # 8. Water clarity — KD490 (same as marlin)
    # ------------------------------------------------------------------
    kd_file = os.path.join(_output_dir, "kd490_raw.nc")
    if os.path.exists(kd_file):
        try:
            kds = xr.open_dataset(kd_file)
            kd_da = kds["KD490"].squeeze()
            k_lons = kd_da.longitude.values if "longitude" in kd_da.dims else kd_da.lon.values
            k_lats = kd_da.latitude.values if "latitude" in kd_da.dims else kd_da.lat.values
            kd_data = _interp_to_grid(kd_da.values.astype(float), k_lons, k_lats)
            kds.close()
            clarity_score = np.clip(1.0 - (kd_data - 0.04) / 0.11, 0, 1)
            _add_score("clarity", clarity_score)
        except Exception as e:
            print(f"[SBT-Hotspots] Clarity scoring failed: {e}")

    # ------------------------------------------------------------------
    # 9. Current scoring — speed-based (no directional bias for SBT)
    # ------------------------------------------------------------------
    cur_file = os.path.join(_output_dir, "currents_raw.nc")
    if os.path.exists(cur_file):
        try:
            cds = xr.open_dataset(cur_file)
            uo_raw = cds["uo"]
            vo_raw = cds["vo"]
            if "depth" in uo_raw.dims and uo_raw.sizes["depth"] > 1:
                uo_raw = uo_raw.isel(depth=0)
                vo_raw = vo_raw.isel(depth=0)
            uo_da = uo_raw.squeeze()
            vo_da = vo_raw.squeeze()
            c_lons = uo_da.longitude.values if "longitude" in uo_da.dims else uo_da.lon.values
            c_lats = uo_da.latitude.values if "latitude" in uo_da.dims else uo_da.lat.values
            uo_data = _interp_to_grid(uo_da.values.astype(float), c_lons, c_lats)
            vo_data = _interp_to_grid(vo_da.values.astype(float), c_lons, c_lats)
            cds.close()
            cur_speed = np.sqrt(uo_data**2 + vo_data**2)

            # SBT: speed-based scoring, no directional bias
            # 0 at <0.03 m/s, 1.0 at >=0.25 m/s
            speed_score = np.clip((cur_speed - 0.03) / 0.22, 0, 1)
            current_score = speed_score
            current_score[land] = np.nan
            _add_score("current", current_score)

            # 10. Convergence — same logic as marlin
            dudx = np.gradient(uo_data, axis=1)
            dvdy = np.gradient(vo_data, axis=0)
            divergence = dudx + dvdy
            conv_score = np.clip(-divergence / 0.005, 0, 1)
            conv_filled = conv_score.copy()
            conv_filled[np.isnan(conv_filled)] = 0
            conv_score = gaussian_filter(conv_filled, sigma=1.0)
            conv_score[land] = np.nan
            # Synergy with current
            synergy = 1.0 + 0.25 * np.clip(current_score, 0, 1)
            conv_score_synergy = np.clip(conv_score * synergy, 0, 1)
            conv_score_synergy[land] = np.nan
            _add_score("convergence", conv_score_synergy)
        except Exception as e:
            print(f"[SBT-Hotspots] Current scoring failed: {e}")

    # ------------------------------------------------------------------
    # 10b. Dissolved oxygen — at depth (95-105m)
    # ------------------------------------------------------------------
    o2_file = os.path.join(_output_dir, "o2_raw.nc")
    if os.path.exists(o2_file):
        try:
            ods = xr.open_dataset(o2_file)
            for ov in ["o2", "O2", "dissolved_oxygen"]:
                if ov in ods:
                    o2_da = ods[ov].squeeze()
                    break
            o_lons = o2_da.longitude.values if "longitude" in o2_da.dims else o2_da.lon.values
            o_lats = o2_da.latitude.values if "latitude" in o2_da.dims else o2_da.lat.values
            o2_data = _interp_to_grid(o2_da.values.astype(float), o_lons, o_lats)
            ods.close()
            # Score: 1.0 at O2>=220 mmol/m3, 0 at O2<=180
            o2_score = np.clip((o2_data - 180) / 40, 0, 1)
            _add_score("o2", o2_score)
        except Exception as e:
            print(f"[SBT-Hotspots] O2 scoring failed: {e}")

    # ------------------------------------------------------------------
    # 11. Feature bands — adapted from marlin with SBT-specific params
    # ------------------------------------------------------------------
    try:
        _grid_step_b = abs(lons[1] - lons[0]) if nx > 1 else 0.083
        band_width_deg = _opt_band_width_nm * 0.0167
        band_width_cells = band_width_deg / _grid_step_b

        _mean_lat = abs(np.mean(lats))
        _cos_lat = np.cos(np.radians(_mean_lat))
        _edt_sampling = [1.0, _cos_lat]
        _upsample = max(1, int(np.ceil(3.0 / max(band_width_cells, 0.1))))

        def _band_score(binary_mask, weight=1.0, width_nm=None):
            if not np.any(binary_mask):
                return np.zeros((ny, nx))
            bwc = (width_nm * 0.0167 / _grid_step_b) if width_nm else band_width_cells
            if _upsample > 1:
                from scipy.ndimage import zoom
                fine_mask = zoom(binary_mask.astype(float), _upsample, order=0) > 0.5
                fine_bwc = bwc * _upsample
                fine_dist = distance_transform_edt(~fine_mask, sampling=_edt_sampling)
                fine_norm = np.clip(fine_dist / fine_bwc, 0, 1)
                fine_band = np.clip(1.0 - fine_norm ** _opt_band_decay, 0, 1)
                band = zoom(fine_band, 1.0 / _upsample, order=1)[:ny, :nx]
            else:
                dist = distance_transform_edt(~binary_mask, sampling=_edt_sampling)
                normalised = np.clip(dist / bwc, 0, 1)
                band = np.clip(1.0 - normalised ** _opt_band_decay, 0, 1)
            band[land] = 0
            return band * weight

        band_layers = {}

        # SST front band
        _thresh = SST_GRADIENT_THRESHOLD
        cross_h = (grad_mag[:, :-1] - _thresh) * (grad_mag[:, 1:] - _thresh) <= 0
        cross_v = (grad_mag[:-1, :] - _thresh) * (grad_mag[1:, :] - _thresh) <= 0
        sst_front_mask = np.zeros((ny, nx), dtype=bool)
        sst_front_mask[:, :-1] |= cross_h
        sst_front_mask[:, 1:] |= cross_h
        sst_front_mask[:-1, :] |= cross_v
        sst_front_mask[1:, :] |= cross_v
        sst_front_mask |= (grad_mag > _thresh)
        sst_front_mask &= ~coast_buf & ~land
        if np.any(sst_front_mask):
            band_layers["sst_front"] = _band_score(sst_front_mask)

        # Isotherm bands — SBT: 17/18/19/20C
        try:
            iso_smooth = gaussian_filter(sst_filled, sigma=0.5)
            for iso_temp in _sbt_isotherms:
                cross_h = (iso_smooth[:, :-1] - iso_temp) * (iso_smooth[:, 1:] - iso_temp) <= 0
                cross_v = (iso_smooth[:-1, :] - iso_temp) * (iso_smooth[1:, :] - iso_temp) <= 0
                cross_mask = np.zeros((ny, nx), dtype=bool)
                cross_mask[:, :-1] |= cross_h
                cross_mask[:, 1:] |= cross_h
                cross_mask[:-1, :] |= cross_v
                cross_mask[1:, :] |= cross_v
                cross_mask &= ~land & ~coast_buf
                if np.any(cross_mask):
                    band_layers[f"isotherm_{iso_temp}C"] = _band_score(cross_mask)
        except Exception:
            pass

        # CHL edge band
        if chl_grid is not None:
            try:
                chl_for_grad = np.log10(np.clip(chl_grid, 0.01, 10))
                chl_for_grad[np.isnan(chl_for_grad) | land] = np.nanmean(chl_for_grad[~land])
                chl_smooth = gaussian_filter(chl_for_grad, sigma=1.5)
                cgx = sobel(chl_smooth, axis=1)
                cgy = sobel(chl_smooth, axis=0)
                chl_grad = np.sqrt(cgx**2 + cgy**2)
                chl_grad[coast_buf] = 0
                cg90 = np.nanpercentile(chl_grad[~coast_buf & ~land], 90)
                chl_edge_mask = (chl_grad / cg90 > _opt_band_chl_thresh) & ~coast_buf & ~land if cg90 > 0 else np.zeros_like(land)
                if np.any(chl_edge_mask):
                    band_layers["chl_edge"] = _band_score(chl_edge_mask)
            except Exception:
                pass

        # CHL contour at 0.25 mg/m3 (SBT bait boundary)
        if chl_grid is not None:
            try:
                chl_for_contour = gaussian_filter(
                    np.where(np.isnan(chl_grid) | land, 0, chl_grid), sigma=0.8)
                cross_h = (chl_for_contour[:, :-1] - _sbt_chl_contour) * (chl_for_contour[:, 1:] - _sbt_chl_contour) <= 0
                cross_v = (chl_for_contour[:-1, :] - _sbt_chl_contour) * (chl_for_contour[1:, :] - _sbt_chl_contour) <= 0
                chl_cross = np.zeros((ny, nx), dtype=bool)
                chl_cross[:, :-1] |= cross_h
                chl_cross[:, 1:] |= cross_h
                chl_cross[:-1, :] |= cross_v
                chl_cross[1:, :] |= cross_v
                chl_cross &= ~land & ~coast_buf
                if np.any(chl_cross):
                    band_layers["chl_025"] = _band_score(chl_cross)
            except Exception:
                pass

        # SLA contour bands — BOTH warm and cold eddy edges (unlike marlin)
        if ssh_grid is not None:
            try:
                ssh_for_grad = ssh_grid.copy()
                ssh_for_grad[np.isnan(ssh_for_grad) | land] = np.nanmean(ssh_grid[~land])
                ssh_smooth_b = gaussian_filter(ssh_for_grad, sigma=1.0)
                eddy_contour = np.zeros((ny, nx), dtype=bool)
                for level in [-0.03, -0.01, 0.03, 0.07, 0.11]:
                    cross_h = (ssh_smooth_b[:, :-1] - level) * (ssh_smooth_b[:, 1:] - level) <= 0
                    cross_v = (ssh_smooth_b[:-1, :] - level) * (ssh_smooth_b[1:, :] - level) <= 0
                    cross = np.zeros((ny, nx), dtype=bool)
                    cross[:, :-1] |= cross_h
                    cross[:, 1:] |= cross_h
                    cross[:-1, :] |= cross_v
                    cross[1:, :] |= cross_v
                    eddy_contour |= cross & ~land & ~coast_buf
                if np.any(eddy_contour):
                    band_layers["eddy_edge"] = _band_score(eddy_contour)
            except Exception:
                pass

        # SSTA edge band — included for SBT (cool anomaly edges)
        try:
            from marlin_data import compute_ssta
            if date_str:
                ssta_for_band = compute_ssta(sst, lats, lons, date_str)
                if ssta_for_band is not None and not np.all(np.isnan(ssta_for_band)):
                    ssta_filled = ssta_for_band.copy()
                    ssta_filled[np.isnan(ssta_filled) | land] = 0
                    ssta_smooth = gaussian_filter(ssta_filled, sigma=1.0)
                    for ssta_level in [-1.0, -0.5, 0.5]:
                        cross_h = (ssta_smooth[:, :-1] - ssta_level) * (ssta_smooth[:, 1:] - ssta_level) <= 0
                        cross_v = (ssta_smooth[:-1, :] - ssta_level) * (ssta_smooth[1:, :] - ssta_level) <= 0
                        cross = np.zeros((ny, nx), dtype=bool)
                        cross[:, :-1] |= cross_h
                        cross[:, 1:] |= cross_h
                        cross[:-1, :] |= cross_v
                        cross[1:, :] |= cross_v
                        cross &= ~land & ~coast_buf
                        if np.any(cross):
                            band_layers[f"ssta_{ssta_level}"] = _band_score(cross)
        except Exception:
            pass

        # Bathymetry contour bands — SBT-weighted
        if tif_path and 'bathy' in dir():
            try:
                depth_master_b = _interp_to_grid(
                    np.where(np.isnan(bathy), 0, -bathy), b_lons, b_lats)
                for depth_m, bw in _bathy_band_weights.items():
                    tol_shore = max(20, depth_m * 0.15)
                    tol_deep = max(30, depth_m * 0.30)
                    contour_mask = ((depth_master_b >= depth_m - tol_shore) &
                                    (depth_master_b <= depth_m + tol_deep) & ~land)
                    if np.any(contour_mask):
                        bathy_band = _band_score(contour_mask, weight=bw)
                        shallow_mask = (depth_master_b < depth_m * 0.50) & ~land
                        bathy_band[shallow_mask] = 0
                        band_layers[f"bathy_{depth_m}m"] = bathy_band
            except Exception as e:
                print(f"[SBT-Hotspots] Bathy contour banding failed: {e}")

        # No FAD bands for SBT

        # Band overlap computation
        if band_layers:
            band_stack = np.array(list(band_layers.values()))
            band_sum = np.sum(band_stack, axis=0)
            band_count = np.sum(band_stack > 0.3, axis=0).astype(float)
            counted_sum = np.sum(np.where(band_stack > 0.3, band_stack, 0), axis=0)
            mean_band = np.where(band_count > 0, counted_sum / np.maximum(band_count, 1), 0)
            _feature_band_count = band_count
            _feature_band_mean = mean_band

            # Floor boost for key feature lines
            _key_feature_floor = 0.58
            key_floor = np.zeros((ny, nx))
            _rel_floor_mask = (grad_mag / g90 > _opt_band_front_thresh) & ~coast_buf & ~land if g90 > 0 else sst_front_mask
            broad_front_mask = sst_front_mask | _rel_floor_mask
            if np.any(broad_front_mask):
                broad_front_band = _band_score(broad_front_mask)
                key_floor = np.maximum(key_floor, np.where(broad_front_band > 0.5, _key_feature_floor, 0))
            if "chl_025" in band_layers:
                key_floor = np.maximum(key_floor, np.where(band_layers["chl_025"] > 0.5, _key_feature_floor, 0))
            _feature_key_floor = key_floor

            n_bands = len(band_layers)
            multi2 = np.sum(band_count[~land] >= 2) / max(np.sum(~land), 1) * 100
            multi3 = np.sum(band_count[~land] >= 3) / max(np.sum(~land), 1) * 100
            print(f"[SBT-Hotspots] Feature bands: {n_bands} types, "
                  f"{multi2:.0f}% in 2+ bands, {multi3:.0f}% in 3+ bands")
        else:
            _feature_band_count = np.zeros((ny, nx))
            _feature_band_mean = np.zeros((ny, nx))
            _feature_key_floor = np.zeros((ny, nx))
    except Exception as e:
        _feature_band_count = np.zeros((ny, nx))
        _feature_band_mean = np.zeros((ny, nx))
        _feature_key_floor = np.zeros((ny, nx))
        print(f"[SBT-Hotspots] Feature banding failed: {e}")

    # ------------------------------------------------------------------
    # Normalize and apply multipliers
    # ------------------------------------------------------------------
    valid = weight_sum > 0
    final = np.full((ny, nx), np.nan)
    final[valid] = score[valid] / weight_sum[valid]
    final[land] = np.nan

    # Depth gate multiplier
    if "depth" in sub_scores:
        depth_mult = sub_scores["depth"]
        dmask = ~np.isnan(depth_mult) & valid
        final[dmask] *= depth_mult[dmask]

    # Shelf break boost
    if "shelf_break" in sub_scores:
        shelf_mult = 1.0 + _opt_shelf_boost * sub_scores["shelf_break"]
        smask = ~np.isnan(shelf_mult) & valid
        final[smask] *= shelf_mult[smask]
        final = np.clip(final, 0, 1)

    # Feature band boost
    if np.any(_feature_band_count > 0):
        _band_overlap_thresh = 2
        no_band_mask = (_feature_band_count < 0.1) & valid & ~land
        final[no_band_mask] *= 0.80
        extra = np.clip(_feature_band_count - _band_overlap_thresh, 0, None)
        band_mult = (1.0
                     + _opt_band_single * _feature_band_count * _feature_band_mean
                     + _opt_band_overlap * extra * _feature_band_mean)
        final[valid] *= band_mult[valid]

    # Sqrt compression + rescale
    raw_max = float(np.nanmax(final[valid & ~land])) if np.any(valid & ~land) else 1.0
    if raw_max > 1.0:
        over = np.clip(final - 1.0, 0, None)
        over = np.sqrt(over)
        final = np.where(final > 1.0, 1.0 + over, final)
        compressed_max = float(np.nanmax(final[valid & ~land]))
        final[valid] = final[valid] / compressed_max
        final = np.clip(final, 0, 1)

    # Floor boost for feature lines
    if np.any(_feature_key_floor > 0):
        final[valid] = np.maximum(final[valid], _feature_key_floor[valid])

    # Spatial smoothing
    _grid_step = abs(lons[1] - lons[0]) if nx > 1 else 0.083
    _smooth_sigma = max(0.6, 0.015 / _grid_step)
    final_filled = final.copy()
    final_filled[np.isnan(final_filled)] = 0
    final_smooth = gaussian_filter(final_filled, sigma=_smooth_sigma)
    final_smooth[land | ~valid] = np.nan

    fmin = float(np.nanmin(final_smooth[~land & valid]))
    fmax = float(np.nanmax(final_smooth[~land & valid]))
    fmean = float(np.nanmean(final_smooth[~land & valid]))
    print(f"[SBT-Hotspots] Score range: {fmin:.3f} - {fmax:.3f} (mean {fmean:.3f})")

    # ------------------------------------------------------------------
    # Build depth mask for clipping
    # ------------------------------------------------------------------
    clip_mask = None
    if tif_path and os.path.exists(tif_path):
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, mapping
            from marlin_data import build_deep_water_mask
            clip_mask = build_deep_water_mask(tif_path, depth_threshold=-30)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Export as contour polygons
    # ------------------------------------------------------------------
    def _sample_scores(coords_list):
        from matplotlib.path import Path
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]
        lon_min, lon_max = min(xs), max(xs)
        lat_min, lat_max = min(ys), max(ys)
        col_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        row_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        if len(col_idx) == 0 or len(row_idx) == 0:
            return 0.0, {}
        sub_lons = lons[col_idx]
        sub_lats = lats[row_idx]
        mesh_lon, mesh_lat = np.meshgrid(sub_lons, sub_lats)
        points = np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()])
        poly_path = Path([(c[0], c[1]) for c in coords_list])
        inside = poly_path.contains_points(points).reshape(mesh_lon.shape)
        region_composite = final_smooth[np.ix_(row_idx, col_idx)]
        valid_composite = region_composite[inside & ~np.isnan(region_composite)]
        actual_intensity = round(float(np.mean(valid_composite)), 2) if len(valid_composite) > 0 else 0.0
        result = {}
        for name, arr in sub_scores.items():
            region = arr[np.ix_(row_idx, col_idx)]
            v = region[inside & ~np.isnan(region)]
            if len(v) > 0:
                w = SBT_WEIGHTS.get(name, 0)
                mean_score = round(float(np.mean(v)), 2)
                if name == "depth":
                    result[name] = {"score": mean_score, "weight": -1}
                elif name == "shelf_break":
                    result[name] = {"score": round(1.0 + _opt_shelf_boost * mean_score, 2), "weight": -2}
                else:
                    result[name] = {"score": mean_score, "weight": w}
        if _depth_grid is not None:
            depth_region = _depth_grid[np.ix_(row_idx, col_idx)]
            depth_valid = depth_region[inside & ~np.isnan(depth_region) & (depth_region > 0)]
            if len(depth_valid) > 0:
                result["depth_m"] = {"score": round(float(np.mean(depth_valid))), "weight": -3}
        bc_region = _feature_band_count[np.ix_(row_idx, col_idx)]
        bc_valid = bc_region[inside & ~np.isnan(bc_region)]
        if len(bc_valid) > 0:
            result["bands"] = {"score": round(float(np.mean(bc_valid)), 1), "weight": -4}
        return actual_intensity, result

    plot_data = final_smooth.copy()
    plot_data[np.isnan(plot_data)] = 0
    levels = [0] + HOTSPOT_BANDS + [1.0]
    fig, ax = plt.subplots()
    cf = ax.contourf(lons, lats, plot_data, levels=levels, extend="neither")
    plt.close(fig)

    features = []
    for band_idx, seg_list in enumerate(cf.allsegs):
        if band_idx == 0:
            continue
        intensity = round((levels[band_idx] + levels[band_idx + 1]) / 2, 2) if band_idx + 1 < len(levels) else 1.0
        band_label = f"{levels[band_idx]:.0%}-{levels[band_idx+1]:.0%}" if band_idx + 1 < len(levels) else f">{levels[band_idx]:.0%}"
        for seg in seg_list:
            if len(seg) < 4:
                continue
            coords = [[round(float(x), 4), round(float(y), 4)] for x, y in seg]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            actual_intensity, breakdown = _sample_scores(coords)
            props = {
                "species": "sbt",
                "type": "hotspot",
                "intensity": actual_intensity,
                "band": band_label,
            }
            for name, info in breakdown.items():
                props[f"s_{name}"] = info["score"]
                props[f"w_{name}"] = info["weight"]
            if clip_mask is not None:
                try:
                    poly = ShapelyPolygon([(c[0], c[1]) for c in coords]).buffer(0)
                    clipped = poly.intersection(clip_mask)
                    if clipped.is_empty:
                        continue
                    geom = mapping(clipped)
                    if geom["type"] == "Polygon":
                        features.append({"type": "Feature", "geometry": geom, "properties": props})
                    elif geom["type"] == "MultiPolygon":
                        for mc in geom["coordinates"]:
                            features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": mc}, "properties": props})
                    continue
                except Exception:
                    pass
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": props,
            })

    geojson = {"type": "FeatureCollection", "features": features}
    output_path = os.path.join(_output_dir, "sbt_hotspots.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[SBT-Hotspots] {len(features)} polygons across {len(HOTSPOT_BANDS)} bands "
          f"-> {output_path}")

    return {
        "path": output_path,
        "grid": final_smooth,
        "lats": lats,
        "lons": lons,
        "sub_scores": sub_scores,
        "weights": {k: v for k, v in SBT_WEIGHTS.items()},
        "band_count": _feature_band_count,
        "band_mean": _feature_band_mean,
    }
