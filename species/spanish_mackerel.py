"""
Spanish Mackerel (Scomberomorus commerson) habitat scoring.

Methodology differs significantly from blue marlin:
- Reef-structure dependent: scoring centered on reef/hard bottom features
- Shallower depth zone: 10-80m (peak 20-50m near reef edges)
- SST preference: 22-28C (broader warm-water tolerance)
- Current-driven: prey concentrates in current breaks around structure
- Chlorophyll: higher productivity preferred (bait aggregation)
- No deep-water canyon features or FAD boosting

Requires high-resolution bathymetry (<=50m) for reef structure detection.
Falls back to GMRT (~200m) but results will lack reef detail.
"""

import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Scoring weights — reef-structure & inshore-focused
# ---------------------------------------------------------------------------
SPANISH_MACKEREL_WEIGHTS = {
    "sst":           0.10,  # SST — uniformly high in summer, low discrimination
    "reef_structure": 0.00, # Applied as MULTIPLIER, not additive (key gate)
    "depth":         0.00,  # Applied as gate/multiplier, not additive
    "chl":           0.20,  # Chlorophyll — bait productivity, moderate discrimination
    "current":       0.15,  # Current speed — prey aggregation in flow breaks
    "ssh":           0.10,  # Sea level anomaly — eddy influence on inshore
    "sst_front":     0.12,  # SST gradient — bait aggregation at fronts
    "mld":           0.05,  # Mixed layer depth
    "current_shear": 0.15,  # Current shear — bait traps at flow boundaries
    "clarity":       0.03,  # Water clarity — uniformly clear in summer, minimal discrimination
}

# Intensity bands for polygon export — higher floor, tighter bands for selective zones
HOTSPOT_BANDS = [0.40, 0.50, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95]

# Optimal parameters
_opt_sst_optimal = 25.0    # Peak SST preference
_opt_sst_sigma = 3.0       # Broad tolerance
_opt_sst_sigma_above = 3.5 # Slightly asymmetric
_opt_chl_optimal = 0.35    # Higher CHL than marlin (more productive water)
_opt_chl_sigma = 0.30      # In log10 space
_opt_mld_optimal = 25.0    # Shallower MLD preferred
_opt_mld_sigma = 15.0
_opt_clarity_optimal = 0.05  # Clearer water preferred (low KD490)
_opt_clarity_sigma = 0.03


def generate_spanish_mackerel_hotspots(bbox, tif_path=None, date_str=None,
                                       output_dir=None, hires_bathy=None):
    """
    Build habitat suitability grid for Spanish Mackerel.

    Parameters
    ----------
    bbox : dict with lon_min, lon_max, lat_min, lat_max
    tif_path : str, path to bathymetry GeoTIFF
    date_str : str, YYYY-MM-DD
    output_dir : str, override output directory
    hires_bathy : str, path to high-resolution bathymetry GeoTIFF for reef detection

    Returns
    -------
    dict with path, grid, lats, lons, sub_scores, weights, band_count, band_mean
    """
    import xarray as xr
    from scipy.ndimage import gaussian_filter, sobel, convolve, distance_transform_edt
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use marlin_data's OUTPUT_DIR if not specified
    if output_dir is None:
        import marlin_data
        _output_dir = marlin_data.OUTPUT_DIR
    else:
        _output_dir = output_dir

    print("[SM-Hotspots] Building Spanish Mackerel habitat suitability...")

    # --- Load SST as master grid ---
    sst_file = os.path.join(_output_dir, "sst_raw.nc")
    if not os.path.exists(sst_file):
        print("[SM-Hotspots] No SST data - skipping")
        return None
    ds = xr.open_dataset(sst_file)
    for var in ["thetao", "analysed_sst", "sst"]:
        if var in ds:
            sst_da = ds[var].squeeze()
            break
    else:
        print("[SM-Hotspots] No SST variable found")
        return None

    lons = sst_da.longitude.values if "longitude" in sst_da.dims else sst_da.lon.values
    lats = sst_da.latitude.values if "latitude" in sst_da.dims else sst_da.lat.values

    from marlin_data import _kelvin_to_celsius
    sst = _kelvin_to_celsius(sst_da.values.copy().astype(float))
    ds.close()

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

    def _add_score(name, values, mask=None):
        w = SPANISH_MACKEREL_WEIGHTS.get(name, 0)
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
    # 1. SST score — broad Gaussian centered at 25C
    # ------------------------------------------------------------------
    sst_filled = sst.copy()
    sst_filled[land] = np.nanmean(sst)
    sst_smooth = gaussian_filter(sst_filled, sigma=0.5)
    sst_smooth[land] = np.nan
    sigma_map = np.where(sst_smooth < _opt_sst_optimal, _opt_sst_sigma, _opt_sst_sigma_above)
    sst_score = np.exp(-0.5 * ((sst_smooth - _opt_sst_optimal) / sigma_map) ** 2)
    _add_score("sst", sst_score)

    # ------------------------------------------------------------------
    # 2. SST front score
    # ------------------------------------------------------------------
    sst_for_grad = sst_filled.copy()
    sst_grad = gaussian_filter(sst_for_grad, sigma=1.5)
    gx = sobel(sst_grad, axis=1)
    gy = sobel(sst_grad, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    from scipy.ndimage import binary_dilation
    coast_buf = binary_dilation(land, iterations=2)
    grad_mag[coast_buf] = 0
    ocean_grad = grad_mag[~coast_buf & ~land]
    g90 = np.nanpercentile(ocean_grad, 90) if len(ocean_grad) > 0 else 0
    if g90 > 0:
        front_score = np.clip(grad_mag / g90, 0, 1)
    else:
        front_score = np.zeros_like(grad_mag)
    front_score = front_score * sst_score
    front_score[land] = np.nan
    _add_score("sst_front", front_score)

    # ------------------------------------------------------------------
    # 3. Chlorophyll — higher productivity preferred for Spanish Mackerel
    # ------------------------------------------------------------------
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
            cds.close()
            chl_log = np.log10(np.clip(chl_data, 0.01, 10))
            optimal_chl = np.log10(_opt_chl_optimal)
            chl_score = np.exp(-0.5 * ((chl_log - optimal_chl) / _opt_chl_sigma) ** 2)
            _add_score("chl", chl_score)
        except Exception as e:
            print(f"[SM-Hotspots] CHL scoring failed: {e}")

    # ------------------------------------------------------------------
    # 4. Depth gating + Reef structure scoring
    # ------------------------------------------------------------------
    _depth_grid = None
    # Prefer high-res bathy for reef detection; auto-detect if not specified
    # Priority: bathy_combined (LiDAR+GMRT) > bathy_hires (GMRT high) > tif_path
    if hires_bathy is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for candidate_name in ["bathy_combined.tif", "bathy_hires.tif"]:
            hires_candidate = os.path.join(script_dir, "data", candidate_name)
            if os.path.exists(hires_candidate):
                hires_bathy = hires_candidate
                break
    bathy_tif = hires_bathy or tif_path
    if bathy_tif and os.path.exists(bathy_tif):
        try:
            import rasterio
            with rasterio.open(bathy_tif) as src:
                bathy = src.read(1).astype(float)
                bt = src.transform
                bw, bh = bathy.shape[1], bathy.shape[0]
                b_lons = np.array([bt.c + (j + 0.5) * bt.a for j in range(bw)])
                b_lats = np.array([bt.f + (i + 0.5) * bt.e for i in range(bh)])
                nd = src.nodata
                if nd is not None:
                    bathy[bathy == nd] = np.nan

            abs_depth = np.where(np.isnan(bathy), 0, -bathy)

            # --- Depth gate: Spanish Mackerel zone 10-80m ---
            # Ramp: 5-10m (0.3->1.0), full 10-50m, taper 50-100m, zero >120m
            depth_score_hr = np.where(abs_depth < 5, 0,
                             np.where(abs_depth < 10, 0.3 + 0.7 * (abs_depth - 5) / 5,
                             np.where(abs_depth < 50, 1.0,
                             np.where(abs_depth < 100, 0.5 + 0.5 * (1.0 - (abs_depth - 50) / 50),
                             np.where(abs_depth < 120, 0.2,
                             0.0)))))

            # Max-pool depth score to master grid
            depth_master = np.zeros((ny, nx))
            depth_val_master = np.zeros((ny, nx))
            dlat = abs(lats[1] - lats[0]) / 2 if ny > 1 else 0.04
            dlon = abs(lons[1] - lons[0]) / 2 if nx > 1 else 0.04
            for yi in range(ny):
                for xi in range(nx):
                    b_row = (b_lats >= lats[yi] - dlat) & (b_lats <= lats[yi] + dlat)
                    b_col = (b_lons >= lons[xi] - dlon) & (b_lons <= lons[xi] + dlon)
                    if np.any(b_row) and np.any(b_col):
                        depth_master[yi, xi] = np.max(depth_score_hr[np.ix_(b_row, b_col)])
                        depth_val_master[yi, xi] = np.mean(abs_depth[np.ix_(b_row, b_col)])
                    else:
                        byi = np.argmin(np.abs(b_lats - lats[yi]))
                        bxi = np.argmin(np.abs(b_lons - lons[xi]))
                        depth_master[yi, xi] = depth_score_hr[byi, bxi]
                        depth_val_master[yi, xi] = abs_depth[byi, bxi]
            depth_master[land] = np.nan
            sub_scores["depth"] = depth_master.copy()
            _depth_grid = depth_val_master

            # --- Reef structure: bathymetric roughness ---
            # High roughness = reef/hard bottom features
            # Use standard deviation of depth in a local window as proxy
            # for structural complexity (rugosity)
            bathy_filled = bathy.copy()
            bathy_filled[np.isnan(bathy_filled)] = 0

            # Compute roughness at high-res then interpolate
            # Roughness = local std dev of elevation (captures reef pinnacles, ledges)
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(bathy_filled, size=5)
            local_sq_mean = uniform_filter(bathy_filled**2, size=5)
            roughness = np.sqrt(np.clip(local_sq_mean - local_mean**2, 0, None))
            roughness[np.isnan(bathy)] = 0

            # Also compute slope (Sobel gradient magnitude)
            dgx = sobel(bathy_filled, axis=1)
            dgy = sobel(bathy_filled, axis=0)
            slope = np.sqrt(dgx**2 + dgy**2)
            slope[np.isnan(bathy)] = 0

            # Combined reef structure = roughness + slope, within SM depth zone
            # Only count structure in 5-120m depth range
            sm_depth_mask = (abs_depth >= 5) & (abs_depth <= 120)
            roughness[~sm_depth_mask] = 0
            slope[~sm_depth_mask] = 0

            # Interpolate to master grid
            rough_master = _interp_to_grid(roughness, b_lons, b_lats)
            slope_master = _interp_to_grid(slope, b_lons, b_lats)

            # Normalize by 90th percentile
            r_ocean = rough_master[~land & ~np.isnan(rough_master)]
            r90 = np.nanpercentile(r_ocean, 90) if len(r_ocean) > 0 else 1
            if r90 > 0:
                reef_rough = np.clip(rough_master / r90, 0, 1)
            else:
                reef_rough = np.zeros((ny, nx))

            s_ocean = slope_master[~land & ~np.isnan(slope_master)]
            s90 = np.nanpercentile(s_ocean, 90) if len(s_ocean) > 0 else 1
            if s90 > 0:
                reef_slope = np.clip(slope_master / s90, 0, 1)
            else:
                reef_slope = np.zeros((ny, nx))

            # Reef structure = 60% roughness + 40% slope
            reef_score = 0.6 * reef_rough + 0.4 * reef_slope
            reef_score[land] = np.nan
            # Store directly — weight=0 means _add_score skips it,
            # but we need it in sub_scores for the reef multiplier gate
            sub_scores["reef_structure"] = np.clip(reef_score, 0, 1).copy()

            reef_pct = np.sum(reef_score[~land & ~np.isnan(reef_score)] > 0.3) / max(np.sum(~land), 1) * 100
            bathy_res = abs(b_lons[1] - b_lons[0]) * 111000 if len(b_lons) > 1 else 0
            print(f"[SM-Hotspots] Reef structure: {reef_pct:.0f}% of cells >30% "
                  f"(bathy resolution ~{bathy_res:.0f}m)")

        except Exception as e:
            print(f"[SM-Hotspots] Depth/reef scoring failed: {e}")
            _depth_grid = None

    # ------------------------------------------------------------------
    # 5. SSH score — eddy influence on inshore
    # ------------------------------------------------------------------
    ssh_file = os.path.join(_output_dir, "ssh_raw.nc")
    if os.path.exists(ssh_file):
        try:
            sds = xr.open_dataset(ssh_file)
            for sv in ["zos", "sla", "adt"]:
                if sv in sds:
                    sla_da = sds[sv].squeeze()
                    break
            sla_lons = sla_da.longitude.values if "longitude" in sla_da.dims else sla_da.lon.values
            sla_lats = sla_da.latitude.values if "latitude" in sla_da.dims else sla_da.lat.values
            sla_data = _interp_to_grid(sla_da.values.astype(float), sla_lons, sla_lats)
            sds.close()
            # Positive SLA = warm water intrusion, generally favorable
            ssh_score = np.clip(np.abs(sla_data) / 0.12, 0, 1)
            ssh_score[land] = np.nan
            _add_score("ssh", ssh_score)
        except Exception as e:
            print(f"[SM-Hotspots] SSH scoring failed: {e}")

    # ------------------------------------------------------------------
    # 6. Current score — speed favorable for prey aggregation
    # ------------------------------------------------------------------
    curr_file = os.path.join(_output_dir, "current_raw.nc")
    if os.path.exists(curr_file):
        try:
            cds = xr.open_dataset(curr_file)
            uo = None
            vo = None
            for uv in ["uo"]:
                if uv in cds:
                    uo_da = cds[uv].squeeze()
                    if "depth" in uo_da.dims:
                        uo_da = uo_da.isel(depth=0)
                    uo_lons = uo_da.longitude.values if "longitude" in uo_da.dims else uo_da.lon.values
                    uo_lats = uo_da.latitude.values if "latitude" in uo_da.dims else uo_da.lat.values
                    uo = _interp_to_grid(uo_da.values.astype(float), uo_lons, uo_lats)
            for vv in ["vo"]:
                if vv in cds:
                    vo_da = cds[vv].squeeze()
                    if "depth" in vo_da.dims:
                        vo_da = vo_da.isel(depth=0)
                    vo = _interp_to_grid(vo_da.values.astype(float), uo_lons, uo_lats)
            cds.close()

            if uo is not None and vo is not None:
                speed = np.sqrt(uo**2 + vo**2)
                # Moderate current (0.1-0.4 m/s) is ideal — too fast or calm = poor
                # Gaussian centered at 0.25 m/s
                current_score = np.exp(-0.5 * ((speed - 0.25) / 0.15) ** 2)
                current_score[land] = np.nan
                _add_score("current", current_score)

                # Current shear — velocity gradient indicates flow breaks
                # where bait concentrates around reef structure
                speed_filled = speed.copy()
                speed_filled[np.isnan(speed_filled)] = 0
                sx = sobel(speed_filled, axis=1)
                sy = sobel(speed_filled, axis=0)
                shear = np.sqrt(sx**2 + sy**2)
                shear[coast_buf] = 0
                sh90 = np.nanpercentile(shear[~coast_buf & ~land], 90)
                if sh90 > 0:
                    shear_score = np.clip(shear / sh90, 0, 1)
                else:
                    shear_score = np.zeros((ny, nx))
                shear_score[land] = np.nan
                _add_score("current_shear", shear_score)
        except Exception as e:
            print(f"[SM-Hotspots] Current scoring failed: {e}")

    # ------------------------------------------------------------------
    # 7. MLD score — shallower preferred
    # ------------------------------------------------------------------
    mld_file = os.path.join(_output_dir, "mld_raw.nc")
    if os.path.exists(mld_file):
        try:
            mds = xr.open_dataset(mld_file)
            for mv in ["mlotst", "mld"]:
                if mv in mds:
                    mld_da = mds[mv].squeeze()
                    break
            mld_lons = mld_da.longitude.values if "longitude" in mld_da.dims else mld_da.lon.values
            mld_lats = mld_da.latitude.values if "latitude" in mld_da.dims else mld_da.lat.values
            mld_data = _interp_to_grid(mld_da.values.astype(float), mld_lons, mld_lats)
            mds.close()
            mld_score = np.exp(-0.5 * ((mld_data - _opt_mld_optimal) / _opt_mld_sigma) ** 2)
            mld_score[land] = np.nan
            _add_score("mld", mld_score)
        except Exception as e:
            print(f"[SM-Hotspots] MLD scoring failed: {e}")

    # ------------------------------------------------------------------
    # 8. Water clarity — visual predator, clearer = better
    # ------------------------------------------------------------------
    kd_file = os.path.join(_output_dir, "kd490_raw.nc")
    if os.path.exists(kd_file):
        try:
            kds = xr.open_dataset(kd_file)
            for kv in ["KD490", "kd490", "kd_490"]:
                if kv in kds:
                    kd_da = kds[kv].squeeze()
                    break
            kd_lons = kd_da.longitude.values if "longitude" in kd_da.dims else kd_da.lon.values
            kd_lats = kd_da.latitude.values if "latitude" in kd_da.dims else kd_da.lat.values
            kd_data = _interp_to_grid(kd_da.values.astype(float), kd_lons, kd_lats)
            kds.close()
            # Lower KD490 = clearer water = better
            clarity_score = np.exp(-0.5 * ((kd_data - _opt_clarity_optimal) / _opt_clarity_sigma) ** 2)
            clarity_score[land] = np.nan
            _add_score("clarity", clarity_score)
        except Exception as e:
            print(f"[SM-Hotspots] Clarity scoring failed: {e}")

    # ------------------------------------------------------------------
    # Normalize and apply depth gate
    # ------------------------------------------------------------------
    valid = weight_sum > 0
    final = np.full((ny, nx), np.nan)
    final[valid] = score[valid] / weight_sum[valid]
    final[land] = np.nan

    # Apply depth gate as multiplier
    if "depth" in sub_scores:
        depth_mult = sub_scores["depth"]
        dmask = ~np.isnan(depth_mult) & valid
        final[dmask] *= depth_mult[dmask]

    # Apply reef structure as multiplier — THE key SM discriminator.
    # No reef = no Spanish Mackerel zone. Cells with zero structure
    # are fully suppressed; weak structure scales down proportionally.
    if "reef_structure" in sub_scores:
        reef = sub_scores["reef_structure"]
        rmask = ~np.isnan(reef) & valid
        # Reef multiplier: 0 reef -> 0.05 (nearly invisible),
        # 0.3 reef -> 0.35, 0.5 reef -> 0.55, 1.0 reef -> 1.05
        reef_mult = 0.05 + reef * 1.0
        final[rmask] *= reef_mult[rmask]
        # Clip back to [0, 1]
        final = np.clip(final, 0, 1)
        reef_pct = np.sum(reef[rmask] > 0.3) / max(np.sum(rmask), 1) * 100
        print(f"[SM-Hotspots] Reef gate: {reef_pct:.0f}% of cells have structure >30%")

    # Spatial smoothing
    _grid_step = abs(lons[1] - lons[0]) if nx > 1 else 0.083
    _smooth_sigma = max(0.6, 0.015 / _grid_step)
    final_filled = final.copy()
    final_filled[np.isnan(final_filled)] = 0
    final_smooth = gaussian_filter(final_filled, sigma=_smooth_sigma)
    final_smooth[land | ~valid] = np.nan

    fmin = float(np.nanmin(final_smooth[~land & valid])) if np.any(~land & valid) else 0
    fmax = float(np.nanmax(final_smooth[~land & valid])) if np.any(~land & valid) else 0
    fmean = float(np.nanmean(final_smooth[~land & valid])) if np.any(~land & valid) else 0
    print(f"[SM-Hotspots] Score range: {fmin:.3f} - {fmax:.3f} (mean {fmean:.3f})")

    # --- No spatial clipping for SM ---
    # The depth gate in scoring already zeros cells deeper than 120m,
    # so polygon export naturally covers only the inshore zone.
    clip_mask = None

    # --- Export as filled contour polygons ---
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

            # Sample actual intensity for this polygon
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            col_idx = np.where((lons >= min(xs)) & (lons <= max(xs)))[0]
            row_idx = np.where((lats >= min(ys)) & (lats <= max(ys)))[0]
            actual_intensity = intensity
            if len(col_idx) > 0 and len(row_idx) > 0:
                region = final_smooth[np.ix_(row_idx, col_idx)]
                v = region[~np.isnan(region)]
                if len(v) > 0:
                    actual_intensity = round(float(np.mean(v)), 2)

            props = {
                "species": "spanish_mackerel",
                "type": "hotspot",
                "intensity": actual_intensity,
                "band": band_label,
            }
            # Add sub-score breakdown
            for name, arr in sub_scores.items():
                region = arr[np.ix_(row_idx, col_idx)] if len(col_idx) > 0 and len(row_idx) > 0 else np.array([])
                v = region[~np.isnan(region)] if region.size > 0 else np.array([])
                if len(v) > 0:
                    w = SPANISH_MACKEREL_WEIGHTS.get(name, 0)
                    props[f"s_{name}"] = round(float(np.mean(v)), 2)
                    props[f"w_{name}"] = w

            if clip_mask is not None:
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon, mapping
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
    output_path = os.path.join(_output_dir, "spanish_mackerel_hotspots.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"[SM-Hotspots] {len(features)} polygons across {len(HOTSPOT_BANDS)} bands -> {output_path}")

    return {
        "path": output_path,
        "grid": final_smooth,
        "lats": lats,
        "lons": lons,
        "sub_scores": sub_scores,
        "weights": {k: v for k, v in SPANISH_MACKEREL_WEIGHTS.items()},
        "band_count": np.zeros((ny, nx)),  # No band system for SM yet
        "band_mean": np.zeros((ny, nx)),
    }
