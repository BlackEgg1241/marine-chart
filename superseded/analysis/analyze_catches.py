"""Comprehensive feature analysis of all blue marlin catch locations."""
import marlin_data, numpy as np, os, csv, json, xarray as xr
from datetime import datetime
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Load catches
with open('data/all_catches.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if 'BLUE MARLIN' in r.get('species', '').upper()]

catches = []
for r in rows:
    lat = r.get('lat', '').strip()
    lon = r.get('lon', '').strip()
    if not lat or not lon:
        continue
    d = r['date'].strip()
    for fmt in ['%d/%m/%Y', '%Y-%m-%d']:
        try:
            dt = datetime.strptime(d, fmt)
            date_str = dt.strftime('%Y-%m-%d')
            break
        except:
            continue
    else:
        continue
    catches.append({'date': date_str, 'lat': float(lat), 'lon': float(lon),
                    'tag': r.get('tag', ''), 'type': r.get('type', '')})

by_date = defaultdict(list)
for c in catches:
    by_date[c['date']].append(c)

bbox = [114.5, -32.5, 115.6, -31.5]
results = []

for date in sorted(by_date.keys()):
    ddir = os.path.join('data', date)
    if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
        ddir = os.path.join('data', 'prediction', date)
    if not os.path.exists(os.path.join(ddir, 'sst_raw.nc')):
        continue

    tif = os.path.join(ddir, 'bathy_gmrt.tif')
    marlin_data.OUTPUT_DIR = ddir
    result = marlin_data.generate_blue_marlin_hotspots(
        bbox, tif_path=tif if os.path.exists(tif) else None, date_str=date)
    grid = result['grid']
    lats, lons = result['lats'], result['lons']
    bc = result['band_count']
    bm = result['band_mean']
    ss = result['sub_scores']

    # Load raw SST
    sst_raw = sst_lats = sst_lons = None
    try:
        ds_sst = xr.open_dataset(os.path.join(ddir, 'sst_raw.nc'))
        for var in ['thetao', 'analysed_sst', 'sst']:
            if var in ds_sst:
                sst_raw = marlin_data._kelvin_to_celsius(ds_sst[var].squeeze().values.astype(float))
                sst_lats = ds_sst.latitude.values if 'latitude' in ds_sst.dims else ds_sst.lat.values
                sst_lons = ds_sst.longitude.values if 'longitude' in ds_sst.dims else ds_sst.lon.values
                break
        ds_sst.close()
    except Exception:
        pass

    # Load raw currents
    uo = vo = cur_lats = cur_lons = None
    try:
        ds_cur = xr.open_dataset(os.path.join(ddir, 'currents_raw.nc'))
        uo = ds_cur['uo'].squeeze().values
        vo = ds_cur['vo'].squeeze().values
        cur_lats = ds_cur.latitude.values if 'latitude' in ds_cur.dims else ds_cur.lat.values
        cur_lons = ds_cur.longitude.values if 'longitude' in ds_cur.dims else ds_cur.lon.values
        ds_cur.close()
    except Exception:
        pass

    # Load raw CHL
    chl_val_raw = None
    try:
        ds_chl = xr.open_dataset(os.path.join(ddir, 'chl_raw.nc'))
        for var in ['CHL', 'chl', 'chlor_a']:
            if var in ds_chl:
                chl_data = ds_chl[var].squeeze().values.astype(float)
                chl_lats = ds_chl.latitude.values if 'latitude' in ds_chl.dims else ds_chl.lat.values
                chl_lons = ds_chl.longitude.values if 'longitude' in ds_chl.dims else ds_chl.lon.values
                break
        ds_chl.close()
    except Exception:
        chl_data = None

    # Load raw MLD
    mld_val_raw = None
    try:
        ds_mld = xr.open_dataset(os.path.join(ddir, 'mld_raw.nc'))
        for var in ['mlotst', 'mld', 'MLD']:
            if var in ds_mld:
                mld_data = ds_mld[var].squeeze().values.astype(float)
                mld_lats = ds_mld.latitude.values if 'latitude' in ds_mld.dims else ds_mld.lat.values
                mld_lons = ds_mld.longitude.values if 'longitude' in ds_mld.dims else ds_mld.lon.values
                break
        ds_mld.close()
    except Exception:
        mld_data = None

    # Load raw SSH
    ssh_val_raw = None
    try:
        ds_ssh = xr.open_dataset(os.path.join(ddir, 'ssh_raw.nc'))
        for var in ['zos', 'sla', 'adt']:
            if var in ds_ssh:
                ssh_data = ds_ssh[var].squeeze().values.astype(float)
                ssh_lats = ds_ssh.latitude.values if 'latitude' in ds_ssh.dims else ds_ssh.lat.values
                ssh_lons = ds_ssh.longitude.values if 'longitude' in ds_ssh.dims else ds_ssh.lon.values
                break
        ds_ssh.close()
    except Exception:
        ssh_data = None

    for c in by_date[date]:
        li = np.argmin(np.abs(lats - c['lat']))
        lo = np.argmin(np.abs(lons - c['lon']))
        score = float(grid[li, lo]) if not np.isnan(grid[li, lo]) else 0
        bands = float(bc[li, lo]) if not np.isnan(bc[li, lo]) else 0
        bmean = float(bm[li, lo]) if not np.isnan(bm[li, lo]) else 0

        row = {
            'date': date, 'tag': c['tag'], 'type': c['type'],
            'lat': c['lat'], 'lon': c['lon'],
            'score': round(score, 3), 'bands': int(bands), 'band_mean': round(bmean, 3),
        }

        # Sub-scores
        for name in ['sst', 'sst_front', 'front_corridor', 'chl', 'chl_curvature',
                      'ssh', 'mld', 'current', 'convergence', 'sst_intrusion',
                      'shelf_break', 'depth']:
            if name in ss and isinstance(ss[name], np.ndarray):
                val = ss[name][li, lo]
                row[f'{name}_score'] = round(float(val), 3) if not np.isnan(val) else ''

        # Raw SST + gradient
        if sst_raw is not None:
            sli = np.argmin(np.abs(sst_lats - c['lat']))
            slo = np.argmin(np.abs(sst_lons - c['lon']))
            row['sst_C'] = round(float(sst_raw[sli, slo]), 2)
            grad_y, grad_x = np.gradient(sst_raw)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            row['sst_gradient'] = round(float(grad_mag[sli, slo]), 4)

        # Raw CHL
        if chl_data is not None:
            cli = np.argmin(np.abs(chl_lats - c['lat']))
            clo = np.argmin(np.abs(chl_lons - c['lon']))
            val = chl_data[cli, clo]
            row['chl_mgm3'] = round(float(val), 4) if not np.isnan(val) else ''

        # Raw MLD
        if mld_data is not None:
            mli = np.argmin(np.abs(mld_lats - c['lat']))
            mlo = np.argmin(np.abs(mld_lons - c['lon']))
            val = mld_data[mli, mlo]
            row['mld_m'] = round(float(val), 1) if not np.isnan(val) else ''

        # Raw SSH (SLA)
        if ssh_data is not None:
            hli = np.argmin(np.abs(ssh_lats - c['lat']))
            hlo = np.argmin(np.abs(ssh_lons - c['lon']))
            val = ssh_data[hli, hlo]
            row['ssh_m'] = round(float(val), 4) if not np.isnan(val) else ''

        # Raw currents + divergence
        if uo is not None:
            cli = np.argmin(np.abs(cur_lats - c['lat']))
            clo = np.argmin(np.abs(cur_lons - c['lon']))
            u_val = float(uo[cli, clo])
            v_val = float(vo[cli, clo])
            speed = np.sqrt(u_val**2 + v_val**2)
            direction = np.degrees(np.arctan2(u_val, v_val)) % 360
            row['current_speed_ms'] = round(speed, 4)
            row['current_dir_deg'] = round(direction, 1)
            dudx = np.gradient(uo, axis=1)
            dvdy = np.gradient(vo, axis=0)
            div_val = dudx[cli, clo] + dvdy[cli, clo]
            row['divergence'] = round(float(div_val), 5)

        # Depth from bathy
        try:
            import rasterio
            if os.path.exists(tif):
                with rasterio.open(tif) as src:
                    py, px = src.index(c['lon'], c['lat'])
                    if 0 <= py < src.height and 0 <= px < src.width:
                        d = src.read(1)[py, px]
                        row['depth_m'] = round(float(-d)) if d < 0 else ''
        except Exception:
            pass

        # Nearest zone score (max within 3nm/0.05deg)
        r1, r2 = max(0, li - 3), min(grid.shape[0], li + 4)
        c1, c2 = max(0, lo - 3), min(grid.shape[1], lo + 4)
        nearby = grid[r1:r2, c1:c2]
        row['nearest_zone_score'] = round(float(np.nanmax(nearby)), 3) if np.any(~np.isnan(nearby)) else ''

        results.append(row)

    print(f'{date} done', flush=True)

# Write CSV
out_path = 'data/catch_feature_analysis.csv'
fields = ['date', 'tag', 'type', 'lat', 'lon', 'score', 'bands', 'band_mean',
          'sst_C', 'sst_score', 'sst_gradient', 'sst_front_score', 'front_corridor_score',
          'chl_mgm3', 'chl_score', 'chl_curvature_score',
          'ssh_m', 'ssh_score', 'mld_m', 'mld_score',
          'current_speed_ms', 'current_dir_deg', 'current_score',
          'convergence_score', 'divergence', 'sst_intrusion_score',
          'depth_m', 'depth_score', 'shelf_break_score',
          'nearest_zone_score']

with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)

print(f'\nWrote {len(results)} rows to {out_path}')

# Summary stats
print(f'\n{"="*60}')
print(f'BLUE MARLIN CATCH FEATURE ANALYSIS ({len(results)} catches)')
print(f'{"="*60}')
for field in fields[5:]:
    vals = [float(r[field]) for r in results if r.get(field) and r[field] != '']
    if vals:
        print(f'{field:25s}: mean={np.mean(vals):8.3f}  med={np.median(vals):8.3f}  '
              f'min={np.min(vals):8.3f}  max={np.max(vals):8.3f}  '
              f'std={np.std(vals):7.3f}')
