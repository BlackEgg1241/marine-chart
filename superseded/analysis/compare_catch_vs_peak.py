"""Compare environmental values at catch locations vs nearby zone peaks.

For each catch, find the highest-scoring cell within 5nm and compare
raw environmental values (SST, CHL, MLD, SSH, current) at both points.
This reveals whether zone peaks are driven by genuine environmental
suitability or by feature-band geometry inflation.
"""
import marlin_data, numpy as np, os, csv, xarray as xr
from datetime import datetime
from collections import defaultdict

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
NM_PER_DEG = 60.0
SEARCH_RADIUS_NM = 5.0

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
    ss = result['sub_scores']
    bc = result['band_count']
    bm = result['band_mean']

    cos_lat = np.cos(np.radians(abs(np.mean(lats))))

    # Load raw SST
    sst_raw = sst_lats = sst_lons = None
    try:
        ds = xr.open_dataset(os.path.join(ddir, 'sst_raw.nc'))
        for var in ['thetao', 'analysed_sst', 'sst']:
            if var in ds:
                sst_raw = marlin_data._kelvin_to_celsius(ds[var].squeeze().values.astype(float))
                sst_lats = ds.latitude.values if 'latitude' in ds.dims else ds.lat.values
                sst_lons = ds.longitude.values if 'longitude' in ds.dims else ds.lon.values
                break
        ds.close()
    except Exception:
        pass

    # Load raw CHL
    chl_data = chl_lats = chl_lons = None
    try:
        ds = xr.open_dataset(os.path.join(ddir, 'chl_raw.nc'))
        for var in ['CHL', 'chl', 'chlor_a']:
            if var in ds:
                chl_data = ds[var].squeeze().values.astype(float)
                chl_lats = ds.latitude.values if 'latitude' in ds.dims else ds.lat.values
                chl_lons = ds.longitude.values if 'longitude' in ds.dims else ds.lon.values
                break
        ds.close()
    except Exception:
        pass

    # Load raw MLD
    mld_data = mld_lats = mld_lons = None
    try:
        ds = xr.open_dataset(os.path.join(ddir, 'mld_raw.nc'))
        for var in ['mlotst', 'mld', 'MLD']:
            if var in ds:
                mld_data = ds[var].squeeze().values.astype(float)
                mld_lats = ds.latitude.values if 'latitude' in ds.dims else ds.lat.values
                mld_lons = ds.longitude.values if 'longitude' in ds.dims else ds.lon.values
                break
        ds.close()
    except Exception:
        pass

    # Load raw SSH
    ssh_data = ssh_lats = ssh_lons = None
    try:
        ds = xr.open_dataset(os.path.join(ddir, 'ssh_raw.nc'))
        for var in ['zos', 'sla', 'adt']:
            if var in ds:
                ssh_data = ds[var].squeeze().values.astype(float)
                ssh_lats = ds.latitude.values if 'latitude' in ds.dims else ds.lat.values
                ssh_lons = ds.longitude.values if 'longitude' in ds.dims else ds.lon.values
                break
        ds.close()
    except Exception:
        pass

    # Load raw currents
    uo = vo = cur_lats = cur_lons = None
    try:
        ds = xr.open_dataset(os.path.join(ddir, 'currents_raw.nc'))
        uo = ds['uo'].squeeze().values
        vo = ds['vo'].squeeze().values
        cur_lats = ds.latitude.values if 'latitude' in ds.dims else ds.lat.values
        cur_lons = ds.longitude.values if 'longitude' in ds.dims else ds.lon.values
        ds.close()
    except Exception:
        pass

    # Depth from bathy
    depth_grid = None
    try:
        import rasterio
        if os.path.exists(tif):
            with rasterio.open(tif) as src:
                depth_grid = src.read(1)
                depth_transform = src.transform
                depth_h, depth_w = depth_grid.shape
    except Exception:
        pass

    def sample_raw(lat, lon):
        """Sample all raw environmental values at a point."""
        row = {}
        if sst_raw is not None:
            si = np.argmin(np.abs(sst_lats - lat))
            sj = np.argmin(np.abs(sst_lons - lon))
            val = sst_raw[si, sj]
            row['sst_C'] = round(float(val), 2) if not np.isnan(val) else None
        if chl_data is not None:
            ci = np.argmin(np.abs(chl_lats - lat))
            cj = np.argmin(np.abs(chl_lons - lon))
            val = chl_data[ci, cj]
            row['chl'] = round(float(val), 4) if not np.isnan(val) else None
        if mld_data is not None:
            mi = np.argmin(np.abs(mld_lats - lat))
            mj = np.argmin(np.abs(mld_lons - lon))
            val = mld_data[mi, mj]
            row['mld'] = round(float(val), 1) if not np.isnan(val) else None
        if ssh_data is not None:
            hi = np.argmin(np.abs(ssh_lats - lat))
            hj = np.argmin(np.abs(ssh_lons - lon))
            val = ssh_data[hi, hj]
            row['ssh'] = round(float(val), 4) if not np.isnan(val) else None
        if uo is not None:
            ci = np.argmin(np.abs(cur_lats - lat))
            cj = np.argmin(np.abs(cur_lons - lon))
            u, v = float(uo[ci, cj]), float(vo[ci, cj])
            row['cur_speed'] = round(np.sqrt(u**2 + v**2), 4)
        if depth_grid is not None:
            try:
                py, px = ~depth_transform * (lon, lat)
                py, px = int(py), int(px)
                if 0 <= py < depth_h and 0 <= px < depth_w:
                    row['depth'] = round(float(-depth_grid[py, px]))
            except Exception:
                pass
        return row

    for c in by_date[date]:
        li = np.argmin(np.abs(lats - c['lat']))
        lo = np.argmin(np.abs(lons - c['lon']))
        catch_score = float(grid[li, lo]) if not np.isnan(grid[li, lo]) else 0
        catch_bands = float(bc[li, lo]) if not np.isnan(bc[li, lo]) else 0
        catch_bmean = float(bm[li, lo]) if not np.isnan(bm[li, lo]) else 0

        # Find highest-scoring cell within 5nm
        search_lat = SEARCH_RADIUS_NM / NM_PER_DEG
        search_lon = SEARCH_RADIUS_NM / (NM_PER_DEG * cos_lat)
        lat_mask = np.abs(lats - c['lat']) <= search_lat
        lon_mask = np.abs(lons - c['lon']) <= search_lon
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

        if len(lat_idx) > 0 and len(lon_idx) > 0:
            sub = grid[np.ix_(lat_idx, lon_idx)]
            if np.any(~np.isnan(sub)):
                peak_pos = np.unravel_index(np.nanargmax(sub), sub.shape)
                peak_li = lat_idx[peak_pos[0]]
                peak_lo = lon_idx[peak_pos[1]]
                peak_score = float(grid[peak_li, peak_lo])
                peak_bands = float(bc[peak_li, peak_lo]) if not np.isnan(bc[peak_li, peak_lo]) else 0
                peak_bmean = float(bm[peak_li, peak_lo]) if not np.isnan(bm[peak_li, peak_lo]) else 0
                peak_lat = float(lats[peak_li])
                peak_lon = float(lons[peak_lo])
            else:
                peak_score = catch_score
                peak_bands = catch_bands
                peak_bmean = catch_bmean
                peak_lat, peak_lon = c['lat'], c['lon']
        else:
            peak_score = catch_score
            peak_bands = catch_bands
            peak_bmean = catch_bmean
            peak_lat, peak_lon = c['lat'], c['lon']

        # Sample sub-scores at both locations
        catch_ss = {}
        peak_ss = {}
        for name in ['sst', 'sst_front', 'front_corridor', 'chl', 'ssh', 'mld',
                      'current', 'convergence', 'depth', 'shelf_break']:
            if name in ss and isinstance(ss[name], np.ndarray):
                cv = ss[name][li, lo]
                catch_ss[name] = round(float(cv), 3) if not np.isnan(cv) else None
                pv = ss[name][peak_li, peak_lo]
                peak_ss[name] = round(float(pv), 3) if not np.isnan(pv) else None

        # Sample raw environmental values at both points
        catch_raw = sample_raw(c['lat'], c['lon'])
        peak_raw = sample_raw(peak_lat, peak_lon)

        # Distance between catch and peak
        dlat_nm = (peak_lat - c['lat']) * NM_PER_DEG
        dlon_nm = (peak_lon - c['lon']) * NM_PER_DEG * cos_lat
        dist_nm = np.sqrt(dlat_nm**2 + dlon_nm**2)

        row = {
            'date': date, 'tag': c['tag'],
            # Catch location
            'catch_lat': c['lat'], 'catch_lon': c['lon'],
            'catch_score': round(catch_score, 3),
            'catch_bands': int(catch_bands),
            'catch_bmean': round(catch_bmean, 3),
            # Peak location
            'peak_lat': round(peak_lat, 4), 'peak_lon': round(peak_lon, 4),
            'peak_score': round(peak_score, 3),
            'peak_bands': int(peak_bands),
            'peak_bmean': round(peak_bmean, 3),
            # Offset
            'offset_nm': round(dist_nm, 1),
            'offset_lat_nm': round(dlat_nm, 1),
            'offset_lon_nm': round(dlon_nm, 1),
        }

        # Add raw environmental comparison
        for key in ['sst_C', 'chl', 'mld', 'ssh', 'cur_speed', 'depth']:
            row[f'catch_{key}'] = catch_raw.get(key, '')
            row[f'peak_{key}'] = peak_raw.get(key, '')
            if catch_raw.get(key) is not None and peak_raw.get(key) is not None:
                row[f'diff_{key}'] = round(peak_raw[key] - catch_raw[key], 4)
            else:
                row[f'diff_{key}'] = ''

        # Add sub-score comparison
        for name in ['sst', 'sst_front', 'front_corridor', 'chl', 'ssh', 'mld',
                      'current', 'convergence', 'depth', 'shelf_break']:
            row[f'catch_s_{name}'] = catch_ss.get(name, '')
            row[f'peak_s_{name}'] = peak_ss.get(name, '')

        results.append(row)

    print(f'{date} done', flush=True)

# Write CSV
out_path = 'data/catch_vs_peak_analysis.csv'
fields = list(results[0].keys()) if results else []
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)

# Summary analysis
print(f'\nWrote {len(results)} rows to {out_path}')
print(f'\n{"="*70}')
print(f'CATCH vs PEAK ENVIRONMENTAL COMPARISON ({len(results)} catches)')
print(f'{"="*70}')

# Score comparison
catch_scores = [r['catch_score'] for r in results]
peak_scores = [r['peak_score'] for r in results]
print(f'\nSCORES:')
print(f'  Catch mean: {np.mean(catch_scores):.3f}  Peak mean: {np.mean(peak_scores):.3f}')
print(f'  Score uplift at peak: {np.mean([p-c for p,c in zip(peak_scores, catch_scores)]):.3f}')

# Band comparison
catch_b = [r['catch_bands'] for r in results]
peak_b = [r['peak_bands'] for r in results]
print(f'\nFEATURE BANDS:')
print(f'  Catch mean: {np.mean(catch_b):.1f}  Peak mean: {np.mean(peak_b):.1f}')
print(f'  Band uplift at peak: {np.mean([p-c for p,c in zip(peak_b, catch_b)]):.1f}')

# Environmental comparison
print(f'\nRAW ENVIRONMENTAL VALUES (catch -> peak, delta):')
for key, unit in [('sst_C', 'C'), ('chl', 'mg/m3'), ('mld', 'm'),
                   ('ssh', 'm'), ('cur_speed', 'm/s'), ('depth', 'm')]:
    catch_vals = [r[f'catch_{key}'] for r in results if r.get(f'catch_{key}') not in ('', None)]
    peak_vals = [r[f'peak_{key}'] for r in results if r.get(f'peak_{key}') not in ('', None)]
    diffs = [r[f'diff_{key}'] for r in results if r.get(f'diff_{key}') not in ('', None)]
    if catch_vals and peak_vals:
        print(f'  {key:12s}: catch={np.mean(catch_vals):7.3f}  peak={np.mean(peak_vals):7.3f}  '
              f'delta={np.mean(diffs):+7.4f} {unit}')

# Sub-score comparison
print(f'\nSUB-SCORE COMPARISON (catch -> peak):')
for name in ['sst', 'sst_front', 'front_corridor', 'chl', 'ssh', 'mld',
              'current', 'convergence', 'depth', 'shelf_break']:
    cv = [r[f'catch_s_{name}'] for r in results if r.get(f'catch_s_{name}') not in ('', None)]
    pv = [r[f'peak_s_{name}'] for r in results if r.get(f'peak_s_{name}') not in ('', None)]
    if cv and pv:
        delta = np.mean(pv) - np.mean(cv)
        print(f'  {name:18s}: catch={np.mean(cv):.3f}  peak={np.mean(pv):.3f}  delta={delta:+.3f}')

# Key diagnostic: what % of score uplift comes from bands vs environment?
print(f'\nDIAGNOSTIC: Band-driven vs Environment-driven peaks')
band_driven = 0
env_driven = 0
mixed = 0
for r in results:
    band_uplift = r['peak_bands'] - r['catch_bands']
    sst_diff = r.get('diff_sst_C', 0) or 0
    # Peak has more bands but WORSE environmental values = band-driven
    # Check if SST at peak is further from optimal (22.52) than at catch
    catch_sst = r.get('catch_sst_C', None)
    peak_sst = r.get('peak_sst_C', None)
    if catch_sst and peak_sst:
        catch_sst_err = abs(catch_sst - 22.52)
        peak_sst_err = abs(peak_sst - 22.52)
        if band_uplift > 0 and peak_sst_err > catch_sst_err:
            band_driven += 1
        elif band_uplift <= 0 and peak_sst_err <= catch_sst_err:
            env_driven += 1
        else:
            mixed += 1

total = band_driven + env_driven + mixed
if total > 0:
    print(f'  Band-driven peaks (more bands, worse SST): {band_driven}/{total} ({band_driven/total*100:.0f}%)')
    print(f'  Env-driven peaks (equal/fewer bands, better SST): {env_driven}/{total} ({env_driven/total*100:.0f}%)')
    print(f'  Mixed: {mixed}/{total} ({mixed/total*100:.0f}%)')

# Distance stats
offsets = [r['offset_nm'] for r in results]
print(f'\nOFFSET STATS:')
print(f'  Mean: {np.mean(offsets):.1f} nm  Median: {np.median(offsets):.1f} nm')
print(f'  Peak within 1nm of catch: {sum(1 for o in offsets if o <= 1)}/{len(offsets)}')
print(f'  Peak within 3nm of catch: {sum(1 for o in offsets if o <= 3)}/{len(offsets)}')
