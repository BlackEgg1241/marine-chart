"""Measure distance from each of 49 blue marlin catches to every visible line type.

Reads the actual GeoJSON line features (sst_fronts, isotherms, chl contours,
bathy contours, ssh contours, ssta contours, mld contours) and computes the
nearest-point distance in nautical miles for each catch.
"""
import json, csv, os, numpy as np
from datetime import datetime
from collections import defaultdict

NM_PER_DEG_LAT = 60.0

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    R_nm = 3440.065
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R_nm * 2 * np.arcsin(np.sqrt(a))

def min_dist_to_features(lat, lon, features):
    """Minimum distance from (lat, lon) to any LineString feature, in nm."""
    best = float('inf')
    for feat in features:
        coords = feat['geometry']['coordinates']
        if feat['geometry']['type'] == 'MultiLineString':
            for line in coords:
                for c in line:
                    d = haversine_nm(lat, lon, c[1], c[0])
                    if d < best:
                        best = d
        else:  # LineString
            for c in coords:
                d = haversine_nm(lat, lon, c[1], c[0])
                if d < best:
                    best = d
    return best if best < 999 else None

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

# Define all line types to measure
LINE_TYPES = [
    # (file, filter_key, filter_value, label)
    ('sst_fronts.geojson', 'type', 'sst_front', 'sst_front'),
    ('sst_fronts.geojson', ('type', 'temperature'), ('isotherm', 22), 'isotherm_22C'),
    ('sst_fronts.geojson', ('type', 'temperature'), ('isotherm', 23), 'isotherm_23C'),
    ('sst_fronts.geojson', ('type', 'temperature'), ('isotherm', 24), 'isotherm_24C'),
    ('chl_edges.geojson', 'concentration', 0.07, 'chl_007'),
    ('chl_edges.geojson', 'concentration', 0.15, 'chl_015'),
    ('chl_edges.geojson', 'concentration', 0.30, 'chl_030'),
    ('chl_edges.geojson', 'concentration', 0.60, 'chl_060'),
    ('ssh_eddies.geojson', 'sla', -0.01, 'sla_cold'),
    ('ssh_eddies.geojson', 'sla', 0.03, 'sla_003'),
    ('ssh_eddies.geojson', 'sla', 0.07, 'sla_007'),
    ('ssh_eddies.geojson', 'sla', 0.11, 'sla_011'),
    ('bathymetry_contours.geojson', 'depth', -100, 'bathy_100m'),
    ('bathymetry_contours.geojson', 'depth', -200, 'bathy_200m'),
    ('bathymetry_contours.geojson', 'depth', -300, 'bathy_300m'),
    ('bathymetry_contours.geojson', 'depth', -400, 'bathy_400m'),
    ('bathymetry_contours.geojson', 'depth', -500, 'bathy_500m'),
    ('bathymetry_contours.geojson', 'depth', -600, 'bathy_600m'),
    ('bathymetry_contours.geojson', 'depth', -700, 'bathy_700m'),
    ('bathymetry_contours.geojson', 'depth', -800, 'bathy_800m'),
    ('bathymetry_contours.geojson', 'depth', -900, 'bathy_900m'),
    ('bathymetry_contours.geojson', 'depth', -1000, 'bathy_1000m'),
    ('mld_contours.geojson', 'mld', 10, 'mld_10m'),
    ('mld_contours.geojson', 'mld', 20, 'mld_20m'),
    ('mld_contours.geojson', 'mld', 30, 'mld_30m'),
    ('ssta_contours.geojson', 'anomaly', -0.5, 'ssta_neg05'),
    ('ssta_contours.geojson', 'anomaly', 0.5, 'ssta_pos05'),
    ('ssta_contours.geojson', 'anomaly', 1.0, 'ssta_pos10'),
]

results = []

for date in sorted(by_date.keys()):
    ddir = os.path.join('data', date)
    if not os.path.exists(ddir):
        ddir = os.path.join('data', 'prediction', date)
    if not os.path.exists(ddir):
        continue

    # Load all GeoJSON files for this date
    geojson_cache = {}
    for filename in set(lt[0] for lt in LINE_TYPES):
        path = os.path.join(ddir, filename)
        if os.path.exists(path):
            with open(path) as f:
                geojson_cache[filename] = json.load(f)
        else:
            geojson_cache[filename] = None

    for c in by_date[date]:
        row = {
            'date': date, 'tag': c['tag'], 'type': c['type'],
            'lat': c['lat'], 'lon': c['lon'],
        }

        for lt in LINE_TYPES:
            filename, filter_key, filter_val, label = lt
            gj = geojson_cache.get(filename)
            if gj is None:
                row[label] = ''
                continue

            # Filter features
            if isinstance(filter_key, tuple):
                # Multi-key filter (e.g., type=isotherm AND temperature=22)
                filtered = [f for f in gj['features']
                           if all(f['properties'].get(k) == v
                                  for k, v in zip(filter_key, filter_val))]
            else:
                filtered = [f for f in gj['features']
                           if f['properties'].get(filter_key) == filter_val]

            if not filtered:
                row[label] = ''
                continue

            dist = min_dist_to_features(c['lat'], c['lon'], filtered)
            row[label] = round(dist, 2) if dist is not None else ''

        results.append(row)

    print(f'{date} done ({len(by_date[date])} catches)', flush=True)

# Write CSV
out_path = 'data/catch_line_proximity.csv'
fields = ['date', 'tag', 'type', 'lat', 'lon'] + [lt[3] for lt in LINE_TYPES]
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)

print(f'\nWrote {len(results)} rows to {out_path}')

# Summary statistics
print(f'\n{"="*80}')
print(f'CATCH-TO-LINE PROXIMITY ANALYSIS ({len(results)} catches)')
print(f'{"="*80}')
print(f'\n{"Line Type":<20s} {"N":>4s} {"Mean":>7s} {"Med":>7s} {"Min":>7s} {"Max":>7s} {"<1nm":>6s} {"<2nm":>6s} {"<3nm":>6s} {"<5nm":>6s}')
print('-' * 80)

line_stats = []
for lt in LINE_TYPES:
    label = lt[3]
    vals = [r[label] for r in results if r.get(label) not in ('', None)]
    if not vals:
        continue
    vals = [float(v) for v in vals]
    n = len(vals)
    within_1 = sum(1 for v in vals if v <= 1.0)
    within_2 = sum(1 for v in vals if v <= 2.0)
    within_3 = sum(1 for v in vals if v <= 3.0)
    within_5 = sum(1 for v in vals if v <= 5.0)
    mean_val = np.mean(vals)
    med_val = np.median(vals)
    print(f'{label:<20s} {n:>4d} {mean_val:>7.2f} {med_val:>7.2f} {min(vals):>7.2f} {max(vals):>7.2f} '
          f'{within_1:>3d}/{n:<2d} {within_2:>3d}/{n:<2d} {within_3:>3d}/{n:<2d} {within_5:>3d}/{n:<2d}')
    line_stats.append((label, mean_val, med_val, n, within_3/n*100))

# Rank by median proximity
print(f'\n{"="*80}')
print(f'RANKED BY MEDIAN PROXIMITY (closest to catches first)')
print(f'{"="*80}')
ranked = sorted(line_stats, key=lambda x: x[2])
for i, (label, mean_val, med_val, n, pct_3nm) in enumerate(ranked):
    bar = '#' * int(min(med_val, 30))
    print(f'{i+1:>2d}. {label:<20s} med={med_val:>5.2f}nm  mean={mean_val:>5.2f}nm  <3nm={pct_3nm:.0f}%  |{bar}')

# Correlation: which lines are catches consistently closest to?
print(f'\n{"="*80}')
print(f'LINES WHERE >50% OF CATCHES ARE WITHIN 3nm')
print(f'{"="*80}')
for label, mean_val, med_val, n, pct_3nm in ranked:
    if pct_3nm >= 50:
        print(f'  {label:<20s}: {pct_3nm:.0f}% within 3nm (median {med_val:.2f}nm)')
