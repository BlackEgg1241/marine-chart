# Marine Chart — Perth WA 🐟⚓

A browser-based marine chart viewer with marlin zone analysis, built for Perth / WA waters.

## Live App

**https://blackegg1241.github.io/marine-chart/**

Works on mobile Safari/Chrome. No API keys required for the base app.

## What's Working Now

- **MapLibre GL** chart viewer with 4 basemaps (GA Bathymetry, ESRI Ocean, Satellite, Street)
- **OpenSeaMap** nautical overlay (buoys, lights, channel markers)
- **GPS tracking** with speed (knots), heading, accuracy circle, track log
- **Simulated GPS** — Fremantle to Rottnest demo route
- **NASA GIBS satellite layers** — SST, Chlorophyll-a, True Colour (no API key)
- **Marlin Zones** — 200m shelf edge line, Perth Canyon & FAD markers, species temp legend
- **Date picker** — navigate satellite data day by day

## Data Pipeline (Tier 2)

The `marlin_data.py` script fetches real ocean data and produces GeoJSON overlays:

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Copernicus Marine (one-time)
copernicusmarine login
```

### Usage

```bash
# Fetch yesterday's data for Perth region, process all layers
python marlin_data.py

# Specific date
python marlin_data.py --date 2026-03-05

# Custom bounding box (lon_min lat_min lon_max lat_max)
python marlin_data.py --bbox 113 -34 117 -30

# With GEBCO bathymetry contours (download GeoTIFF from download.gebco.net first)
python marlin_data.py --gebco gebco_perth.tif

# Re-process existing data without re-downloading
python marlin_data.py --skip-fetch
```

### Outputs (data/ folder)

| File | Description |
|------|-------------|
| `sst_fronts.geojson` | SST gradient lines (temp breaks) + marlin isotherms |
| `chl_edges.geojson` | Chlorophyll concentration edges (productivity boundaries) |
| `currents.geojson` | Current vector arrows (speed + direction) |
| `bathymetry_contours.geojson` | 50m, 100m, 200m, 500m, 1000m depth contours |
| `report.json` | Summary — SST range, marlin zone availability |

### GEBCO Bathymetry Contours

For accurate depth contours (200m shelf edge etc):

1. Go to **https://download.gebco.net/**
2. Select region: roughly 113°E–117°E, 30°S–34°S
3. Download as **GeoTIFF**
4. Run: `python marlin_data.py --gebco gebco_2025_n-30_s-34_w113_e117.tif`

Or use GDAL directly:
```bash
gdal_contour -fl -50 -100 -200 -500 -1000 gebco_perth.tif contours.geojson -f GeoJSON
```

## Data Sources

| Source | Data | Cost | API Key |
|--------|------|------|---------|
| NASA GIBS | SST, Chlorophyll, Satellite imagery | Free | None |
| Copernicus Marine | SST, Currents, Chlorophyll (raw NetCDF) | Free | Account required |
| Geoscience Australia | Bathymetry basemap tiles | Free | None |
| ESRI | Ocean basemap, Satellite imagery | Free | None |
| OpenSeaMap | Nautical marks overlay | Free | None |
| OpenStreetMap | Street basemap | Free | None |
| GEBCO | Global bathymetry grid (for contours) | Free | None |
| IMOS OceanCurrent | AU-specific SST, Chlorophyll | Free | Account required |

## Potential Upgrades

- **Navionics Web API** — proper nautical charts with HD bathymetry (free with navKey registration at webapiv2.navionics.com)
- **Computed marlin heatmap** — score each pixel based on SST range + gradient + chlorophyll edge + bathymetry proximity
- **Current particle animation** — Windy.com-style animated flow over the map
- **IMOS high-res SST** — 2km resolution Australian-specific SST from IMOS/BoM
- **Offline tile caching** — save tiles for use without mobile data (Service Worker)
- **Native iOS wrapper** — WKWebView app for App Store distribution

## Architecture

```
┌─────────────────────────────────┐
│  Browser (Mobile Safari/Chrome) │
│  ┌───────────────────────────┐  │
│  │   MapLibre GL JS          │  │
│  │   ├── Basemap tiles       │  │
│  │   ├── OpenSeaMap overlay   │  │
│  │   ├── NASA GIBS WMS       │  │
│  │   ├── GeoJSON overlays ←──┼──┼── marlin_data.py outputs
│  │   └── GPS (browser API)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Python Pipeline (your PC)      │
│  ├── Copernicus Marine API      │
│  ├── SST front detection        │
│  ├── Chlorophyll edge detection  │
│  ├── Current vector processing   │
│  ├── GEBCO contour extraction   │
│  └── → GeoJSON files            │
└─────────────────────────────────┘
```

## Marlin Fishing Science

The app targets temperature breaks and productivity edges — the key indicators:

- **Blue Marlin**: 23–29°C, warm side of temp breaks
- **Striped Marlin**: 21–24°C, rides the Leeuwin Current along the shelf edge
- **SST gradients**: Sharp temperature changes = convergence zones where bait concentrates
- **Chlorophyll edges**: Boundary between productive green water and clean blue water
- **Shelf edge + canyons**: Perth Canyon (200m→4000m), Rottnest FADs in 200m water

Perth Canyon is approximately 22km west of Rottnest Island, where the shelf drops from 200m to over 1000m within a few kilometres. Striped marlin have been recorded at the FADs behind Rottnest in 200m water, travelling with the Leeuwin Current.
