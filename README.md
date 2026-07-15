# MarLEEn Tracker — Perth Blue Marlin Forecast

A browser-based marine chart and automated forecasting system for blue marlin fishing off Perth, Western Australia. Combines satellite ocean data, ML trajectory analysis, and weather forecasting into a single-page app with daily automated email reports.

## Live App

**https://blackegg1241.github.io/marine-chart/**

Works on mobile Safari/Chrome. No API keys required for the base app.

---

## Features

### Interactive Map (index.html)
- **MapLibre GL JS** map with 3 basemaps: Ocean (ArcGIS/GEBCO bathymetry), Satellite (Maxar), Street (OSM)
- **OpenSeaMap** nautical overlay (buoys, lights, channel markers)
- **GPS tracking** with speed (knots), heading, accuracy circle, track log
- **Blue marlin hotspot zones** — scored habitat suitability overlays (0–100%)
- **11 data layers**: SST, Chlorophyll, Height Map (SSH), SST Fronts, CHL Edges, Currents, Bathymetry contours, Clarity (KD490), Eddies, MLD, O2
- **NASA GIBS satellite tiles** — SST (MUR L4), Chlorophyll-a (MODIS Aqua), CMEMS SLA
- **Catch browser** — historical blue marlin catch records with prev/next navigation
- **7-day forecast panel** — Zone Peak + Conditions chart, trend analysis, best day recommendation
- **Weather charts** — wind speed/direction and swell/wave height with sunlight strips
- **Boating comfort rating** — 0–100% score for a 5.2m boat
- **Measurement tool** — right-click/long-press for pin-based Haversine distance (nautical miles)
- **Key marks** — Perth Canyon, Rottnest Trench, FAD buoys, fishing spots
- **Date navigation** — browse historical observation data and forecast days
- **Accessible Trench Zone** — dashed purple boundary showing the primary fishing area

### Automated Daily Pipeline (run_daily.py)
- Runs via Windows Task Scheduler at 04:00 AWST daily
- Fetches weather, generates 7-day hotspot forecasts, archives for verification
- Auto-commits and pushes data to GitHub so the live app updates
- Sends HTML email report with inline charts

### Email Report
- **Blue Marlin 7-Day Forecast** — Habitat Score, Marlin (ML trend) Score, Comfort, and a blended rating of **60% habitat + 40% comfort** (the ML trend score is shown for reference, not folded into the rating); labelled GREAT/GOOD/FAIR/POOR
- **FADs Go/No-Go** — boating safety assessment with wind, swell, comfort, sun times
- **Inline charts** — wind speed & direction arrows, swell & wave height, trench zone scores
- Dark-themed HTML with sunlight strips (dawn/day/dusk/night)

---

## Architecture

```
Browser (Mobile Safari/Chrome)
  MapLibre GL JS
    Basemap tiles (ArcGIS Ocean/Satellite, OSM)
    OpenSeaMap nautical overlay
    NASA GIBS WMTS (SST, CHL, SLA tiles)
    GeoJSON overlays (hotspots, fronts, currents, contours...)
    GPS (browser Geolocation API)
    Forecast panel (zone scores, ML predictions)
    Weather charts (wind, swell, comfort)

Python Pipeline (local PC, scheduled daily)
  fetch_marine_weather.py    Open-Meteo APIs (wind, swell, comfort)
  fetch_prediction.py        CMEMS ANFC model (7-day hotspot maps)
  generate_forecast_summary.py  Zone-max scores for UI
  archive_forecast.py        Archive for verification
  run_daily.py               Orchestrator + email + git push

Scoring & Analysis (offline)
  marlin_data.py             Main data pipeline (observations)
  optimize_scoring.py        Optuna parameter optimization (200 trials)
  validate_scoring.py        Catch validation against GFAA/DPI records (46 blue marlin)
  analyze_trends_v2.py       ML prediction model (paired within-event)
  fetch_lookback.py          7-day lookback data for catch events
```

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `copernicusmarine`, `xarray`, `numpy`, `scipy`, `netCDF4`, `matplotlib`, `rasterio`, `shapely`, `geojson`

Additional for optimization: `optuna`, `scikit-learn`

### Copernicus Marine (for observation data pipeline)

```bash
copernicusmarine login
```

### Daily Pipeline

```bash
# Run manually
python run_daily.py

# Run without email
python run_daily.py --no-email

# Install Windows scheduled task (04:00 AWST daily)
python run_daily.py --install

# Remove scheduled task
python run_daily.py --uninstall
```

Email requires `MARLEEN_GMAIL_APP_PASSWORD` environment variable (Gmail app password).

### Observation Data Pipeline

```bash
# Fetch yesterday's data, generate all overlays
python marlin_data.py

# Specific date
python marlin_data.py --date 2026-03-05

# Custom bounding box (lon_min lat_min lon_max lat_max)
python marlin_data.py --bbox 113 -34 117 -30

# With GEBCO bathymetry contours
python marlin_data.py --gebco gebco_perth.tif

# Re-process without re-downloading
python marlin_data.py --skip-fetch
```

---

## Data Pipeline Outputs

### Observation Data (data/)

| File | Description |
|------|-------------|
| `sst_fronts.geojson` | SST gradient lines (temp breaks) + marlin isotherms |
| `chl_edges.geojson` | Chlorophyll concentration edges (productivity boundaries) |
| `currents.geojson` | Current vector arrows (speed + direction) |
| `blue_marlin_hotspots.geojson` | Scored habitat suitability zones (0–100%) |
| `bathymetry_contours.geojson` | 50m, 100m, 200m, 500m, 1000m depth contours |
| `report.json` | Summary — SST range, marlin zone availability |
| `marine_weather.json` | 7-day hourly wind, swell, comfort, sun times |

### Forecast Data (data/prediction/)

| File | Description |
|------|-------------|
| `YYYY-MM-DD/blue_marlin_hotspots.geojson` | Hotspot map per forecast day |
| `prediction_results.json` | ML model scores per day (Marlin Score) |
| `forecast_summary.json` | Zone-max scores for UI (Habitat Score) |
| `forecast_zone.geojson` | Accessible Trench Zone boundary polygon |

### Historical Data

| Directory | Description |
|-----------|-------------|
| `data/YYYY-MM-DD/` | Observation data per date (66 historical + recent) |
| `data/lookback/YYYY-MM-DD/` | 7-day rolling windows for catch events |

---

## Scoring Algorithm

See [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md) for full technical details. (The older [ALGORITHM_REPORT.md](ALGORITHM_REPORT.md) describes a superseded 10-variable model and should not be cited.)

**Summary:** The deployed model ("v22") combines **14 ocean features** into a normalized weighted sum (0–100%), optimized via Optuna against **46 GPS-verified blue marlin catches** (25 unique locations). Top weights: SST (0.150), Current Shear (0.129), Chlorophyll (0.107), CHL Curvature (0.100), SSH (0.100), Salinity Front (0.100). SST uses an **asymmetric Gaussian centred at 23.75°C** (sigma 2.50 below the optimum, 4.0 above) — a cool poleward-edge value for the Perth Canyon, not the global tropical 25–30°C band. Multiplicative modifiers for depth (gate below ~80m) and shelf break (up to a 12% boost).

**Accuracy (honest note):** The old headline "mean 84–89% at catch locations" is *in-sample training error*, not predictive skill — the Optuna objective maximizes that same quantity and there is no held-out set. The repo now includes `evaluate_model.py` + [EVALUATION.md](EVALUATION.md), which instead report presence/background discrimination: AUC ≈ **0.93** versus random ocean (in-sample), ≈ **0.91** leave-one-year-out, but only ≈ **0.78** within already-rendered zones (regional selection is good; fine-grained spot selection is weaker). A true out-of-sample figure needs a weight-refit cross-validation, which is not yet done. Note also that the same grid scores **striped** marlin equally well (AUC 0.93 vs 0.93) — it is really a shared shelf-edge / Leeuwin-front **billfish** index, not a blue-specific model.

**Prediction model:** Paired within-event ML analysis detects warm water homogenization patterns in the 7-day lead-up to catches. Leave-one-out CV: 61.1% mean accuracy. This ML "Marlin Score" is shown for reference and is **not** folded into the email's blended rating.

---

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| Copernicus Marine | SST, Currents, SSH, CHL, MLD, O2, Clarity (NetCDF) | Free (account required) |
| IMOS AODN | High-res Australian SST (0.02°) | Free (account required) |
| NASA GIBS | SST, Chlorophyll, satellite imagery (WMTS tiles) | Free |
| CMEMS SLA WMTS | Sea level anomaly tiles (NRT daily) | Free |
| Open-Meteo | Wind, swell, temperature, precipitation forecasts | Free |
| Geoscience Australia | Bathymetry basemap tiles | Free |
| ESRI | Ocean basemap, satellite imagery | Free |
| OpenSeaMap | Nautical marks overlay | Free |
| GEBCO | Global bathymetry grid (for contours) | Free |
| GMRT | High-res bathymetry GeoTIFF | Free |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `marlin_data.py` | Main data pipeline: fetch ocean data, generate all GeoJSON overlays |
| `fetch_prediction.py` | 7-day forecast: ANFC model data, hotspot maps per day |
| `fetch_marine_weather.py` | Weather data from Open-Meteo (wind, swell, comfort rating) |
| `generate_forecast_summary.py` | Zone-max scores from hotspot GeoJSONs |
| `archive_forecast.py` | Archive daily Open-Meteo marine weather forecasts for verification |
| `verify_forecast.py` | Verify forecast accuracy against BOM/buoy observations |
| `run_daily.py` | Daily automation orchestrator + email + git push |
| `validate_scoring.py` | Validate scoring against the 46 GPS-verified blue marlin catches |
| `evaluate_model.py` | Honest presence/background evaluation (AUC, lift) — see `EVALUATION.md` |
| `optimize_scoring.py` | Optuna optimization of scoring weights (200 trials) |
| `analyze_trends_v2.py` | ML prediction model (paired within-event, RF/GB) |
| `analyze_trends.py` | Initial trend analysis (superseded by v2) |
| `fetch_lookback.py` | Fetch 7-day lookback ocean data for catch events |

---

## Marlin Fishing Science

The app targets the intersection of multiple ocean features that concentrate marlin prey:

- **SST (23.75°C optimal, asymmetric — sigma 2.50 below, 4.0 above)**: Blue marlin follow warm Leeuwin Current water along the shelf edge. This is a cool poleward-edge value for the Perth Canyon, well below the global tropical 25–30°C band — the median SST at catch locations (~22.9°C) sits at the southern thermal limit of the Leeuwin Current, which is also why the same grid scores striped marlin equally well (a shared shelf-edge billfish index rather than a blue-specific one).
- **Temperature fronts**: Sharp SST gradients aggregate baitfish at thermal boundaries
- **Leeuwin Current intrusion**: Warm water pushed inshore toward the canyon signals active current transport
- **Chlorophyll edges**: Boundary between productive green water and clean blue water
- **Shelf edge + canyons**: Perth Canyon drops from 200m to >4000m within kilometres
- **Current convergence**: Converging flow at the canyon head traps bait
- **Warm water homogenization**: 7-day warming pattern with narrowing SST range precedes catches

Perth Canyon is approximately 22km west of Rottnest Island. The Accessible Trench Zone (Rottnest Trench to Club Marine, ~35km span) is the primary fishing area reachable by trailer boats from Perth metro boat ramps.
