# Blue Marlin Habitat Suitability Algorithm

## Technical Report — Perth Canyon Region, Western Australia

---

## 1. Overview

The algorithm produces a **composite habitat suitability score (0–100%)** for each grid cell across the Perth Canyon region (113.5–116.5°E, 30.5–33.5°S). It combines 10 ocean variables into a weighted sum, applies static depth and shelf-break multipliers, and exports the result as GeoJSON polygons with intensity bands for display on a web-based marine chart.

The scoring weights and parameters were **optimized using Optuna** (200 trials) against historical catch data, maximizing a composite objective of mean score, floor score, and percentage above 70%.

### Validation

Validated against **71 historical marlin catch records** (45 blue, 19 striped, 7 black) from the Game Fishing Association of Australia (GFAA) tagging database, spanning 2000–2023 across 53 unique dates.

**Important:** The GFAA CSV uses **degrees-minutes format** (DDM), not decimal degrees. Raw value "31.49S" = 31°49' = -31.8167°. All catch coordinates are converted via `ddm_to_dd()` before validation.

| Metric | All Species (71) | Blue Marlin Only (45) |
|--------|:-----------------:|:---------------------:|
| Mean score at catch locations | **89%** | **88%** |
| Median score | **92%** | **90%** |
| Catches scoring >= 50% | **100%** (71/71) | **100%** (45/45) |
| Catches scoring >= 70% | **97%** (69/71) | **96%** (43/45) |
| Catches in a scored zone | **100%** (71/71) | **100%** (45/45) |
| Score range | 68%–100% | 68%–100% |

---

## 2. Data Sources

All ocean data is fetched from the **Copernicus Marine Service** via their Python API. Bathymetry is from **GMRT** (Global Multi-Resolution Topography) GeoTIFF.

| Variable | Dataset | Resolution | Update |
|----------|---------|-----------|--------|
| SST | METOFFICE-GLO-SST-L4-NRT-OBS (satellite L4) | ~0.05° (~5km) | Daily |
| SST (forecasts) | CMEMS ANFC model (thetao) | 0.083° (~9km) | Daily |
| SST (high-res obs) | IMOS AODN L3S | 0.02° (~2km) | Daily |
| Currents (u,v) | CMEMS GLO-PHY model | 0.083° (~9km) | Daily |
| Sea Level Anomaly | CMEMS SLA L4 DUACS (multi-satellite) | 0.125° | Daily |
| Chlorophyll-a | CMEMS OC L4 gapfree (satellite) | ~4km | Daily |
| Mixed Layer Depth | CMEMS GLO-PHY model | 0.083° | Daily |
| Dissolved Oxygen | CMEMS GLO-BGC model | 0.25° | Daily |
| Water Clarity (KD490) | CMEMS OC (satellite) | ~4km | Daily |
| Bathymetry | GMRT GeoTIFF | ~100m | Static |

The SST grid serves as the **master grid** — all other variables are interpolated onto it using bilinear interpolation (`scipy.interpolate.RegularGridInterpolator`).

---

## 3. Scoring Components

All weights below are the **Optuna-optimized values** (200 trials, composite objective).

### 3.1 SST Score (weight: 0.31)

The primary habitat driver. Blue marlin in the Perth region are caught predominantly at 22–24°C, with a median SST of 22.9°C at catch locations.

**Method:** Gaussian function centered at **23.4°C** with **sigma=2.97°C**:

```
sst_score = exp(-0.5 × ((SST - 23.4) / 2.97)²)
```

The wide sigma (2.97) reflects the broad tolerance range observed in the catch data — marlin were caught at SSTs from 21°C to 27°C. The SST field is lightly smoothed (Gaussian σ=0.5 pixels) before scoring to reduce satellite noise while preserving gradients.

### 3.2 SST Front Score (weight: 0.09)

Temperature fronts aggregate baitfish at thermal boundaries, concentrating prey for marlin.

**Method:**
1. Compute SST gradient magnitude using Sobel operators on a smoothed SST field (Gaussian σ=1.5)
2. Mask a 2-pixel buffer around coastline to suppress false fronts from land-sea contrast
3. Normalize by the **90th percentile** of ocean gradient values (more temporally stable than max-based normalization)
4. Modulate by SST suitability — fronts in cold water are less useful for marlin, so `front_score = front_score × sst_score` (continuous weighting, not a hard gate)
5. Apply a **warm-water floor** of **0.07** — in warm water (SST score > 0.6), even the absence of a strong front gets a minimum score

### 3.3 Cross-Shelf SST Gradient — Leeuwin Current Intrusion (weight: 0.03)

Measures whether the Leeuwin Current is actively pushing warm water inshore (toward the shelf edge and canyon). Provides independent predictive signal (partial r=0.42 after controlling for SST score).

**Method:**
For each grid cell, compare mean SST of the 3–4 cells to the east (inshore) vs 2–5 cells to the west (offshore) along the same latitude:

```
cross_gradient = mean(SST_inshore) - mean(SST_offshore)
intrusion_score = clip(cross_gradient / 0.50, 0, 1)
```

In warm water (SST score > 0.5), a **baseline floor of 0.24** is applied even without a clear cross-shelf gradient — the Leeuwin Current may be present without a strong E-W temperature difference. A positive gradient (warm inshore) indicates active Leeuwin Current intrusion — the mechanism that delivers warm tropical water to the Perth Canyon.

### 3.4 Chlorophyll Score (weight: 0.08)

Moderate chlorophyll indicates the bait productivity zone at the shelf edge.

**Method:** Gaussian in log-space, peaking at **0.22 mg/m³** with **sigma=0.79**:

```
chl_score = exp(-0.5 × ((log10(CHL) - log10(0.22)) / 0.79)²)
```

### 3.5 SSH / Sea Level Anomaly Score (weight: 0.19)

Sea level anomaly is a proxy for warm water mass presence (absolute) and mesoscale eddy structure (relative).

**Method:** Blended 50/50 absolute + relative:

- **Absolute SLA:** Score 0 at SLA≤0m, score 1.0 at SLA≥0.15m. High absolute SLA indicates a warm water mass sitting over the canyon.
- **Relative SLA:** Local SLA minus a smoothed background (Gaussian σ=4 pixels). Score = clip(relative_SLA / 0.04, 0, 1). Detects eddy edges where marlin hunt along the interface of warm and cool water masses.

### 3.6 Current Favorability Score (weight: 0.12)

Currents that transport warm water into the Perth Canyon aggregate bait and create productive fishing conditions.

**Method:** Three-factor composite:

1. **Speed score:** clip((speed - 0.03) / 0.17, 0, 1) — stronger currents transport more warm water
2. **Upstream SST:** For each cell, sample SST 2 grid cells upstream (opposite to flow direction). Score using the same Gaussian as the main SST score. This measures *what temperature the current is delivering*.
3. **Eastward bonus:** Onshore/eastward flow (positive u-component) gets a **18% bonus**: `1.0 + 0.18 × east_score`

```
current_score = clip(speed_score × upstream_temp_score × east_bonus, 0, 1)
```

The eastward bonus is soft (not a gate) because validation showed marlin are caught in all current directions — but eastward Leeuwin Current flow is associated with better conditions.

### 3.7 Current Convergence Score (weight: 0.04)

Convergent flow concentrates baitfish at the canyon head.

**Method:**
1. Compute divergence: `div = du/dx + dv/dy` (finite differences on the current grid)
2. Convergence score: `clip(-divergence / 0.005, 0, 1)` — negative divergence = converging flow
3. Smooth with Gaussian σ=1.0 to reduce grid-scale artefacts
4. Apply **bait trap synergy**: convergence is amplified when current is also strong:

```
synergy = 1.0 + 0.36 × current_score
convergence_final = clip(convergence × synergy, 0, 1)
```

### 3.8 Mixed Layer Depth Score (weight: 0.08)

Shallower mixed layers compress marlin and bait into a thinner surface layer, making encounters more likely.

**Method:**

```
mld_score = clip(1.0 - (MLD - 20) / 80, 0, 1)
```

Score 1.0 at MLD ≤ 20m, 0.5 at 50m, 0 at ≥ 100m.

### 3.9 Dissolved Oxygen Score (weight: 0.04)

Oxygen at depth (100m) — below a threshold, marlin avoid the area. Rarely limiting in the well-oxygenated Perth Canyon region.

**Method:**

```
o2_score = clip((O2 - 100) / 100, 0, 1)
```

Score 0 at ≤100 mmol/m³, 0.5 at 150, 1.0 at ≥200.

### 3.10 Water Clarity Score (weight: 0.04)

Based on diffuse attenuation coefficient KD490. Clearer water is preferred by blue marlin (visual predators). Rarely limiting in offshore waters.

**Method:**

```
clarity_score = clip(1.0 - (KD490 - 0.04) / 0.11, 0, 1)
```

Score 1.0 at KD490 < 0.04 (very clear), 0 at > 0.15 (turbid).

### Weight Summary

| Component | Weight | Primary Driver |
|-----------|:------:|----------------|
| SST | 0.31 | Temperature preference |
| SSH (SLA) | 0.19 | Warm water mass / eddy detection |
| Current | 0.12 | Warm water transport |
| SST Front | 0.09 | Bait aggregation at temp breaks |
| Chlorophyll | 0.08 | Productivity zone |
| Mixed Layer Depth | 0.08 | Vertical compression |
| Convergence | 0.04 | Bait trapping |
| Dissolved Oxygen | 0.04 | Habitat viability |
| Water Clarity | 0.04 | Visual hunting |
| SST Intrusion | 0.03 | Leeuwin Current signal |
| **Total** | **1.00** | |

---

## 4. Static Multipliers

Two bathymetric factors are applied as **multipliers** (not additive weights) to the composite score.

### 4.1 Depth Gate (multiplier: 0 to 1.0)

Blue marlin in this region are only caught over deep water (>100m). The depth score gates the composite:

```
depth_score:
  < 50m:       0.0  (zero — too shallow)
  50–80m:      linear ramp (depth - 50) / 30
  80–800m:     1.0  (optimal — continental slope/canyon)
  800–2000m:   1.0 → 0.70  (linear taper: 0.85 + 0.15 × (1 - (depth-800)/1200))
  > 2000m:     0.70  (still viable but less productive)
```

**Max-pooling:** Depth scoring is computed at the native bathymetry resolution (~100m) first, then max-pooled to the coarser SST grid via `_maxpool_to_grid()`. This prevents shelf-edge catches from being scored as shallow due to linear interpolation averaging deep canyon pixels with nearby shallow/land pixels.

### 4.2 Shelf Break Boost (multiplier: 1.0 to 1.53)

Steep bathymetric gradients (canyon walls, shelf edge) drive upwelling and bait aggregation:

```
shelf_multiplier = 1.0 + 0.53 × shelf_score
```

Canyon walls and the shelf edge can receive up to a **53% boost** to their composite score.

---

## 5. Score Composition

The final composite score for each grid cell is:

```
raw_score = Σ(weight_i × score_i) / Σ(weight_i)    (for available variables)
gated_score = raw_score × depth_score
boosted_score = gated_score × (1.0 + 0.53 × shelf_break_score)
final_score = clip(boosted_score, 0, 1)
```

The normalization by `Σ(weight_i)` ensures graceful degradation — if some data sources are missing (e.g., SSH failed to download), the remaining scores are re-weighted proportionally.

A light spatial smoothing (Gaussian σ=0.8 pixels) is applied to the final composite to reduce pixelation.

---

## 6. GeoJSON Export

The continuous score grid is converted to **filled contour polygons** using matplotlib's `contourf` at these intensity bands:

| Band | Score Range | Description |
|------|------------|-------------|
| 1 | 15–25% | Very low potential |
| 2 | 25–35% | Low potential |
| 3 | 35–45% | Moderate potential |
| 4 | 45–55% | Fair potential |
| 5 | 55–65% | Good potential |
| 6 | 65–75% | High potential |
| 7 | 75–85% | Very high potential |
| 8 | 85–100% | Prime zone |

Polygons below 15% (background noise) are excluded. Each polygon is clipped to the deep water mask (>50m depth) so no hotspot zones appear over shallow water or land.

Each GeoJSON feature includes:
- `intensity`: composite score (0–1)
- `band`: human-readable band label (e.g., "75%–85%")
- `sub_scores`: breakdown of each component's score and weight for the polygon area
- Color coding from yellow (low) through orange to deep red (prime)

---

## 7. Prediction Model — Marlin Score (ML)

In addition to the spatial habitat score, a **machine learning prediction model** assesses whether ocean parameter trajectories match historical patterns observed before blue marlin catches.

### 7.1 Design

**Paired within-event analysis** — each catch event serves as its own control:
- 7-day lookback windows (days -7 to 0) of ocean data are compared
- Early period (days -7 to -5) vs late period (days -2 to 0) are tested for significant shifts
- This controls for seasonal and location effects (each catch = own baseline)

### 7.2 Statistical Tests

- **Wilcoxon signed-rank test**: 4 significant paired shifts detected (p < 0.05):
  1. `sst_min` increases (+0.8%, p=0.038) — cold water retreating
  2. `sst_range` decreases (-7.1%, p=0.032) — thermal homogenization
  3. `sst_skew` decreases (p=0.036) — distribution normalizing
  4. `sst_p25` increases (+0.8%, p=0.023) — cold tail warming
- **Permutation test** (5000 shuffles): 15 significant trajectories identified

### 7.3 ML Classifiers

- **Random Forest** and **Gradient Boosting** classifiers (scikit-learn)
- Features: trajectory slopes, acceleration, momentum over the 7-day lookback
- Leave-one-out cross-validation on 11 catch events (events with >= 6/8 days of data):
  - Mean accuracy: **61.1%**
  - Events >= 50% accuracy: **73%** (8/11)
- Top features: `sst_max|accel`, `mld_mean|slope3`

### 7.4 Physical Story

The 7-day pattern before catches shows a consistent warm water homogenization sequence:
- **Phase 1 (-7 to -5)**: Baseline thermal structure, normal variability
- **Phase 2 (-4 to -2)**: Warm water intrusion begins — SST min rises, range narrows, Leeuwin Current strengthens
- **Phase 3 (-1 to 0)**: Homogeneous warm water established, elevated currents, shallow MLD, anticyclonic vorticity — marlin arrive to feed

### 7.5 Forecast Scoring

The `fetch_prediction.py` pipeline evaluates each forecast day against the ML model:
- **Absolute value match** (40% weight) — do current conditions match catch patterns?
- **Trend direction match** (35% weight) — are parameters moving in the right direction?
- **Early-to-late shift** (25% weight) — does the 7-day trajectory match the homogenization pattern?

Output: 0–100% "Marlin Score" per day, with confidence level and top/bottom contributing parameters.

---

## 8. Daily Forecast Pipeline

The `run_daily.py` orchestrator runs automatically via Windows Task Scheduler at **04:00 AWST** daily (after ECMWF model runs complete).

### 8.1 Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `fetch_marine_weather.py` | Wind, swell, temperature from Open-Meteo (Rottnest + Hillarys) |
| 2 | `fetch_prediction.py` | 8-day lookback + 7-day forward hotspot maps from CMEMS ANFC |
| 3 | `generate_forecast_summary.py` | Zone-max scores for UI from hotspot GeoJSONs |
| 4 | `archive_forecast.py` | Archive today's Open-Meteo marine weather forecast for later verification |

### 8.2 Weather Data

`fetch_marine_weather.py` fetches from Open-Meteo APIs:
- **Marine API**: wave height, swell (primary + secondary), wind waves, SST
- **Weather API**: wind speed/direction/gusts, temperature, precipitation, weather code, visibility
- Combines primary + secondary swell via energy-weighted RSS (height) and energy-weighted average (period/direction)
- Computes **boating comfort rating** (0–100) for a 5.2m boat:
  - Wind speed (30%), swell height (25%), gusts (10%), wave height (10%), wind direction (10%), rain/storms (10%), visibility (5%)
- Includes civil twilight times (first light, sunrise, sunset, last light)

### 8.3 Accessible Trench Zone

The forecast summary computes max habitat scores within the **Accessible Trench Zone** — the primary fishing area reachable by trailer boats:

| Corner | Mark | Coordinates |
|--------|------|-------------|
| North | North Metro 04 | 115.1754°E, 31.7287°S |
| West | Rottnest Trench | 114.98°E, 32.01°S |
| South | Fibrelite Boats | 115.1667°E, 32.1667°S |
| East | Club Marine | 115.3333°E, 32.05°S |

Zone scoring uses **bounding box overlap test** (not centroid) to avoid missing cells at zone boundaries.

### 8.4 Email Report

Daily email includes:
- **Pipeline status** — pass/fail for each step with timing
- **Blue Marlin 7-Day Forecast** table: Habitat Score, Marlin Score, Comfort, blended Rating
  - Rating = 60% Marlin (trajectory) + 25% Habitat (spatial) + 15% Comfort (boating)
  - Labels: GREAT (>=75), GOOD (>=60), FAIR (>=45), POOR (<45)
- **FADs Go/No-Go** assessment for 5.2m boat:
  - GO: swell <1.5m, wind <15kn, gusts <25kn, no persistent northerlies, no storms
  - MARGINAL: borderline conditions (swell 1.2–1.5m, wind 10–15kn, isolated northerly, drizzle)
  - NO GO: any hard limit exceeded
  - Includes sun times (first light / sunrise / sunset / last light)
- **Inline charts**: Wind speed & direction, Swell & wave height, Trench zone scores — all with sunlight strips

### 8.5 Auto-Publish

On successful pipeline completion, updated data files are automatically committed and pushed to GitHub so the hosted app at `https://blackegg1241.github.io/marine-chart/` refreshes.

---

## 9. Validation Methodology

Validation was performed by:

1. Loading 71 historical marlin catch records (45 blue, 19 striped, 7 black) with GPS coordinates and dates from the GFAA tagging database CSV
2. Converting DDM coordinates to decimal degrees via `ddm_to_dd()`
3. For each unique catch date (53 dates spanning 2000–2023), fetching the same ocean data from Copernicus Marine reanalysis products
4. Running the full scoring algorithm for that date
5. Sampling the raw score grid at the **exact pixel** nearest to each catch location (not polygon-averaged)
6. Recording the composite score and all sub-scores to `data/validation_results.csv`

**Pixel-level sampling** was critical — polygon-averaged sampling underestimated accuracy by ~16 percentage points because polygon averaging dilutes the score across areas that may include both high and low scoring cells.

### Optimization (Optuna)

`optimize_scoring.py` runs 200 trials using Tree-structured Parzen Estimator (TPE):
- **Objective**: 0.6 × mean_score + 0.2 × pct_above_70 + 0.2 × (1 - (1 - min_score)²)
- **Search space**: All 10 component weights, SST optimal/sigma, CHL optimal/sigma, front floor, intrusion threshold/baseline, shelf boost, east bonus, synergy factor
- **Constraint**: Weights must sum to 1.0

### Key calibration changes from initial to optimized:

| Parameter | Initial | Optimized | Impact |
|-----------|---------|-----------|--------|
| SST weight | 0.22 | 0.31 | SST is dominant signal |
| SST optimal | 23.5°C | 23.4°C | Minor adjustment |
| SST sigma | 2.0 | 2.97 | Broader tolerance |
| SSH weight | 0.15 | 0.19 | SLA more important than expected |
| Front weight | 0.12 | 0.09 | Fronts less critical |
| Front floor | 0.15 | 0.07 | Lower warm-water baseline |
| Convergence weight | 0.08 | 0.04 | Less signal in convergence |
| East bonus | 0.30 | 0.18 | Softer directional preference |
| Shelf boost | 0.50 | 0.53 | Slightly stronger shelf edge signal |
| Synergy | 0.40 | 0.36 | Slightly less interaction |
| CHL optimal | 0.20 | 0.22 mg/m³ | Minor adjustment |
| CHL sigma | 0.40 | 0.79 | Much broader CHL tolerance |

### Metrics that were tested but rejected:

| Metric | Result | Reason |
|--------|--------|--------|
| SST complexity (local variance) | r = 0.807 with fronts | Confounded — no independent signal after controlling for fronts |
| Bathymetric curvature | r = 0.043 | No correlation with catch success |
| Moon phase | No significant pattern | Too few catches per phase for statistical power |
| CHL inversion | Negative raw correlation | Location-confounded — not actionable |

---

## 10. Key Design Principles

1. **Additive weighted scoring with multiplicative gates**: Ocean conditions contribute additively (can compensate for each other), but depth is a hard gate — no amount of warm water makes 20m depth viable.

2. **Graceful degradation**: Missing data sources are excluded and weights re-normalized, rather than failing or using zeros.

3. **Physically-motivated interactions**: The bait trap synergy (current × convergence) and warm-water front floor encode understood biological mechanisms, not arbitrary curve fitting.

4. **Robust normalization**: Using percentile-based normalization (90th for fronts) rather than max-based ensures temporal consistency — a weak day shouldn't have its highest gradient scored as 100%.

5. **Resolution-aware processing**: Max-pooling for depth (where the highest value in a coarse cell matters) vs linear interpolation for shelf break (where average steepness matters).

6. **Dual scoring approach**: Spatial habitat suitability (where are conditions good today?) combined with temporal ML trajectory analysis (are conditions trending toward a catch pattern?) — addressing both the "where" and "when" of marlin prediction.

---

## 11. Limitations

- **Validation sample**: 71 catches across 53 dates is a modest sample. The catch database is biased toward successful trips — we cannot validate false positive rates (high-scoring areas with no fish).
- **Spatial resolution**: SST grid is ~5km. Sub-kilometer features (tide rips, localised bait schools) are not resolved.
- **Temporal lag**: Satellite data has 1-day latency. Conditions may shift between the data snapshot and the fishing day.
- **Species assumption**: Primarily calibrated for blue marlin. Striped and black marlin catches also validate well (included in the 71-catch dataset) but optimal parameters may differ slightly.
- **No lunar/tidal component**: Although moon phase and tidal state affect bite windows, the available data was insufficient to model these effects statistically.
- **ML model CV**: The prediction model achieves 61.1% mean accuracy with limited training data (11 events with sufficient lookback). Performance should improve as more catch events are added.
- **Forecast degradation**: Hotspot maps generated from ANFC forecast data (days +1 to +7) are inherently less accurate than observation-based maps due to model forecast uncertainty, particularly beyond day 3.
