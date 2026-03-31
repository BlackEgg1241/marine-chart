# MarLEEn Habitat Scoring Methodology

## A Spatial Habitat Suitability Model for Blue Marlin (*Makaira nigricans*) in the Perth Canyon, Western Australia

---

## Abstract

This document presents the theoretical framework, computational methodology, and empirical validation of a spatial habitat suitability model developed for predicting blue marlin distribution in the Perth Canyon region off southwestern Western Australia. The model integrates 14 oceanographic variables derived from satellite altimetry, ocean reanalysis products, and high-resolution bathymetry into a weighted composite score on a 0.02-degree (~2 km) grid. Feature scoring employs value-space Gaussian edge transforms, Lagrangian particle advection (FTLE), and vorticity-based shear detection to capture the dynamic boundary between the Leeuwin Current and the Capes Undercurrent -- the primary oceanographic driver of blue marlin aggregation at this site. Bayesian hyperparameter optimization (Optuna, 400 trials) on 25 unique GPS-verified catch locations yields a validation mean of 84% and 80% of catches scoring above 70% habitat suitability. The model is operationally deployed with 7-day forecasts from the CMEMS ANFC ocean model.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Study Area: The Perth Canyon System](#2-study-area-the-perth-canyon-system)
3. [Data Sources](#3-data-sources)
4. [Scoring Architecture](#4-scoring-architecture)
5. [Feature Variables](#5-feature-variables)
   - 5.1 [Sea Surface Temperature (SST)](#51-sea-surface-temperature)
   - 5.2 [SST Fronts and Front Corridors](#52-sst-fronts-and-front-corridors)
   - 5.3 [Chlorophyll-a Concentration](#53-chlorophyll-a-concentration)
   - 5.4 [Chlorophyll Curvature](#54-chlorophyll-curvature)
   - 5.5 [Sea Surface Height Anomaly (SSH/SLA)](#55-sea-surface-height-anomaly)
   - 5.6 [Current Shear (Relative Vorticity)](#56-current-shear-relative-vorticity)
   - 5.7 [Okubo-Weiss Parameter](#57-okubo-weiss-parameter)
   - 5.8 [Finite-Time Lyapunov Exponents (FTLE)](#58-finite-time-lyapunov-exponents)
   - 5.9 [Upwelling Edge Detection](#59-upwelling-edge-detection)
   - 5.10 [Salinity Front](#510-salinity-front)
   - 5.11 [Vertical Velocity](#511-vertical-velocity)
   - 5.12 [SST-CHL Bivariate Kernel](#512-sst-chl-bivariate-kernel)
   - 5.13 [Shelf Break Proximity](#513-shelf-break-proximity)
   - 5.14 [Bathymetric Contour Band System](#514-bathymetric-contour-band-system)
6. [Depth Gate](#6-depth-gate)
7. [Edge-Scoring Transform](#7-edge-scoring-transform)
8. [Lunar Phase Modifier](#8-lunar-phase-modifier)
9. [Post-Processing and Spatial Smoothing](#9-post-processing-and-spatial-smoothing)
10. [Optimization Framework](#10-optimization-framework)
11. [Validation](#11-validation)
12. [Discussion](#12-discussion)
13. [References](#13-references)

---

## 1. Introduction

Blue marlin (*Makaira nigricans*) are apex pelagic predators whose distribution is strongly governed by mesoscale oceanographic features including thermal fronts, current shear boundaries, and eddy structures (Graves et al., 2002; Rooker et al., 2012). In the Perth Canyon system off southwestern Western Australia, the interaction between the poleward-flowing Leeuwin Current (LC) and the equatorward-flowing Capes Undercurrent (LUC) creates a unique convergence of warm tropical surface water over cold, nutrient-rich subsurface water -- conditions that concentrate baitfish and attract pelagic predators (Rennie et al., 2007; Woo & Pattiaratchi, 2008).

This model addresses a practical question: **given today's ocean conditions, where within the Perth Canyon fishing zone are blue marlin most likely to be encountered?** Unlike species distribution models that predict presence/absence at coarse scales, this system produces a continuous 0-100% habitat suitability map at ~2 km resolution, updated daily with a 7-day forecast horizon.

The model was developed iteratively through 22 optimization cycles totalling approximately 5,000 Optuna trials, progressively incorporating new oceanographic variables, refining scoring geometry, and validating against 46 GPS-verified blue marlin catch records spanning 2000-2026.

### 1.1 Design Philosophy

A key empirical finding shapes the model architecture: **blue marlin catches occur at the edges of high-scoring zones, not at their peaks.** Analysis of 47 catches shows they sit at approximately 87% of the local peak score, with an average distance of 7 nm from the nearest peak. This is not a model artifact but reflects the biology -- marlin patrol the interface between offshore dynamic features (strong currents, deep shear) and the shelf-edge productivity zone (upwelling, chlorophyll gradients). The model therefore employs value-space Gaussian edge transforms that reward intermediate feature values characteristic of transition zones, rather than extreme values at feature cores.

![Composite habitat map showing catch locations at zone edges](Screenshots/composite/2016-01-16.png)
*Figure 1: Composite habitat suitability map for 16 January 2016. Catch locations (green markers) cluster at zone edges rather than peak scoring regions, consistent with the edge-hunting hypothesis. The graduated colour scale runs from 50% (cool) to 97% (warm).*

---

## 2. Study Area: The Perth Canyon System

### 2.1 Geographic Setting

The Perth Canyon is a submarine canyon system located approximately 45 nautical miles west of Fremantle, Western Australia (31.9S, 115.1E). The canyon incises the continental shelf from approximately 200 m depth to over 4,000 m, with the canyon head at approximately 115.08E, 31.92S. The study domain encompasses 114.8-115.5E, 31.6-32.2S, covering the accessible fishing zone from the canyon head to the Fish Aggregating Device (FAD) network east of the shelf break.

### 2.2 Oceanographic Setting

The Perth Canyon sits at the confluence of two opposing current systems that create the dynamic oceanographic environment exploited by pelagic predators:

**The Leeuwin Current (LC):** A warm (24-26C), low-salinity (34.5-35.0 PSU) surface current flowing southward along the Western Australian shelf edge. The LC is the dominant surface current in the region, transporting tropical water poleward and suppressing coastal upwelling that would otherwise occur on this eastern boundary coast (Feng et al., 2003). The LC jet runs strongest along the 200-500 m isobaths, with typical velocities of 0.2-0.5 m/s.

**The Capes Undercurrent (LUC):** A cold (14-19C), nutrient-rich subsurface current flowing equatorward at depths of 250-600 m. When the LUC encounters the canyon topography, it is deflected upward, creating cold upwelling plumes that penetrate through the warm LC surface layer (Rennie et al., 2007). This upwelling drives a nutrient-phytoplankton-baitfish-predator trophic cascade.

**The Interaction Zone:** The lateral shear boundary between the southward LC and northward LUC creates a persistent vorticity front at the canyon rim. This shear zone concentrates prey through several mechanisms:

1. **Mechanical aggregation:** Convergent flow at the shear boundary physically stacks baitfish
2. **Nutrient injection:** Upwelled LUC water fertilises the euphotic zone, driving chlorophyll blooms
3. **Thermal front formation:** The mixing of LC and LUC water creates persistent SST gradients of 0.15C/km or greater
4. **Eddy shedding:** Instabilities in the LC-LUC shear generate warm-core anticyclonic eddies that trap and transport prey

Analysis of 46 historical catches confirms this pattern: 78% occur in southward-flowing current (Leeuwin Current dominance), catches cluster at 200-500 m depth (the shelf-slope transition), and current direction becomes significantly more variable on the shallow shelf (standard deviation 27 vs 13 in deeper water) -- indicating the bathymetric steering that generates shear.

![Depth distribution showing catch concentrations at shelf edge](Screenshots/depth/2012-01-28.png)
*Figure 2: Depth scoring overlay for 28 January 2012. The graduated depth gate shows the transition from suppressed shallow water (<100 m) through the prime catch zone (200-500 m) to the deep offshore taper. Catch locations cluster along the 200-300 m contour at the shelf-slope transition.*

---

## 3. Data Sources

### 3.1 Ocean Model Data

All ocean state variables are sourced from the Copernicus Marine Environment Monitoring Service (CMEMS):

| Variable | Product | Resolution | Coverage |
|----------|---------|------------|----------|
| SST, Currents, SSH, MLD | `cmems_mod_glo_phy_my_0.083deg_P1D-m` (reanalysis) | 0.083 (~9 km) | 1993-present |
| SST (NRT) | `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` | 0.05 (~5 km) | Recent 2 years |
| Chlorophyll-a | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | 4 km | 1997-present |
| Salinity | `cmems_mod_glo_phy_my_0.083deg_P1D-m` (variable `so`) | 0.083 | 1993-present |
| Forecasts | CMEMS ANFC (Australian NearCoast Forecast) | 0.083 | 8 days lookback + 7 forward |

### 3.2 Bathymetry

High-resolution bathymetry is sourced from the Global Multi-Resolution Topography (GMRT) synthesis via REST API, providing sub-kilometre resolution in the canyon system. The bathymetry is used for the depth gate, shelf break gradient computation, contour band system, and rugosity (VRM) calculation.

### 3.3 Catch Data

46 GPS-verified blue marlin catch records from the DPI/GFAA tagging database (Export.csv), spanning 2000-2026. Coordinates are recorded in degrees-minutes (DDM) format and converted to decimal degrees for spatial analysis. Of the 46 catches, 25 represent unique GPS locations; the remaining 21 share locations with other catches (duplicate positions from the same fishing grounds on different dates).

---

## 4. Scoring Architecture

### 4.1 Grid System

All input data is interpolated to a common 0.02-degree master grid (~2.2 km at 32S) spanning the study domain. Interpolation uses `scipy.interpolate.RegularGridInterpolator` with linear method to preserve gradients while avoiding interpolation artifacts.

**Critical design decision:** Spatial gradients (Sobel operators for SST fronts, current shear, etc.) are computed at the *native* resolution of each data source (0.083 for currents, 4 km for CHL) *before* interpolation to the master grid. Computing gradients after interpolation produces artifact edges at the interpolation scale that do not correspond to real oceanographic features.

### 4.2 Weighted Sum Composite

The composite habitat score is computed as a normalised weighted sum:

$$S = \frac{\sum_{i=1}^{n} w_i \cdot s_i}{\sum_{i=1}^{n} w_i}$$

where $s_i \in [0,1]$ is the normalised score for feature $i$ and $w_i$ is its optimised weight. The denominator handles missing data gracefully -- if a feature is unavailable for a given date, its weight is excluded from the normalisation.

### 4.3 Multiplicative Modifiers

After the weighted sum, several multiplicative modifiers are applied sequentially:

1. **Depth gate** (Section 6): Suppresses shallow water, tapers deep water
2. **Shelf break boost**: `score *= (1 + 0.12 * shelf_score)`
3. **Multi-feature edge overlap**: Up to +15% for cells with 3+ transitioning features
4. **Band overlap system**: Graduated multiplier based on bathymetric contour proximity
5. **Lunar phase modifier** (Section 8): +5-10% at new moon
6. **Score-gradient reward**: +20% at transition zones within good habitat
7. **Feature line floor**: Minimum 0.62 on major SST fronts and CHL contours
8. **Gaussian spatial smoothing**: ~1 nm physical smoothing

### 4.4 Current Feature Weights (v22)

| Feature | Weight | Contribution |
|---------|--------|-------------|
| SST | 0.150 | 15% |
| Current Shear | 0.129 | 13% |
| CHL | 0.107 | 11% |
| CHL Curvature | 0.100 | 10% |
| SSH | 0.100 | 10% |
| Salinity Front | 0.100 | 10% |
| FTLE | 0.093 | 9% |
| Front Corridor | 0.064 | 6% |
| Vertical Velocity | 0.057 | 6% |
| Upwelling Edge | 0.043 | 4% |
| SST-CHL Bivariate | 0.021 | 2% |
| Okubo-Weiss | 0.021 | 2% |
| Shelf Break | 0.007 | 1% |
| SST Front | 0.007 | 1% |

---

## 5. Feature Variables

### 5.1 Sea Surface Temperature

**Ecological rationale:** SST is the primary environmental driver of blue marlin distribution globally (Graves et al., 2002). In the Perth Canyon, the Leeuwin Current delivers warm tropical water (24-26C) to latitudes where surface temperatures would otherwise be 18-20C. Blue marlin are tropical/subtropical predators with a thermal preference centred around 23-24C, though they tolerate a wider range on the warm side.

**Scoring function:** An asymmetric Gaussian centred on the optimal temperature:

$$S_{SST} = \exp\left(-\frac{1}{2}\left(\frac{T - T_{opt}}{\sigma(T)}\right)^2\right)$$

where:
- $T_{opt} = 23.75$C (optimised from catch data, consistent with literature range 22-25C)
- $\sigma(T) = 2.50$ if $T < T_{opt}$ (tighter below -- cold water is less tolerable)
- $\sigma(T) = 4.00$ if $T \geq T_{opt}$ (wider above -- marlin tolerate warm water well)

The asymmetric sigma reflects the biological reality that blue marlin are constrained by minimum temperature thresholds (thermoregulatory limits) but can exploit a wide range of warm water.

**Optimisation history:** Early iterations used $T_{opt} = 22.0$C with $\sigma = 1.0$ (very tight), which penalised 24% of catches that occurred in water slightly outside this narrow band. The v22 optimisation widened tolerance significantly, recognising that at this latitude, any water warmed by the Leeuwin Current is potentially suitable.

![SST scoring showing Gaussian temperature preference](Screenshots/sst/2015-02-28.png)
*Figure 3: SST suitability score for 28 February 2015. The Leeuwin Current appears as the warm (high-scoring) band along the shelf edge. The asymmetric Gaussian produces a gradual transition on the warm side but sharper cutoff in cooler offshore water. Catch locations sit within the warm LC plume.*

---

### 5.2 SST Fronts and Front Corridors

**Ecological rationale:** Thermal fronts -- sharp SST gradients exceeding 0.15C/km -- are among the strongest predictors of billfish aggregation (Podesta et al., 1993). In the Perth Canyon, fronts form where Leeuwin Current water meets cooler ambient water, where upwelling plumes reach the surface, and at eddy boundaries. Analysis of 47 catches shows 86% score above 0.70 on front corridor presence.

**SST Front computation:**

1. Gaussian smooth SST at native resolution ($\sigma = 1.5$ grid cells)
2. Apply Sobel edge detection: $G = \sqrt{G_x^2 + G_y^2}$
3. Normalise by 90th percentile of ocean cells
4. Widen influence with Gaussian blur ($\sigma = 3.5$ grid cells) -- catches sit *adjacent* to fronts, not on the sharpest gradient pixel
5. Modulate by SST suitability -- fronts in cold water do not score
6. Apply warm-water floor: if $S_{SST} > 0.6$, minimum front score of 0.07

**Front Corridor computation:**

Front corridors identify locations where fronts converge from multiple directions -- "pinch points" that funnel prey:

1. Threshold front score at 85th percentile to create binary front mask
2. Compute Euclidean distance transform from nearest front
3. Create proximity score: $P = \text{clip}(1 - d/4, 0, 1)$
4. Apply quadrant kernel ($5 \times 5$) to count how many cardinal directions contain front presence
5. Corridor score: $C = P \times \text{clip}((D_{count} - 1) / 2, 0, 1)$
6. Modulate by SST suitability

The corridor approach captures the observation that catches cluster at the intersection of multiple fronts, not along isolated frontal lines.

![Front corridor analysis showing multi-directional convergence](Screenshots/front_corridor/2016-01-16.png)
*Figure 4: Front corridor scoring for 16 January 2016. High scores (warm colours) indicate locations where SST fronts converge from multiple directions. Catches cluster at corridor intersections where prey is funnelled into concentrated feeding zones.*

---

### 5.3 Chlorophyll-a Concentration

**Ecological rationale:** Chlorophyll-a concentration serves as a proxy for primary productivity and, by trophic extension, baitfish availability. Blue marlin are not directly attracted to high chlorophyll water but to the interface between productive and oligotrophic water -- where baitfish aggregate at the edge of food-rich plumes. The Perth Canyon upwelling system drives chlorophyll to 0.3-0.6 mg/m3 in the plume core, while Leeuwin Current water is typically 0.05-0.10 mg/m3.

**Scoring function:** A log-space Gaussian centred on the transitional concentration:

$$S_{CHL} = \exp\left(-\frac{1}{2}\left(\frac{\log_{10}(\text{CHL}) - \log_{10}(C_{opt})}{\sigma_{CHL}}\right)^2\right)$$

where:
- $C_{opt} = 0.20$ mg/m3 (the oligotrophic/mesotrophic transition -- "clean blue" water at the productive edge)
- $\sigma_{CHL} = 0.45$ (log-space units, accommodating the wide natural range)

The log-space transform is essential because chlorophyll concentrations span several orders of magnitude (0.01 to >10 mg/m3), and the biological response is proportional to log-concentration, not absolute concentration.

![Chlorophyll scoring showing baitfish productivity zones](Screenshots/chl/2012-01-28.png)
*Figure 5: Chlorophyll-a suitability score for 28 January 2012. The scoring Gaussian peaks at 0.20 mg/m3, corresponding to the transition zone between the oligotrophic Leeuwin Current and productive upwelled water. Catches align with this transitional boundary.*

---

### 5.4 Chlorophyll Curvature

**Ecological rationale:** While chlorophyll concentration identifies the background productivity field, the *curvature* (second spatial derivative) of the chlorophyll field identifies dynamic features -- the edges of plankton patches, peninsulas of productive water extending into blue water, and pockets where nutrients pool. These geometric features of the productivity field represent active mixing zones where baitfish concentrate.

**Computation:**

1. Apply log transform to CHL field: $\text{CHL}_{log} = \log_{10}(\text{CHL})$
2. Gaussian smooth ($\sigma = 2.0$)
3. Compute Laplacian: $\nabla^2 \text{CHL} = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$
4. Take absolute value (both concave and convex features are interesting)
5. Normalise by 90th percentile
6. Apply edge-scoring transform (Section 7)

**Edge-scoring parameters:** centre = 0.50, width = 0.80. The wide Gaussian width means the curvature score is broadly tolerant, rewarding any cell with moderate curvature rather than requiring extreme values.

![CHL curvature showing plankton patch edges](Screenshots/chl_curvature/2015-02-08.png)
*Figure 6: Chlorophyll curvature analysis for 8 February 2015. High curvature values (warm colours) identify edges and geometric features of phytoplankton patches -- peninsulas, pockets, and filaments that indicate active mixing. These dynamic features concentrate baitfish at their margins.*

---

### 5.5 Sea Surface Height Anomaly

**Ecological rationale:** Sea level anomaly (SLA) from satellite altimetry is a powerful proxy for mesoscale oceanographic structure. Positive SLA indicates warm-core anticyclonic eddies -- rotating water masses that trap and transport warm water, nutrients, and prey organisms. The Leeuwin Current sheds warm-core eddies as it flows past the Perth Canyon, and these eddies are known aggregation sites for pelagic predators (Waite et al., 2007).

**Scoring function:** A blend of absolute and relative SLA components:

$$S_{SSH} = \alpha \cdot S_{abs} + (1 - \alpha) \cdot S_{rel}$$

where:
- $S_{abs} = \text{clip}(\text{SLA} / 0.12, 0, 1)$ -- positive SLA indicates warm water mass
- $S_{rel} = \text{clip}((\text{SLA} - \text{SLA}_{bg}) / 0.04, 0, 1)$ -- local anomaly above Gaussian-smoothed background
- $\alpha = 0.20$ (20% absolute, 80% relative)

The relative component dominates because absolute SLA exhibits a strong offshore gradient (deep water has higher SLA regardless of mesoscale features). The relative component isolates local eddy structure by subtracting a broad-scale background ($\sigma = 4$ grid cells).

**Note:** SSH is a *peak-scored* feature (catches at high values), not an edge-scored feature. This distinguishes it from most other variables where catches occur at transitions.

![SSH scoring showing eddy structure](Screenshots/ssh/2016-01-16.png)
*Figure 7: Sea surface height anomaly scoring for 16 January 2016. The relative SLA component highlights warm-core eddy structure (orange-red) embedded in the Leeuwin Current. Catches cluster on the high-SLA side of mesoscale features, consistent with warm-core eddy association.*

---

### 5.6 Current Shear (Relative Vorticity)

**Ecological rationale:** Relative vorticity -- the vertical component of the curl of horizontal velocity -- is the mathematical signature of current shear. In the Perth Canyon, the southward Leeuwin Current at the surface overlies the northward Capes Undercurrent at depth, creating a persistent lateral shear zone along the canyon rim. This shear boundary is the primary physical mechanism concentrating baitfish for blue marlin predation.

The shear zone acts as a "bait wall" through three mechanisms:
1. **Mechanical trapping:** Opposing flows create convergent secondary circulation that pins organisms at the boundary
2. **Nutrient injection:** Undercurrent water entrained into the shear zone brings nutrients to the euphotic zone
3. **Ecotone effect:** The shear boundary marks the transition between two water masses, maximising prey diversity

**Computation:**

Vorticity is computed at native current resolution (0.083) before interpolation:

$$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

where $u$ and $v$ are the zonal and meridional velocity components. The absolute vorticity $|\zeta|$ is used because both cyclonic and anticyclonic shear are biologically relevant.

Post-processing:
1. Gaussian smooth ($\sigma = 1.0$ grid cells)
2. Normalise by 90th percentile
3. Apply depth-dependent modifier: zero at <60 m, linear ramp to full at 300 m
4. Apply edge-scoring transform (centre = 0.80, width = 0.60)

**Depth modifier rationale:** Current shear at depths <60 m is dominated by wind-driven surface friction, not the Leeuwin/Undercurrent interaction. The depth ramp ensures only bathymetrically-steered shear is scored.

**Weight:** 12.9% -- the second-highest contributor, reflecting the central importance of the LC/LUC interaction.

![Current shear showing Leeuwin/Undercurrent boundary](Screenshots/current_shear/2017-02-04.png)
*Figure 8: Current shear (vorticity) for 4 February 2017. High vorticity values (warm colours) trace the Leeuwin Current / Undercurrent boundary along the canyon rim and shelf edge. The depth modifier suppresses shallow nearshore shear that is wind-driven rather than bathymetrically steered. Catches cluster at the shear boundary where baitfish are mechanically concentrated.*

---

### 5.7 Okubo-Weiss Parameter

**Ecological rationale:** The Okubo-Weiss parameter $W$ discriminates between strain-dominated regions (fronts, filaments) where passive tracers (and baitfish) accumulate, and vorticity-dominated regions (eddy cores) where rotation dominates. This is distinct from simple vorticity (Section 5.6) which cannot distinguish between the interior of a coherent eddy (stable rotation, no aggregation) and a deformation zone between eddies (strong shear, active aggregation).

**Computation:**

$$W = S_n^2 + S_s^2 - \omega^2$$

where:
- $S_n = \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}$ (normal strain rate)
- $S_s = \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}$ (shear strain rate)
- $\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ (relative vorticity)

Interpretation:
- $W > 0$: **Strain-dominated** -- deformation zones, fronts, filaments. Passive tracers (baitfish) are stretched and accumulated along convergent axes.
- $W < 0$: **Vorticity-dominated** -- eddy cores. Rotation dominates; organisms are trapped but not actively concentrated.
- $W \approx 0$: Background field with no dominant dynamic feature.

Only positive $W$ values are scored (strain-dominated regions), normalised by the 90th percentile. Edge-scoring (centre = 0.30, width = 0.65) peaks at moderate strain values, consistent with catches occurring at the *margins* of strain zones rather than at maximum deformation.

![Okubo-Weiss parameter showing strain vs rotation](Screenshots/okubo_weiss/2016-01-16.png)
*Figure 9: Okubo-Weiss parameter for 16 January 2016. Strain-dominated regions (positive W, warm colours) mark fronts and deformation zones where baitfish accumulate. Vorticity-dominated regions (cool/absent) correspond to eddy cores where stable rotation prevents aggregation. The distinction is ecologically critical -- marlin hunt the boundaries, not the centres.*

---

### 5.8 Finite-Time Lyapunov Exponents (FTLE)

**Ecological rationale:** FTLE analysis identifies **Lagrangian Coherent Structures (LCS)** -- material transport barriers in the ocean that act as "invisible walls" concentrating passively drifting organisms. Unlike Eulerian diagnostics (SST fronts, vorticity), FTLE captures the *actual paths* water parcels follow over time, revealing accumulation lines that may not be visible in instantaneous velocity snapshots. This is the mathematical framework for understanding **bait stacking** -- the passive accumulation of planktonic organisms and larval fish at transport barriers, which in turn attracts baitfish schools and their pelagic predators.

**Theoretical framework:**

Consider a fluid parcel at position $\mathbf{x}_0$ at time $t_0$. After advection for time $T$, it arrives at position $\mathbf{x}(t_0 + T; \mathbf{x}_0, t_0)$. The flow map gradient (deformation gradient tensor) is:

$$\mathbf{F} = \frac{\partial \mathbf{x}(t_0 + T)}{\partial \mathbf{x}_0}$$

The right Cauchy-Green strain tensor captures the total stretching:

$$\mathbf{C} = \mathbf{F}^T \mathbf{F}$$

The FTLE field is defined as:

$$\text{FTLE}(\mathbf{x}_0, t_0, T) = \frac{1}{|T|} \ln \sqrt{\lambda_{max}(\mathbf{C})}$$

where $\lambda_{max}$ is the maximum eigenvalue of $\mathbf{C}$.

**Implementation:**

1. **Particle seeding:** Initialise a regular grid of passive tracers across the study domain
2. **RK4 integration:** Advect particles forward through 3 consecutive daily velocity fields using fourth-order Runge-Kutta:

$$\mathbf{k}_1 = \Delta t \cdot \mathbf{v}(\mathbf{x}_n, t_n)$$
$$\mathbf{k}_2 = \Delta t \cdot \mathbf{v}(\mathbf{x}_n + \tfrac{1}{2}\mathbf{k}_1, t_n + \tfrac{1}{2}\Delta t)$$
$$\mathbf{k}_3 = \Delta t \cdot \mathbf{v}(\mathbf{x}_n + \tfrac{1}{2}\mathbf{k}_2, t_n + \tfrac{1}{2}\Delta t)$$
$$\mathbf{k}_4 = \Delta t \cdot \mathbf{v}(\mathbf{x}_n + \mathbf{k}_3, t_n + \Delta t)$$
$$\mathbf{x}_{n+1} = \mathbf{x}_n + \tfrac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

3. **Deformation gradient:** Compute $\mathbf{F}$ from finite differences of final vs initial positions
4. **Strain tensor:** $\mathbf{C} = \mathbf{F}^T \mathbf{F}$, solve for maximum eigenvalue
5. **FTLE field:** Normalise by 95th percentile (FTLE ridges are sparse by nature)

**Bait Stacking Mechanism:**

FTLE ridges represent *attracting* material lines where fluid parcels converge from both sides. Planktonic organisms, larval fish, and small baitfish that are passively advected by currents accumulate at these ridges over multi-day timescales. The result is a concentration of prey biomass along narrow bands that may extend for tens of kilometres. Blue marlin, as active predators, exploit these prey concentration lines.

This is distinct from convergence (Section 5.11), which captures *instantaneous* convergent flow. FTLE captures the *time-integrated* effect, revealing accumulation structures that persist even when instantaneous convergence is weak.

**Weight:** 9.3% -- the fourth-highest contributor, reflecting the importance of Lagrangian transport in prey aggregation.

![FTLE showing Lagrangian transport barriers](Screenshots/ftle/2016-01-16.png)
*Figure 10: Finite-Time Lyapunov Exponents for 16 January 2016. High FTLE ridges (warm colours) mark Lagrangian Coherent Structures -- transport barriers where passively drifting organisms accumulate ("bait stacking"). These ridges often align with, but are not identical to, Eulerian features like SST fronts. Catches cluster near FTLE ridges where prey has been concentrated by multi-day transport.*

---

### 5.9 Upwelling Edge Detection

**Ecological rationale:** Canyon upwelling brings cold, nutrient-rich Undercurrent water to the surface, driving productivity blooms. However, the cold upwelling core itself is too cold for blue marlin. Catches occur on the *warm side* of the upwelling boundary -- where Leeuwin Current water meets the cool, productive upwelling plume. This interface offers the optimal combination: warm water within the marlin's thermal tolerance AND elevated baitfish concentrations from the adjacent upwelling.

**Computation:**

1. **Identify upwelling core:**
   - SST cooling signature: $S_{cool} = \text{clip}((\bar{T}_{ocean} - T) / 1.5, 0, 1)$
   - CHL enrichment: $S_{CHL} = \text{clip}((\text{CHL} - \text{CHL}_{med}) / (2 \cdot \text{CHL}_{med}), 0, 1)$
   - Combined core: $U_{core} = S_{cool} \times S_{CHL}$ (requires BOTH cooling AND enrichment)

2. **MLD reinforcement:** Shallow mixed layer depth indicates strong stratification break from upwelling:
   $U_{core} \times (0.5 + 0.5 \cdot \text{clip}(1 - (\text{MLD} - 15)/35, 0, 1))$

3. **Edge detection:** Sobel gradient of the smoothed upwelling core
4. **Widen influence:** Gaussian blur ($\sigma = 4.0$) -- catches sit adjacent to the edge, not on it
5. **Warm-water filter:** Multiply by SST suitability score -- only the warm side scores

**Edge-scoring:** centre = 0.75, width = 0.10. The narrow width concentrates scoring tightly around the optimal edge intensity, reflecting the precision with which catches track the upwelling boundary.

![Upwelling edge showing warm-side catch association](Screenshots/upwelling_edge/2011-02-26.png)
*Figure 11: Upwelling edge detection for 26 February 2011. The algorithm identifies boundaries where cool, productive upwelling water meets warm Leeuwin Current water. Scoring is filtered to the warm side only, matching the observed catch pattern of marlin hunting the productive edge while remaining in thermally suitable water.*

---

### 5.10 Salinity Front

**Ecological rationale:** The Leeuwin Current carries low-salinity tropical water (34.5-35.0 PSU) into the higher-salinity Indian Ocean (35.5-36.0 PSU). This halocline gradient provides a robust marker of the LC boundary that persists even when SST fronts are obscured by solar heating in summer. Salinity fronts serve as an independent tracer of the water mass boundary that generates the shear and productivity features exploited by pelagic predators.

**Computation:**

1. Gaussian smooth salinity field ($\sigma = 1.5$)
2. Sobel gradient: $G_{sal} = \sqrt{G_x^2 + G_y^2}$
3. Normalise by 90th percentile
4. Widen influence with Gaussian blur ($\sigma = 4.0$)
5. Re-normalise by 90th percentile

**Note:** Salinity front is *peak-scored* (not edge-scored). Unlike vorticity or curvature where catches sit at intermediate values, catches associate with the strongest salinity gradients -- the sharpest water mass boundary.

**Weight:** 10.0% -- equal to SSH and CHL curvature, reflecting its importance as an LC boundary marker.

![Salinity front showing halocline gradient](Screenshots/salinity_front/2015-03-02.png)
*Figure 12: Salinity front scoring for 2 March 2015. The halocline gradient (warm colours) traces the Leeuwin Current boundary. In summer, when solar heating obscures SST fronts, salinity gradients provide the most robust marker of the water mass interface. Catches align with the strongest salinity gradients.*

---

### 5.11 Vertical Velocity

**Ecological rationale:** Vertical water motion drives the upwelling-downwelling cycle that concentrates nutrients and prey. Upwelling zones bring nutrients to the surface; downwelling zones concentrate floating prey. The vertical velocity field complements the upwelling edge detection (Section 5.9) by capturing the *dynamic* upwelling signal rather than the *thermal/chemical* signature.

**Computation:**

Vertical velocity is not directly available from CMEMS products. It is derived from the continuity equation for an incompressible fluid:

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

Rearranging:

$$\frac{\partial w}{\partial z} \approx -\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = -\nabla_H \cdot \mathbf{u}$$

Positive horizontal divergence implies upwelling (surface water moving apart, replaced by water from below). The proxy score is:

$$S_w = \text{clip}\left(\frac{\nabla_H \cdot \mathbf{u}}{0.005}, 0, 1\right)$$

Only positive divergence (upwelling) is scored. The threshold of $5 \times 10^{-3}$ s$^{-1}$ corresponds to significant vertical motion at the mesoscale.

![Vertical velocity showing upwelling proxy](Screenshots/vertical_velocity/2016-01-16.png)
*Figure 13: Vertical velocity proxy for 16 January 2016. Positive horizontal divergence (warm colours) indicates upwelling -- locations where subsurface water is being drawn to the surface. These upwelling centres drive the nutrient injection that supports the baitfish food chain.*

---

### 5.12 SST-CHL Bivariate Kernel

**Ecological rationale:** While SST and chlorophyll are scored independently, their *interaction* carries additional information. The optimal blue marlin habitat is warm, moderately productive water ("clean blue at the productive edge"). Neither warm-barren water (Leeuwin Current core, high SST but low CHL) nor cold-productive water (upwelling core, high CHL but low SST) is optimal. A 2D Gaussian kernel on the SST-CHL plane captures this interaction without needing to increase either individual feature's weight.

**Scoring function:**

The bivariate score penalises conditions that are individually suitable but jointly unsuitable:

$$S_{biv} = \exp\left(-\frac{1}{2}\left[\frac{(T - T_{opt})^2}{\sigma_T^2} + \frac{(\log C - \log C_{opt})^2}{\sigma_C^2}\right]\right)$$

This 2D Gaussian peaks at the intersection of optimal SST and optimal CHL, falling off in all four quadrants. The +0.50 correlation with catch scores (second highest of all features) confirmed its predictive value, leading to re-activation in v22.

**Weight:** 2.1% -- modest but significant. The bivariate kernel provides a refinement that helps distinguish the interface zone from either water mass individually.

![SST-CHL bivariate showing optimal habitat overlap](Screenshots/sst_chl_bivariate/2012-01-28.png)
*Figure 14: SST-CHL bivariate kernel for 28 January 2012. The highest scores (warm colours) occur where warm SST and moderate CHL coincide -- the transitional zone between the warm, oligotrophic Leeuwin Current and the cool, productive upwelling plume. This captures the "clean blue at the productive edge" habitat preference.*

---

### 5.13 Shelf Break Proximity

**Ecological rationale:** Blue marlin catches cluster strongly around the 200-300 m depth contour -- the shelf-slope transition where the Leeuwin Current jet runs strongest and bathymetric steering generates shear. Analysis of 47 catches shows a median depth of 229 m, with 40% at 100-200 m and 67% within 3 nm of the 200 m contour.

**Computation:**

A hybrid of bathymetric gradient (Sobel) and depth proximity (Gaussian):

$$S_{shelf} = (1 - \beta) \cdot S_{gradient} + \beta \cdot S_{proximity}$$

where:
- $S_{gradient}$: Sobel gradient of bathymetry, normalised by 90th percentile. Peaks at steepest slope (typically 300-500 m on the canyon wall).
- $S_{proximity} = \exp\left(-\frac{1}{2}\left(\frac{d - d_{target}}{\sigma_d}\right)^2\right)$: Gaussian centred on target depth.
- $\beta = 0.80$ (80% proximity, 20% gradient)
- $d_{target} = 270$ m, $\sigma_d = 50$ m

The high proximity blend ($\beta = 0.80$) was introduced because catches sit on the *flat shelf lip*, not the steepest canyon wall slope. The Sobel gradient peak (300-500 m) is 2-3 nm offshore of the primary catch zone. The tight sigma (50 m) concentrates scoring around the validated catch depth.

![Shelf break showing catch clustering at shelf lip](Screenshots/shelf_break/2012-01-28.png)
*Figure 15: Shelf break scoring for 28 January 2012. The blend of bathymetric gradient and depth proximity concentrates scoring around the 270 m contour where catches cluster. The proximity component (80% weight) pulls the scoring peak shoreward from the steepest slope to the actual catch zone.*

---

### 5.14 Bathymetric Contour Band System

**Ecological rationale:** Beyond the shelf break itself, multiple bathymetric contours (200 m, 300 m, 500 m, 700 m, 1000 m, etc.) influence prey distribution. Canyon topography steers currents along isobaths, creating persistent current corridors that concentrate baitfish. The band system scores proximity to each contour independently, then uses overlap count as a habitat quality indicator.

**Computation:**

For each depth contour $d_m$ with weight $w_m$:

1. **Detect contour band:** Identify cells within depth tolerance: $d_m - \text{tol}_{shore} \leq d \leq d_m + \text{tol}_{deep}$
   - $\text{tol}_{shore} = \max(20, d_m \times 0.30)$
   - $\text{tol}_{deep} = \max(30, d_m \times 0.30)$
2. **Soft taper:** Replace hard mask with gradual onset at $d_m \times 0.65$
3. **Distance-weighted band score:** Decay from contour centreline
4. **Accumulate:** Track band count and mean band score per cell

**Band overlap boost:** Applied as a graduated multiplier:
- 0 bands: $\times 0.55$ (featureless open water -- suppressed)
- 1 band: $\times 0.80$ (single contour association -- mild suppression)
- 2+ bands: $\times 1.0$ + additive boost for convergence

This system ensures that cells near multiple bathymetric features (e.g., canyon head where multiple isobaths converge) receive higher scores than cells along a single, isolated contour.

---

## 6. Depth Gate

The depth gate is a multiplicative modifier applied to the composite score that restricts habitat suitability to biologically plausible depths. Blue marlin are surface-oriented predators that feed in the upper water column, but they are associated with specific bathymetric zones:

| Depth Zone | Multiplier | Justification |
|------------|-----------|---------------|
| < 80 m | 0.00 | No blue marlin catches. Nearshore, too shallow for LC interaction. |
| 80 - 180 m | 0.50 - 1.00 | Linear ramp. Some catches in this zone; shelf-edge habitat. |
| 180 - 500 m | 1.00 | Prime zone: 65% of catches. Full scoring. |
| 500 - 1500 m | 1.00 - 0.80 | Catches decrease. Gradual taper. |
| > 1500 m | 0.95 | Deep offshore. Effectively no penalty (v22: depth gate relaxed after analysis showed larger-scale features extend beyond canyon structure). |

**Design evolution:** Early versions applied aggressive deep-water suppression (floor = 0.40 at 2000+ m), reflecting an assumption that catches only occur near the canyon. Analysis revealed a *triple penalty* -- the depth gate, shelf break additive weight, and shelf multiplicative boost each independently suppressed deep water, creating a compound ~45% penalty at 2000 m. Since many oceanographic features (eddies, FTLE ridges, salinity fronts) extend well beyond the canyon, v22 relaxed the depth gate to floor = 0.95, effectively deferring depth-based discrimination to the shelf break feature itself.

---

## 7. Edge-Scoring Transform

### 7.1 Motivation

A fundamental observation from catch analysis is that blue marlin do not occupy the *peaks* of oceanographic features but their *edges*. Catches sit at approximately 87% of the local peak score, in zones of steep spatial gradient. The fish patrol the interface between feature extremes, exploiting the ecotone between different water masses.

For four features -- Okubo-Weiss, upwelling edge, current shear, and chlorophyll curvature -- a value-space Gaussian transform is applied that peaks at an intermediate raw value rather than at the maximum:

$$S_{edge}(v) = \exp\left(-\frac{1}{2}\left(\frac{v - c}{\sigma}\right)^2\right)$$

where $v$ is the raw normalised feature value, $c$ is the "sweet spot" centre, and $\sigma$ is the width.

### 7.2 Parameters

| Feature | Centre | Width | Interpretation |
|---------|--------|-------|----------------|
| Okubo-Weiss | 0.30 | 0.65 | Moderate strain, not extreme deformation |
| Upwelling Edge | 0.75 | 0.10 | Precisely at the edge (narrow peak) |
| Current Shear | 0.80 | 0.60 | High but not maximum shear (boundary, not core) |
| CHL Curvature | 0.50 | 0.80 | Broadly tolerant of any curvature level |

### 7.3 Contrast with Peak-Scored Features

Two features are explicitly *peak-scored* (catches at maximum values):
- **SSH:** Catches associate with the highest SLA values (warm-core eddy interiors)
- **Salinity front:** Catches associate with the strongest halocline gradients

This biological distinction -- some features are edge-associated, others are peak-associated -- is encoded directly in the scoring architecture rather than being left to weight optimisation alone.

---

## 8. Lunar Phase Modifier

### 8.1 Theory: Diel Vertical Migration (DVM)

The deep scattering layer (DSL) -- a dense aggregation of mesopelagic organisms -- undergoes diel vertical migration, ascending toward the surface at dusk and descending at dawn. Lunar illumination modulates this behaviour:

- **New moon (dark):** The DSL rises higher and stays longer in the upper water column, compressed against the thermocline. This concentrates prey biomass in the surface layer accessible to marlin. Historical catch data shows captures at 261 m average depth with MLD of 17 m during new moon phases.

- **Full moon (bright):** Moonlight inhibits DSL ascent, keeping prey dispersed deeper in the water column. Captures average 392 m depth with MLD of 23 m.

Additionally, spring tides (which occur at both new and full moon) intensify tidal currents, enhancing shear zones and upwelling at the canyon.

### 8.2 Implementation

The lunar cycle is modelled as a sinusoidal illumination function:

$$I_{moon} = \frac{1}{2}\left(1 - \cos\left(\frac{2\pi \cdot d_{cycle}}{29.53}\right)\right)$$

where $d_{cycle}$ is the phase position within the 29.53-day synodic cycle (0 = new moon, 0.5 = full moon). The reference new moon is 6 January 2000.

The habitat modifier:

$$M_{lunar} = 1 + \lambda \cdot (1 - I_{moon})$$

where $\lambda = 0.10$ (10% maximum boost range). This yields:
- New moon: $\times 1.10$ (maximum boost -- prey compressed at surface)
- Full moon: $\times 1.00$ (no boost -- prey dispersed)

The effect is subtle (5-10%) but consistent with empirical observations of increased catch rates during darker phases.

---

## 9. Post-Processing and Spatial Smoothing

### 9.1 Multi-Feature Edge Overlap

Analysis shows catches have 24% more features simultaneously transitioning than peak locations (3.81 vs 3.07 features with high spatial gradients). Cells where 3+ features exhibit steep transitions receive a multiplicative boost of up to +15%.

For each of 9 diagnostic features, spatial gradient magnitude is computed and thresholded at the 75th percentile. Cells exceeding this threshold in 3+ features are boosted:

$$M_{edge} = \text{clip}(1 + 0.05 \times (n_{transitioning} - 2), 1.0, 1.15)$$

### 9.2 Score-Gradient Reward

Catches occur at zones of steep spatial gradient in the composite score itself (~87% of local peak, with 72% showing gradient >5 points per 2 nm). The score-gradient reward boosts transition zones:

1. Compute spatial gradient of smoothed composite score
2. Normalise by 99th percentile
3. Apply only where composite score > 0.5 (avoid boosting poor habitat)
4. Multiplicative boost: $M_{grad} = 1 + 0.20 \times G_{norm}$

### 9.3 Feature Line Floor

Cells on major oceanographic feature lines (SST fronts, CHL 0.15 mg/m3 contour) are lifted to a minimum score of 0.62. This prevents important structural features from being suppressed by low values in one or two other variables. Analysis shows 87% of ocean cells are on at least one feature line, and catches preferentially associate with these features.

### 9.4 Spatial Smoothing

Final Gaussian smoothing at approximately 1 nm physical scale ($\sigma \approx 0.015 / \Delta_{grid}$ degrees) removes grid-scale noise while preserving mesoscale structure. Land cells and invalid data are masked before smoothing to prevent contamination.

---

## 10. Optimization Framework

### 10.1 Optuna Bayesian Optimization

The model's approximately 50 tuneable parameters are optimised using Optuna's Tree-structured Parzen Estimator (TPE) sampler. The optimisation maximises a multi-component objective function:

$$J = \bar{S}_{catch} + B_{coverage} + L_{lift} + G_{gradient} + R_{ratio} + P_{percentile}$$

Components:
- $\bar{S}_{catch}$: Mean habitat score at catch locations
- $B_{coverage}$: Percentage of catches scoring $\geq$ 70%
- $L_{lift}$: Score at catches minus ocean background mean (discrimination)
- $G_{gradient}$: Mean spatial gradient at catch locations (edge structure)
- $R_{ratio}$: Catch score / ocean mean ratio
- $P_{percentile}$: Median catch percentile vs all ocean cells

### 10.2 Training Set

v22 uses **unique-only training**: 25 catches with unique GPS coordinates. The 21 catches sharing locations with other records are excluded from the objective function to prevent spatial bias (repeatedly rewarding the model for scoring well at the same physical location).

### 10.3 Optimization History

| Version | Trials | Key Change | Objective | Validation Mean |
|---------|--------|------------|-----------|-----------------|
| v4-v5 | 200 | Weight tuning only | -- | 80% |
| v6 | 200 | Edge-aligned objective | 28.1 | 80% |
| v7 | 200 | +rugosity, bivariate, FTLE | 27.5 | 84% |
| v8 | 200 | +stratification, thermocline | 28.4 | 84% |
| v11 | 200 | FTLE fixed, all features | 29.3 | -- |
| v17 | 200 | Value-space edge scoring | 30.3 | -- |
| v18 | 2000 | 16 workers, geometry rework | 33.7 | 88% |
| v22 | 400 | Unique-only, +bivariate, wider SST | 25.4 | 84% |

**Note:** The objective function changed between versions, so raw objective values are not directly comparable. v22's lower absolute objective reflects the unique-only training (harder problem with fewer catches) and stricter validation criteria.

---

## 11. Validation

### 11.1 Two-Tier Scoring

Validation uses a two-tier approach to handle the 21 catches that share GPS coordinates with other catches:

- **Unique GPS (24 catches):** Score is sampled at the exact grid cell containing the catch coordinates
- **Duplicate locations (22 catches):** Score is the maximum within a 1 nm radius of the catch coordinates (proximity max)

This prevents the validation from penalising the model for imprecise GPS coordinates at frequently-fished locations.

### 11.2 Results (v22)

| Metric | Value |
|--------|-------|
| Mean score at catch locations | 84% |
| Median score | 86% |
| Catches scoring >= 70% | 80% (37/46) |
| Catches within a hotspot zone | 87% (40/46) |
| Minimum score | 55% |
| Maximum score | 95% |
| Unique GPS mean | 83% |
| Duplicate location mean | 86% |

### 11.3 Seasonal Backtest

Independent validation via 950 weekly habitat scores (2010-2026) confirms the model correctly identifies marlin season without using catch data:
- Peak scores: March-April (91-92% zone mean) -- coincides with peak marlin season
- Trough scores: August-September (63-69%) -- off-season

![Seasonal backtest showing catch alignment](data/backtest/catches_vs_habitat.png)
*Figure 16: Habitat score time series from the 15-year weekly backtest overlaid with actual catch dates. Peak habitat scores align with the known January-April blue marlin season in Perth.*

![Season tracker](data/backtest/season_tracker.png)
*Figure 17: Seasonal tracker showing the annual cycle of habitat suitability. The model correctly reproduces the warm-season peak without any explicit seasonal weighting.*

---

## 12. Discussion

### 12.1 The Edge-Hunting Paradigm

The most significant finding from this work is that blue marlin habitat cannot be understood through a "peak-seeking" framework. The fish do not occupy the locations where any single oceanographic variable is maximised. Instead, they exploit the *ecotone* between the warm, oligotrophic Leeuwin Current and the cool, productive upwelling/shear zone. This interface offers:

1. **Thermal suitability** (warm enough for the marlin's metabolic requirements)
2. **Prey availability** (adjacent to productivity-driven baitfish concentrations)
3. **Hydrodynamic structure** (shear boundaries and transport barriers that concentrate prey)

The edge-scoring transform (Section 7) encodes this understanding directly into the scoring architecture, peaking at intermediate feature values rather than extremes.

### 12.2 The Leeuwin Current as Habitat Engine

The model's weight structure reflects the central importance of the LC/LUC interaction. The top three weighted features -- current shear (13%), SST (15%), and salinity front (10%) -- all trace aspects of the Leeuwin Current boundary. Including FTLE (9%), SSH (10%), and upwelling edge (4%), features directly related to the LC system account for over 60% of the total weight.

This is consistent with the oceanographic understanding of the Perth Canyon: the Leeuwin Current is the primary driver of warm-water pelagic habitat at this latitude, and its interaction with the canyon topography creates the dynamic features that concentrate prey.

### 12.3 Lagrangian vs Eulerian Perspectives

The inclusion of FTLE (Section 5.8) at 9.3% weight represents a philosophically different approach from the other features. While SST, CHL, and vorticity are *Eulerian* diagnostics (what does the ocean look like at this instant?), FTLE captures the *Lagrangian* transport history (where has the water been, and where are organisms being concentrated?). The bait stacking mechanism -- passive accumulation at Lagrangian transport barriers -- operates over multi-day timescales and may not be visible in any single snapshot of ocean state.

### 12.4 Limitations

1. **Spatial resolution:** The 0.083 (9 km) native resolution of current and SSH data is coarser than the mesoscale features (1-10 km) that likely drive catch-scale habitat selection. IMOS nearshore products (0.02) improve SST resolution but are not available for all variables.

2. **Temporal resolution:** Daily-mean fields may not capture submesoscale features that develop and dissipate within hours. Tidal effects are averaged out.

3. **Training set size:** 25 unique catch locations, while spanning 26 years, is a small sample for a 50-parameter model. The risk of overfitting is mitigated by the Bayesian optimisation framework and the physical constraints on parameter ranges, but remains a concern.

4. **Depth integration:** The model uses surface or near-surface fields exclusively. Subsurface structure (thermocline depth, undercurrent position) is inferred from surface signatures rather than measured directly.

5. **Species specificity:** The model is calibrated exclusively for blue marlin. Other pelagic species (striped marlin, wahoo, yellowfin tuna) likely have different habitat preferences and would require separate calibration.

---

## 13. References

Feng, M., Meyers, G., Pearce, A., & Wijffels, S. (2003). Annual and interannual variations of the Leeuwin Current at 32S. *Journal of Geophysical Research*, 108(C11), 3355.

Graves, J.E., Luckhurst, B.E., & Prince, E.D. (2002). An evaluation of pop-up satellite tags for estimating post-release survival of blue marlin. *Fishery Bulletin*, 100(1), 134-142.

Podesta, G.P., Browder, J.A., & Hoey, J.J. (1993). Exploring the association between swordfish catch rates and thermal fronts on US longline grounds in the western North Atlantic. *Continental Shelf Research*, 13(2-3), 253-277.

Rennie, S.J., Pattiaratchi, C.B., & McCauley, R.D. (2007). Eddy formation through the interaction between the Leeuwin Current, Leeuwin Undercurrent and topography. *Deep Sea Research Part II*, 54(8-10), 818-836.

Rooker, J.R., Dance, M.A., Wells, R.J.D., Quigg, A., Hill, R.L., Appeldoorn, R.S., ... & Stunz, G.W. (2012). Impacts of the Deepwater Horizon oil spill on pelagic Sargassum communities: short-term and long-term effects. *Marine Environmental Research*, 79, 105-116.

Waite, A.M., Thompson, P.A., Pesant, S., Feng, M., Beckley, L.E., Domingues, C.M., ... & Koslow, J.A. (2007). The Leeuwin Current and its eddies: An introductory overview. *Deep Sea Research Part II*, 54(8-10), 789-796.

Woo, M., & Pattiaratchi, C. (2008). Hydrography and water masses off the western Australian coast. *Deep Sea Research Part I*, 55(9), 1090-1104.

---

*Document generated: 26 March 2026*
*Model version: v22 (Optuna 400-trial, unique-only training)*
*Validation: mean 84%, median 86%, 80% >= 70%*
