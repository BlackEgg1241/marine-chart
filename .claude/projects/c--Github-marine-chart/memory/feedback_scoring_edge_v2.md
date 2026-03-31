---
name: Edge-hunting scoring rollback reference (v2)
description: Pre-edge-hunting code snapshots for MLD, shelf_break, and current scoring — saved before applying edge-gradient modifications on 2026-03-22
type: feedback
---

## Pre-change scoring code (2026-03-22)

### MLD (line ~1217-1233)
```python
# 6. MLD score — shallower = better (marlin compressed at surface)
mld_score = np.clip(1.0 - (mld_data - 20) / 80, 0, 1)
# Score: 1.0 at MLD<20m, 0.5 at 50m, 0 at 100m
_add_score("mld", mld_score)
```
Pure value-based: shallow MLD = high score, deep MLD = low score.

### Shelf break (line ~1090-1119)
```python
# Sobel gradient on raw bathy
bathy_filled = bathy.copy()
bathy_filled[np.isnan(bathy_filled)] = 0
dgx = sobel(bathy_filled, axis=1)
dgy = sobel(bathy_filled, axis=0)
depth_gradient = np.sqrt(dgx**2 + dgy**2)
depth_gradient[np.isnan(bathy)] = 0
shelf_break = _interp_to_grid(depth_gradient, b_lons, b_lats)
shelf_score = np.clip(shelf_break / 100, 0, 1)
_add_score("shelf_break", shelf_score)
```
Already gradient-based on bathymetry itself, but scores the steepest slope, not the edge of the slope.

### Current (line ~1256-1334)
```python
# Speed score: 0 at <0.03 m/s, 1.0 at >=0.20 m/s
speed_score = np.clip((cur_speed - 0.03) / 0.17, 0, 1)
# East score: 0 at uo<=0, 1.0 at uo>=0.15 m/s
east_score = np.clip(uo_data / 0.15, 0, 1)
# Upstream SST: sample ~20km upstream, Gaussian score
upstream_temp_score = np.exp(-0.5 * ((upstream_sst - optimal_temp) / sigma_map_up) ** 2)
# Combined: speed × upstream_temp × eastward bonus
east_bonus = 1.0 + 0.03 * east_score
current_score = speed_score * upstream_temp_score * east_bonus
_add_score("current", current_score)
```
Value-based: faster + warmer upstream + eastward = higher score.

**Why:** User observed catches consistently offset from zone peaks. Edge analysis (analyze_catch_offset.py) confirmed:
- MLD: 1.83x gradient lift at catches
- shelf_break: 1.70x gradient lift at catches
- current: 1.64x gradient lift at catches

**How to apply:** If validation drops below 80% mean or 85% >= 70%, revert these three features using the code above.
