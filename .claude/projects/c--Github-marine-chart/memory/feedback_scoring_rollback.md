---
name: scoring_discrimination_changes
description: Changes made to marlin scoring discrimination on 2026-03-21 — rollback reference
type: feedback
---

Changes to `marlin_data.py` band boost / feature floor to improve zone discrimination:

**Before (rollback values):**
1. Band boost rescaling (line ~1688): `final[valid] = final[valid] / compressed_max` — divides ALL scores by max, flattening distribution
2. Feature line floor (line ~1603): `_key_feature_floor = 0.62` — lifts 76% of cells to ≥0.62
3. No-band suppression (line ~1662): `final[no_band_mask] *= 0.80` — 20% dampen for cells without feature bands

**After (new values):**
1. Band boost rescaling: only rescale cells >1.0, preserve cells ≤1.0 unchanged
2. Feature line floor: lowered to 0.50
3. No-band suppression: increased to 0.70 (30% dampen)

**Why:** Non-overlapping polygon rendering exposed that scoring had a massive cliff at 0.62-0.65 (5% of cells above 0.65 vs 62% above 0.60). Band boost rescaling was dividing ALL scores by the compressed max (~1.6), dragging everything below 0.65. Feature floor at 0.62 lifted 76% of cells to a plateau. Combined effect: almost no discrimination above the display floor.

**How to apply:** If zones are too sparse or catches miss zones, revert these three values. If zones are too broad, keep them.
