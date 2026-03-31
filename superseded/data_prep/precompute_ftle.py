"""Pre-compute FTLE fields for all catch dates and save as .npz cache files.
This avoids recomputing the expensive RK4 particle advection on every Optuna trial.
Each date takes ~0.8s; 44 dates = ~35s total (vs 9,400 recomputations in 200 trials).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from optimize_visual import load_catches
from marlin_data import compute_ftle, DEFAULT_BBOX as BBOX
import marlin_data

dates = sorted(set(c['date'] for c in load_catches()))
print(f"Pre-computing FTLE for {len(dates)} catch dates...")

ok = 0
skip = 0
fail = 0

for i, d in enumerate(dates, 1):
    out_dir = os.path.join("data", d)
    cache_file = os.path.join(out_dir, "ftle_cache.npz")

    if os.path.exists(cache_file):
        skip += 1
        continue

    marlin_data.OUTPUT_DIR = out_dir

    try:
        result = compute_ftle(d, BBOX, window_days=3)
        if result is not None:
            ftle_field, ftle_lons, ftle_lats = result
            np.savez_compressed(cache_file,
                                ftle=ftle_field,
                                lons=ftle_lons,
                                lats=ftle_lats)
            ok += 1
            print(f"  [{i}/{len(dates)}] {d} -> cached ({ftle_field.shape})")
        else:
            fail += 1
            print(f"  [{i}/{len(dates)}] {d} -> no data")
    except Exception as e:
        fail += 1
        print(f"  [{i}/{len(dates)}] {d} -> FAILED: {e}")

print(f"\nDone: {ok} computed, {skip} already cached, {fail} failed")
