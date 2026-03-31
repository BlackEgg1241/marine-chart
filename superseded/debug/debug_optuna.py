"""Debug: why do Optuna worker processes fail?"""
import sys, os, traceback, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize_visual import _score_one_date, load_catches, BBOX, EVAL_BBOX
from collections import defaultdict

def debug_worker(args):
    """Same as _score_one_date but prints errors instead of swallowing."""
    date_str = args[0]
    try:
        r = _score_one_date(args)
        # Re-enable stdout to print result
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if r is None:
            print(f"  {date_str}: returned None (internal error)")
        else:
            print(f"  {date_str}: OK, {len(r['catches'])} catches")
        return r
    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"  {date_str}: EXCEPTION: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    catches = load_catches()
    by_date = defaultdict(list)
    for c in catches:
        by_date[c["date"]].append(c)

    SCRIPT_DIR = os.path.abspath(".")

    # Non-baseline params (what Optuna would generate)
    params = {
        "w_sst": 0.20, "w_sst_front": 0.05, "w_front_corridor": 0.08,
        "w_chl": 0.12, "w_chl_curvature": 0.04,
        "w_ssh": 0.10, "w_shelf": 0.10, "w_current": 0.06,
        "w_shear": 0.10, "w_upwell": 0.05,
        "w_ftle": 0.06, "w_vert_vel": 0.02,
        "w_salinity_front": 0.10, "w_okubo_weiss": 0.06,
        "shelf_prox_blend": 0.3, "shelf_prox_sigma": 80,
        "depth_shallow_full": 140, "pool_pct": 85,
        "shelf_boost": 0.04,
        "depth_shallow_floor": 0.80, "depth_taper_start": 350,
        "depth_taper_mid": 900, "depth_floor": 0.35,
        "shelf_prox_depth": 180, "edge_shelf_sigma": 2.5,
        "sst_optimal": 22.75, "sst_sigma": 1.75,
        "sst_sigma_above": 3.5,
        "chl_threshold": 0.16, "chl_sigma": 0.25,
        "band_width_nm": 3.5, "band_boost": 0.30,
        "band_decay": 0.75, "front_floor": 0.06,
        "band_shore_ratio": 0.20, "band_deep_ratio": 0.20,
        "shallow_cut": 0.45,
        "shear_depth_thresh": 70, "shear_depth_full": 175,
        "sst_shelf_interact": 0.05,
        "lunar_boost": 0.04,
        "bathy_w_200": 0.7, "bathy_w_500": 0.3,
        "edge_spread": 4.0,
        "ssh_edge_center": 0.6, "ssh_edge_width": 0.30,
        "current_edge_center": 0.6, "current_edge_width": 0.30,
        "okubo_weiss_edge_center": 0.6, "okubo_weiss_edge_width": 0.30,
        "upwelling_edge_edge_center": 0.6, "upwelling_edge_edge_width": 0.30,
        "salinity_front_edge_center": 0.6, "salinity_front_edge_width": 0.30,
        "current_shear_edge_center": 0.6, "current_shear_edge_width": 0.30,
        "sst_front_edge_center": 0.6, "sst_front_edge_width": 0.30,
        "chl_curvature_edge_center": 0.6, "chl_curvature_edge_width": 0.30,
    }

    # Test 2 dates in spawned processes
    test_dates = ["2022-02-19", "2017-02-13"]
    work_items = []
    for d in test_dates:
        if d in by_date:
            work_items.append((d, by_date[d], params, SCRIPT_DIR, BBOX, EVAL_BBOX))

    print(f"Testing {len(work_items)} dates in spawned workers...")
    ctx = multiprocessing.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        futures = {pool.submit(debug_worker, item): item[0] for item in work_items}
        results = []
        for f in as_completed(futures):
            results.append(f.result())

    print(f"\nResults: {sum(1 for r in results if r)} / {len(results)} succeeded")
