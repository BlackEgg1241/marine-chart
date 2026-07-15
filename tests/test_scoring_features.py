#!/usr/bin/env python3
"""
Golden tests for the pure scoring transforms in scoring_features.py.

Runs under pytest (CI) or standalone: `python tests/test_scoring_features.py`.
These pin the numeric behaviour of each feature scorer so refactors of the
2,000-line engine can be verified against known inputs -- the safety net the
project currently lacks (there are no other tests in the repo).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring_features as sf
import scoring_config as sc


def test_sst_peaks_at_optimum():
    assert sf.sst_gaussian(23.75) == 1.0
    # asymmetric: 3C warmer scores higher than 3C cooler
    warm = sf.sst_gaussian(23.75 + 3)
    cool = sf.sst_gaussian(23.75 - 3)
    assert warm > cool
    # a striped-marlin-range 22C still scores well (cool-edge tolerance) ~0.78
    assert sf.sst_gaussian(22.0) > 0.75


def test_sst_symmetric_when_no_upper_sigma():
    a = sf.sst_gaussian(20.0, optimal=23.0, sigma_below=2.0, sigma_above=None)
    b = sf.sst_gaussian(26.0, optimal=23.0, sigma_below=2.0, sigma_above=None)
    assert abs(a - b) < 1e-12


def test_chl_log_gaussian_peak_and_symmetry_in_log():
    assert abs(sf.chl_log_gaussian(0.20) - 1.0) < 1e-12
    # symmetric in log10: 0.20*10 and 0.20/10 score equally
    hi = sf.chl_log_gaussian(2.0)
    lo = sf.chl_log_gaussian(0.02)
    assert abs(hi - lo) < 1e-9


def test_edge_transform_peaks_at_center():
    assert abs(sf.edge_transform(0.8, 0.8, 0.6) - 1.0) < 1e-12
    # a value at the feature MAXIMUM scores below the center (edge-hunting)
    assert sf.edge_transform(1.0, 0.8, 0.6) < 1.0
    assert sf.edge_transform(0.3, 0.3, 0.65) == 1.0


def test_depth_gate_piecewise():
    g = sf.depth_gate
    assert g(50) == 0.0                      # below zero_cut
    assert g(79.9) == 0.0                    # just below cut
    # at exactly zero_cut the strict `< zero_cut` is false, so the ramp begins
    # at shallow_floor (matches engine's np.where(d < _dt_zero, 0, ...))
    assert abs(g(80) - 0.50) < 1e-9          # ramp starts at shallow_floor
    assert abs(g(180) - 1.0) < 1e-9          # full at shallow_full
    assert g(300) == 1.0                     # prime zone
    assert abs(g(1500) - 0.80) < 1e-9        # knee
    assert abs(g(3000) - 0.95) < 1e-9        # floor beyond 2*mid
    # non-increasing on the real taper (500 -> knee at 1500)...
    taper = [g(d) for d in [500, 800, 1200, 1500]]
    assert all(taper[i] >= taper[i + 1] - 1e-12 for i in range(len(taper) - 1))
    # ...then it RISES back to the relaxed deep floor (0.80 knee -> 0.95 abyssal).
    # This non-monotonic shape is a v22 quirk worth revisiting, but the pure
    # function must reproduce the engine exactly, so the test documents it.
    assert g(3000) > g(1500)


def test_depth_gate_uses_config_defaults():
    # the pure function's defaults must match the authoritative config
    assert sf.depth_gate.__defaults__[0] == sc.DEFAULTS["_depth_zero_cut"]
    assert abs(sf.depth_gate(3000) - sc.DEFAULTS["_depth_floor"]) < 1e-9


def test_ssh_blend_bounds_and_weighting():
    sla = np.array([0.0, 0.06, 0.12, 0.30])
    bg = np.zeros_like(sla)
    out = sf.ssh_blend(sla, bg, abs_blend=0.2)
    assert np.all(out >= 0) and np.all(out <= 1)
    # pure absolute component monotonic in SLA
    only_abs = sf.ssh_blend(sla, sla, abs_blend=1.0)  # rel term = 0
    assert only_abs[0] <= only_abs[1] <= only_abs[2] <= only_abs[3]


def test_clarity_score_bounds():
    assert sf.clarity_score(0.02) == 1.0
    assert sf.clarity_score(0.20) == 0.0
    assert 0 < sf.clarity_score(0.09) < 1


def test_lunar_illumination_new_and_full():
    # reference new moon -> illumination ~0
    assert sf.lunar_illumination("2000-01-06") < 0.05
    # ~14.75 days later -> near full
    assert sf.lunar_illumination("2000-01-21") > 0.9


def test_weighted_composite_graceful_degradation():
    shape = (4, 4)
    a = np.full(shape, 0.8)
    b = np.full(shape, 0.4)
    weights = {"a": 0.6, "b": 0.4}
    comp, frac = sf.weighted_composite({"a": a, "b": b}, weights)
    assert np.allclose(comp, 0.6 * 0.8 + 0.4 * 0.4)
    assert abs(frac - 1.0) < 1e-12
    # drop feature b -> renormalised over a only, and contributed fraction reports it
    comp2, frac2 = sf.weighted_composite({"a": a}, weights)
    assert np.allclose(comp2, 0.8)
    assert abs(frac2 - 0.6) < 1e-12


def test_config_has_no_conflicts_and_validates_typos():
    # every default is resolvable
    for k in sc.DEFAULTS:
        assert sc.get(sys.modules[__name__], k) == sc.DEFAULTS[k] or not hasattr(sys.modules[__name__], k)

    class M:  # a fake engine module with a typo'd override
        _opt_sst_optiml = 22.0   # typo
        _opt_sst_optimal = 23.75  # correct
    warnings = []
    unknown = sc.validate_overrides(M, warn=warnings.append)
    assert "_opt_sst_optiml" in unknown
    assert "_opt_sst_optimal" not in unknown


def _run_standalone():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_standalone()
