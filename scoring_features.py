#!/usr/bin/env python3
"""
scoring_features.py -- pure, unit-testable feature-scoring transforms.

These are the pointwise (per-cell) scoring functions extracted verbatim from the
2,000-line `generate_blue_marlin_hotspots()` so they can be tested against known
inputs without running the full Copernicus/scipy pipeline. Each returns a 0-1 score
array the same shape as its input.

The engine imports and calls these; the golden tests in tests/test_scoring_features.py
pin their behaviour. Extracting them is the first step toward making the god-function
testable (see review). Everything here depends only on numpy.
"""

from __future__ import annotations

import numpy as np


def sst_gaussian(sst, optimal=23.75, sigma_below=2.50, sigma_above=4.0):
    """Asymmetric Gaussian SST suitability.

    Tighter below the optimum (cold water is less tolerable) and wider above
    (billfish tolerate warm water). `optimal` for the Perth Canyon is a *cool-edge*
    blue-marlin value (~23.75C), not the global tropical 25-30C 'prime' band -- see
    MARLIN_TEMPS and the species note in SCORING_METHODOLOGY.md. Pass sigma_above=None
    for a symmetric Gaussian.
    """
    sst = np.asarray(sst, dtype=float)
    if sigma_above is None:
        return np.exp(-0.5 * ((sst - optimal) / sigma_below) ** 2)
    sigma_map = np.where(sst < optimal, sigma_below, sigma_above)
    return np.exp(-0.5 * ((sst - optimal) / sigma_map) ** 2)


def chl_log_gaussian(chl, optimal=0.20, sigma=0.45):
    """Log-space Gaussian chlorophyll suitability, peaking at the oligotrophic/
    mesotrophic transition (`optimal` mg/m3). CHL is clipped to [0.01, 10] before the
    log, matching the engine."""
    chl = np.asarray(chl, dtype=float)
    chl_log = np.log10(np.clip(chl, 0.01, 10))
    return np.exp(-0.5 * ((chl_log - np.log10(optimal)) / sigma) ** 2)


def edge_transform(v, center, width):
    """Value-space Gaussian 'edge' transform: score peaks at an intermediate raw
    value `center` (the ecotone), not at the feature maximum. Used for okubo_weiss,
    upwelling_edge, current_shear, chl_curvature. Input is clipped to [0,1] first."""
    v = np.clip(np.asarray(v, dtype=float), 0, 1)
    out = np.exp(-0.5 * ((v - center) / max(width, 0.01)) ** 2)
    return np.clip(out, 0, 1)


def depth_gate(depth_m, zero_cut=80, shallow_floor=0.50, shallow_full=180,
               taper_start=500, taper_mid=1500, floor=0.95, knee=0.80):
    """Multiplicative depth gate as a function of positive depth in metres.

    Piecewise: 0 below `zero_cut`; linear ramp `shallow_floor`->1 to `shallow_full`;
    1.0 to `taper_start`; linear 1.0->`knee` to `taper_mid`; linear `knee`->`floor`
    to 2*`taper_mid`; `floor` beyond. Matches marlin_data.py:1563-1569 exactly.
    """
    d = np.asarray(depth_m, dtype=float)
    return np.where(d < zero_cut, 0.0,
           np.where(d < shallow_full,
                    shallow_floor + (1.0 - shallow_floor) * (d - zero_cut) / max(shallow_full - zero_cut, 1),
           np.where(d < taper_start, 1.0,
           np.where(d < taper_mid, 1.0 - (1.0 - knee) * (d - taper_start) / max(taper_mid - taper_start, 1),
           np.where(d < taper_mid * 2, knee - (knee - floor) * (d - taper_mid) / max(taper_mid, 1),
           floor)))))


def ssh_blend(sla, sla_background, abs_blend=0.2, abs_scale=0.12, rel_scale=0.04):
    """Blend absolute SLA (warm-water-mass proxy) with relative SLA above a smoothed
    background (eddy-edge proxy). Matches marlin_data.py:1594-1606. Caller supplies the
    smoothed `sla_background` (the engine uses a Gaussian sigma=4)."""
    sla = np.asarray(sla, dtype=float)
    abs_score = np.clip(sla / abs_scale, 0, 1)
    rel_score = np.clip((sla - np.asarray(sla_background, dtype=float)) / rel_scale, 0, 1)
    return abs_blend * abs_score + (1.0 - abs_blend) * rel_score


def clarity_score(kd490, clear=0.04, span=0.11):
    """Water-clarity score from KD490: 1.0 at very clear (<=`clear`), 0 at turbid.
    Matches marlin_data.py:1668."""
    kd = np.asarray(kd490, dtype=float)
    return np.clip(1.0 - (kd - clear) / span, 0, 1)


def lunar_illumination(date, ref_new_moon=None):
    """Moon illumination in [0,1] (0=new, 1=full) from the 29.53-day synodic cycle.

    NOTE: the habitat *boost* built on this is disabled by default (see review: it was
    a spatially uniform daily multiplier with no within-day discrimination, and the
    project's own analysis found no moon-phase signal). This helper is retained for
    diagnostics only.
    """
    from datetime import datetime
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    if ref_new_moon is None:
        ref_new_moon = datetime(2000, 1, 6)
    days_since = (date - ref_new_moon).days
    lunar_cycle = (days_since % 29.53) / 29.53
    return 0.5 * (1 - np.cos(2 * np.pi * lunar_cycle))


def weighted_composite(sub_scores, weights, valid_mask=None):
    """Normalised weighted sum with graceful degradation: sum(w_i * s_i) / sum(w_i)
    over features that are present and non-NaN. Mirrors the engine's accumulation +
    final = score/weight_sum. `sub_scores` and `weights` are name->array / name->float.

    Returns (composite, contributed_weight_fraction) so callers can detect when a
    failed feature has silently changed the scoring basis (see review: days become
    non-comparable when a feature drops out)."""
    names = [n for n in weights if weights[n] > 0 and n in sub_scores]
    if not names:
        raise ValueError("no positive-weight features present")
    shape = np.asarray(sub_scores[names[0]]).shape
    score = np.zeros(shape, dtype=float)
    wsum = np.zeros(shape, dtype=float)
    total_w = sum(weights[n] for n in names)
    for n in names:
        v = np.clip(np.asarray(sub_scores[n], dtype=float), 0, 1)
        ok = ~np.isnan(v)
        if valid_mask is not None:
            ok &= valid_mask
        score[ok] += weights[n] * v[ok]
        wsum[ok] += weights[n]
    composite = np.full(shape, np.nan)
    nz = wsum > 0
    composite[nz] = score[nz] / wsum[nz]
    contributed_fraction = total_w / sum(w for w in weights.values() if w > 0)
    return composite, contributed_fraction
