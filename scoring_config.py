#!/usr/bin/env python3
"""
scoring_config.py -- single authoritative source for every tunable scoring parameter.

Background
----------
The scoring engine reads ~58 tunable parameters as implicit module globals via
`getattr(sys.modules[__name__], '_opt_x', DEFAULT)`, and the optimizer injects them
with `setattr(marlin_data, '_opt_x', value)`. Two problems that this module fixes:

  1. A typo in an injected name (e.g. `_opt_sst_optiml`) silently falls back to the
     default with no error -- a wrong run that looks identical to a correct one.
     `validate_overrides()` catches this by warning on any unknown `_opt_*`/`_edge_*`
     override set on the module.

  2. The default values were duplicated in `marlin_data.py` and `optimize_visual.py`
     and had drifted apart (e.g. sst_optimal 23.75 vs 22.5, depth_floor 0.95 vs 0.25).
     DEFAULTS below is the one place they live; both modules should read from it.

This is intentionally a plain dict + helpers (not a hard refactor of all 58 call sites,
which cannot be runtime-verified in this environment). It removes the fragility without
risking the numerically-sensitive engine.
"""

from __future__ import annotations

# Authoritative v22 defaults. Keys are the exact module-attribute names the engine reads.
DEFAULTS: dict[str, float | int | str | bool] = {
    # --- SST ---
    "_opt_sst_optimal": 23.75,       # deg C. Cool-edge blue-marlin optimum off Perth.
    "_opt_sst_sigma": 2.50,          # Gaussian sigma below optimum (tighter: cold penalty)
    "_opt_sst_sigma_above": 4.0,     # Gaussian sigma above optimum (wider: warm tolerance)
    # --- Chlorophyll ---
    "_opt_chl_threshold": 0.20,      # mg/m3, log-Gaussian centre
    "_opt_chl_sigma": 0.45,          # log10 units
    "_opt_bivariate_rho": 0.0,       # SST-CHL bivariate correlation
    # --- SST front / corridor / intrusion ---
    "_edge_front_sigma": 3.5,        # px, front-widening blur
    "_opt_front_floor": 0.07,        # warm-water minimum front score
    "_opt_corridor_pct": 85,         # percentile threshold for corridor mask
    "_opt_intrusion_threshold": 0.45,
    "_opt_intrusion_baseline": 0.29,
    # --- SSH ---
    "_ssh_abs_blend": 0.2,           # weight on absolute vs relative SLA
    # --- Currents / convergence / shear ---
    "_edge_current_blend": 0.4,
    "_edge_conv_blend": 0.0,
    "_opt_east_bonus": 0.03,
    "_opt_synergy_factor": 0.25,
    "_opt_shear_depth_thresh": 60,   # m, below which shear is suppressed
    "_opt_shear_depth_full": 300,    # m, above which shear scores fully
    # --- MLD ---
    "_edge_mld_blend": 0.0,
    # --- Upwelling / shelf ---
    "_edge_upwell_sigma": 4.0,
    "_edge_shelf_sigma": 2.0,
    "_shelf_prox_blend": 0.80,       # proximity vs gradient blend
    "_shelf_prox_depth": 270,        # m, target depth
    "_shelf_prox_sigma": 50,         # m
    "_opt_shelf_boost": 0.12,        # multiplicative shelf boost
    "_opt_sst_shelf_interact": 0.0,
    # --- Depth gate ---
    "_depth_zero_cut": 80,           # m, hard zero below
    "_depth_shallow_floor": 0.50,    # ramp start value
    "_depth_shallow_full": 180,      # m, full score above
    "_depth_taper_start": 500,       # m, deep taper start
    "_depth_taper_mid": 1500,        # m, deep taper knee
    "_depth_floor": 0.95,            # deep-water floor
    # --- Bathymetric band system ---
    "_opt_band_shore_ratio": 0.30,
    "_opt_band_deep_ratio": 0.30,
    "_opt_shallow_cut": 0.65,
    "_opt_bathy_w_200": 1.0,
    "_opt_bathy_w_500": 0.7,
    "_opt_band_width_nm": 4.5,
    "_opt_band_decay": 0.60,
    "_opt_band_chl_thresh": 0.45,
    "_opt_band_front_thresh": 0.25,
    "_opt_band_boost": 0.40,
    "_opt_band_single": 0.06,
    "_opt_band_overlap": 0.20,
    "_opt_zero_band_mult": 0.55,
    "_opt_one_band_mult": 0.80,
    # --- Post-processing multipliers ---
    "_opt_pool_percentile": 75,
    "_opt_key_feature_floor": 0.40,
    "_opt_score_grad_blend": 0.20,
    "_edge_boost_strength": 0.05,
    "_opt_lunar_boost": 0.0,         # DISABLED by default (was 0.10). See review:
                                     # uniform daily multiplier, no within-day skill;
                                     # project's own analysis rejected moon phase.
    # --- Subsurface (zero-weight features, kept for completeness) ---
    "_opt_strat_strong": 6.0,
    "_opt_strat_weak": 2.0,
    "_opt_thermo_cold": 8.0,
    "_opt_thermo_warm": 16.0,
    # --- Modes ---
    "_scoring_mode": "weighted_sum",
    "_profile_pure": False,
}

# Attribute prefixes that denote a tunable scoring override on the engine module.
_OVERRIDE_PREFIXES = ("_opt_", "_edge_", "_depth_", "_shelf_", "_ssh_", "_scoring_", "_profile_")

# Names with an override prefix that are legitimately NOT in DEFAULTS (computed/paths).
_ALLOWED_EXTRA = {"_profile_path", "_default_profile"}


def get(module, key):
    """Resolve a parameter: runtime override on `module` wins, else authoritative default."""
    if key not in DEFAULTS:
        raise KeyError(f"Unknown scoring parameter: {key!r}")
    return getattr(module, key, DEFAULTS[key])


def apply_defaults(module):
    """Set any missing default as an explicit attribute on `module` (idempotent).
    Existing values (e.g. optimizer injections) are preserved."""
    for key, val in DEFAULTS.items():
        if not hasattr(module, key):
            setattr(module, key, val)


def validate_overrides(module, warn=print):
    """Warn about override-prefixed attributes on `module` that are not known parameters.
    Catches optimizer typos that would otherwise silently use a default.
    Returns the list of unknown names."""
    unknown = []
    for name in dir(module):
        if not any(name.startswith(p) for p in _OVERRIDE_PREFIXES):
            continue
        if name in DEFAULTS or name in _ALLOWED_EXTRA:
            continue
        # ignore callables/imported modules that happen to match a prefix
        val = getattr(module, name, None)
        if callable(val):
            continue
        unknown.append(name)
    for name in unknown:
        warn(f"[scoring_config] WARNING: '{name}' is set on the engine but is not a "
             f"known parameter -- it has no effect (possible typo).")
    return unknown


if __name__ == "__main__":
    print(f"{len(DEFAULTS)} authoritative scoring parameters:")
    for k, v in DEFAULTS.items():
        print(f"  {k:28s} = {v}")
