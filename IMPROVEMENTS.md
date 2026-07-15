# Accuracy & Meaningfulness Improvements

Work done on branch `improve/accuracy-meaningfulness` in response to the
marine-science + software review. Grouped by the review's tiers. Each item is
marked **[verified]** (runs/tested in this environment) or **[written]** (correct
by construction but needs the full Copernicus/scipy pipeline to run end-to-end).

## A guiding constraint

The scoring model is **calibrated** — its weights were Optuna-tuned against the
catch set. So "fixing" a feature's numerics silently changes predictions that were
tuned around the old behaviour. Changes here are therefore split:

- **Output-neutral / pure bugs** (safe to apply now): applied and, where possible,
  verified numerically identical or strictly-better.
- **Calibration-affecting** (need a re-tune to stay honest): implemented safely
  (behind a flag / disabled) or documented as `HONESTY NOTE` in code, not silently
  changed. These are called out below.

## Tier 1 — Measure accuracy honestly

- **[verified] `evaluate_model.py` + `EVALUATION.md`** — presence/background
  discrimination from the committed maps (numpy only). Reports mean *including*
  out-of-zone misses, AUC vs random ocean and vs random in-zone points, TSS, a
  capture/lift curve, and leave-one-year-out AUC. Deployed grid (in-sample):
  blue AUC **0.93** vs ocean, **0.78** within zones, **0.91 ± 0.10** LOYO; the same
  grid scores striped marlin equally well (0.93) → it is a shared shelf-edge
  billfish index, not blue-specific.
- **[verified] `validate_scoring.py`** — headline mean now counts misses (was
  in-zone-only, survivorship-biased); labels the old number; flags in-sample;
  removes the hardcoded `C:\Users\User\…` path (env var + fallback); documents the
  upward-biased `_proximity_max`.
- **[written] `verify_forecast.py`** (new, wired into `run_daily.py`) — scores each
  archived forecast (+1..+7) against observations, writing per-lead MAE/bias and a
  rating hit to `data/forecast_verification/`. Closes the loop the README claimed
  existed. Skill numbers accrue once a few days of archives exist.
- **Still open:** a *real* out-of-sample number needs a weight-refit CV loop
  (leave-one-year-out, refit with `optimize_visual.py` on train years, score the
  held-out year through `evaluate_model.py`). Scaffolding is in place; running it
  needs the ocean-data pipeline.

## Tier 2 — Fix the science

- **[verified] Species/thermal identity documented** — `MARLIN_TEMPS` and the docs
  now state the composite uses a local cool-edge optimum (23.75 °C) that overlaps
  the striped band, distinct from the global 24–27 °C "prime" band; the model does
  not distinguish blue from striped.
- **[verified] Lunar boost disabled by default** — it was a spatially uniform daily
  multiplier (no within-day discrimination) and the project's own analysis found no
  moon signal. Kept behind `_opt_lunar_boost` (now 0.0).
- **[written] Honesty notes in code** for the over-sold features: `current_shear`
  (surface-only vorticity, not vertical LC/undercurrent shear), `vertical_velocity`
  (index-space divergence, no warm-side gate), FTLE (under-resolved + mis-scaled
  Jacobian). These need physical-unit gradients + a re-tune, so they are documented
  rather than silently changed.

## Tier 3 — Make the output meaningful

- **[verified] Monotonic band-boost rescale** — replaced the discontinuous
  `[1.0,max]→[0.75,1.0]` top-only map (which could rank a more-boosted cell *below*
  a less-boosted one) with a continuous soft-knee that preserves ordering.
- **[verified] Feature-line-floor logging/comments** now report the actual applied
  value instead of a hardcoded `≥0.62` that disagreed with the 0.40 default.
- **[verified] Lead-time confidence** — `fetch_prediction.py` decays confidence for
  days +4..+7; `index.html` styles those rows as lower-confidence.
- **[verified] Fail-loud on dropped features** — the engine now warns and exposes a
  completeness fraction when a failed feature silently drops out (days were
  otherwise non-comparable).

## Tier 4 — Engineering / reproducibility

- **[verified] `scoring_features.py` + `tests/test_scoring_features.py`** — pure,
  unit-tested per-cell transforms (SST, CHL, edge, depth gate, SSH, clarity,
  weighted composite). 11 golden tests. The engine now calls these for SST / edge /
  depth (verified numerically identical). First step to making the 2,000-line
  function testable.
- **[verified] `scoring_config.py`** — one authoritative table of the 58 tunable
  defaults + `validate_overrides()` that warns on unknown `_opt_*` attributes so an
  optimizer typo can't silently use a default. `optimize_visual.py` fallbacks now
  read from it (they had drifted, e.g. `sst_optimal` 22.5 vs 23.75).
- **[verified] Boating comfort safety cap** — a per-factor ceiling so a hazard
  (e.g. 2.5 m swell) can't be averaged away to "GOOD"; only ever lowers the score.
- **[verified] Dependencies pinned** (major-version caps + lockfile note);
  hardcoded catch-CSV paths made env-overridable; dead `marine-chart-v6.html`
  removed.

## How to run the new tooling

```bash
python evaluate_model.py                 # honest discrimination metrics (numpy only)
python tests/test_scoring_features.py    # golden tests (or: pytest tests/)
python scoring_config.py                 # dump the 58 authoritative defaults
python verify_forecast.py --summary      # rolling forecast-skill table (once archives exist)
```
