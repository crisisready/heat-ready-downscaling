# Provenance

This repository is not a from-scratch project. Its model-training code was extracted from a
long-lived, unmerged branch of the private serving repository
(`crisisready/heat-risk-data-api`), rather than written fresh here — extracted at tip rather than
merged, because that branch is ~200 commits behind its own `main` and a real merge would revert
~95k unrelated lines. `git tag archive/<branch-name>` was applied to the source branch's tip in
the private repository at extraction time, specifically so it stops looking mergeable to a future
reader.

Every file listed below carries the source SHA it was extracted from in its own file header
comment, in addition to this table — so provenance survives a copy of an individual file, not
just a reading of this document.

## Files extracted from `crisisready/heat-risk-data-api`

All ten extracted 2026-07-27 from `origin/feature/downscaling-phase4-model-training` at tip
commit `9d8a678c594fbe2878033373b750cc8465a9d80e`, verbatim (`git show <sha>:<path>`, no merge).

| File in this repository | Extracted from (private repo path) |
|---|---|
| `scripts/train_downscaling.py` | `scripts/train_downscaling.py` |
| `scripts/sweep_qrf_hyperparams.py` | `scripts/sweep_qrf_hyperparams.py` |
| `scripts/sweep_gbm.py` | `scripts/sweep_gbm.py` |
| `scripts/backfill_wind.py` | `scripts/backfill_wind.py` |
| `scripts/build_training_set.py` | `scripts/build_training_set.py` (forward-ported onto main's current `src/` API afterward — see this file's own header for the call-site fixes applied post-extraction) |
| `tests/test_train_downscaling.py` | `tests/test_train_downscaling.py` |
| `tests/test_sweep_qrf_hyperparams.py` | `tests/test_sweep_qrf_hyperparams.py` |
| `tests/test_sweep_gbm.py` | `tests/test_sweep_gbm.py` |
| `tests/test_backfill_wind.py` | `tests/test_backfill_wind.py` |
| `tests/test_build_training_set.py` | `tests/test_build_training_set.py` — **not in the original plan's section 5.1 file list**; found and extracted separately during the build_training_set.py forward-port, since it exists on the branch with real coverage for the single riskiest file in this port |

The private repository tagged `archive/downscaling-phase4-model-training` at this same SHA at
extraction time, specifically so the branch stops looking mergeable to a future reader there.

## `src/heatready_downscaling/` — extracted from `main`, not the branch

Unlike the section above, these came from `crisisready/heat-risk-data-api`'s **`main`** branch at
commit `57479e5` (2026-07-27), not from `origin/feature/downscaling-phase4-model-training` — they
are the already-shipped scoring/inference code, extracted so this package can run standalone
without importing the private repo's `src/`. Each module's own docstring names its exact source
file/line range; this table is the index.

| Module | Extracted/adapted from (private repo, `main`) |
|---|---|
| `features.py` | `src/downscaling.py` (`FEATURE_ORDER`, `_doy_trig`, `build_feature_matrix`) |
| `koppen.py` | `src/ghcn.py` (Köppen classification block) |
| `contract.py` | `src/downscaling.py` (`predict_downscaled`, `_feature_importance_weights`, `_aoa_dissimilarity`, `_confidence_class`, `_not_applied_result`, `derive_zones_passing_cv_gate`, `load_model`/`load_model_metadata`) — restructured into a `ModelAdapter` protocol + `QRFModelAdapter` concrete class rather than a free function taking a raw bundle dict; see the module's own docstring for why |
| `score.py` | `scripts/validate_lagfill_downscaling.py` (`score_band`, `fidelity_report`, and their constants) — `validate_forecast_downscaling.py` imported this same function rather than cloning it, so there was only ever one implementation to extract |
| `gates.py` | `scripts/publish_band_gate.py` (`build_gate`) — the blend-gate jsonschema is genuinely new (no `build_gate`-equivalent existed for `publish_blend_gate.py`, which only did an ad-hoc key-presence check) |
| `report.py` | `scripts/validate_lagfill_downscaling.py` (`_build_report`) — canonicalized against a real inconsistency found between the two validation scripts' report shapes (`rows_nrt_paired` vs `rows_lead_paired`); see the module's own docstring |
| `snapshot.py` | New. No private-repo equivalent — schema matches the private repo's `docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md` sections 6.1/6.2 |

Two required changes made during this extraction, not present in the private repo's original code
(both mandated by the crowdsourced model-improvement program's design, not incidental):
`score.score_band` takes a required `fold_salt` keyword argument (snapshot-version-salted fold
assignment, closing a fold-shopping gap), and takes a `contract.ModelAdapter` instead of a raw
joblib bundle dict.

## A note on the source branch

`origin/feature/downscaling-phase4-model-training` in the private repository is **not merged and
should not be** — see the private repo's own memory/plan documentation for why (F1/F2 findings,
`docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md`). This repository is the
resolution: rather than trying to reconcile that branch with its own `main`, the branch's useful
Phase-4 work moves here, where it becomes the single source of truth for training/validation code,
and the private repo consumes it as a pinned dependency instead of vendoring a divergent copy.
