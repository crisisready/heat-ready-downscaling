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

## Phase 1.3 smoke-test result (2026-07-27)

Ran `snapshot_covariates_for_stations` (the covariate half of `build_training_set.py`'s
pipeline, called directly to bypass a real CDS cost-ceiling 403 blocking the unrelated
ERA5-Land fetch) for Nicaragua's 8 active GHCN stations, and diffed the output against the
live `ghcn_training` table's existing rows for the same stations.

**Exact match, all 8 stations**: `elevation_mean_m`, `slope_deg`, `aspect_deg`, `wc_built_frac`,
`wc_tree_frac`, `wc_water_frac`, `ghsl_urban_fraction`, `canopy_height_mean_m`,
`canopy_frac_over_3m` — zero mismatches. This is the strongest available evidence that the
`dem.extract_elevation`/`vulnerability.extract_worldcover`/`extract_ghsl_smod`/`extract_canopy`
call-site fixes applied during this forward-port (see this file's own header note on
`build_training_set.py`, and the file's own inline "FORWARD-PORT FIX" comments) are faithful.

**`pop_density_per_km2` mismatched on all 8 stations**, with no consistent ratio (e.g. 727 vs
379, 5139 vs 3183, 3888 vs 4049 people/km²) — **not a forward-port defect.** The private repo's
`src/landscan.py` was modified the day before this smoke test ran, by the LandScan
population-conservation fix (PR #285, commits `7b3e34d`/`6753944`/`a2ab12d`, merged
2026-07-26), which specifically corrected the sub-pixel/small-polygon extraction path — the
exact case a station's ~0.01°(~1km) buffer hits. Section 5.3's plan text ("`_population_density_by_station`
ports across unchanged, so the training build stays bit-for-bit consistent with what
`ds-2026.07-rf4` learned") was accurate when written but was invalidated by that same-week fix.
The live `ghcn_training` rows reflect pre-fix extraction; this smoke run's covariates reflect
post-fix extraction — both are internally correct, they're just computed by two different
(one buggy, one fixed) versions of the same function.

**Implication tracked, not resolved here**: `ghcn_training`'s full 907-station corpus — and the
currently-deployed `ds-2026.07-rf4` model trained on it — likely has a systematically-off
`pop_density_per_km2` feature for any station whose buffer falls in the small-polygon case PR
#285 fixed (plausibly most/all of them, given the buffer size). This is a real data-quality
question independent of the crowdsourced-model-improvement program, tracked as a backlog item
for a future full-corpus refresh + retrain-impact assessment — deliberately not actioned as
part of Phase 1, given low current user volume and the value of getting the retrain evaluation
right rather than fast.

**LST** (`lst_warm_season_anomaly_c`, tolerance-checked per the plan, not exact-match): 7/8
stations within 1°C; one station (NIM00065271) at 1.42°C, accepted as within the noise the plan
itself attributes to LST (Landsat scene availability moving between runs).

**One station absent**: `NIM00065082` has zero rows in `ghcn_training` — excluded from the
original training build (most likely by an activity-date filter at build time), not a diff bug.

## Phase 1.4 — validate/publish scripts moved from the private repo (2026-07-27)

Five more files moved from `crisisready/heat-risk-data-api`'s `main` (not the archived branch —
these were never part of the branch; they were written directly against the shipped
`ds-2026.07-rf4` model) and deleted there in the same close-out PR:

| File in this repository | Extracted from (private repo `main`) |
|---|---|
| `scripts/validate_lagfill_downscaling.py` | `scripts/validate_lagfill_downscaling.py` @ `a31ec14863273904ae6a9de7a6c34cb77f84f4f4` |
| `scripts/validate_forecast_downscaling.py` | `scripts/validate_forecast_downscaling.py` @ `a31ec14863273904ae6a9de7a6c34cb77f84f4f4` |
| `scripts/validate_station_blend.py` | `scripts/validate_station_blend.py` @ `e6cc1d2a338fd9c87ef51cbc1daf6cb1ed5e2b3b` |
| `scripts/publish_band_gate.py` | `scripts/publish_band_gate.py` @ `a31ec14863273904ae6a9de7a6c34cb77f84f4f4` |
| `scripts/publish_blend_gate.py` | `scripts/publish_blend_gate.py` @ `e6cc1d2a338fd9c87ef51cbc1daf6cb1ed5e2b3b` |

Each now delegates its scoring/gate-building logic to the already-extracted package
(`heatready_downscaling.score`/`gates`/`report`/`contract`) instead of carrying its own copy — see
each file's own "REFACTORED during the move" docstring note for exactly what changed. One real
consequence of this refactor: `validate_station_blend.py` had only ever imported private-repo-only
modules for `downscaling.load_model`/`predict_downscaled` (now `contract.QRFModelAdapter`) and
`ghcn.koppen_broad_group_letter_from_zone` (now `koppen.koppen_broad_group_letter_from_zone`,
already present in this package) — with both swapped out, it has **no private-repo-only imports
left** and is fully runnable/testable standalone in this repo, unlike its two `validate_lagfill_
downscaling.py`/`validate_forecast_downscaling.py` siblings (still gated on `db`/`heat_calcs`/
`open_meteo`/`api_call_manager`, see their own "NOT RUNNABLE STANDALONE" docstring note and
`conftest.py`'s `collect_ignore`).

Tests for `test_validate_lagfill_downscaling.py`/`test_validate_forecast_downscaling.py` were
ported largely as-is, minus the score_band/fidelity_report-specific test classes now redundant
with `tests/test_score.py` (which already covers that logic — it was extracted from these same
scripts in Phase 1.2, before this Phase 1.4 move). `test_publish_band_gate.py` was rewritten as a
CLI-level test (the private repo's original directly unit-tested `build_gate`, now redundant with
`tests/test_gates.py`); `test_publish_blend_gate.py` ported as-is, since it was already CLI-level.
`validate_station_blend.py` had no test file in the private repo to port.

## Phase 2 — original code, not extracted (2026-07-27)

`scripts/build_band_paired_snapshot.py` and the `write_stations`/`compute_holdout`/`write_holdout`
additions to `src/heatready_downscaling/snapshot.py` are new code written for this repository,
not extracted or adapted from the private repo (unlike everything above) — there is no private-
repo equivalent to port from; `snapshot.py`'s own reader/writer/manifest primitives (Phase 1.2)
already anticipated this script in their own docstrings ("not yet written -- Phase 2"). The real
`v2026.07` snapshot in this repository's first GitHub Release was built and verified end-to-end
with this script against the live, freshly re-exported `ghcn_training` corpus (274,249 rows,
post pop_density retrain) — see the script's own module docstring and commit message for the
verification detail (partition counts, manifest checksums, holdout exclusion correctness).

## A note on the source branch

`origin/feature/downscaling-phase4-model-training` in the private repository is **not merged and
should not be** — see the private repo's own memory/plan documentation for why (F1/F2 findings,
`docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md`). This repository is the
resolution: rather than trying to reconcile that branch with its own `main`, the branch's useful
Phase-4 work moves here, where it becomes the single source of truth for training/validation code,
and the private repo consumes it as a pinned dependency instead of vendoring a divergent copy.
