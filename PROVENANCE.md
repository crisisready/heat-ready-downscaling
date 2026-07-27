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

All nine extracted 2026-07-27 from `origin/feature/downscaling-phase4-model-training` at tip
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

The private repository tagged `archive/downscaling-phase4-model-training` at this same SHA at
extraction time, specifically so the branch stops looking mergeable to a future reader there.

## What was NOT extracted (written fresh in this repository)

Everything under `src/heatready_downscaling/` (the installable package) is new code written for
this repository, even where it consolidates logic that previously existed only inline in one of
the extracted scripts (e.g. `score.py`'s `score_band`, deduplicated from
`validate_lagfill_downscaling.py` and a near-clone in the forecast validator). Where a new module
is a genuine extraction of existing logic rather than new logic, its own docstring says so and
names the source file/line range it came from.

## A note on the source branch

`origin/feature/downscaling-phase4-model-training` in the private repository is **not merged and
should not be** — see the private repo's own memory/plan documentation for why (F1/F2 findings,
`docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md`). This repository is the
resolution: rather than trying to reconcile that branch with its own `main`, the branch's useful
Phase-4 work moves here, where it becomes the single source of truth for training/validation code,
and the private repo consumes it as a pinned dependency instead of vendoring a divergent copy.
