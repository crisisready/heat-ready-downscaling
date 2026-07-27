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

| File in this repository | Extracted from (private repo path) | Source branch tip SHA | Extracted |
|---|---|---|---|
| _to be filled in during extraction — see the private repo's `docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md` section 5.1_ | | | |

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
