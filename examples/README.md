# Worked examples

Two complete, runnable examples for the two contribution rungs open today (`GOVERNANCE.md`'s
Status section / `CONTRIBUTING.md`). Both run against nothing but this public repo and the
published snapshot release -- no private-repo access, no AWS credentials, no paid API key. Both
run in CI (`.github/workflows/examples.yml`) on every push and PR, so they cannot silently rot: if
a future snapshot release changes the data enough that either example's claim no longer holds, CI
goes red rather than the docs quietly going stale.

## Rung A: `rung_a_evaluate_dark_cell.py`

Re-runs the same evaluation the referee runs (`score.score_band` via
`contract.FrozenPredictionAdapter`, imported directly from `scripts/run_submission.py`'s own
`reproduce()` -- not reimplemented) against a real dark cell, and writes a real
`claimed_report.json`.

```
python examples/rung_a_evaluate_dark_cell.py
```

Named cell: `tmax`/`BWh`/`lag_fill`, model `ds-2026.07-rf5`. Pass `--target`/`--zone` to point it at
any other zone the snapshot's `lag_fill` band covers.

## Rung B / L1: `rung_b_l1_covariate_linear_fit.py`

Two separate halves, on purpose (see the script's own module docstring for why this distinction
matters): **fits** a covariate-linear correction (a plain least-squares intercept+slope) on a
subset of the zone's stations, then **scores** that declared value on the *other, held-out*
stations the same way the referee does (`score_band`'s `covariate_linear` path never fits anything
itself -- it only evaluates an already-declared value out-of-sample, and fitting and scoring on the
same rows would make that evaluation optimistically in-sample).

```
python examples/rung_b_l1_covariate_linear_fit.py
```

Named example: `tmax`/`Am`/`lag_fill` corrected on `wc_tree_frac` (tree-canopy land-cover
fraction), fit on 7 of the 12 GHCN stations the snapshot carries for that zone and scored on the
other 5. Pass `--zone`/`--target`/`--covariate`/`--n-fit-stations` to fit against a different cell,
covariate, or split (covariate must be one of `score.STATIC_COVARIATE_ALLOWLIST` -- see
`CONTRIBUTING.md`'s Rung B section for why that list is what it is).

## What happens when you run either

1. Downloads the public `snapshot-v2026.07` GitHub Release asset (cached under `.cache/snapshots/`
   -- gitignored, re-run is instant once cached).
2. Prints a short summary of the result.
3. Writes a real `claimed_report.json` to `examples/output/` (gitignored -- this is a scratch
   artifact of running the example, not something to commit).

Both scripts exit non-zero if the named example's claim no longer holds against whatever snapshot
they're run against (`--require-beats-grid`/`--require-pass`, on by default; pass
`--no-require-beats-grid`/`--no-require-pass` to just look at the numbers while exploring a
different cell) -- this is what makes "every example runs in CI" (roadmap Phase 2's own acceptance
bar) a real check rather than a script that merely doesn't crash.

## Turning either into a real submission

Neither script writes a `manifest.yaml` -- that's deliberately a separate step (a
`submit`/`make submit` scaffolding helper is its own follow-up ticket, not part of this slice; see
the repo's open issues). See `CONTRIBUTING.md`'s "Submission format" for the exact shape, and
`submissions/2026-07/001-nish-kishore-lagfill-bwk-tmin/manifest.yaml` for a real merged one to copy
from. The `claimed_report.json` either example writes is usable as-is for that submission's own
`claimed_report.json` field.
