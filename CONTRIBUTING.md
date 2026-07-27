# Contributing

Thanks for helping close the dark cells. This document is the scoring contract this program is
being built around — read it before writing a submission, not after a provisional score surprises
you.

**Status: the automated submission pipeline described below is not open yet.** The band-paired
snapshot and the `heatready_downscaling` scoring library (`score.py`, `gates.py`, `contract.py`,
etc.) are real and installable today — `score_band`'s docstring and `tests/test_score.py` are the
actual, current scoring behavior, not a plan. But `run_submission.py` (the referee that would
reproduce a submission's claim and post a provisional score), `score_forward_eval.py` (the monthly
official cycle), and the `ledger/`/`docs/leaderboard.md` files this document references do not
exist in this repository yet — that automation is the next phase of this program. Until it ships,
opening a PR against `submissions/` will not be scored by anything automated. If you want to
experiment now, you can write a short script against the installed package (`snapshot.
read_band_partitions` + `score.score_band` + `contract.QRFModelAdapter`) and open a GitHub issue
with what you find — genuinely useful groundwork, just not yet a credited submission. Watch this
repository for when automated intake opens.

## Before you start

1. Read the [README](README.md) for the contribution ladder (Rung A/B/C) and the two tracks
   (serving-ready / research).
2. Download the current snapshot from this repository's latest GitHub Release (or resolve the
   Zenodo DOI for a permanent, citable copy). `snapshots/<version>/MANIFEST.json` pins the exact
   data your submission is scored against — a mismatched `manifest_sha256` is an automatic reject.
3. `pip install -e .` from a checkout of this repository to get `heatready_downscaling` at the
   version pinned in your target snapshot's manifest.

## Submission format

**This is the target format `run_submission.py` will consume once it exists (see the status note
above) — not a schema you can submit against today.** `--from-snapshot` in particular is not yet a
real flag on any script in this repository; the validator's current `--phase score` reads live
model output against a `--paired-in` JSON file, not a Parquet snapshot partition directly. Wiring a
snapshot-aware scoring entrypoint is part of what shipping the referee involves.

Once it exists, create `submissions/{YYYY-MM}/{NNN}-{your-github-username}-{slug}/manifest.yaml`:

```yaml
schema_version: 1
submission_id: "2026-08-001"
author:
  github: your-github-username
  name: "Your Name"
  orcid: null                # optional; used for credit attribution if present
  affiliation: null
track: serving-ready          # serving-ready | research
rung: A                       # A | B | C (C is not yet open — see README)
snapshot:
  version: "v2026.07"          # the current real snapshot version -- check the latest GitHub Release
  manifest_sha256: "…"        # pins the exact data; mismatch = auto-reject
claims:
  - model_version: "ds-2026.07-rf5"
    band_key: "lag_fill"
    targets: ["tmax", "tmin"]
    zones: ["Cfb", "BWk"]     # the (target, zone) cells you're claiming
method:
  kind: rerun-validator        # rerun-validator (A) | parameters (B) | model (C, not yet open)
  entrypoint: "scripts/validate_lagfill_downscaling.py"
  args: ["--phase", "score", "--from-snapshot", "snapshots/v2026.07",
         "--model-version", "ds-2026.07-rf5", "--band-key", "lag_fill"]
  package_version: "0.1.0"
  code_ref: null               # rung C only, once opened
  extra_covariates: []         # research track only: {name, source, url, license, global,
                                #   cadence, reproducible_fetch}
claimed_report: "claimed_report.json"
tolerance:
  rmse_improvement_pct_debiased_cv: 0.002
  bias_correction_c: 0.01
  rmse_qrf_c: 0.005
reproducibility:
  seed: 20260721
  runtime_notes: "…"
```

`claimed_report.json` must be exactly `heatready_downscaling.report.build_report`'s shape (the same
report envelope `validate_lagfill_downscaling.py`/`validate_forecast_downscaling.py` already
produce by calling it) — `report_schema_version`, `generated_by{tool,version,git_commit}`,
`snapshot_version`, `band_key`, `sample_requested`, `rows_sampled`, `rows_paired`,
`fidelity_check`, `by_target`. Do not invent a second report schema — reusing the existing one is
what lets `heatready_downscaling.gates.build_gate` (wrapped by `publish_band_gate.py`'s CLI) work
unchanged on your output.

Open a pull request with your `submissions/{...}/` directory. `validate-submission.yml` lints the
manifest and report schema on every push — it executes nothing. `run_submission.py` then
reproduces your claim from the public snapshot, scores it against the holdout, and posts a
provisional score as a PR comment.

## Hard constraints — read before you argue with a provisional score

**Provisional scores are for ranking and feedback only. They are never a gate decision.** The
provisional check is a zone-stratified 15% station holdout, computed in minutes so you get fast
feedback. `_MIN_ZONE_N` (minimum stations per zone) and `_BIAS_CV_MIN_STATIONS` (minimum distinct
stations for the bias cross-validation) will **not** be met on a 15% slice for thin zones — a thin
zone can show a promising provisional number that the real (official, monthly, forward-eval) check
will not confirm. This is expected, not a bug in the provisional check.

**Official scoring is monthly and forward-only.** Each cycle scores every active candidate against
station-days that did not exist in *any* snapshot version at the candidate's submission time
(verifiable from `snapshot.manifest_sha256`) — the only uncheatable holdout, since GHCN-Daily is a
free public dataset and a classic hidden test set is structurally impossible here. **A candidate
must win 2 consecutive official cycles before promotion to production.**

**Fold assignment is salted per snapshot version**, not fixed — `md5(f"{snapshot_version}:
{station_id}")`, deterministic within one snapshot but not gameable across snapshot versions by
re-submitting until a favorable station split appears. This salt, the `_AUTO_ENABLE_MARGIN`
constant, and `_MIN_ZONE_N` are all public in this repository's own source — that's deliberate,
not an oversight; the incentives this creates (e.g., "don't bother claiming a zone with <30
stations") are meant to be visible to contributors, not hidden gatekeeping.

**Per-band cycle lag is real and asymmetric.** CDS `reanalysis-era5-land` lags 2–3 months behind
present, and GHCN quality-control settles over several weeks. Concretely: **month M's official
cycle closes at M+2 for Open-Meteo-sourced bands (`lag_fill`, `forecast_lead*`), and at M+4 for the
`era5` band.** If you claim an `era5`-band cell, expect roughly twice the wait for your first
official result compared to a lag-fill claim — this is not your submission stalling.

**Zone list and band list are fixed by code you cannot change from a submission.** `_BAND_KEYS`
is a hardcoded tuple, coupled to `publish_band_gate.py`'s CLI `choices` and to the private serving
repo's own forecast-lead-days default. Lighting a new **zone** for an existing band is a data
publish (your submission can do this). Adding a new **lead day** or an entirely new **band** is a
code change in the private serving repository, out of scope for any submission here — don't submit
a `forecast_lead8` claim expecting it to be scoreable; it isn't a recognized `band_key` yet.

## What "Rung A" actually asks of you

Rung A is evaluation coverage: rerun the existing, unmodified validator (`validate_lagfill_downscaling.py`
or `validate_forecast_downscaling.py`) against a dark `(target, zone, band)` cell, using the
snapshot to fetch the ground truth and base values our infrastructure would otherwise need a paid
Open-Meteo key to fetch live. If the cell passes the SAME gate the model's already-live cells
passed, you've earned credit for lighting it — no new code, no new parameters, just running the
existing bar against data nobody had gotten to yet.

## What "Rung B" actually asks of you

Rung B is a published parameter: a `bias_correction[target][zone]` float (station-grouped-CV
validated, see `heatready_downscaling.score.score_band`'s own docstring for the exact
recalibration this represents), or a blend-kernel `(L_km, R_km, tau)` triple for the
distance-weighted nearby-station residual blend. Same scoring path as Rung A; the difference is
what your `manifest.yaml`'s `method` block declares you ran.

## Research track

New covariates are welcome if they are global (not one-country-specific), reproducibly fetchable
(a documented, automatable source — not a one-off manual download), and either openly licensed or
licensable by us for redistribution. CI enforces `extra_covariates[].license` against an allowlist
of SPDX identifiers plus a `proprietary-licensed` escape hatch that requires a named licensor and
flags the submission for manual review. Research-track wins are advisory: they do not promote to
production automatically, but they do feed a ranked roadmap and (with your permission) public
credit for the finding.

**Seeded first research-track issue: serving-consistent population density.** Training's
`pop_density_per_km2` comes from LandScan Global over a ~1 km station buffer; the live serving
path computes the analogous feature from WorldPop over each polygon's own area — a real,
disclosed mismatch (see `DATA_LICENSE`). Swapping the numerator, denominator, or both, and
measuring the effect, is a fully self-contained research-track contribution — no new data pipeline
needed, since WorldPop's rasters are already mirrored and referenced in the relevant GitHub issue.

## Questions

Open a GitHub issue. For anything security-relevant, see `GOVERNANCE.md`.
