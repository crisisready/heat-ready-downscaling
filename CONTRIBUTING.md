# Contributing

Thanks for helping close the dark cells. This document is the scoring contract this program is
built around. Read it before writing a submission, not after a provisional score surprises you.

**Status: the automated submission pipeline described below is built, wired, and live-verified.**
The band-paired snapshot, the `heatready_downscaling` scoring library (`score.py`, `gates.py`,
`contract.py`, etc.), `run_submission.py` (the referee, run by `referee.yml` on every submission
PR), `score_forward_eval.py` (the monthly official cycle, run by `score-forward-eval.yml` on a
cron), and the `ledger/`/`docs/leaderboard.md` files are all real, tested, and live in this
repository. Opening a PR against `submissions/` is scored automatically; the first submission
(`2026-07-001`) went through the full referee and ledger pipeline end to end. If the referee gets
something wrong, open a GitHub issue with the submission ID and what you expected.

## Before you start

1. Read the [README](README.md) for the contribution ladder (Rung A/B/C) and the two tracks
   (serving-ready / research).
2. Download the current snapshot from this repository's latest GitHub Release, or resolve the
   Zenodo DOI for a permanent, citable copy. `snapshots/<version>/MANIFEST.json` pins the exact
   data your submission is scored against. A mismatched `manifest_sha256` is an automatic reject.
3. `pip install -e .` from a checkout of this repository to get `heatready_downscaling` at the
   version pinned in your target snapshot's manifest.

## Submission format

This is the real schema `run_submission.py` validates every submission PR against
(`heatready_downscaling.submission.MANIFEST_SCHEMA`). A mismatch fails CI immediately. Note that
`method.entrypoint`/`args` are provenance metadata only: they document what you ran locally to
produce `claimed_report.json`, but the referee does not execute them. It independently reproduces
your claim from the snapshot via `score.score_band` and `contract.FrozenPredictionAdapter`, then
compares the two reports within `tolerance`. Your own claimed numbers are never trusted directly.

Create `submissions/{YYYY-MM}/{NNN}-{your-github-username}-{slug}/manifest.yaml`:

```yaml
schema_version: 1
submission_id: "2026-08-001"
author:
  github: your-github-username
  name: "Your Name"
  orcid: null                # optional; used for credit attribution if present
  affiliation: null
track: serving-ready          # serving-ready | research
rung: A                       # A | B | C -- A and B (bias_correction/delta_scale shape only, see
                              # "What Rung B actually asks of you" below) are scoreable by the
                              # automated referee today; C is not yet open at all, see README
snapshot:
  version: "v2026.07"          # the current real snapshot version -- check the latest GitHub Release
  manifest_sha256: "e0660fff6397c5e4760927fdeb6f40b4cc241562df82d5cfca2e54f2e084a204"  # v2026.07's real, current value (post --phase predict); pins the exact data, mismatch = auto-reject
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
  candidate: null               # rung B only -- see the Rung B example below
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

`claimed_report.json` must be exactly `heatready_downscaling.report.build_report`'s shape, the same
report envelope `validate_lagfill_downscaling.py`/`validate_forecast_downscaling.py` already
produce by calling it: `report_schema_version`, `generated_by{tool,version,git_commit}`,
`snapshot_version`, `band_key`, `sample_requested`, `rows_sampled`, `rows_paired`,
`fidelity_check`, `by_target`. Do not invent a second report schema. Reusing the existing one is
what lets `heatready_downscaling.gates.build_gate` (wrapped by `publish_band_gate.py`'s CLI) work
unchanged on your output.

Open a pull request with your `submissions/{...}/` directory. `referee.yml` runs on every push:
schema validation first (manifest and report shape, no code executed), then `run_submission.py`
reproduces your claim from the public snapshot, scores it against the holdout, and posts a
provisional score as a PR comment.

## Hard constraints, read before you argue with a provisional score

**Provisional scores are for ranking and feedback only. They are never a gate decision.** The
provisional check is a zone-stratified 15% station holdout, computed in minutes so you get fast
feedback. `_MIN_ZONE_N` (minimum paired ROWS per zone -- not stations; see the two-distinct-stations note in the Rung B section, which is a separate and stricter bar) and `_BIAS_CV_MIN_STATIONS` (minimum distinct
stations for the bias cross-validation) will not be met on a 15% slice for thin zones. A thin zone
can show a promising provisional number that the real, official, monthly forward-eval check will
not confirm. This is expected, not a bug in the provisional check.

**Official scoring is monthly and forward-only.** Each cycle scores every active candidate against
station-days that did not exist in any snapshot version at the candidate's submission time
(verifiable from `snapshot.manifest_sha256`). This is the only uncheatable holdout, since GHCN-Daily
is a free public dataset and a classic hidden test set is structurally impossible here. A candidate
must win 2 consecutive official cycles before promotion to production.

**Fold assignment is salted per snapshot version, not fixed.**
`md5(f"{snapshot_version}:{station_id}")` is deterministic within one snapshot but not gameable
across snapshot versions by re-submitting until a favorable station split appears. This salt, the
`_AUTO_ENABLE_MARGIN` constant, and `_MIN_ZONE_N` are all public in this repository's own source.
That is deliberate. The incentives this creates (for example, don't bother claiming a zone with
fewer than 30 stations) are meant to be visible to contributors, not hidden gatekeeping.

**Per-band cycle lag is real and asymmetric.** CDS `reanalysis-era5-land` lags 2 to 3 months behind
present, and GHCN quality control settles over several weeks. Concretely: month M's official cycle
closes at M+2 for Open-Meteo-sourced bands (`lag_fill`, `forecast_lead*`), and at M+4 for the
`era5` band. If you claim an `era5`-band cell, expect roughly twice the wait for your first
official result compared to a lag-fill claim. That is not your submission stalling.

**Zone list and band list are fixed by code you cannot change from a submission.** `_BAND_KEYS` is
a hardcoded tuple, coupled to `publish_band_gate.py`'s CLI `choices` and to the private serving
repo's own forecast-lead-days default. Lighting a new zone for an existing band is a data publish;
your submission can do this. Adding a new lead day or an entirely new band is a code change in the
private serving repository, out of scope for any submission here. Don't submit a `forecast_lead8`
claim expecting it to be scoreable; it isn't a recognized `band_key` yet.

## What "Rung A" actually asks of you

Rung A is evaluation coverage: reproduce the existing, unmodified scoring bar
(`heatready_downscaling.score.score_band`, applying the shipped model via
`contract.FrozenPredictionAdapter`) against a dark (target, zone, band) cell, using the public
snapshot to fetch the ground truth and base values our infrastructure would otherwise need a paid
Open-Meteo key to fetch live -- exactly what the referee itself runs (`run_submission.py`'s
`reproduce()`), and exactly what `examples/rung_a_evaluate_dark_cell.py` does end to end: see
`examples/README.md` for a worked, CI-executed run against a real dark cell.

**Correction (2026-08-26):** this section previously named
`validate_lagfill_downscaling.py`/`validate_forecast_downscaling.py` as what to rerun. Neither is
snapshot-runnable -- both need private-repo modules and live Aurora/Open-Meteo access (see their
own docstrings) -- and `method.entrypoint` naming one is provenance metadata only, never executed
by the referee (see "Submission format" above). If you have access to run them from the private
repo's own bastion, that route still exists; from this public repo alone, `score_band` +
`FrozenPredictionAdapter` against the snapshot -- the example script above -- is the actual path.
If the cell passes the same gate the model's already-live cells passed, you've earned credit for
lighting it. No new code, no new parameters, just running the existing bar against data nobody had
gotten to yet.

Not every dark cell clears the bar just because you rerun the validator against it. The first real
submission through this pipeline (`2026-07-001`, `lag_fill`/`BWk`/`tmin`) targeted the `lag_fill`
band's 9 missing zones and found that only one of the nine, `BWk`, actually beats the grid baseline
with the current model. The rest, including `Cfb`, do not clear the gate yet. Check the real
numbers before writing a manifest, not just whether a cell is dark.

## What "Rung B" actually asks of you

**Status (2026-08-25): open for a `bias_correction[target][zone]` float, a `delta_scale`
`{scale, offset}` affine correction, or a covariate-linear correction that varies continuously
with a static per-location covariate (see below).** Declare your proposed value in `manifest.yaml`'s
`method.candidate` (`{target: {zone: {"bias_correction_c": float}}}` or
`{target: {zone: {"scale": float, "offset": float}}}`, `method.kind: parameters`) -- the referee
scores that DECLARED value out-of-sample (`heatready_downscaling.score.score_band`'s
`proposed_correction` parameter; see its own docstring for the exact scoring, and
`score_forward_eval.py`'s docstring for how the monthly official cycle uses the same declared
value, not a freshly re-derived one). This is deliberately a smaller bar than Rung A's own
mechanical fit: the referee never trusts your claimed numbers, but it also never fits a value for
you -- station-grouped CV validates whether YOUR number generalizes, not whether some other number
would have scored better.

**Fitting the value is on you; the referee only scores it.** `score.score_band`'s
`covariate_linear` path (used below) independently evaluates and CIs an *already-declared*
intercept/slope -- it does not fit one from data. `examples/rung_b_l1_covariate_linear_fit.py` is a
complete worked example of both halves against a real snapshot: a small OLS fit on a named station
subset, then the same `score_band` reproduction the referee runs, producing a real
`claimed_report.json`. See `examples/README.md`.

**Also open, as of this change**: a correction that varies continuously with a static
per-location covariate, instead of being one constant for a whole zone. A flat number cannot
express a correction that gets stronger as you move inland from the coast, and there is no zone or
subzone small enough to fake it.

One caveat on `coast_dist_km` specifically, because it will bite somebody: it measures distance to
the nearest **ocean** coastline and does not know about lakes. Chicago, sitting on Lake Michigan,
measures about 950 km. That is the right number for the sea and the wrong number for the water
Chicago actually responds to, so this is not the covariate to reach for on a large-lake city. Declare `basis`, an `intercept`, and one
covariate term:

```yaml
    tmin:
      BSh:
        basis: raw_grid                  # or model_delta -- see below, this choice matters
        intercept: 0.782
        terms:
          - covariate: coast_dist_km
            slope: -0.0413               # degrees C per km inland
        valid_range: [[0.0, 25.0]]       # the covariate range your fit is actually evidence over
```

Three things to know before you use it:

- **`basis` decides what your correction is added to, and it is not a formatting detail.**
  `model_delta` adds to the model's own predicted delta, which is what the two flat shapes
  always did. `raw_grid` adds directly to the raw grid value. If the cell you are claiming is one
  where the model does not currently apply -- a zone that fails its own cross-validation gate, or
  one with no measured stations -- then a `model_delta` correction has nothing to attach to and
  will score as "not scored", while a `raw_grid` one is scoreable. Several of the darkest cells
  are exactly like this.
- **Your covariate must be on the allowlist in `score.STATIC_COVARIATE_ALLOWLIST`.** Everything
  else is excluded for a stated reason, and `score.COVARIATE_EXCLUSIONS` gives the reason for
  each. Four kinds of exclusion, so you can tell in advance whether something you want is
  admissible: day-varying quantities (wind, humidity, the grid value itself), because a static
  covariate is what lets a promotion precompute one value per served polygon with nothing new
  running in the serving path; the model's internal derived feature names, because the scorer
  reads real snapshot *columns* off each row and a derived name silently matches nothing;
  station-scoped columns like `elevation_m`, because a served polygon has no station and so has
  no value to compile (`elevation_mean_m` is the per-location form, and it is allowlisted); and
  compass bearing `aspect_deg`, because a straight-line slope on a circular variable is not
  interpretable. One more, `pop_density_per_km2`, is held back for a different reason: it has a
  known training/serving mismatch (this repo's issue #1), so a slope fitted on snapshot values
  would be applied against different values in production.
- **One covariate term, two at the most.** This is a low cap on purpose. Our own two-covariate
  fit for Valencia was measurably *worse* on held-out stations than the one-covariate version --
  three free parameters against eight training stations per fold. The referee also reports what
  your correction would have scored with the intercept alone, so a covariate term that does not
  beat a flat constant will show up as not earning its keep.

Scoring for this shape reports more than one number, because one number was hiding real results.
You get metrics for all days and for hot days separately (hot days are where a heat product
actually matters, and an effect concentrated there gets washed out by whole-year averaging), a
95% confidence interval on the improvement from resampling whole stations, and a verdict of
`pass`, `candidate`, or `fail`. `candidate` means the improvement is real but the interval still
includes zero -- usually a sign you need more stations, not that the finding is wrong. A `pass`
requires the interval to exclude zero, which is a stricter bar than the point estimate alone.

**That interval requirement is what the monthly cycle actually enforces, for every Rung B
shape** -- not just this one. A proposal whose interval includes zero does not win a cycle, so it
cannot accumulate the two consecutive wins promotion needs. Two practical consequences worth
knowing before you submit:

- **You need at least two distinct stations among the rows your correction actually applies
  to** -- not merely in the zone. The interval comes from resampling whole stations, and it is
  computed over the scored rows only, so a narrow `valid_range` or a sparsely populated covariate
  can leave you with a dozen stations in the zone and one among your scored rows. One station
  produces no interval, and no interval cannot exclude zero. Row count is not a substitute:
  forty station-days from a single station still gives you nothing to resample. A cell in that
  position is reported as `insufficient_n`, not as a loss -- "we cannot measure this" and "this
  is not good enough" are different answers.
- **This was not always true.** Until 2026-08-26 the gate was the point estimate alone, while
  this paragraph already promised the interval bar. The documented bar was stricter than the
  enforced one. It is now the same bar, and this note stays because a contributor calibrating how
  much to trust our stated gates deserves to know which ones have been fixed rather than always
  been true.

**Not yet open**: the blend-kernel `(L_km, R_km, tau)` triple for the distance-weighted
nearby-station residual blend (`validate_station_blend.py`'s own scoring path needs a parallel
extension to accept a declared triple instead of only grid-searching one -- tracked separately,
not part of the 2026-08-25 Rung B opening above).

A Rung B manifest's `method` block declares its candidate alongside the other Rung A fields:

```yaml
rung: B
method:
  kind: parameters
  entrypoint: "scripts/validate_lagfill_downscaling.py"   # what you ran locally to arrive at this value -- provenance only, not executed
  args: []
  package_version: "0.1.0"
  candidate:
    tmax:
      Cfb: {bias_correction_c: 0.8}
    tmin:
      Cfb: {scale: 0.92, offset: 0.1}
```

## Rung C, when it opens — what to prepare

**Rung C is not open.** `GOVERNANCE.md` records the decision that governs it and exactly which
opening condition is still missing. This section exists so that if you are working toward a
locally-trained model, you build the right thing rather than discovering the rules afterwards.

**Your code will never run on our infrastructure.** A Rung C submission is a *recipe plus data*:
the training procedure described well enough that our own pipeline can reproduce it, and the
inputs declared through the same `data_sources` block the licensing gate already checks. We
retrain from that recipe and promote what we built. This is not a sandbox we have not got round
to building — it is the decision, and it is what keeps the pickle-deserialisation surface out of
the programme entirely.

The practical consequence, worth knowing before you invest months: **if your procedure cannot be
described well enough for us to reproduce it, it cannot be submitted, however good your model
is.** A model artifact alone is not a Rung C submission.

Two things you can do now that will not be wasted:

- **Register what you have.** `registry/` takes an entry for a model that is not promoted and not
  serving — `local/seoul-sdot-v1` is exactly that, a real local model recorded with its evidence,
  its holdout design, and its blockers stated in the open. An entry at status `registered` is a
  record of something that exists, and needs no artifact checksum.
- **Hold your evidence to the bar that already exists.** A genuine held-out validation, a
  cluster-bootstrap interval that excludes zero, and an honestly stated coverage bound are what
  the scoring vocabulary already reports for Rung B, and they are what a Rung C claim will be
  read against.

## Research track

New covariates are welcome if they are global (not one-country-specific), reproducibly fetchable
(a documented, automatable source, not a one-off manual download), and either openly licensed or
licensable by us for redistribution.

## Declaring data you bring

Anything you use beyond the published snapshot has to be declared, and the declaration is
checked. Two places to put it:

- `method.extra_covariates[]` for a new model covariate (the research track).
- `method.data_sources[]` for any other data input: a local model's training data, a sensor
  feed, a coastline layer.

A `data_sources` entry needs a `name`, a `license`, a `reproducible_fetch` URL, and a
`redistribution_tier` of `unrestricted`, `attribution-required`, or `no-redistribution`:

```yaml
method:
  data_sources:
    - name: "Natural Earth 1:10m physical coastline"
      license: "CC0-1.0"
      redistribution_tier: unrestricted
      reproducible_fetch: "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
      sha256: "bfa04cdbcbef07ef90dfca1dabb48062eca29900a113df0f389303e255484017"
```

`license` must be an SPDX identifier from the allowlist in
`heatready_downscaling.licensing.SPDX_ALLOWLIST`, and identifiers are case-sensitive
(`CC-BY-4.0`, not `cc-by-4.0`). Non-commercial variants are not accepted: HeatReady's own
`DATA_LICENSE` carries no such restriction, so accepting NC data would make the published
snapshot's stated terms wrong for part of its own contents.

**Your tier has to match your licence.** A licence carrying an attribution obligation
(`CC-BY-4.0`, `CC-BY-3.0`, `ODC-BY-1.0`, `OGL-UK-3.0`, `Apache-2.0`, `MIT`, `BSD-3-Clause`)
cannot be declared `unrestricted` -- use `attribution-required`, so the snapshot carries the
notice forward the way `DATA_LICENSE` already does for LandScan/ORNL. Declaring
`unrestricted` for one of those is rejected rather than quietly accepted, because the
alternative is republishing someone's data with the attribution stripped.

**Share-alike licences route to a maintainer rather than passing automatically.** `ODbL-1.0`
and the `CC-BY-SA`/`GPL`/`AGPL` family are real and usable, but they impose terms on
derivative work that conflict with `DATA_LICENSE`'s CC BY 4.0 "No additional restrictions"
clause. That is a decision with legal weight, so it goes to a person.

If your data has a real licence with no SPDX identifier -- common for municipal and
national-agency agreements -- use `license: proprietary-licensed` and name who granted it in
`licensor`. That is not a way around the allowlist; it routes the submission to a maintainer
instead of passing automatically. `no-redistribution` data does the same: a local model may
legitimately train on data we cannot republish, but somebody has to decide that, so it never
passes silently.

**Note on this section's history, since it matters for how much you should trust our docs.**
This paragraph previously said CI enforced an SPDX allowlist and a HEAD check. It did not --
the check was documented from July and only built on 2026-08-25. Any licence string passed
until then. Nothing wrong was ingested (no submission had declared a covariate), but the
control was real only on paper, and we would rather say so than quietly fix it. It is real
now: enforced in `validate_manifest` itself, so the referee rejects a bad licence, and again
in the `Data licensing` workflow, which additionally checks that each `reproducible_fetch`
URL resolves.

Research-track wins are advisory: they do not promote to
production automatically, but they do feed a ranked roadmap and, with your permission, public
credit for the finding.

**Seeded first research-track issue: serving-consistent population density.** Training's
`pop_density_per_km2` comes from LandScan Global over a roughly 1 km station buffer. The live
serving path computes the analogous feature from WorldPop over each polygon's own area, a real,
disclosed mismatch (see `DATA_LICENSE`). Swapping the numerator, denominator, or both, and
measuring the effect, is a fully self-contained research-track contribution. No new data pipeline
is needed, since WorldPop's rasters are already mirrored and referenced in the relevant GitHub
issue.

## Questions

Open a GitHub issue. For anything security-relevant, see `GOVERNANCE.md`.
