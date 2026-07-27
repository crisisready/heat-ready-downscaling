# heat-ready-downscaling

Neighborhood-resolution downscaling for [HeatReady](https://crisisready.io)'s heat-risk model —
opened to outside contributors. This repository is canonical for the downscaling model's training,
validation, and scoring code; [`heat-risk-data-api`](https://github.com/crisisready/heat-risk-data-api)
(private) consumes it as a pinned package and owns serving only.

**Current status.** The band-paired snapshot and the `heatready_downscaling` scoring library
(`score.py`/`gates.py`/`report.py`/`contract.py`) are live and real — download the snapshot from
this repository's latest Release and score against it directly with the package. The **automated**
submission pipeline this document also describes (a PR-based referee, monthly forward-eval,
the credit ledger and leaderboard) has not shipped yet — that's the next phase of this program, not
built at the time of this repository's first public release. Until it ships, treat the submission
format/workflow sections below as the design contributors will submit against, not something you
can open a PR for today. Watch this repository or the linked GitHub issues for when it opens.

## Why this exists

The downscaling model gates corrections per **(target, zone, band)**: 2 targets (`tmax`/`tmin`) ×
19 Köppen climate zones × 9 bands (base ERA5 + `lag_fill` + `forecast_lead1..7`) = **342 cells**.
As of this repository's creation, 166 pass validation; 176 are dark — not because the model can't
work there, but because nobody has yet run the validator against those cells with enough data to
clear the bar.

The model drifts heterogeneously: there is no single global summer, and every region continuously
accrues new ground-truth observations (GHCN-Daily, free and public) that could re-validate a dark
cell. Rather than scale an internal modeling team to chase this, HeatRisk/HeatReady opens the
problem to the scientific community: contributors work here, are scored against a bar that already
exists in code, and (once the scoring harness ships) will earn durable per-cell author credit with
a recorded tenure — visible in `ledger/credit.jsonl` and `docs/leaderboard.md`, neither of which
exists yet (see "Current status" above).

Two properties make this cheap rather than risky:

1. **Gate zone keys are free-form strings**, read from S3 with a fail-closed default. Lighting a
   dark cell is **a data publish, not a code deploy** to the serving side.
2. **This repo already supports model-free contribution.** A `bias_correction[target][zone]` float,
   station-grouped-CV-validated, is how several `tmax` zones per forecast lead already pass —
   no retrained model required to move the needle.

## Contribution ladder

| Rung | What | Status |
|---|---|---|
| **A** | Evaluation coverage — run the validator on a dark cell using the published snapshot | designed; automated intake not yet open |
| **B** | Published parameters — bias constants, blend-kernel `L_km`/`R_km`/`tau` | designed; automated intake not yet open |
| **C** | New model code | deferred — see `docs/rung-c.md` (not yet published) |

**v1 executes only this repository's own code.** A submission declares *what to run* (an
entrypoint + args against the published snapshot); the referee runs `heatready_downscaling` at a
pinned version. No contributor Python executes anywhere in v1 — this makes the joblib-pickle RCE
surface irrelevant rather than something review must contain, and removes an entire class of
"your environment differs from mine" disputes. (The referee itself — `run_submission.py` — is part
of the not-yet-shipped automation above; see "Current status.")

## Two tracks

- **serving-ready** — the existing 21 features only. A winning submission is promotable to
  production immediately (after independent re-verification — see below).
- **research** — new covariates allowed if global, reproducibly fetchable, and either open or
  licensable by us. Wins are advisory and feed a ranked roadmap; they do not promote automatically.

## How scoring will work

This is the design the not-yet-built automated referee will implement (see "Current status")
— `heatready_downscaling.score.score_band` itself is real and already scoreable by hand today:

- **Provisional** (minutes, on submission): a zone-stratified station holdout. Ranking and
  feedback only — **never a gate decision**. See `CONTRIBUTING.md` for why thin zones can't
  clear the real bar on a 15% slice.
- **Official** (monthly): a rolling forward-evaluation on station-days that did not exist in any
  snapshot at the candidate's submission time — the only uncheatable holdout, since GHCN-Daily is
  free and anyone can verify the eval set. **A candidate is promoted to official status after 2
  consecutive winning cycles.**

Every promotion will be independently re-derived against a private copy of the snapshot before
anything reaches production — a submission's own claimed numbers are never trusted directly.

## Getting the snapshot

The band-paired training/validation snapshot (GHCN ground truth + every band's base value, paired
per station-day) is distributed as a GitHub Release asset plus a Zenodo DOI — **not** hosted on
crisisready.io's own infrastructure, so a citable, versioned copy always exists independent of us.
See the latest release for the current `snapshot_version` and download link.

## Getting started

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full submission process,
[`GOVERNANCE.md`](GOVERNANCE.md) for how decisions and disputes are handled, and
[`PROVENANCE.md`](PROVENANCE.md) for where each file in this repository actually came from —
several were extracted from a private, unmerged branch rather than written fresh here.

## License

Code: [Apache-2.0](LICENSE). Data (snapshots, submissions, ledgers): [CC BY 4.0](DATA_LICENSE),
with a required LandScan/ORNL attribution clause — see `DATA_LICENSE` for the exact terms and a
disclosed train/serve population-density mismatch.
