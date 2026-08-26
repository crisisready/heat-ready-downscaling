# Governance

## How this document is maintained

**Every rule here describes something that exists in code, or is explicitly marked as not yet
built. No exceptions, and the distinction is never left to inference.**

This principle is written down because the repository has already failed it. From July until
2026-08-25, `CONTRIBUTING.md` told contributors that "CI enforces `extra_covariates[].license`
against an allowlist of SPDX identifiers plus a `proprietary-licensed` escape hatch", and
`submission.py`'s own schema comment said the same. Neither was true: there was no allowlist, no
escape-hatch check, and no licensing logic in any workflow, so any licence string at all passed
for six weeks. Separately, until 2026-08-26 this file's own status note said two scripts had not
been built when both had been merged and were in use, and `CONTRIBUTING.md` promised that a
passing submission "requires the interval to exclude zero" while the gate deciding monthly wins
used the point estimate alone.

None of those caused visible harm, because no external contributor had exercised them yet. That
is luck, not a defence. A programme whose entire premise is that its gates are real and publicly
inspectable cannot have documented gates that do not exist — the docs are the product, as much
as the scoring code is. **A control that is documented and unimplemented is worse than an
admitted gap**, because a reader calibrates their trust on it.

So: if you are editing this file and cannot point at the code that enforces the sentence you are
writing, mark it. "Designed, not built" and "not open yet" are respectable statements. An
aspiration written in the present tense is not.

**Status (2026-08-26).**

Real, tested, and live:

- `run_submission.py` and `score_forward_eval.py` — the automated referee, live-verified end to
  end (submission `2026-07-001` merged through the full pipeline).
- `replay_downscaling.py`, and the private repo's `scripts/promote_from_public.py` — the
  production-promotion side. This file previously said neither had been built; both have been
  merged since, and the claim was stale rather than cautious.
- The licensing gate (`heatready_downscaling.licensing`, `scripts/check_data_licensing.py`, the
  `Data licensing` workflow) — the control this section's principle exists because of.
- The model registry (`heatready_downscaling.registry`, `registry/`, the `Registry` workflow),
  with three retroactively registered entries.

Designed and merged as a design, NOT implemented:

- **Rung D** (contributed data). `docs/design-2026-08-25-rung-d-contributed-data.md`. No code.
- **The internal-evidence track** (below). No code.

Not open:

- **Rung C** (contributed model code). See its own section below for exactly what remains.

## Roles

- **Maintainer (Nishant Kishore / CrisisReady).** Final decision-maker on this repository: license
  terms, what merges, when Rung C opens, and any dispute the automated process below doesn't
  resolve. Also a contributor. Submissions authored by the maintainer go through the identical
  public path as anyone else's (see `CONTRIBUTING.md`). No submission from any author, maintainer
  included, skips independent re-verification before promotion.
- **The referee (automated).** `scripts/run_submission.py` (on submission) and
  `scripts/score_forward_eval.py` (monthly) are the actual scoring path. Neither is a person. They
  are deterministic code, reviewable in this repository like anything else. If you think the
  referee scored something wrong, the fix is a PR against the referee's code, with a test proving
  the bug, not an appeal to a human to override a specific submission's number.
- **Promotion to production** happens in the private serving repository
  (`crisisready/heat-risk-data-api`, `scripts/promote_from_public.py`), which independently
  re-derives every claimed metric against a private copy of the snapshot before publishing
  anything. A submission's own reported numbers are never trusted directly, regardless of who
  submitted them.

## How decisions get made

- **Provisional/official scoring**: fully mechanical, per `CONTRIBUTING.md`. No human judgment
  call in the ordinary case. A submission either clears the published bar or it doesn't.
- **Promotion (production gate publish)**: requires the independent re-derivation above to match
  the claim within the manifest's stated tolerance, then a human (the maintainer) reviews the
  tier-mix/delta-distribution diff `replay_downscaling.py` produces before confirming the publish.
  This human-in-the-loop step is deliberate, see `promote_from_public.py`'s own docstring, and it
  is not a rubber stamp. A genuinely bad diff (for example a large `out_of_distribution` rate
  spike) is grounds to hold a promotion even after the numbers technically clear tolerance.
- **New covariates (research track)**: CI enforces the license/global/reproducible-fetch
  requirements mechanically. Whether a finding is interesting enough to prioritize on the roadmap
  is a maintainer judgment call, made in the open on the relevant GitHub issue.
- **Rung C (new model code)**: **still not open.** This document previously said the decision
  on executing untrusted model code "will be published here before Rung C opens". Publishing it
  now, along with an honest account of which opening conditions are met.

  **The decision: contributor code never executes, anywhere, at any rung.** A Rung C submission
  is a *recipe plus data*, not an artifact we load. What gets reviewed, retrained, and promoted
  is a model **our own pipeline built** from the contributor's declared training procedure and
  declared inputs. This keeps the `joblib`-pickle deserialisation surface out of the programme
  entirely, which is the same reasoning that made v1 refuse to execute contributor Python — the
  answer to "how do we run untrusted model code safely" turns out to be that we do not run it.

  A consequence contributors need before they invest effort: **credit and promotion attach to
  the recipe and the data, not to your artifact.** If your procedure cannot be described well
  enough for us to reproduce it, it cannot be submitted at Rung C, however good the model is.

  Opening conditions, stated against reality rather than intent:

  | Condition | State |
  |---|---|
  | `contract.ModelAdapter` load-bearing, not a stub | **Met.** `score.score_band` depends only on the protocol, and `contract.FrozenPredictionAdapter` is a real second implementation of it. |
  | An explicit decision on executing untrusted model code | **Met** — the decision above. |
  | A harness that retrains from a declared recipe | **Not built.** This is what remains. |
  | A way to state a model's claims and evidence machine-readably | **Met.** `heatready_downscaling.registry`, with `local/seoul-sdot-v1` registered as a worked example of an `artifact_route` entry. |

  So Rung C is one missing component away, not a research question. It stays closed until that
  component exists, and this table is the thing to update when it does.

## The internal-evidence track — DESIGNED, NOT BUILT

**No code implements any of this.** It is recorded here because it is a decided direction that
changes what the public path means, and a contributor reading the rules deserves to know it is
coming rather than discovering it in a changelog.

**The problem it addresses.** Official scoring is monthly and forward-only, and a candidate must
win two consecutive cycles before promotion. Band lag makes the real wall-clock worse than "two
months": month M's cycle closes at M+2 for Open-Meteo bands and M+4 for `era5`, so two
consecutive `era5` wins is roughly five to six months. During a heat emergency that is not a
tenable clock for a correction we already have strong evidence for.

**The insight is that the two-cycle rule is an anti-gaming device, not an accuracy device.**
Read what it defends against: a contributor picking a favourable slice of history, or
resubmitting until a fold split flatters them. That is why the holdout is forward-only — GHCN-Daily
is public and growing, so a classic hidden test set is structurally impossible — and why the fold
hash is salted per snapshot version. None of that threat model describes our own team's research,
where we control the data collection, the fit, and the review.

So the design is **not** "make the public clock faster". The public clock should not move; it is
the only uncheatable thing available. It is to stop asking one gate two different questions:

| Track | Question | Threat | Defence |
|---|---|---|---|
| Public | Is this contributor's number trustworthy? | Gaming | Forward-only holdout, salted folds, two cycles, no human judgement. **Unchanged.** |
| Internal | Is this evidence strong enough to serve? | Self-deception — an under-powered fit, a leaked holdout, a base mismatch, a result that only exists whole-year-averaged | A written evidence bar plus a named human who signs it |

**What would make it safe, if it is built.** Eligibility is structural rather than trust-based:
the submitter must be a repo maintainer or collaborator (CI can check the org), and the
validation design — holdout, strata, target, covariate — must be **pre-registered** in the
registry manifest before the numbers are filled in, with the pre-registration's own git commit in
the record. Pre-registration is what substitutes for the forward-only holdout: it removes the
same "kept trying until it worked" degree of freedom, in days rather than months. That is the
mechanism, not "we trust ourselves".

The bar, all required: a genuine held-out validation of the pre-registered design; a cluster
bootstrap CI on the improvement that **excludes zero**, per target, in the stratum claimed;
the result reported both at the stratum the product serves and whole-year; a stated coverage
bound and failure mode; independent re-derivation through `promote_from_public.py`'s existing
machinery, **unchanged**; the `replay_downscaling.py` diff reviewed by the maintainer and
`--confirm` typed by a human, **unchanged**; and a **time-boxed** promotion that re-enters the
ordinary public cycle in parallel and comes down if that cycle later contradicts it.

It shortens the waiting, never the checking. And it would not enter the credit ledger's
`tenure_start` path — internal promotions would get their own record type, so the leaderboard and
the ledger stay a clean record of *public* contribution.

**Not decided, and deliberately not designed:** an emergency path that lowers the *evidence* bar
rather than the waiting. The standing answer during an emergency is to serve the raw grid and say
so, which is what the system already does, and which follows from the principle the whole design
rests on: a wrong-but-confident correction is worse than an honest fallback.

## Disputes

If you believe a scoring result is wrong, open a GitHub issue with the submission ID, the exact
metric you're disputing, and, if possible, a minimal reproduction. Because the referee's fold
assignment is a deterministic, salted function of `snapshot_version` and `station_id` (see
`CONTRIBUTING.md`), any dispute should be independently reproducible by re-running the same
`heatready_downscaling.score.score_band` call. If it isn't, that's itself a bug worth a report.

## Security

No contributor Python executes anywhere in v1 (see the README's contribution-ladder section).
Submissions declare what to run from this repository's own code, at a pinned version, against a
published snapshot. This is a deliberate design choice, not an oversight: it removes the
joblib-pickle deserialization RCE surface that a "run arbitrary contributed model code" design
would otherwise require code review to contain. Report security concerns about the referee or the
promotion pipeline via GitHub's private vulnerability reporting on this repository, not a public
issue.

## Code of conduct

Be direct, be specific, argue with data. Disagreements about a scoring result or a model approach
are expected and welcome. Resolve them by pointing at the deterministic, reproducible scoring path
above, not by escalation. Anything else, personal conduct issues, contact the maintainer directly.
