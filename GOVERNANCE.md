# Governance

**Status:** this document describes the intended governance model once the automated referee and
promotion pipeline exist — see `CONTRIBUTING.md`'s own status note. `run_submission.py`,
`score_forward_eval.py`, `replay_downscaling.py`, and the private repo's `promote_from_public.py`
have not been built yet. Today, the maintainer role below is real and active; the "automated"
referee role is aspirational until that code ships.

## Roles

- **Maintainer (Nishant Kishore / CrisisReady).** Final decision-maker on this repository:
  license terms, what merges, when Rung C opens, and any dispute the automated process below
  doesn't resolve. Also a contributor — submissions authored by the maintainer go through the
  identical public path as anyone else's (see `CONTRIBUTING.md`); no submission from any author,
  maintainer included, skips independent re-verification before promotion.
- **The referee (automated).** `scripts/run_submission.py` (on submission) and
  `scripts/score_forward_eval.py` (monthly) are the actual scoring path. Neither is a person —
  they're deterministic code, reviewable in this repository like anything else. If you think the
  referee scored something wrong, the fix is a PR against the referee's code (with a test proving
  the bug), not an appeal to a human to override a specific submission's number.
- **Promotion to production** happens in the private serving repository
  (`crisisready/heat-risk-data-api`, `scripts/promote_from_public.py`), which independently
  re-derives every claimed metric against a private copy of the snapshot before publishing
  anything — a submission's own reported numbers are never trusted directly, regardless of who
  submitted them.

## How decisions get made

- **Provisional/official scoring**: fully mechanical, per `CONTRIBUTING.md`. No human judgment
  call in the ordinary case — a submission either clears the published bar or it doesn't.
- **Promotion (production gate publish)**: requires the independent re-derivation above to match
  the claim within the manifest's stated tolerance, then a human (the maintainer) reviews the
  tier-mix / delta-distribution diff `replay_downscaling.py` produces before confirming the
  publish. This human-in-the-loop step is deliberate — see `promote_from_public.py`'s own
  docstring — and is not a rubber stamp: a genuinely bad diff (e.g. a large `out_of_distribution`
  rate spike) is grounds to hold a promotion even after the numbers technically clear tolerance.
- **New covariates (research track)**: CI enforces the license/global/reproducible-fetch
  requirements mechanically; whether a finding is *interesting enough* to prioritize on the
  roadmap is a maintainer judgment call, made in the open on the relevant GitHub issue.
- **Rung C (new model code)**: not yet open. Opening it requires `contract.py`'s `ModelAdapter`
  protocol to be load-bearing (not just a stub) and an explicit decision on how untrusted model
  code is executed safely. That decision, and the process for reviewing a Rung C submission once
  it exists, will be published here before Rung C opens — it is out of scope for this document
  today.

## Disputes

If you believe a scoring result is wrong: open a GitHub issue with the submission ID, the exact
metric you're disputing, and (if possible) a minimal reproduction. Because the referee's fold
assignment is a deterministic, salted function of `snapshot_version` and `station_id` (see
`CONTRIBUTING.md`), any dispute should be independently reproducible by re-running the same
`heatready_downscaling.score.score_band` call — if it isn't, that's itself a bug worth a report.

## Security

No contributor Python executes anywhere in v1 (see the README's contribution-ladder section) —
submissions declare *what to run* from this repository's own code, at a pinned version, against a
published snapshot. This is a deliberate design choice, not an oversight: it removes the
joblib-pickle deserialization RCE surface that a "run arbitrary contributed model code" design
would otherwise require code review to contain. Report security concerns about the referee or the
promotion pipeline via GitHub's private vulnerability reporting on this repository, not a public
issue.

## Code of conduct

Be direct, be specific, argue with data. Disagreements about a scoring result or a model
approach are expected and welcome — resolve them by pointing at the deterministic, reproducible
scoring path above, not by escalation. Anything else (personal conduct issues) — contact the
maintainer directly.
