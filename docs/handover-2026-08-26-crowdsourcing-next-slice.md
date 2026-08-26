# Handover: the approved next slice for the crowdsourcing program

Written by `gu-dev_heatready-crowdsourcing-next` for whichever worker picks this up next. Nishant
approved the scope below (via `AskUserQuestion`, 2026-08-26) but asked that execution be handed
to a fresh worker rather than continued in the same session. This doc plus the GitHub board are
that handover — read both before writing code.

## Ground truth to read first, in this order

1. `GOVERNANCE.md`'s "Status (2026-08-26)" section — what's real, what's designed-not-built, what's
   not open, as of this handover.
2. `crisisready/heat-risk-data-api:research/crowdsourced-modeling/ROADMAP.md` (private repo) —
   **the approved design governing this whole program**. Marked `DESIGN APPROVED 2026-08-21`. Do
   not skip this; a prior worker built the Rung B extension without reconciling against it and
   shipped an incomplete subset (see Claude memory `crowdsource-cell-key-vs-value` /
   `approved-crowdsource-roadmap-location` in this repo's memory namespace).
3. This doc.

## Where the program actually stands (verified 2026-08-26, not assumed)

Cross-referencing `GOVERNANCE.md` against the roadmap's own phase acceptance criteria:

- **ROADMAP.md Phase 1 (registry + retroactive pilots): mostly done, one gap.** The registry
  (`heatready_downscaling.registry`, `registry/global/ds-2026.07-rf5`,
  `registry/local/seoul-sdot-v1`, `registry/local/valencia-coast-v1`) is real, CI-checked
  (`.github/workflows/registry.yml`), and matches the roadmap's three named retroactive entries.
  **What's missing**: Phase 1 acceptance criteria explicitly requires "the rendered models page
  shows real evidence numbers matching the source reports." No such page exists —
  `scripts/render_leaderboard.py` only renders `ledger/credit.jsonl` into
  `docs/leaderboard.{md,json}`; it does not read `registry/`. There is no public models page
  anywhere in `docs/`.
- **ROADMAP.md Phase 2 (contributor front door): not started, and it's the next unstarted phase.**
  Roadmap text: "This phase is load-bearing, not cosmetic (Nishant's explicit direction)." Four
  deliverables, none built: (1) `CONTRIBUTING.md` rewrite around the full A–D/L1/L2 ladder, (2)
  worked runnable CI-executed examples per contribution type, (3) a quickstart tested to be
  under 10 minutes, (4) a `submit`/`make submit` scaffolding helper.
  - **Important, already resolved**: Phase 2 item 2 names a "prerequisite deliverable" — extracting
    "the L1 CV + cluster-bootstrap harness from `fit_valencia_local_correction.py`... this is also
    what closes the standing 'Rung B designed, not yet scoreable' gap." **This is already done.**
    PRs #27–#29 built exactly this as `score.py`'s `covariate_linear` correction path
    (`_bootstrap_reduction_ci`, hot-day stratification, per-target CI reporting — see
    `score.py` lines ~198–460, ~800–1170). Don't re-derive it; the harness already exists, it just
    has no worked example pointing at it yet.
- **Two stale GitHub issues, already closed by this worker**: #19 ("Wire Rung B scoring...") and
  #20 ("Build replay_downscaling.py...") were still open even though the project board already
  showed them `Done` (shipped in PRs #24/#25). Closed with comments citing the shipping PRs.

## The approved slice (what to actually build)

Three pieces, approved as one slice but implementable as separate PRs (recommended, for cleaner
review):

### 1. Housekeeping — already coded, just needs a PR

A commit already sits on local branch `docs/crowdsourcing-housekeeping` in this worktree
(`heat-ready-downscaling-worktrees/crowdsourcing-housekeeping`, commit `f41dab2`): fixes
`README.md`'s contribution-ladder table, which pointed at a `docs/rung-c.md` that was never
published, to point at `GOVERNANCE.md`'s real Rung C section instead, and adds the missing Rung D
row (the ladder table had no D row at all despite the design doc existing since PR #32). This
handover doc is committed on the same branch. Open the PR, run the standing review process below,
merge when clean.

### 2. Close Phase 1's open gap: render the registry into a public models page

Extend the existing leaderboard/docs pipeline (`scripts/render_leaderboard.py` is the closest
precedent for "read structured data, render markdown/json into `docs/`, CI-checked") to read
`registry/*/*/manifest.yaml` and produce a `docs/models.md` (name to match whatever `README.md`/
`GOVERNANCE.md` already call it, or introduce the term) listing each entry's `model_id`, rung/L-tier,
claimed scope (cells), and its evidence summary (metric, CI, n_stations — read straight from the
manifest's evidence block, don't recompute). This closes Phase 1's last unmet acceptance line.
Keep it derived/regenerable like `docs/leaderboard.md` already is — never hand-edited.

### 3. Phase 2's core deliverable, scoped down: worked examples for the two rungs open today

Full Phase 2 is large (4 sub-deliverables, plus a full `CONTRIBUTING.md` prose rewrite that
Nishant's own roadmap says needs his voice/review). This slice scopes to just the "key deliverable"
named in the roadmap — worked, runnable, **CI-executed** examples — for the two rungs actually
open today:

- **Rung A example**: a small, complete, runnable case that re-runs the existing validator
  (`scripts/validate_lagfill_downscaling.py` or `validate_forecast_downscaling.py`) against a
  named dark cell using the published snapshot, producing a real `claimed_report.json`.
- **Rung B/L1 example**: fit and submit a covariate-linear (or plain affine) correction on a
  provided station subset using the `covariate_linear` scoring path already built in `score.py`,
  producing a real `claimed_report.json` a reader could actually open a submission PR with.

Roadmap's own acceptance bar: "a cold-context agent (or colleague, e.g. Tiger) executes each
example end-to-end from the docs alone, without help; every example runs in CI." Wire both into a
GitHub Actions job so they can't silently rot (`tests.yml` or a new `examples.yml`).

**Explicitly deferred, do not attempt in this slice** (their prerequisites don't exist yet — note
this in the PR description so nobody assumes it's an oversight):

- L2 and Rung D worked examples — Rung D isn't implemented at all yet (design-only,
  `docs/design-2026-08-25-rung-d-contributed-data.md`, with real unresolved conflicts flagged
  in it — e.g. consumer-network geolocation vs. the PII-stripping policy in the private roadmap's
  §5 — that need a maintainer decision, not an executor's judgment call).
- The quickstart 10-minute timing test and the `submit`/`make submit` scaffolding helper — real
  Phase 2 items, left for the next slice after this one.
- The full `CONTRIBUTING.md` A–D/L1/L2 prose rewrite — per the roadmap, this is gated on Nishant's
  own authorship/voice review, not something to draft speculatively in this slice. A small,
  factual pointer to the new examples directory in the existing Rung A/B sections is in scope;
  a wholesale rewrite is not.

## Board state (as of this handover)

- Issues #19, #20 closed (see above).
- New tickets to create for this slice, labeled `area:crowdsourcing`, `repo:downscaling`,
  `priority:p0` or `p1`, `colleague-ok` if appropriate: "Render registry into a public models
  page (Phase 1 closeout)" and "Worked, CI-executed examples for Rung A + Rung B/L1 (Phase 2 core
  deliverable)". Add both to the HeatReady project board (`crisisready` org, project #2,
  `PVT_kwDOBFQ8Ac4Bg_8i`), status `Ready`.
- After this slice: the next board items to open are Phase 2's remaining three deliverables
  (quickstart, submit helper, full CONTRIBUTING rewrite — flag the last as needing Nishant's
  direct involvement, not a colleague-ok item), and, further out, Phase 3 (model selection served)
  and Phase 4 (Rung D implementation, which has the PII/geolocation conflict above still unresolved
  and needs a maintainer decision before it can be scoped into an executable slice).

## Correction (2026-08-26, gu-dev_crowdsourcing-slice-exec, caught by `review.py`'s codex adapter on PR #36)

Three operational claims above turned out not to hold once checked against the actual scripts.
Left in place above rather than rewritten, since this doc records what was approved and reasoned
at handover time — corrected here instead:

- **The Rung A example cannot point at `validate_lagfill_downscaling.py` or
  `validate_forecast_downscaling.py`.** Both scripts' own docstrings say so directly: neither
  accepts a snapshot argument, both stamp `ghcn_training-live`, and both are explicitly "NOT
  RUNNABLE STANDALONE IN THIS REPO" — they import `db`/`heat_calcs`/`open_meteo` modules that
  only exist in the private `heat-risk-data-api` repo, with live Aurora/Open-Meteo access.
  `CONTRIBUTING.md` line ~242 confirms `method.entrypoint` naming one of these is *provenance
  metadata only* — "what you ran locally... not executed" — never something the referee (or a
  worked example meant to run from the public repo alone) actually invokes. The runnable,
  public-snapshot-only path is what `run_submission.py` and `score_forward_eval.py` already use:
  `contract.FrozenPredictionAdapter` over a downloaded snapshot release, scored through
  `score.score_band`. The Rung A worked example needs to be a new small script built on that same
  path, not a wrapper around either validator.
- **The L1 CV/cluster-bootstrap harness is NOT a fitting harness — it's scoring-only.** PRs
  #27–#29's `covariate_linear` path (`_covariate_linear_effect`, `_bootstrap_reduction_ci` in
  `score.py`) takes an *already-declared* `intercept`/`slope` per term from the manifest and
  evaluates/CIs it against held-out data — it never fits a slope from a station subset. No public
  script fits a covariate-linear candidate today. The Rung B/L1 worked example therefore needs its
  own (small — this is a 1–2 term OLS/affine fit, not new infrastructure) fitting step in addition
  to calling the existing scoring path; "don't re-derive it" above applies to the scoring/CI math
  only, not to fitting.
- **There is no existing "leaderboard staleness" CI check to mirror.** `docs/leaderboard.md` is
  regenerated and diffed only inside `score-forward-eval.yml`'s monthly bot-PR job, which is a
  different shape (decides whether to open a PR) from a per-PR staleness gate. `docs/models.md`'s
  CI check is new, not a copy of an existing pattern.

## Standing process reminders (from this repo's own conventions)

- Every PR gets code review before merge, even Tier 1: `/code-review medium` +
  `infrastructure/scripts/codex-review.sh` in round 1's fan-out, plus the mandatory
  `infrastructure/scripts/review/review.py <worktree> <base-sha> <head-sha> --tier <tier>` run
  (non-gating during calibration, but the run itself is required every time).
- `git fetch origin main` and fast-forward before reviewing — a stale local `main` produces
  false-positive findings.
- Use `git worktree add`, never `git checkout`/`switch` in the shared checkout — this repo's
  checkout is shared across concurrent sessions on this machine.
- Commits in this repo are always authored as Nishant Kishore, never with any AI co-author
  trailer — see this repo's own `CLAUDE.md` hard gate.
