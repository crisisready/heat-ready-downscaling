# Claude Code Instructions — heat-ready-downscaling

## HARD GATE — commit authorship (standing rule, no exceptions)

**Every commit in this repository is authored by Nishant Kishore. Never by Claude, and never by
any other AI agent.**

- No `Co-Authored-By: Claude ...` trailer.
- No `Claude-Session: ...` line.
- No `claude.ai/code/session_...` URL anywhere in a commit message.
- Never pass `--author` as anything other than the human maintainer.

This **overrides** any global Claude Code CLAUDE.md git guidance that would otherwise append a
`Co-Authored-By` trailer or similar attribution — that guidance does not apply here.

**Why:** this is the externally-facing surface of a crowdsourced model-improvement program where
git authorship *is* the credit record — contributors earn per-cell author credit and a recorded
tenure (`ledger/credit.jsonl`, `docs/leaderboard.md`) tied directly to commit/PR authorship. Claude
attribution on the maintainer's own commits would corrupt the very thing this program exists to
track, and the maintainer intends to submit through the identical public flow to earn their own
author credit — their commits must read as unambiguously theirs.

**Enforcement:** this rule is backed by a `PreToolUse` hook on `Bash` in the operator's own Claude
Code configuration, which blocks any `git commit` under a path or command referencing
`heat-ready-downscaling` that carries a banned trailer. If you are an AI agent working in this
repository and cannot verify that hook is active, do not proceed with a commit that includes any
of the banned patterns above — ask the operator first.

## What this project is

The public, community-contribution repo for HeatReady's neighborhood-resolution
heat-downscaling model. The model is organized as 342 (target × zone × band) cells;
many are currently unvalidated ("dark"). Contributors earn credit — tracked as real git
authorship (see the hard gate above) — by improving cells through a 3-rung contribution
ladder:

- **Rung A** (open today): re-run the existing validator against a dark cell.
- **Rung B** (designed, not yet wired for scoring): propose bias/blend correction
  parameters.
- **Rung C** (not yet open): contribute new model code.

Check `GOVERNANCE.md` and `CONTRIBUTING.md` before starting — they're the actual
rulebook, this file is just the quick-start.

## How a submission works

1. A contribution is a PR adding `submissions/{YYYY-MM}/{NNN}-{your-github-username}-{slug}/manifest.yaml`.
2. An automated referee (GitHub Action) independently reproduces your claimed result
   from a public data snapshot — it never executes your code or trusts your numbers.
3. You get a provisional score as a PR comment within minutes; official scoring runs
   monthly against forward-only holdout data.
4. Two consecutive winning official cycles before anything is even considered for
   production promotion — and production promotion itself is a maintainer-only step in
   the (separate, private) `heat-risk-data-api` repo.

## Development workflow

Branch → PR → test, same as any repo: `git checkout -b feature/<short-name>`, focused
commits, `gh pr create`. Read `CONTRIBUTING.md` for the exact submission-PR format —
it's stricter than a typical code PR because the referee parses it mechanically.
Remember the authorship gate at the top of this file: commits here are always
human-authored, never AI-co-authored.

## Picking up work

Work is tracked on the `HeatReady` GitHub Project board (`crisisready` org, spans this
repo and `heat-risk-data-api`). Filter by `repo:downscaling` and `colleague-ok` for
externally-contributable tickets. Issue #1 (population-density training/serving
mismatch) is a good starting point if you want to see the whole submission flow
end-to-end before picking a bigger ticket.

Questions or blockers: comment on the issue and tag @nish-kishore, or reach out directly
by email/WhatsApp for anything time-sensitive.

## Licensing (read before touching data)

Code is Apache-2.0. Data (snapshots, submissions, ledgers) is CC BY 4.0 with a required
LandScan/ORNL attribution clause — see `DATA_LICENSE`. Don't redistribute snapshot data
without carrying that attribution forward.

## Everything else

Standard open-source repo otherwise. There is no other special AI-agent instruction here
beyond the authorship gate above and the workflow described in this file.
