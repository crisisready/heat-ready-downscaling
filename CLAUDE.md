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

## Everything else

Standard open-source repo. See `README.md`, `CONTRIBUTING.md`, and `GOVERNANCE.md` for how the
project itself works. There is no other special AI-agent instruction here beyond the authorship
gate above.
