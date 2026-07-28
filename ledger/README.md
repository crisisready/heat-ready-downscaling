# Ledger

Three append-only JSONL files, never edited, only appended to. Schemas live in
[`src/heatready_downscaling/ledger.py`](../src/heatready_downscaling/ledger.py)
(`SUBMISSIONS_LINE_SCHEMA`, `CYCLES_LINE_SCHEMA`, `CREDIT_LINE_SCHEMA`), which is also the single
source of truth `.github/workflows/check-ledger-append.yml` validates every PR's diff against. See
that module's own docstring for the full shape of each line.

- **`submissions.jsonl`**: one line per submission, appended by `scripts/append_ledger_entry.py`
  after a submission PR merges (never by the contributor's own PR; see
  `scripts/run_submission.py`'s module docstring for why).
- **`cycles.jsonl`**: one line per (monthly cycle, cell, candidate), appended by the monthly
  `scripts/score_forward_eval.py`.
- **`credit.jsonl`**: `tenure_start`/`tenure_end` events. An end is always a new line, never an
  edit to its own start line.

The first real submission (`2026-07-001`) went through the pipeline 2026-07-28, so
`submissions.jsonl` has its first line. `cycles.jsonl` and `credit.jsonl` are still empty: the
first candidate hasn't been through an official monthly cycle yet. `docs/leaderboard.{md,json}`
are derived from these files and safe to regenerate or overwrite; they are not ledger data
themselves.
