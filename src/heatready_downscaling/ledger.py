"""
The three append-only ledger line schemas (plan section 7.3) + the
append-only integrity check both `scripts/append_ledger_entry.py` and
`.github/workflows/check-ledger-append.yml` (via `scripts/
check_ledger_append_only.py`) depend on. New in this package (Phase 3, no
private-repo equivalent). See PROVENANCE.md.

Three files, never edited, only appended to:
  - ledger/submissions.jsonl -- one line per submission, written by
    scripts/append_ledger_entry.py at merge time (never by the
    contributor's own PR -- see run_submission.py's own module docstring
    for why the PR number in a submission's own line can only be resolved
    AFTER merge).
  - ledger/cycles.jsonl -- one line per (cycle x cell x candidate),
    written by the monthly scripts/score_forward_eval.py.
  - ledger/credit.jsonl -- tenure_start/tenure_end events. An end is
    always a NEW line, never an edit to its own start line.

docs/leaderboard.{md,json} are DERIVED from these and safe to overwrite --
they are not ledger data themselves and have no schema here.
"""

SUBMISSIONS_LINE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "ts", "submission_id", "author_github", "track", "rung", "model_version", "band_key",
        "snapshot_version", "manifest_sha256", "claimed_report_sha256", "reproduced", "pr", "runner_commit",
    ],
    "properties": {
        "ts": {"type": "string"},
        "submission_id": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{3}$"},
        "author_github": {"type": "string"},
        "track": {"enum": ["serving-ready", "research"]},
        "rung": {"enum": ["A", "B", "C"]},
        "model_version": {"type": "string"},
        "band_key": {"type": "string"},
        "snapshot_version": {"type": "string"},
        "manifest_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "claimed_report_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "reproduced": {"type": "boolean"},
        "max_abs_deviation": {"type": "object"},
        "pr": {"type": "string"},  # "{owner}/{repo}#{number}"
        "runner_commit": {"type": ["string", "null"]},  # None for a local/dry-run invocation outside CI
    },
}

_CELL_SCHEMA: dict = {
    "type": "object",
    "required": ["model_version", "band_key", "target", "zone"],
    "properties": {
        "model_version": {"type": "string"}, "band_key": {"type": "string"},
        "target": {"enum": ["tmax", "tmin"]}, "zone": {"type": "string"},
    },
}

CYCLES_LINE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "ts", "cycle", "eval_month", "submission_id", "author_github", "cell",
        "n_forward", "n_stations", "rmse_grid_c", "status", "snapshot_version", "runner_commit", "package_version",
    ],
    "properties": {
        "ts": {"type": "string"},
        "cycle": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
        "eval_month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
        "submission_id": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{3}$"},
        "author_github": {"type": "string"},
        "cell": _CELL_SCHEMA,
        "n_forward": {"type": "integer"}, "n_stations": {"type": "integer"},
        "rmse_grid_c": {"type": ["number", "null"]}, "rmse_qrf_c": {"type": ["number", "null"]},
        "rmse_debiased_cv_c": {"type": ["number", "null"]},
        "rmse_improvement_pct_debiased_cv": {"type": ["number", "null"]},
        "bias_correction_c": {"type": ["number", "null"]},
        "spatial_skill": {"type": ["boolean", "null"]},
        "gated_insufficient_n": {"type": "boolean"},
        "status": {"enum": ["win", "loss", "insufficient_n"]},
        "incumbent_submission_id": {"type": ["string", "null"]},
        "snapshot_version": {"type": "string"},
        "runner_commit": {"type": ["string", "null"]},  # None for a local/dry-run invocation outside CI
        "package_version": {"type": "string"},
    },
}

CREDIT_LINE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["ts", "event", "cell", "author_github"],
    "properties": {
        "ts": {"type": "string"},
        "event": {"enum": ["tenure_start", "tenure_end"]},
        "cell": _CELL_SCHEMA,
        "author_github": {"type": "string"},
    },
    # tenure_start and tenure_end carry different required fields beyond
    # the shared ones above -- oneOf a start-shaped and an end-shaped
    # variant, keyed on `event`, rather than making everything optional
    # (which would let a malformed line of either kind slip through).
    "allOf": [
        {
            "if": {"properties": {"event": {"const": "tenure_start"}}}, "then": {
                "required": ["submission_id", "start_month", "end_month"],
                "properties": {
                    "author_name": {"type": ["string", "null"]}, "orcid": {"type": ["string", "null"]},
                    "submission_id": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{3}$"},
                    "start_month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
                    "end_month": {"const": None},
                    "score_at_start": {"type": "object"},
                    "cycles_won": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "if": {"properties": {"event": {"const": "tenure_end"}}}, "then": {
                "required": ["start_month", "end_month"],
                "properties": {
                    "start_month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
                    "end_month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
                    "superseded_by": {"type": ["string", "null"]},
                },
            },
        },
    ],
}

_LEDGER_SCHEMAS = {
    "submissions": SUBMISSIONS_LINE_SCHEMA,
    "cycles": CYCLES_LINE_SCHEMA,
    "credit": CREDIT_LINE_SCHEMA,
}


def validate_ledger_line(kind: str, line: dict) -> None:
    """kind is one of "submissions"/"cycles"/"credit" -- matches the
    ledger/{kind}.jsonl filename. Raises jsonschema.ValidationError on a
    malformed line."""
    import jsonschema

    if kind not in _LEDGER_SCHEMAS:
        raise ValueError(f"unrecognized ledger kind {kind!r} -- must be one of {sorted(_LEDGER_SCHEMAS)}")
    jsonschema.validate(line, _LEDGER_SCHEMAS[kind])


def parse_jsonl(text: str) -> list[dict]:
    """Parse a .jsonl file's contents into a list of dicts, one per
    non-blank line. Shared by the append-only check and anything reading a
    ledger file (score_forward_eval.py, render_leaderboard.py)."""
    import json

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def check_append_only(base_text: str, head_text: str, kind: str) -> list[str]:
    """The required CI check (plan section 7.3): a PR's diff to a
    ledger/*.jsonl file must be ONLY appended lines -- no edit, no
    reorder, no deletion. Returns a list of violation strings (empty =
    OK). Checked at the LINE level (not a raw byte-prefix check) so a file
    that gained a trailing newline, or whose last line is completed by a
    later write, isn't a false-positive violation -- what matters is that
    every base line still appears, unchanged, as a prefix of head's own
    line sequence.

    Every appended line is also schema-validated (validate_ledger_line)
    and checked for duplicate identifiers within the same kind (a
    resubmitted submission_id, a cycle line the ledger already has for the
    same (cycle, cell, submission_id))."""
    base_lines = parse_jsonl(base_text)
    head_lines = parse_jsonl(head_text)

    violations: list[str] = []
    if len(head_lines) < len(base_lines):
        violations.append(f"ledger/{kind}.jsonl shrank ({len(base_lines)} -> {len(head_lines)} lines) -- deletions are never allowed")
        return violations

    for i, base_line in enumerate(base_lines):
        if head_lines[i] != base_line:
            violations.append(f"ledger/{kind}.jsonl line {i + 1} changed -- edits to existing lines are never allowed")

    if violations:
        return violations

    appended = head_lines[len(base_lines):]
    seen_ids = {_line_identity(kind, line) for line in base_lines}
    for i, line in enumerate(appended):
        try:
            validate_ledger_line(kind, line)
        except Exception as exc:
            violations.append(f"ledger/{kind}.jsonl appended line {i + 1} failed schema validation: {exc}")
            continue
        identity = _line_identity(kind, line)
        if identity is not None and identity in seen_ids:
            violations.append(f"ledger/{kind}.jsonl appended line {i + 1} duplicates an existing entry: {identity}")
        seen_ids.add(identity)

    return violations


def _line_identity(kind: str, line: dict) -> tuple | None:
    """A natural uniqueness key per ledger kind, for the duplicate check
    above. None means "no uniqueness constraint for this kind" (credit.jsonl
    legitimately has more than one line per cell over time -- tenure starts
    and ends -- so it has no single natural key here)."""
    if kind == "submissions":
        return (line.get("submission_id"),)
    if kind == "cycles":
        cell = line.get("cell", {})
        return (line.get("cycle"), cell.get("model_version"), cell.get("band_key"), cell.get("target"), cell.get("zone"), line.get("submission_id"))
    return None
