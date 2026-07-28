"""Unit tests for heatready_downscaling.ledger -- the three ledger line
schemas and the append-only integrity check the required CI check
(plan section 7.3) depends on."""

import pytest

from heatready_downscaling import ledger


def _submissions_line(**overrides) -> dict:
    base = {
        "ts": "2026-08-03T11:02:14Z", "submission_id": "2026-08-001", "author_github": "nishkishore",
        "track": "serving-ready", "rung": "A", "model_version": "ds-2026.07-rf5", "band_key": "lag_fill",
        "snapshot_version": "v2026.07", "manifest_sha256": "a" * 64, "claimed_report_sha256": "b" * 64,
        "reproduced": True, "max_abs_deviation": {"rmse_qrf_c": 0.0004},
        "pr": "crisisready/heat-ready-downscaling#12", "runner_commit": "abc1234",
    }
    base.update(overrides)
    return base


def _cycles_line(**overrides) -> dict:
    base = {
        "ts": "2026-10-05T06:11:03Z", "cycle": "2026-10", "eval_month": "2026-08",
        "submission_id": "2026-08-001", "author_github": "nishkishore",
        "cell": {"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "target": "tmax", "zone": "Cfb"},
        "n_forward": 4128, "n_stations": 47,
        "rmse_grid_c": 1.912, "rmse_qrf_c": 1.744, "rmse_debiased_cv_c": 1.731,
        "rmse_improvement_pct_debiased_cv": 0.0947, "bias_correction_c": 0.412,
        "spatial_skill": True, "gated_insufficient_n": False,
        "status": "win", "incumbent_submission_id": None,
        "snapshot_version": "v2026.10", "runner_commit": "abc1234", "package_version": "0.3.0",
    }
    base.update(overrides)
    return base


def _tenure_start(**overrides) -> dict:
    base = {
        "ts": "2026-11-05T06:20:01Z", "event": "tenure_start",
        "cell": {"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "target": "tmax", "zone": "Cfb"},
        "author_github": "nishkishore", "author_name": "Nishant Kishore", "orcid": None,
        "submission_id": "2026-08-001", "start_month": "2026-11", "end_month": None,
        "score_at_start": {"rmse_improvement_pct_debiased_cv": 0.0947}, "cycles_won": ["2026-10", "2026-11"],
    }
    base.update(overrides)
    return base


def _tenure_end(**overrides) -> dict:
    base = {
        "ts": "2027-03-05T06:20:01Z", "event": "tenure_end",
        "cell": {"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "target": "tmax", "zone": "Cfb"},
        "author_github": "nishkishore", "start_month": "2026-11", "end_month": "2027-02",
        "superseded_by": "2027-01-004",
    }
    base.update(overrides)
    return base


class TestValidateLedgerLine:
    def test_well_formed_submissions_line_passes(self):
        ledger.validate_ledger_line("submissions", _submissions_line())

    def test_well_formed_cycles_line_passes(self):
        ledger.validate_ledger_line("cycles", _cycles_line())

    def test_well_formed_tenure_start_passes(self):
        ledger.validate_ledger_line("credit", _tenure_start())

    def test_well_formed_tenure_end_passes(self):
        ledger.validate_ledger_line("credit", _tenure_end())

    def test_tenure_start_missing_submission_id_raises(self):
        line = _tenure_start()
        del line["submission_id"]
        with pytest.raises(Exception):
            ledger.validate_ledger_line("credit", line)

    def test_tenure_end_missing_end_month_raises(self):
        line = _tenure_end()
        del line["end_month"]
        with pytest.raises(Exception):
            ledger.validate_ledger_line("credit", line)

    def test_unrecognized_kind_raises(self):
        with pytest.raises(ValueError, match="unrecognized ledger kind"):
            ledger.validate_ledger_line("not-a-kind", {})

    def test_bad_submission_id_shape_raises(self):
        with pytest.raises(Exception):
            ledger.validate_ledger_line("submissions", _submissions_line(submission_id="not-an-id"))


class TestParseJsonl:
    def test_parses_multiple_lines(self):
        text = '{"a": 1}\n{"a": 2}\n'
        assert ledger.parse_jsonl(text) == [{"a": 1}, {"a": 2}]

    def test_blank_lines_skipped(self):
        text = '{"a": 1}\n\n{"a": 2}\n'
        assert ledger.parse_jsonl(text) == [{"a": 1}, {"a": 2}]

    def test_empty_file_returns_empty_list(self):
        assert ledger.parse_jsonl("") == []


class TestCheckAppendOnly:
    def _jsonl(self, *lines) -> str:
        import json
        return "\n".join(json.dumps(line) for line in lines) + "\n"

    def test_pure_append_is_clean(self):
        base = self._jsonl(_submissions_line())
        head = self._jsonl(_submissions_line(), _submissions_line(submission_id="2026-08-002"))
        assert ledger.check_append_only(base, head, "submissions") == []

    def test_no_changes_is_clean(self):
        base = self._jsonl(_submissions_line())
        assert ledger.check_append_only(base, base, "submissions") == []

    def test_edited_existing_line_flagged(self):
        base = self._jsonl(_submissions_line())
        head = self._jsonl(_submissions_line(reproduced=False))
        violations = ledger.check_append_only(base, head, "submissions")
        assert any("changed" in v for v in violations)

    def test_deleted_line_flagged(self):
        base = self._jsonl(_submissions_line(), _submissions_line(submission_id="2026-08-002"))
        head = self._jsonl(_submissions_line())
        violations = ledger.check_append_only(base, head, "submissions")
        assert any("shrank" in v for v in violations)

    def test_reordered_lines_flagged(self):
        a = _submissions_line(submission_id="2026-08-001")
        b = _submissions_line(submission_id="2026-08-002")
        base = self._jsonl(a, b)
        head = self._jsonl(b, a)
        violations = ledger.check_append_only(base, head, "submissions")
        assert any("changed" in v for v in violations)

    def test_malformed_appended_line_flagged(self):
        base = self._jsonl(_submissions_line())
        bad = _submissions_line(submission_id="2026-08-002")
        del bad["manifest_sha256"]
        head = self._jsonl(_submissions_line(), bad)
        violations = ledger.check_append_only(base, head, "submissions")
        assert any("schema validation" in v for v in violations)

    def test_duplicate_submission_id_flagged(self):
        base = self._jsonl(_submissions_line())
        head = self._jsonl(_submissions_line(), _submissions_line())  # same submission_id twice
        violations = ledger.check_append_only(base, head, "submissions")
        assert any("duplicates" in v for v in violations)

    def test_empty_base_all_appended_lines_checked(self):
        head = self._jsonl(_submissions_line())
        assert ledger.check_append_only("", head, "submissions") == []

    def test_credit_ledger_allows_multiple_lines_per_cell(self):
        """credit.jsonl legitimately has a tenure_start AND a later
        tenure_end for the same cell -- _line_identity returns None for
        "credit", so this must never be flagged as a duplicate."""
        base = self._jsonl(_tenure_start())
        head = self._jsonl(_tenure_start(), _tenure_end())
        assert ledger.check_append_only(base, head, "credit") == []
