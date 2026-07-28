"""Unit tests for scripts/append_ledger_entry.py's pure-logic pieces --
build_submission_line + append_line. The full main() (re-running the
referee against a real snapshot download) is not exercised here, same as
run_submission.py's own network-touching functions -- this script was
hand-verified separately."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import append_ledger_entry as ale


def _manifest(**overrides) -> dict:
    base = {
        "submission_id": "2026-08-001",
        "author": {"github": "nishkishore"},
        "track": "serving-ready", "rung": "A",
        "snapshot": {"version": "v2026.07", "manifest_sha256": "a" * 64},
        "claims": [{"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "targets": ["tmax"], "zones": ["Cfb"]}],
        "claimed_report": "claimed_report.json",
    }
    base.update(overrides)
    return base


class TestBuildSubmissionLine:
    def test_builds_a_valid_ledger_line(self, tmp_path):
        (tmp_path / "claimed_report.json").write_text(json.dumps({"x": 1}))
        line = ale.build_submission_line(
            _manifest(), str(tmp_path), "pass", {"rmse_qrf_c": 0.001}, 12,
            "crisisready/heat-ready-downscaling", "abc1234",
        )
        assert line["submission_id"] == "2026-08-001"
        assert line["author_github"] == "nishkishore"
        assert line["reproduced"] is True
        assert line["pr"] == "crisisready/heat-ready-downscaling#12"
        assert line["runner_commit"] == "abc1234"
        assert len(line["claimed_report_sha256"]) == 64

    def test_failing_status_is_not_reproduced(self, tmp_path):
        (tmp_path / "claimed_report.json").write_text(json.dumps({"x": 1}))
        line = ale.build_submission_line(_manifest(), str(tmp_path), "fail", {}, 12, "o/r", "abc")
        assert line["reproduced"] is False

    def test_claimed_report_sha256_matches_actual_file_bytes(self, tmp_path):
        content = json.dumps({"a": 1, "b": 2})
        (tmp_path / "claimed_report.json").write_text(content)
        line = ale.build_submission_line(_manifest(), str(tmp_path), "pass", {}, 1, "o/r", "abc")
        assert line["claimed_report_sha256"] == ale._sha256_file(str(tmp_path / "claimed_report.json"))

    def test_missing_max_abs_deviation_defaults_to_empty_dict(self, tmp_path):
        (tmp_path / "claimed_report.json").write_text("{}")
        line = ale.build_submission_line(_manifest(), str(tmp_path), "pass", None, 1, "o/r", "abc")
        assert line["max_abs_deviation"] == {}


class TestAppendLine:
    def test_appends_a_valid_line(self, tmp_path):
        line = {
            "ts": "2026-08-03T11:02:14Z", "submission_id": "2026-08-001", "author_github": "nishkishore",
            "track": "serving-ready", "rung": "A", "model_version": "ds-2026.07-rf5", "band_key": "lag_fill",
            "snapshot_version": "v2026.07", "manifest_sha256": "a" * 64, "claimed_report_sha256": "b" * 64,
            "reproduced": True, "max_abs_deviation": {}, "pr": "o/r#1", "runner_commit": "abc",
        }
        ale.append_line(str(tmp_path), "submissions", line)
        content = (tmp_path / "submissions.jsonl").read_text()
        assert json.loads(content.strip()) == line

    def test_appends_without_overwriting_existing_lines(self, tmp_path):
        line1 = {
            "ts": "2026-08-03T11:02:14Z", "submission_id": "2026-08-001", "author_github": "a",
            "track": "serving-ready", "rung": "A", "model_version": "m", "band_key": "lag_fill",
            "snapshot_version": "v1", "manifest_sha256": "a" * 64, "claimed_report_sha256": "b" * 64,
            "reproduced": True, "max_abs_deviation": {}, "pr": "o/r#1", "runner_commit": "abc",
        }
        line2 = {**line1, "submission_id": "2026-08-002"}
        ale.append_line(str(tmp_path), "submissions", line1)
        ale.append_line(str(tmp_path), "submissions", line2)
        lines = (tmp_path / "submissions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["submission_id"] == "2026-08-001"
        assert json.loads(lines[1])["submission_id"] == "2026-08-002"

    def test_malformed_line_raises_before_writing(self, tmp_path):
        import pytest
        bad_line = {"ts": "2026-08-03T11:02:14Z"}  # missing every other required field
        with pytest.raises(Exception):
            ale.append_line(str(tmp_path), "submissions", bad_line)
        assert not (tmp_path / "submissions.jsonl").exists()
