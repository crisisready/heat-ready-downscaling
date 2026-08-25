"""Unit tests for scripts/append_ledger_entry.py's pure-logic pieces --
build_submission_line + append_line. The full main() (re-running the
referee against a real snapshot download) is not exercised here, same as
run_submission.py's own network-touching functions -- this script was
hand-verified separately.

TestMainCandidateWiring (2026-08-25) is the one exception: it DOES drive
main() end to end, with the network/heavy pieces (download_snapshot,
verify_snapshot, reproduce) mocked out -- added specifically to cover the
single most important fix in PR #24 (round-2 code review's own explicit
observation: this wiring "has no direct unit test... a future regression
there wouldn't be caught")."""

import json
import os
import sys

import yaml

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


class TestMainCandidateWiring:
    """main()'s reproduce() call must forward the manifest's own
    method.candidate, exactly like run_submission.py's PR-time call --
    the fix for the most severe finding on PR #24 (Codex adversarial
    review): without it, "reproduced": true could be written to the
    ledger having never actually verified a Rung B submission's declared
    correction. download_snapshot/verify_snapshot/reproduce are mocked --
    this is about the WIRING (what main() passes to reproduce), not
    re-testing reproduce()'s own scoring logic (covered in
    test_run_submission.py)."""

    def _write_full_submission(self, tmp_path, candidate):
        sub_dir = tmp_path / "submissions" / "2026-08" / "001-alice-lagfill-cfb"
        sub_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1, "submission_id": "2026-08-001",
            "author": {"github": "alice", "name": None, "orcid": None, "affiliation": None},
            "track": "serving-ready", "rung": "B" if candidate is not None else "A",
            "snapshot": {"version": "v2026.08", "manifest_sha256": "a" * 64},
            "claims": [{"model_version": "ds-test", "band_key": "lag_fill", "targets": ["tmax"], "zones": ["Cfb"]}],
            "method": {
                "kind": "parameters" if candidate is not None else "rerun-validator",
                "entrypoint": "scripts/run_submission.py", "args": [], "package_version": "0.1.0",
                "code_ref": None, "extra_covariates": [],
                **({"candidate": candidate} if candidate is not None else {}),
            },
            "claimed_report": "claimed_report.json",
            "tolerance": {"rmse_qrf_c": 0.005},
        }
        (sub_dir / "manifest.yaml").write_text(yaml.dump(manifest))
        (sub_dir / "claimed_report.json").write_text(json.dumps({
            "report_schema_version": 1, "model_version": "ds-test", "band_key": "lag_fill",
            "snapshot_version": "v2026.08", "sample_requested": 0, "rows_sampled": 10, "rows_paired": 10,
            "fidelity_check": {"n": 0}, "by_target": {"tmax": {}, "tmin": {}},
        }))
        return sub_dir

    def _run_main_capturing_reproduce_kwargs(self, tmp_path, sub_dir, monkeypatch):
        captured = {}

        def fake_reproduce(snapshot_dir, model_version, band_key, snapshot_version, candidate=None):
            captured["candidate"] = candidate
            return {
                "report_schema_version": 1, "model_version": model_version, "band_key": band_key,
                "snapshot_version": snapshot_version, "sample_requested": 0, "rows_sampled": 10,
                "rows_paired": 10, "fidelity_check": {"n": 0}, "by_target": {"tmax": {}, "tmin": {}},
            }

        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        monkeypatch.setattr(ale.rs, "download_snapshot", lambda version, cache_root=None: str(tmp_path / "cache"))
        monkeypatch.setattr(ale.rs, "verify_snapshot", lambda snapshot_dir, sha: [])
        monkeypatch.setattr(ale.rs, "reproduce", fake_reproduce)
        monkeypatch.setattr(ale.rs, "_package_version", lambda: "0.1.0")
        monkeypatch.setattr(sys, "argv", [
            "append_ledger_entry.py", "--submission-dir", str(sub_dir),
            "--repo", "crisisready/heat-ready-downscaling", "--pr-number", "24", "--pr-author", "alice",
            "--ledger-dir", str(ledger_dir),
        ])
        ale.main()
        return captured

    def test_reproduce_called_with_manifest_candidate(self, tmp_path, monkeypatch):
        candidate = {"tmax": {"Cfb": {"bias_correction_c": 0.8}}}
        sub_dir = self._write_full_submission(tmp_path, candidate)
        captured = self._run_main_capturing_reproduce_kwargs(tmp_path, sub_dir, monkeypatch)
        assert captured["candidate"] == candidate

    def test_reproduce_called_with_none_for_rung_a(self, tmp_path, monkeypatch):
        """A Rung A submission has no method.candidate at all -- must
        pass None through, not KeyError or an empty dict standing in for
        "no candidate.\""""
        sub_dir = self._write_full_submission(tmp_path, candidate=None)
        captured = self._run_main_capturing_reproduce_kwargs(tmp_path, sub_dir, monkeypatch)
        assert captured["candidate"] is None
