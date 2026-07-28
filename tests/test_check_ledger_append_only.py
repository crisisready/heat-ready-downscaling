"""Unit tests for scripts/check_ledger_append_only.py -- uses a REAL tiny
git repo in tmp_path (git init + real commits) rather than mocking
subprocess, so the actual `git show {ref}:{path}` invocation gets genuine
coverage, not just a mocked stand-in for it."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import check_ledger_append_only as cla


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


_VALID_LINE = {
    "ts": "2026-08-03T11:02:14Z", "submission_id": "2026-08-001", "author_github": "nishkishore",
    "track": "serving-ready", "rung": "A", "model_version": "m", "band_key": "lag_fill",
    "snapshot_version": "v1", "manifest_sha256": "a" * 64, "claimed_report_sha256": "b" * 64,
    "reproduced": True, "max_abs_deviation": {}, "pr": "o/r#1", "runner_commit": "abc",
}


class TestCheckPaths:
    def test_pure_append_passes(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        ledger_dir = repo / "ledger"
        ledger_dir.mkdir()
        path = ledger_dir / "submissions.jsonl"
        path.write_text(json.dumps(_VALID_LINE) + "\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")

        line2 = {**_VALID_LINE, "submission_id": "2026-08-002"}
        with open(path, "a") as f:
            f.write(json.dumps(line2) + "\n")

        monkeypatch.chdir(repo)
        result = cla.check_paths("HEAD", ["ledger/submissions.jsonl"])
        assert result == []

    def test_edited_line_fails(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        ledger_dir = repo / "ledger"
        ledger_dir.mkdir()
        path = ledger_dir / "submissions.jsonl"
        path.write_text(json.dumps(_VALID_LINE) + "\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")

        path.write_text(json.dumps({**_VALID_LINE, "reproduced": False}) + "\n")

        monkeypatch.chdir(repo)
        result = cla.check_paths("HEAD", ["ledger/submissions.jsonl"])
        assert any("changed" in v for v in result)

    def test_brand_new_ledger_file_is_not_a_violation(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        (repo / "README.md").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")

        ledger_dir = repo / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps(_VALID_LINE) + "\n")

        monkeypatch.chdir(repo)
        result = cla.check_paths("HEAD", ["ledger/submissions.jsonl"])
        assert result == []


class TestGitShow:
    def test_missing_file_at_ref_returns_empty_string(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        (repo / "README.md").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        monkeypatch.chdir(repo)
        assert cla.git_show("HEAD", "nonexistent.jsonl") == ""

    def test_existing_file_at_ref_returns_its_content(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        (repo / "a.txt").write_text("hello\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        monkeypatch.chdir(repo)
        assert cla.git_show("HEAD", "a.txt") == "hello\n"
