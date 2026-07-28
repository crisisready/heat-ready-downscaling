"""Unit tests for scripts/render_leaderboard.py -- pure derivation logic
over ledger/credit.jsonl + ledger/cycles.jsonl, no I/O beyond what main()
itself does (tested separately via a tmp_path integration test)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import render_leaderboard as rl


def _cell(model_version="ds-2026.07-rf5", band_key="lag_fill", target="tmax", zone="Cfb"):
    return {"model_version": model_version, "band_key": band_key, "target": target, "zone": zone}


def _tenure_start(author="alice", start_month="2026-11", cell=None):
    return {
        "ts": "2026-11-05T06:20:01Z", "event": "tenure_start", "cell": cell or _cell(),
        "author_github": author, "start_month": start_month, "end_month": None,
    }


def _tenure_end(author="alice", start_month="2026-11", end_month="2027-02", cell=None):
    return {
        "ts": "2027-03-05T06:20:01Z", "event": "tenure_end", "cell": cell or _cell(),
        "author_github": author, "start_month": start_month, "end_month": end_month,
    }


class TestActiveTenures:
    def test_open_tenure_is_active(self):
        lines = [_tenure_start()]
        active = rl.active_tenures(lines)
        assert len(active) == 1
        assert active[0]["author_github"] == "alice"

    def test_closed_tenure_is_not_active(self):
        lines = [_tenure_start(), _tenure_end()]
        assert rl.active_tenures(lines) == []

    def test_superseding_tenure_replaces_the_previous_holder(self):
        lines = [
            _tenure_start(author="alice", start_month="2026-11"),
            _tenure_end(author="alice", start_month="2026-11", end_month="2027-01"),
            _tenure_start(author="bob", start_month="2027-02"),
        ]
        active = rl.active_tenures(lines)
        assert len(active) == 1
        assert active[0]["author_github"] == "bob"

    def test_different_cells_tracked_independently(self):
        lines = [
            _tenure_start(author="alice", cell=_cell(zone="Cfb")),
            _tenure_start(author="bob", cell=_cell(zone="BWh")),
        ]
        active = rl.active_tenures(lines)
        assert {t["author_github"] for t in active} == {"alice", "bob"}

    def test_mismatched_tenure_end_does_not_clear_a_different_start(self):
        """A tenure_end whose start_month doesn't match the currently
        tracked open tenure is a data inconsistency -- must not silently
        clear an unrelated, still-open tenure."""
        lines = [
            _tenure_start(author="alice", start_month="2026-11"),
            _tenure_end(author="someone-else", start_month="2099-01", end_month="2099-02"),
        ]
        active = rl.active_tenures(lines)
        assert len(active) == 1
        assert active[0]["author_github"] == "alice"

    def test_empty_ledger_yields_no_active_tenures(self):
        assert rl.active_tenures([]) == []


class TestCreditCounts:
    def test_counts_tenure_starts_per_author(self):
        lines = [_tenure_start(author="alice"), _tenure_start(author="alice", cell=_cell(zone="BWh")), _tenure_start(author="bob")]
        assert rl.credit_counts(lines) == {"alice": 2, "bob": 1}

    def test_tenure_end_does_not_affect_count(self):
        """All-time count includes cells the contributor no longer holds
        -- a tenure_end must not decrement it."""
        lines = [_tenure_start(author="alice"), _tenure_end(author="alice")]
        assert rl.credit_counts(lines) == {"alice": 1}

    def test_empty_ledger_yields_no_counts(self):
        assert rl.credit_counts([]) == {}


class TestRecentCycleActivity:
    def test_returns_only_the_most_recent_n_cycles(self):
        lines = [{"cycle": c} for c in ("2026-08", "2026-09", "2026-10", "2026-11")]
        recent = rl.recent_cycle_activity(lines, n_cycles=2)
        assert {r["cycle"] for r in recent} == {"2026-10", "2026-11"}

    def test_empty_ledger_yields_no_activity(self):
        assert rl.recent_cycle_activity([]) == []


class TestRenderMarkdown:
    def test_empty_state_shows_no_credits_yet(self):
        md = rl.render_markdown([], {})
        assert "No cells have been credited yet" in md
        assert "No cell currently has an active tenure" in md

    def test_populated_state_shows_table_rows(self):
        active = [{"cell": _cell(), "author_github": "alice", "start_month": "2026-11"}]
        counts = {"alice": 3}
        md = rl.render_markdown(active, counts)
        assert "alice" in md
        assert "Cfb" in md
        assert "3" in md


class TestRenderJson:
    def test_shape(self):
        active = [{"cell": _cell(), "author_github": "alice", "start_month": "2026-11"}]
        counts = {"alice": 3, "bob": 1}
        result = rl.render_json(active, counts, [])
        assert result["active_tenures"] == active
        # sorted descending by count
        assert list(result["credit_counts"].keys()) == ["alice", "bob"]


class TestMainIntegration:
    def test_generates_both_files_from_empty_ledgers(self, tmp_path, monkeypatch):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "credit.jsonl").write_text("")
        (ledger_dir / "cycles.jsonl").write_text("")
        docs_dir = tmp_path / "docs"

        monkeypatch.setattr(sys, "argv", [
            "render_leaderboard.py", "--ledger-dir", str(ledger_dir), "--docs-dir", str(docs_dir),
        ])
        rl.main()

        assert (docs_dir / "leaderboard.md").exists()
        assert (docs_dir / "leaderboard.json").exists()
        data = json.loads((docs_dir / "leaderboard.json").read_text())
        assert data["active_tenures"] == []
        assert data["credit_counts"] == {}

    def test_generates_from_populated_ledgers(self, tmp_path, monkeypatch):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "credit.jsonl").write_text(json.dumps(_tenure_start(author="alice")) + "\n")
        (ledger_dir / "cycles.jsonl").write_text("")
        docs_dir = tmp_path / "docs"

        monkeypatch.setattr(sys, "argv", [
            "render_leaderboard.py", "--ledger-dir", str(ledger_dir), "--docs-dir", str(docs_dir),
        ])
        rl.main()

        md = (docs_dir / "leaderboard.md").read_text()
        assert "alice" in md
