"""Unit tests for scripts/score_forward_eval.py -- the monthly official
scorer. Covers the pure-logic pieces directly, plus one full end-to-end
integration test building two synthetic snapshot versions and running
main() across two cycles to verify the win-streak/promotion/supersession
logic holds together, not just each piece in isolation."""

import json
import os
import sys
from datetime import date

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import score_forward_eval as sfe

from heatready_downscaling import snapshot as snap


def _cell(model_version="ds-test", band_key="lag_fill", target="tmax", zone="Cfb"):
    return {"model_version": model_version, "band_key": band_key, "target": target, "zone": zone}


class TestFindSubmissionManifest:
    def test_finds_by_zero_padded_sequence_prefix(self, tmp_path):
        d = tmp_path / "submissions" / "2026-08" / "001-nishkishore-lagfill-cfb"
        d.mkdir(parents=True)
        (d / "manifest.yaml").write_text("x: 1")
        result = sfe._find_submission_manifest(str(tmp_path / "submissions"), "2026-08-001")
        assert result == str(d / "manifest.yaml")

    def test_username_containing_digits_does_not_false_match(self, tmp_path):
        """A naive substring-based glob could confuse a username like
        'user001' with sequence 001 -- the zero-padded PREFIX match must
        not be fooled by this."""
        d1 = tmp_path / "submissions" / "2026-08" / "002-user001-slug"
        d1.mkdir(parents=True)
        (d1 / "manifest.yaml").write_text("x: 1")
        result = sfe._find_submission_manifest(str(tmp_path / "submissions"), "2026-08-001")
        assert result is None

    def test_missing_submission_returns_none(self, tmp_path):
        (tmp_path / "submissions" / "2026-08").mkdir(parents=True)
        assert sfe._find_submission_manifest(str(tmp_path / "submissions"), "2026-08-999") is None


class TestLoadActiveCandidates:
    def _write_submission(self, tmp_path, submission_id, github, model_version, band_key, targets, zones, candidate=None):
        year_month, seq = submission_id.rsplit("-", 1)
        d = tmp_path / "submissions" / year_month / f"{seq}-{github}-slug"
        d.mkdir(parents=True)
        manifest = {
            "claims": [{"model_version": model_version, "band_key": band_key, "targets": targets, "zones": zones}],
        }
        if candidate is not None:
            manifest["method"] = {"candidate": candidate}
        (d / "manifest.yaml").write_text(yaml.dump(manifest))

    def _ledger_line(self, submission_id, github, snapshot_version="v2026.08", reproduced=True, ts="2026-08-01T00:00:00Z"):
        return {
            "ts": ts, "submission_id": submission_id, "author_github": github,
            "snapshot_version": snapshot_version, "reproduced": reproduced,
        }

    def test_expands_claim_into_cells(self, tmp_path):
        self._write_submission(tmp_path, "2026-08-001", "alice", "ds-test", "lag_fill", ["tmax", "tmin"], ["Cfb", "BWh"])
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps(self._ledger_line("2026-08-001", "alice")) + "\n")

        active = sfe.load_active_candidates(str(ledger_dir), str(tmp_path / "submissions"))
        assert len(active) == 4  # 2 targets x 2 zones
        assert ("ds-test", "lag_fill", "tmax", "Cfb") in active

    def test_unreproduced_submission_is_not_active(self, tmp_path):
        self._write_submission(tmp_path, "2026-08-001", "alice", "ds-test", "lag_fill", ["tmax"], ["Cfb"])
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps(self._ledger_line("2026-08-001", "alice", reproduced=False)) + "\n")

        active = sfe.load_active_candidates(str(ledger_dir), str(tmp_path / "submissions"))
        assert active == {}

    def test_later_submission_supersedes_earlier_for_same_cell(self, tmp_path):
        self._write_submission(tmp_path, "2026-08-001", "alice", "ds-test", "lag_fill", ["tmax"], ["Cfb"])
        self._write_submission(tmp_path, "2026-09-001", "bob", "ds-test", "lag_fill", ["tmax"], ["Cfb"])
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        lines = [
            self._ledger_line("2026-08-001", "alice", ts="2026-08-01T00:00:00Z"),
            self._ledger_line("2026-09-001", "bob", ts="2026-09-01T00:00:00Z"),
        ]
        (ledger_dir / "submissions.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")

        active = sfe.load_active_candidates(str(ledger_dir), str(tmp_path / "submissions"))
        assert active[("ds-test", "lag_fill", "tmax", "Cfb")]["author_github"] == "bob"

    def test_rung_a_candidate_has_no_proposed_correction_entry(self, tmp_path):
        """2026-08-25: a Rung A manifest has no method.candidate block at
        all -- load_active_candidates must report None, not KeyError."""
        self._write_submission(tmp_path, "2026-08-001", "alice", "ds-test", "lag_fill", ["tmax"], ["Cfb"])
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps(self._ledger_line("2026-08-001", "alice")) + "\n")

        active = sfe.load_active_candidates(str(ledger_dir), str(tmp_path / "submissions"))
        assert active[("ds-test", "lag_fill", "tmax", "Cfb")]["proposed_correction_entry"] is None

    def test_extracts_proposed_correction_entry_for_rung_b(self, tmp_path):
        """The single most important wiring this Codex-review-fixed gap
        was about: score_forward_eval must be able to read a Rung B
        submission's OWN declared value back out of its manifest."""
        self._write_submission(
            tmp_path, "2026-08-001", "alice", "ds-test", "lag_fill", ["tmax", "tmin"], ["Cfb"],
            candidate={"tmax": {"Cfb": {"bias_correction_c": 0.8}}},
        )
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps(self._ledger_line("2026-08-001", "alice")) + "\n")

        active = sfe.load_active_candidates(str(ledger_dir), str(tmp_path / "submissions"))
        assert active[("ds-test", "lag_fill", "tmax", "Cfb")]["proposed_correction_entry"] == {"bias_correction_c": 0.8}
        # tmin has no entry in the candidate block -- must stay None, not inherit tmax's
        assert active[("ds-test", "lag_fill", "tmin", "Cfb")]["proposed_correction_entry"] is None


class TestBandMonthsAndNewMonths:
    def _row(self, station_id="A", d=date(2023, 1, 1)):
        return {
            "station_id": station_id, "date": d, "band": "lag_fill", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            "grid_tmax_c": 29.0, "grid_tmin_c": 19.0, "grid_specific_humidity_kgkg": 0.01, "nighttime_wind_ms": 2.0,
            "base_source": "x", "base_model": None, "humidity_source": "band", "wind_source": "band",
            "lon": -112.0, "lat": 33.4, "elevation_m": 340.0, "region": "US", "climate_zone": "Cfb",
            "koppen_main_group_code": 3, "obs_window_shift_days": 0,
            "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
            "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
            "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5, "elevation_mean_m": 340.0,
            "slope_deg": 3.5, "aspect_deg": 180.0, "pop_density_source": "landscan_global",
            "pop_density_buffer_deg": 0.01, "lst_reference_radius_km": 75.0, "snapshot_version": "v1",
        }

    def test_new_months_is_the_set_difference(self, tmp_path):
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        snap.write_partition(str(old_dir), "lag_fill", "2023-06", [self._row(d=date(2023, 6, 1))])
        snap.write_partition(str(new_dir), "lag_fill", "2023-06", [self._row(d=date(2023, 6, 1))])
        snap.write_partition(str(new_dir), "lag_fill", "2023-07", [self._row(d=date(2023, 7, 1))])

        new_months = sfe.new_months_since_submission(str(new_dir), str(old_dir), "lag_fill")
        assert new_months == ["2023-07"]

    def test_no_new_months_returns_empty(self, tmp_path):
        d = tmp_path / "snap"
        snap.write_partition(str(d), "lag_fill", "2023-06", [self._row()])
        assert sfe.new_months_since_submission(str(d), str(d), "lag_fill") == []


class TestConsecutiveWins:
    def _cycle_line(self, submission_id, cycle, status, cell=None):
        return {"submission_id": submission_id, "cycle": cycle, "status": status, "cell": cell or _cell()}

    def test_two_consecutive_wins_counted(self):
        lines = [
            self._cycle_line("s1", "2026-09", "win"),
            self._cycle_line("s1", "2026-10", "win"),
        ]
        assert sfe.consecutive_wins(lines, "s1", ("ds-test", "lag_fill", "tmax", "Cfb")) == 2

    def test_streak_broken_by_a_loss(self):
        lines = [
            self._cycle_line("s1", "2026-08", "loss"),
            self._cycle_line("s1", "2026-09", "win"),
            self._cycle_line("s1", "2026-10", "win"),
        ]
        assert sfe.consecutive_wins(lines, "s1", ("ds-test", "lag_fill", "tmax", "Cfb")) == 2

    def test_streak_broken_by_a_different_submission_winning_more_recently(self):
        lines = [
            self._cycle_line("s1", "2026-09", "win"),
            self._cycle_line("s2", "2026-10", "win"),
        ]
        assert sfe.consecutive_wins(lines, "s1", ("ds-test", "lag_fill", "tmax", "Cfb")) == 0

    def test_different_cells_do_not_interfere(self):
        lines = [
            self._cycle_line("s1", "2026-10", "win", cell=_cell(zone="BWh")),
        ]
        assert sfe.consecutive_wins(lines, "s1", ("ds-test", "lag_fill", "tmax", "Cfb")) == 0

    def test_no_history_is_zero(self):
        assert sfe.consecutive_wins([], "s1", ("ds-test", "lag_fill", "tmax", "Cfb")) == 0


class TestCurrentTenureHolder:
    def test_open_tenure_is_the_holder(self):
        lines = [{"event": "tenure_start", "cell": _cell(), "author_github": "alice", "submission_id": "s1", "start_month": "2026-11"}]
        holder = sfe.current_tenure_holder(lines, ("ds-test", "lag_fill", "tmax", "Cfb"))
        assert holder["author_github"] == "alice"

    def test_closed_tenure_has_no_holder(self):
        lines = [
            {"event": "tenure_start", "cell": _cell(), "author_github": "alice", "submission_id": "s1", "start_month": "2026-11"},
            {"event": "tenure_end", "cell": _cell(), "author_github": "alice"},
        ]
        assert sfe.current_tenure_holder(lines, ("ds-test", "lag_fill", "tmax", "Cfb")) is None

    def test_no_history_returns_none(self):
        assert sfe.current_tenure_holder([], ("ds-test", "lag_fill", "tmax", "Cfb")) is None


class TestBuildLedgerLines:
    def _metrics(self):
        return {
            "eval_month": "2026-08", "n_forward": 100, "n_stations": 20, "rmse_grid_c": 1.9, "rmse_qrf_c": 1.7,
            "rmse_debiased_cv_c": 1.68, "rmse_improvement_pct_debiased_cv": 0.1,
            "bias_correction_c": 0.2, "spatial_skill": True, "gated_insufficient_n": False, "status": "win",
        }

    def test_cycle_line_is_schema_valid(self):
        from heatready_downscaling import ledger
        candidate = {"submission_id": "2026-08-001", "author_github": "alice"}
        line = sfe.build_cycle_line("2026-10", "v2026.10", candidate, ("ds-test", "lag_fill", "tmax", "Cfb"), self._metrics(), "0.1.0", "abc")
        ledger.validate_ledger_line("cycles", line)  # must not raise

    def test_tenure_start_is_schema_valid(self):
        from heatready_downscaling import ledger
        candidate = {"submission_id": "2026-08-001", "author_github": "alice"}
        line = sfe.build_tenure_start("2026-10", candidate, ("ds-test", "lag_fill", "tmax", "Cfb"), self._metrics(), ["2026-09", "2026-10"])
        ledger.validate_ledger_line("credit", line)

    def test_tenure_end_is_schema_valid(self):
        from heatready_downscaling import ledger
        prior = {"author_github": "alice", "start_month": "2026-05"}
        line = sfe.build_tenure_end(("ds-test", "lag_fill", "tmax", "Cfb"), prior, "2026-10", "2026-08-002")
        ledger.validate_ledger_line("credit", line)


class TestMainIntegration:
    """End-to-end: two synthetic snapshot versions (v1 has June only, v2
    adds July), one active submission whose OWN snapshot is v1, run main()
    for two consecutive cycles against v2 -- proves the whole pipeline
    (load candidates -> new-months diff -> score -> ledger append ->
    win-streak -> promotion) holds together."""

    def _row(self, station_id, d, tmax=30.0, tmin=20.0, grid_tmax=25.0, grid_tmin=15.0):
        return {
            "station_id": station_id, "date": d, "band": "lag_fill", "station_tmax_c": tmax, "station_tmin_c": tmin,
            "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin, "grid_specific_humidity_kgkg": 0.01, "nighttime_wind_ms": 2.0,
            "base_source": "x", "base_model": None, "humidity_source": "band", "wind_source": "band",
            "lon": -112.0, "lat": 33.4, "elevation_m": 340.0, "region": "US", "climate_zone": "Cfb",
            "koppen_main_group_code": 3, "obs_window_shift_days": 0,
            "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
            "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
            "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5, "elevation_mean_m": 340.0,
            "slope_deg": 3.5, "aspect_deg": 180.0, "pop_density_source": "landscan_global",
            "pop_density_buffer_deg": 0.01, "lst_reference_radius_km": 75.0, "snapshot_version": "v2026.10",
        }

    def _prediction_row(self, station_id, d, target, delta_c):
        return {
            "station_id": station_id, "date": d, "target": target, "delta_c": delta_c, "ci95_c": 0.4,
            "confidence": "high", "applied": True, "out_of_distribution": False,
            "covariates_missing": json.dumps([]), "cv_gate_passed": True, "model_version": "ds-test",
        }

    def _build_snapshot(self, snapshot_dir, months_and_rows):
        for month, rows in months_and_rows.items():
            snap.write_partition(str(snapshot_dir), "lag_fill", month, rows)
        # Enough distinct stations/rows per zone for the plain gate
        # (MIN_ZONE_N=30) to actually evaluate rather than gating closed.
        all_rows = [r for rows in months_and_rows.values() for r in rows]
        predictions = [
            self._prediction_row(r["station_id"], r["date"], target, delta_c=4.0)
            for r in all_rows for target in ("tmax", "tmin")
        ]
        snap.write_predictions_partition(str(snapshot_dir), "ds-test", "lag_fill", list(months_and_rows)[0], predictions)

    def test_full_cycle_promotes_after_two_wins(self, tmp_path, monkeypatch):
        # Build 40 station-days for June (v1's own submission-time snapshot)
        # and 40 MORE for July (only in v2 -- the "new" forward-eval data).
        june_rows = [self._row(f"S{i:03d}", date(2023, 6, 1)) for i in range(40)]
        july_rows = [self._row(f"S{i:03d}", date(2023, 7, 1)) for i in range(40)]

        v1_dir = tmp_path / "cache" / "v2026.08"
        v2_dir = tmp_path / "cache" / "v2026.10"
        self._build_snapshot(v1_dir, {"2023-06": june_rows})
        self._build_snapshot(v2_dir, {"2023-06": june_rows, "2023-07": july_rows})

        # A submission whose own snapshot_version is v2026.08 (the OLD one).
        submissions_root = tmp_path / "submissions"
        sub_dir = submissions_root / "2026-08" / "001-alice-lagfill-cfb"
        sub_dir.mkdir(parents=True)
        (sub_dir / "manifest.yaml").write_text(yaml.dump({
            "claims": [{"model_version": "ds-test", "band_key": "lag_fill", "targets": ["tmax"], "zones": ["Cfb"]}],
        }))

        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps({
            "ts": "2026-08-01T00:00:00Z", "submission_id": "2026-08-001", "author_github": "alice",
            "snapshot_version": "v2026.08", "reproduced": True,
        }) + "\n")
        (ledger_dir / "cycles.jsonl").write_text("")
        (ledger_dir / "credit.jsonl").write_text("")

        monkeypatch.setattr(sfe.rs, "download_snapshot", lambda version, cache_root=None: str(tmp_path / "cache" / version))

        # Cycle 1 (2026-09): should score a WIN and record one cycle line, no promotion yet.
        monkeypatch.setattr(sys, "argv", [
            "score_forward_eval.py", "--cycle", "2026-09", "--current-snapshot-version", "v2026.10",
            "--ledger-dir", str(ledger_dir), "--submissions-root", str(submissions_root),
        ])
        sfe.main()

        with open(ledger_dir / "cycles.jsonl") as f:
            cycles_after_1 = [json.loads(line) for line in f if line.strip()]
        with open(ledger_dir / "credit.jsonl") as f:
            credit_after_1 = [json.loads(line) for line in f if line.strip()]
        assert len(cycles_after_1) == 1
        assert cycles_after_1[0]["status"] == "win"
        assert credit_after_1 == []  # only one win so far -- not promoted yet

        # Cycle 2 (2026-10): re-run against the SAME "new" data (still v2026.10) --
        # a second win for the same submission should trigger promotion.
        monkeypatch.setattr(sys, "argv", [
            "score_forward_eval.py", "--cycle", "2026-10", "--current-snapshot-version", "v2026.10",
            "--ledger-dir", str(ledger_dir), "--submissions-root", str(submissions_root),
        ])
        sfe.main()

        with open(ledger_dir / "credit.jsonl") as f:
            credit_after_2 = [json.loads(line) for line in f if line.strip()]
        assert len(credit_after_2) == 1
        assert credit_after_2[0]["event"] == "tenure_start"
        assert credit_after_2[0]["author_github"] == "alice"

    def test_rung_b_candidate_status_decided_by_declared_value_not_mechanical_fit(self, tmp_path, monkeypatch):
        """The exact gap Codex's adversarial review caught in the original
        design: this same fixture (grid=25, station=30, delta_c=4.0) is a
        huge MECHANICAL win (qrf_err~=1.0 vs rmse_grid=5.0, way past
        AUTO_ENABLE_MARGIN). A Rung B candidate that declares a wildly
        wrong bias_correction_c must still LOSE the cycle -- if this
        script fell back to qrf_beats_grid_with_margin (the mechanically-
        fit number) instead of the candidate's own declared value, this
        would incorrectly come out a 'win'."""
        june_rows = [self._row(f"S{i:03d}", date(2023, 6, 1)) for i in range(40)]
        july_rows = [self._row(f"S{i:03d}", date(2023, 7, 1)) for i in range(40)]

        v1_dir = tmp_path / "cache" / "v2026.08"
        v2_dir = tmp_path / "cache" / "v2026.10"
        self._build_snapshot(v1_dir, {"2023-06": june_rows})
        self._build_snapshot(v2_dir, {"2023-06": june_rows, "2023-07": july_rows})

        submissions_root = tmp_path / "submissions"
        sub_dir = submissions_root / "2026-08" / "001-alice-lagfill-cfb"
        sub_dir.mkdir(parents=True)
        (sub_dir / "manifest.yaml").write_text(yaml.dump({
            "claims": [{"model_version": "ds-test", "band_key": "lag_fill", "targets": ["tmax"], "zones": ["Cfb"]}],
            # Wildly wrong: overcorrects by 20C on top of an already-small ~1C residual.
            "method": {"candidate": {"tmax": {"Cfb": {"bias_correction_c": -20.0}}}},
        }))

        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps({
            "ts": "2026-08-01T00:00:00Z", "submission_id": "2026-08-001", "author_github": "alice",
            "snapshot_version": "v2026.08", "reproduced": True,
        }) + "\n")
        (ledger_dir / "cycles.jsonl").write_text("")
        (ledger_dir / "credit.jsonl").write_text("")

        monkeypatch.setattr(sfe.rs, "download_snapshot", lambda version, cache_root=None: str(tmp_path / "cache" / version))
        monkeypatch.setattr(sys, "argv", [
            "score_forward_eval.py", "--cycle", "2026-09", "--current-snapshot-version", "v2026.10",
            "--ledger-dir", str(ledger_dir), "--submissions-root", str(submissions_root),
        ])
        sfe.main()

        with open(ledger_dir / "cycles.jsonl") as f:
            cycles_after_1 = [json.loads(line) for line in f if line.strip()]
        assert len(cycles_after_1) == 1
        assert cycles_after_1[0]["status"] == "loss"
        assert cycles_after_1[0]["proposed_correction_beats_grid_with_margin"] is False
