"""Unit tests for scripts/run_submission.py's pure-logic pieces --
find_submission_dir, cross_check, fidelity_rows_for_band,
coverage_violations, render_comment -- plus one integration-style test of
reproduce() against a tiny real (tmp_path) snapshot. download_snapshot/
verify_snapshot (network + real sha256 of a live release asset) are
deliberately not exercised here -- this script was hand-verified against
the real crisisready/heat-ready-downscaling snapshot-v2026.07 release
separately."""

import json
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_submission as rs

from heatready_downscaling import snapshot as snap


def _manifest(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "submission_id": "2026-08-001",
        "author": {"github": "nishkishore", "name": "Nishant Kishore", "orcid": None, "affiliation": None},
        "track": "serving-ready",
        "rung": "A",
        "snapshot": {"version": "v2026.07", "manifest_sha256": "a" * 64},
        "claims": [{"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "targets": ["tmax"], "zones": ["Cfb"]}],
        "method": {
            "kind": "rerun-validator", "entrypoint": "scripts/run_submission.py",
            "args": [], "package_version": "0.1.0", "code_ref": None, "extra_covariates": [],
        },
        "claimed_report": "claimed_report.json",
        "tolerance": {"rmse_qrf_c": 0.005},
        "reproducibility": {"seed": 1, "runtime_notes": None},
    }
    base.update(overrides)
    return base


def _claimed_report(**overrides) -> dict:
    base = {
        "report_schema_version": 1, "model_version": "ds-2026.07-rf5", "band_key": "lag_fill",
        "snapshot_version": "v2026.07", "sample_requested": 0, "rows_sampled": 10, "rows_paired": 10,
        "fidelity_check": {"n": 0}, "by_target": {"tmax": {}, "tmin": {}},
    }
    base.update(overrides)
    return base


class TestFindSubmissionDir:
    def test_finds_exactly_one(self, tmp_path, monkeypatch):
        d = tmp_path / "submissions" / "2026-08" / "001-nishkishore-lagfill-cfb"
        d.mkdir(parents=True)
        (d / "manifest.yaml").write_text("x: 1")
        monkeypatch.chdir(tmp_path)
        assert rs.find_submission_dir("submissions") == os.path.join("submissions", "2026-08", "001-nishkishore-lagfill-cfb")

    def test_zero_submissions_raises(self, tmp_path, monkeypatch):
        (tmp_path / "submissions").mkdir()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit, match="found 0"):
            rs.find_submission_dir("submissions")

    def test_two_submissions_raises(self, tmp_path, monkeypatch):
        for name in ("001-a-x", "002-b-y"):
            d = tmp_path / "submissions" / "2026-08" / name
            d.mkdir(parents=True)
            (d / "manifest.yaml").write_text("x: 1")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit, match="found 2"):
            rs.find_submission_dir("submissions")


class TestCrossCheck:
    _DIR = "submissions/2026-08/001-nishkishore-lagfill-cfb"

    def test_well_formed_submission_has_no_violations(self):
        assert rs.cross_check(_manifest(), _claimed_report(), self._DIR, "nishkishore") == []

    def test_submission_id_mismatch_flagged(self):
        m = _manifest(submission_id="2026-08-999")
        violations = rs.cross_check(m, _claimed_report(), self._DIR, "nishkishore")
        assert any("submission_id" in v for v in violations)

    def test_author_mismatch_with_directory_flagged(self):
        m = _manifest()
        m["author"]["github"] = "someone-else"
        violations = rs.cross_check(m, _claimed_report(), self._DIR, "nishkishore")
        assert any("doesn't start with author.github" in v for v in violations)

    def test_pr_author_mismatch_with_directory_flagged(self):
        violations = rs.cross_check(_manifest(), _claimed_report(), self._DIR, "a-different-user")
        assert any("PR's actual author" in v for v in violations)

    def test_rung_c_rejected(self):
        violations = rs.cross_check(_manifest(rung="C"), _claimed_report(), self._DIR, "nishkishore")
        assert any("not yet open" in v for v in violations)

    def test_rung_b_rejected(self):
        """Rung B is schema-valid (submission.py) but not yet scoreable --
        score_band has no input for a contributor-proposed correction. v1
        intake is Rung A only, a deliberate scope decision, not an
        oversight (see cross_check's own comment)."""
        violations = rs.cross_check(_manifest(rung="B"), _claimed_report(), self._DIR, "nishkishore")
        assert any("not yet open" in v for v in violations)

    def test_rung_a_accepted(self):
        violations = rs.cross_check(_manifest(rung="A"), _claimed_report(), self._DIR, "nishkishore")
        assert not any("not yet open" in v for v in violations)

    def test_multiple_claims_rejected(self):
        m = _manifest()
        m["claims"] = m["claims"] * 2
        violations = rs.cross_check(m, _claimed_report(), self._DIR, "nishkishore")
        assert any("exactly one claims" in v for v in violations)

    def test_claimed_report_band_key_mismatch_flagged(self):
        violations = rs.cross_check(_manifest(), _claimed_report(band_key="forecast_lead1"), self._DIR, "nishkishore")
        assert any("band_key" in v for v in violations)

    def test_claimed_report_model_version_mismatch_flagged(self):
        violations = rs.cross_check(_manifest(), _claimed_report(model_version="ds-2026.07-rf4"), self._DIR, "nishkishore")
        assert any("model_version" in v for v in violations)

    def test_claimed_report_snapshot_version_mismatch_flagged(self):
        violations = rs.cross_check(_manifest(), _claimed_report(snapshot_version="v2026.08"), self._DIR, "nishkishore")
        assert any("snapshot_version" in v for v in violations)

    def test_package_version_mismatch_flagged(self, monkeypatch):
        monkeypatch.setattr(rs, "_package_version", lambda: "9.9.9")
        violations = rs.cross_check(_manifest(), _claimed_report(), self._DIR, "nishkishore")
        assert any("package_version" in v for v in violations)

    def test_malformed_directory_path_flagged(self):
        violations = rs.cross_check(_manifest(), _claimed_report(), "not/a/valid/path", "nishkishore")
        assert any("doesn't match" in v for v in violations)


class TestCheckSubmissionIdUnique:
    def test_no_ledger_file_is_fine(self, tmp_path):
        assert rs.check_submission_id_unique("2026-08-001", str(tmp_path / "ledger")) == []

    def test_empty_ledger_is_fine(self, tmp_path):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text("")
        assert rs.check_submission_id_unique("2026-08-001", str(ledger_dir)) == []

    def test_new_id_is_fine(self, tmp_path):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps({"submission_id": "2026-08-001"}) + "\n")
        assert rs.check_submission_id_unique("2026-08-002", str(ledger_dir)) == []

    def test_duplicate_id_flagged(self, tmp_path):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "submissions.jsonl").write_text(json.dumps({"submission_id": "2026-08-001"}) + "\n")
        violations = rs.check_submission_id_unique("2026-08-001", str(ledger_dir))
        assert any("already exists" in v for v in violations)


class TestFidelityRowsForBand:
    def test_joins_on_station_and_date(self):
        era5_rows = [{"station_id": "A", "date": date(2023, 1, 1), "grid_tmax_c": 30.0, "grid_tmin_c": 20.0}]
        band_rows = [{"station_id": "A", "date": date(2023, 1, 1), "grid_tmax_c": 31.0, "grid_tmin_c": 19.5}]
        rows = rs.fidelity_rows_for_band(band_rows, era5_rows)
        assert len(rows) == 1
        assert rows[0]["era5_tmax"] == 30.0
        assert rows[0]["nrt_tmax"] == 31.0

    def test_unmatched_station_day_dropped(self):
        era5_rows = [{"station_id": "A", "date": date(2023, 1, 1), "grid_tmax_c": 30.0, "grid_tmin_c": 20.0}]
        band_rows = [{"station_id": "B", "date": date(2023, 1, 1), "grid_tmax_c": 31.0, "grid_tmin_c": 19.5}]
        assert rs.fidelity_rows_for_band(band_rows, era5_rows) == []

    def test_none_values_dropped(self):
        era5_rows = [{"station_id": "A", "date": date(2023, 1, 1), "grid_tmax_c": None, "grid_tmin_c": 20.0}]
        band_rows = [{"station_id": "A", "date": date(2023, 1, 1), "grid_tmax_c": 31.0, "grid_tmin_c": 19.5}]
        assert rs.fidelity_rows_for_band(band_rows, era5_rows) == []


class TestCoverageViolations:
    def test_missing_zone_flagged(self):
        manifest = _manifest()
        reproduced = {"by_target": {"tmax": {}}}
        violations = rs.coverage_violations(manifest, reproduced)
        assert any("Cfb" in v and "not present" in v for v in violations)

    def test_zero_applied_rows_flagged(self):
        manifest = _manifest()
        reproduced = {"by_target": {"tmax": {"Cfb": {"n_qrf_applied": 0}}}}
        violations = rs.coverage_violations(manifest, reproduced)
        assert any("zero applied rows" in v for v in violations)

    def test_covered_zone_is_not_flagged(self):
        manifest = _manifest()
        reproduced = {"by_target": {"tmax": {"Cfb": {"n_qrf_applied": 42}}}}
        assert rs.coverage_violations(manifest, reproduced) == []


class TestRenderComment:
    def test_hard_reject_renders_rejection_section(self):
        comment = rs.render_comment(_manifest(), ["some violation"], None, None, [])
        assert "Rejected" in comment
        assert "some violation" in comment

    def test_passing_submission_renders_pass_section(self):
        result = report_result(passed=True, max_abs_deviation={"rmse_qrf_c": 0.001}, violations=[])
        reproduced = {"by_target": {"tmax": {"Cfb": _metrics()}, "tmin": {}}}
        comment = rs.render_comment(_manifest(), [], result, reproduced, [])
        assert "Reproduced within tolerance" in comment
        assert "Cfb" in comment

    def test_failing_submission_renders_violations(self):
        result = report_result(
            passed=False, max_abs_deviation={"rmse_qrf_c": 0.5},
            violations=[{"target": "tmax", "zone": "Cfb", "metric": "rmse_qrf_c", "claimed": 1.0, "reproduced": 1.5, "abs_diff": 0.5, "allowed": 0.005}],
        )
        reproduced = {"by_target": {"tmax": {"Cfb": _metrics()}, "tmin": {}}}
        comment = rs.render_comment(_manifest(), [], result, reproduced, [])
        assert "Did not fully reproduce" in comment
        assert "Tolerance violations" in comment


def _metrics():
    return {
        "n_qrf_applied": 42, "rmse_grid_c": 1.9, "rmse_qrf_c": 1.7,
        "rmse_improvement_pct_debiased_cv": 0.1, "qrf_beats_grid_with_margin": True,
    }


def report_result(*, passed, max_abs_deviation, violations):
    from heatready_downscaling.report import ToleranceResult
    return ToleranceResult(passed=passed, max_abs_deviation=max_abs_deviation, violations=violations)


class TestReproduceIntegration:
    """Builds a tiny real snapshot (era5 + lag_fill bands, frozen
    predictions) in tmp_path and runs the actual reproduce() pipeline end
    to end -- proves read_band_partitions -> FrozenPredictionAdapter ->
    score_band -> fidelity_report -> build_report hold together, not just
    each piece in isolation."""

    def _row(self, station_id, d, band, grid_tmax, grid_tmin, station_tmax=32.0, station_tmin=21.0):
        return {
            "station_id": station_id, "date": d, "band": band,
            "station_tmax_c": station_tmax, "station_tmin_c": station_tmin,
            "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin,
            "grid_specific_humidity_kgkg": 0.012, "nighttime_wind_ms": 2.0,
            "base_source": "x", "base_model": None, "humidity_source": "band", "wind_source": "band",
            "lon": -112.0, "lat": 33.4, "elevation_m": 340.0, "region": "US", "climate_zone": "Cfb",
            "koppen_main_group_code": 3, "obs_window_shift_days": 0,
            "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
            "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
            "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5, "elevation_mean_m": 340.0,
            "slope_deg": 3.5, "aspect_deg": 180.0,
            "pop_density_source": "landscan_global", "pop_density_buffer_deg": 0.01,
            "lst_reference_radius_km": 75.0, "snapshot_version": "v2026.07",
        }

    def _prediction_row(self, station_id, d, target, delta_c):
        return {
            "station_id": station_id, "date": d, "target": target, "delta_c": delta_c, "ci95_c": 0.4,
            "confidence": "high", "applied": True, "out_of_distribution": False,
            "covariates_missing": json.dumps([]), "cv_gate_passed": True, "model_version": "ds-test",
        }

    def test_reproduce_end_to_end(self, tmp_path):
        d = date(2023, 7, 1)
        era5_rows = [self._row("A", d, "era5", 30.0, 20.0)]
        lag_fill_rows = [self._row("A", d, "lag_fill", 30.5, 20.5)]
        snap.write_partition(str(tmp_path), "era5", "2023-07", era5_rows)
        snap.write_partition(str(tmp_path), "lag_fill", "2023-07", lag_fill_rows)

        # Both targets' frozen predictions go in ONE write_predictions_partition
        # call -- separate calls for the same (model_version, band, month)
        # would each overwrite the same part-0.parquet file rather than
        # accumulate.
        snap.write_predictions_partition(
            str(tmp_path), "ds-test", "lag_fill", "2023-07",
            [self._prediction_row("A", d, target, delta_c=1.0) for target in ("tmax", "tmin")],
        )

        result = rs.reproduce(str(tmp_path), "ds-test", "lag_fill", "v2026.07")
        assert result["model_version"] == "ds-test"
        assert result["band_key"] == "lag_fill"
        assert result["fidelity_check"]["n"] == 1  # joined against the era5 row
        assert "Cfb" in result["by_target"]["tmax"]
        assert result["by_target"]["tmax"]["Cfb"]["n_qrf_applied"] == 1

    def test_era5_band_has_no_fidelity_check(self, tmp_path):
        d = date(2023, 7, 1)
        snap.write_partition(str(tmp_path), "era5", "2023-07", [self._row("A", d, "era5", 30.0, 20.0)])
        snap.write_predictions_partition(
            str(tmp_path), "ds-test", "era5", "2023-07",
            [self._prediction_row("A", d, target, delta_c=1.0) for target in ("tmax", "tmin")],
        )
        result = rs.reproduce(str(tmp_path), "ds-test", "era5", "v2026.07")
        assert result["fidelity_check"] == {"n": 0}
