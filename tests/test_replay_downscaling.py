"""Unit tests for scripts/replay_downscaling.py -- the tier-mix/
delta-distribution diff GOVERNANCE.md requires for promotion review."""

import json
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import replay_downscaling as rd

from heatready_downscaling import snapshot as snap


class TestLoadGate:
    def test_none_path_returns_fail_closed_empty(self):
        gate = rd.load_gate(None)
        assert gate["tmax"] == {} and gate["tmin"] == {}
        assert gate["bias_correction"] == {"tmax": {}, "tmin": {}}
        assert gate["delta_scale"] == {"tmax": {}, "tmin": {}}

    def test_valid_gate_file_loads(self, tmp_path):
        gate = {
            "tmax": {"Cfb": True}, "tmin": {"Cfb": True},
            "bias_correction": {"tmax": {"Cfb": 0.5}, "tmin": {}},
            "spatial_skill": {"tmax": {"Cfb": True}, "tmin": {}},
        }
        p = tmp_path / "gate.json"
        p.write_text(json.dumps(gate))
        loaded = rd.load_gate(str(p))
        assert loaded == gate

    def test_malformed_gate_file_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"tmax": {"Cfb": "not-a-bool"}}))
        with pytest.raises(Exception):
            rd.load_gate(str(p))


class TestSubzoneCode:
    def test_normal_station_id(self):
        assert rd.subzone_code("FR000012345") == "FR"

    def test_none_returns_none(self):
        assert rd.subzone_code(None) is None

    def test_short_string_returns_none(self):
        assert rd.subzone_code("F") is None


class TestResolveDeltaScale:
    def _gate(self):
        return {
            "delta_scale": {"tmax": {"Cfb": {"scale": 1.0, "offset": 0.0}}},
            "delta_scale_subzone": {"tmax": {"Cfb": {"FR": {"scale": 0.98, "offset": 0.1}}}},
        }

    def test_subzone_entry_wins_over_flat(self):
        result = rd.resolve_delta_scale(self._gate(), "tmax", "Cfb", "FR")
        assert result == {"scale": 0.98, "offset": 0.1}

    def test_falls_back_to_flat_when_no_subzone_entry(self):
        result = rd.resolve_delta_scale(self._gate(), "tmax", "Cfb", "DE")
        assert result == {"scale": 1.0, "offset": 0.0}

    def test_falls_back_to_flat_when_subzone_none(self):
        result = rd.resolve_delta_scale(self._gate(), "tmax", "Cfb", None)
        assert result == {"scale": 1.0, "offset": 0.0}

    def test_none_when_neither_published(self):
        assert rd.resolve_delta_scale({}, "tmax", "BWh", "US") is None


class TestSubzoneStatus:
    def test_distinguishes_applied_vs_informational_only(self):
        gate = {
            "delta_scale_subzone": {"tmax": {"Cfb": {"FR": {"scale": 0.9, "offset": 0.1}}}},
            "bias_correction_subzone": {"tmax": {"Cfb": {"FR": 0.5}}},
        }
        status = rd.subzone_status(gate, "tmax")
        assert status["delta_scale_subzone_cells_applied"] == ["Cfb.FR"]
        assert status["bias_correction_subzone_cells_published_but_not_applied"] == ["Cfb.FR"]


class TestTierMix:
    def _pred(self, applied, confidence=None, ood=False, cv_gate_passed=True, delta_c=1.0):
        return {"applied": applied, "confidence": confidence, "delta_c": delta_c if applied else None,
                "out_of_distribution": ood, "cv_gate_passed": cv_gate_passed}

    def test_counts_and_rates(self):
        preds = [
            self._pred(True, "high"), self._pred(True, "high"), self._pred(True, "medium", ood=True),
            self._pred(False, cv_gate_passed=False),
        ]
        m = rd.tier_mix(preds)
        assert m["n"] == 4
        assert m["n_applied"] == 3
        assert m["applied_rate"] == pytest.approx(0.75)
        assert m["confidence_counts"] == {"high": 2, "medium": 1, "low": 0}
        assert m["cv_gate_passed_rate"] == pytest.approx(0.75)
        assert m["out_of_distribution_rate"] == pytest.approx(1 / 3)

    def test_empty_input(self):
        m = rd.tier_mix([])
        assert m["n"] == 0
        assert m["applied_rate"] is None
        assert m["out_of_distribution_rate"] is None


class TestDeltaDistribution:
    def test_basic_stats(self):
        preds = [{"applied": True, "delta_c": v} for v in (1.0, 2.0, 3.0)] + [{"applied": False, "delta_c": None}]
        d = rd.delta_distribution(preds)
        assert d["n"] == 3
        assert d["mean"] == pytest.approx(2.0)

    def test_empty_returns_none_stats(self):
        d = rd.delta_distribution([])
        assert d["n"] == 0
        assert d["mean"] is None


class TestPairedDeltaDiff:
    def test_only_pairs_applied_under_both(self):
        old = [{"applied": True, "delta_c": 1.0}, {"applied": False, "delta_c": None}, {"applied": True, "delta_c": 2.0}]
        new = [{"applied": True, "delta_c": 1.5}, {"applied": True, "delta_c": 9.0}, {"applied": True, "delta_c": 2.0}]
        d = rd.paired_delta_diff(old, new)
        assert d["n"] == 2  # index 1 excluded -- not applied under `old`
        assert d["mean"] == pytest.approx((0.5 + 0.0) / 2)


class TestBoundaryDiscontinuity:
    def _row(self, zone, lat, lon, d=date(2023, 7, 1)):
        return {"climate_zone": zone, "lat": lat, "lon": lon, "date": d}

    def test_close_cross_zone_pair_counted(self):
        rows = [self._row("Cfb", 48.0, 2.0), self._row("BWh", 48.001, 2.001)]
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 1
        assert result["old_mean_abs_jump_c"] == pytest.approx(5.0)
        assert result["new_mean_abs_jump_c"] == pytest.approx(2.0)

    def test_same_zone_pair_excluded(self):
        rows = [self._row("Cfb", 48.0, 2.0), self._row("Cfb", 48.001, 2.001)]
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 0
        assert result["old_mean_abs_jump_c"] is None

    def test_far_apart_pair_excluded(self):
        rows = [self._row("Cfb", 0.0, 0.0), self._row("BWh", 50.0, 50.0)]  # thousands of km apart
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 0

    def test_different_dates_never_paired(self):
        rows = [self._row("Cfb", 48.0, 2.0, d=date(2023, 7, 1)), self._row("BWh", 48.001, 2.001, d=date(2023, 7, 2))]
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 0


class TestPredictWithGateAndReplayBandIntegration:
    """Builds a tiny real snapshot (frozen predictions + band rows) and
    drives predict_with_gate/replay_band end to end -- proves the subzone
    splicing actually reaches the right rows, and that the two arms
    (old/new) produce a visibly different tier-mix/delta-distribution."""

    def _row(self, station_id, d, zone="Cfb", grid_tmax=25.0, grid_tmin=15.0):
        return {
            "station_id": station_id, "date": d, "band": "lag_fill",
            "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin,
            "grid_specific_humidity_kgkg": 0.012, "nighttime_wind_ms": 2.0,
            "base_source": "x", "base_model": None, "humidity_source": "band", "wind_source": "band",
            "lon": 2.0, "lat": 48.0, "elevation_m": 340.0, "region": "US", "climate_zone": zone,
            "koppen_main_group_code": 3, "obs_window_shift_days": 0,
            "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
            "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
            "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5, "elevation_mean_m": 340.0,
            "slope_deg": 3.5, "aspect_deg": 180.0, "pop_density_source": "landscan_global",
            "pop_density_buffer_deg": 0.01, "lst_reference_radius_km": 75.0, "snapshot_version": "v2026.07",
        }

    def _prediction_row(self, station_id, d, target, delta_c):
        return {
            "station_id": station_id, "date": d, "target": target, "delta_c": delta_c, "ci95_c": 0.4,
            "confidence": "high", "applied": True, "out_of_distribution": False,
            "covariates_missing": json.dumps([]), "cv_gate_passed": True, "model_version": "ds-test",
        }

    def _build_snapshot(self, tmp_path, station_ids):
        d = date(2023, 7, 1)
        rows = [self._row(sid, d) for sid in station_ids]
        snap.write_partition(str(tmp_path), "lag_fill", "2023-07", rows)
        preds = [self._prediction_row(sid, d, target, delta_c=4.0) for sid in station_ids for target in ("tmax", "tmin")]
        snap.write_predictions_partition(str(tmp_path), "ds-test", "lag_fill", "2023-07", preds)
        return rows

    def test_subzone_override_applies_only_to_matching_station(self, tmp_path):
        from heatready_downscaling import contract
        rows = self._build_snapshot(tmp_path, ["FR001", "US001"])
        adapter = contract.FrozenPredictionAdapter.from_snapshot(str(tmp_path), "ds-test", "lag_fill")
        gate = {
            "tmax": {"Cfb": True}, "delta_scale": {"tmax": {"Cfb": {"scale": 1.0, "offset": 0.0}}},
            "delta_scale_subzone": {"tmax": {"Cfb": {"FR": {"scale": 2.0, "offset": 100.0}}}},
        }
        preds = rd.predict_with_gate(adapter, rows, "tmax", gate)
        fr_pred = next(p for r, p in zip(rows, preds) if r["station_id"] == "FR001")
        us_pred = next(p for r, p in zip(rows, preds) if r["station_id"] == "US001")
        assert fr_pred["delta_c"] == pytest.approx(4.0 * 2.0 + 100.0)  # subzone fit applied
        assert us_pred["delta_c"] == pytest.approx(4.0 * 1.0 + 0.0)    # flat fit applied

    def test_replay_band_shows_a_real_diff_between_arms(self, tmp_path):
        self._build_snapshot(tmp_path, ["S001", "S002"])
        old_gate = rd.load_gate(None)  # nothing published -- fail closed
        new_gate = {
            "tmax": {"Cfb": True}, "tmin": {"Cfb": True},
            "bias_correction": {"tmax": {"Cfb": 0.5}, "tmin": {"Cfb": 0.5}},
            "delta_scale": {"tmax": {}, "tmin": {}},
        }
        result = rd.replay_band(str(tmp_path), "ds-test", "lag_fill", old_gate, new_gate)
        tmax = result["by_target"]["tmax"]
        assert tmax["overall"]["tier_mix"]["old"]["applied_rate"] == pytest.approx(0.0)  # fail-closed baseline
        assert tmax["overall"]["tier_mix"]["new"]["applied_rate"] == pytest.approx(1.0)  # newly lit
        assert "Cfb" in tmax["by_zone"]

    def test_render_summary_does_not_raise_and_mentions_band(self, tmp_path):
        self._build_snapshot(tmp_path, ["S001"])
        old_gate = rd.load_gate(None)
        new_gate = {"tmax": {"Cfb": True}, "tmin": {"Cfb": True}, "bias_correction": {"tmax": {}, "tmin": {}}, "delta_scale": {"tmax": {}, "tmin": {}}}
        result = rd.replay_band(str(tmp_path), "ds-test", "lag_fill", old_gate, new_gate)
        summary = rd.render_summary(result)
        assert "lag_fill" in summary
        assert "ds-test" in summary
