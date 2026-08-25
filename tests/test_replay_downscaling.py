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
    def _row(self, zone, lat, lon, d=date(2023, 7, 1), station_id="US001"):
        return {"climate_zone": zone, "lat": lat, "lon": lon, "date": d, "station_id": station_id}

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
        assert result["cross_subzone"]["n_pairs"] == 0  # zone not in subzoned_zones (default: none)

    def test_same_zone_different_subzone_counted_when_zone_is_subzoned(self):
        """Codex adversarial review finding, PR #25: French Cfb vs.
        neighboring German Cfb straddles exactly the boundary a
        delta_scale_subzone correction introduces -- the ORIGINAL
        cross-ZONE-only definition would find zero pairs for this case."""
        rows = [self._row("Cfb", 48.0, 2.0, station_id="FR001"), self._row("Cfb", 48.001, 2.001, station_id="DE001")]
        result = rd.boundary_discontinuity(
            rows, old_served=[20.0, 25.0], new_served=[20.0, 30.0], max_km=50.0, subzoned_zones=frozenset({"Cfb"}),
        )
        assert result["cross_subzone"]["n_pairs"] == 1
        assert result["cross_subzone"]["old_mean_abs_jump_c"] == pytest.approx(5.0)
        assert result["cross_subzone"]["new_mean_abs_jump_c"] == pytest.approx(10.0)
        assert result["cross_zone"]["n_pairs"] == 0  # same zone -- never double-counted as cross-zone

    def test_same_zone_different_subzone_excluded_when_zone_not_subzoned(self):
        """A zone with NO subzone-scoped correction published at all must
        not generate cross-subzone noise from an arbitrary country split
        that was never a real correction boundary."""
        rows = [self._row("Cfb", 48.0, 2.0, station_id="FR001"), self._row("Cfb", 48.001, 2.001, station_id="DE001")]
        result = rd.boundary_discontinuity(
            rows, old_served=[20.0, 25.0], new_served=[20.0, 30.0], max_km=50.0, subzoned_zones=frozenset(),
        )
        assert result["cross_subzone"]["n_pairs"] == 0

    def test_far_apart_pair_excluded(self):
        rows = [self._row("Cfb", 0.0, 0.0), self._row("BWh", 50.0, 50.0)]  # thousands of km apart
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 0

    def test_different_dates_never_paired(self):
        rows = [self._row("Cfb", 48.0, 2.0, d=date(2023, 7, 1)), self._row("BWh", 48.001, 2.001, d=date(2023, 7, 2))]
        result = rd.boundary_discontinuity(rows, old_served=[20.0, 25.0], new_served=[20.0, 22.0], max_km=50.0)
        assert result["n_cross_zone_pairs_within_km"] == 0


class TestCheckPartitionCoverage:
    """Codex adversarial review finding, PR #25: a mistyped --model-version
    produces a FrozenPredictionAdapter with zero frozen predictions --
    every row comes back not-applied under BOTH arms, and this tool would
    otherwise print a valid-looking 'nothing changes' report instead of
    failing loudly."""

    def test_zero_predictions_with_real_rows_raises(self):
        with pytest.raises(SystemExit, match="ZERO frozen predictions"):
            rd.check_partition_coverage([{"station_id": "S001"}], n_predictions=0, model_version="ds-typo", band_key="lag_fill")

    def test_some_predictions_does_not_raise(self):
        rd.check_partition_coverage([{"station_id": "S001"}], n_predictions=1, model_version="ds-test", band_key="lag_fill")  # must not raise

    def test_no_rows_at_all_does_not_raise(self):
        """Distinct from "rows exist but nothing predicted" -- an empty
        band is caught earlier in replay_band by its own separate
        'snapshot has no rows' check, not this one."""
        rd.check_partition_coverage([], n_predictions=0, model_version="ds-test", band_key="lag_fill")  # must not raise


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

    def test_replay_band_raises_on_mistyped_model_version(self, tmp_path):
        """End-to-end version of TestCheckPartitionCoverage -- proves
        replay_band itself refuses to produce a misleading report for a
        model_version with zero matching frozen predictions, not just
        the pure-logic check function in isolation."""
        self._build_snapshot(tmp_path, ["S001"])
        old_gate, new_gate = rd.load_gate(None), rd.load_gate(None)
        with pytest.raises(SystemExit, match="ZERO frozen predictions"):
            rd.replay_band(str(tmp_path), "ds-typo-does-not-exist", "lag_fill", old_gate, new_gate)

    def test_by_zone_hides_a_subzone_specific_regression_by_subzone_does_not(self, tmp_path):
        """Codex adversarial review finding, PR #25: a delta_scale_subzone
        regression confined to one country can be diluted/hidden in the
        (target, zone) aggregate by every other country's unaffected
        rows in the same zone -- by_subzone must isolate it."""
        self._build_snapshot(tmp_path, ["FR001", "US001"])
        old_gate = {  # a real published baseline, not the fail-closed default -- both arms must be
            "tmax": {"Cfb": True}, "tmin": {"Cfb": True},   # "applied" for paired_delta_diff to be meaningful
            "bias_correction": {"tmax": {}, "tmin": {}}, "delta_scale": {"tmax": {}, "tmin": {}},
        }
        new_gate = {
            "tmax": {"Cfb": True}, "tmin": {"Cfb": True},
            "bias_correction": {"tmax": {}, "tmin": {}}, "delta_scale": {"tmax": {}, "tmin": {}},
            # FR gets a wildly-overcorrecting subzone fit; US keeps whatever
            # the (absent) flat delta_scale/bias_correction would give it
            # (i.e. the raw model delta, unchanged) -- a real per-subzone
            # regression that a whole-Cfb average could still average away.
            "delta_scale_subzone": {"tmax": {"Cfb": {"FR": {"scale": 10.0, "offset": 100.0}}}, "tmin": {}},
        }
        result = rd.replay_band(str(tmp_path), "ds-test", "lag_fill", old_gate, new_gate)
        cfb = result["by_target"]["tmax"]["by_zone"]["Cfb"]
        assert "by_subzone" in cfb
        fr_diff = cfb["by_subzone"]["FR"]["paired_delta_diff"]["mean"]
        us_diff = cfb["by_subzone"]["US"]["paired_delta_diff"]["mean"]
        assert fr_diff is not None and abs(fr_diff) > 50  # the overcorrection is real and large
        assert us_diff is not None and abs(us_diff) < 10  # US is essentially unaffected

    def test_render_summary_does_not_raise_and_mentions_band(self, tmp_path):
        self._build_snapshot(tmp_path, ["S001"])
        old_gate = rd.load_gate(None)
        new_gate = {"tmax": {"Cfb": True}, "tmin": {"Cfb": True}, "bias_correction": {"tmax": {}, "tmin": {}}, "delta_scale": {"tmax": {}, "tmin": {}}}
        result = rd.replay_band(str(tmp_path), "ds-test", "lag_fill", old_gate, new_gate)
        summary = rd.render_summary(result)
        assert "lag_fill" in summary
        assert "ds-test" in summary
