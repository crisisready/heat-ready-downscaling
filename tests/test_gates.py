"""Unit tests for heatready_downscaling.gates -- ported from
crisisready/heat-risk-data-api's tests/test_publish_band_gate.py. See
PROVENANCE.md."""

import pytest

from heatready_downscaling import gates


def _metrics(qrf_beats_grid_with_margin, rmse_improvement_pct_debiased_cv=None, bias_correction_c=None,
             qrf_beats_grid=False):
    return {
        "qrf_beats_grid_with_margin": qrf_beats_grid_with_margin,
        "rmse_improvement_pct_debiased_cv": rmse_improvement_pct_debiased_cv,
        "bias_correction_c": bias_correction_c,
        "qrf_beats_grid": qrf_beats_grid,
    }


class TestBuildGate:
    def test_only_true_verdicts_are_published(self):
        report = {"by_target": {
            "tmax": {
                "Cfb": _metrics(True),
                "Cfa": _metrics(False),
                "BSk": _metrics(None),
            },
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"] == {"Cfb": True}
        assert "Cfa" not in gate["tmax"]
        assert "BSk" not in gate["tmax"]

    def test_cv_validated_zone_publishes_its_bias_correction(self):
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(True, rmse_improvement_pct_debiased_cv=0.05, bias_correction_c=0.412)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["bias_correction"]["tmax"]["Cfb"] == 0.412

    def test_fallback_path_zone_publishes_no_bias_correction(self):
        # rmse_improvement_pct_debiased_cv is None -- too few distinct
        # stations for CV, so no correction was ever validated.
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(True, rmse_improvement_pct_debiased_cv=None, bias_correction_c=None)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert "Cfb" not in gate["bias_correction"]["tmax"]

    def test_empty_report_yields_empty_gate_with_bias_correction_shape_present(self):
        gate = gates.build_gate({"by_target": {"tmax": {}, "tmin": {}}})
        assert gate == {
            "tmax": {}, "tmin": {},
            "bias_correction": {"tmax": {}, "tmin": {}},
            "spatial_skill": {"tmax": {}, "tmin": {}},
        }

    def test_both_targets_handled_independently(self):
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(True)},
            "tmin": {"Cfb": _metrics(False)},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"] == {"Cfb": True}
        assert gate["tmin"] == {}


class TestSpatialSkill:
    def test_raw_qrf_beats_grid_is_genuine_spatial_skill(self):
        report = {"by_target": {
            "tmax": {"Am": _metrics(True, qrf_beats_grid=True)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["spatial_skill"]["tmax"]["Am"] is True

    def test_raw_qrf_fails_but_margin_passes_is_bias_correction_only(self):
        """raw qrf_beats_grid is False (the spatial delta alone is worse
        than grid) but qrf_beats_grid_with_margin is True (the debiased/
        bias-corrected value clears the bar) -- must be labeled False, not
        silently treated the same as genuine spatial skill."""
        report = {"by_target": {
            "tmax": {"Cwa": _metrics(True, rmse_improvement_pct_debiased_cv=0.253,
                                      bias_correction_c=-2.518, qrf_beats_grid=False)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"]["Cwa"] is True  # still published -- it's a valid, CV-proven correction
        assert gate["spatial_skill"]["tmax"]["Cwa"] is False  # but NOT genuine spatial skill
        assert gate["bias_correction"]["tmax"]["Cwa"] == -2.518

    def test_spatial_skill_only_recorded_for_published_zones(self):
        report = {"by_target": {
            "tmax": {"Dfc": _metrics(False, qrf_beats_grid=False)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert "Dfc" not in gate["tmax"]
        assert "Dfc" not in gate["spatial_skill"]["tmax"]


class TestBuildGateBandKeyMismatch:
    def test_mismatched_band_key_raises(self):
        report = {"band_key": "lag_fill", "by_target": {"tmax": {}, "tmin": {}}}
        with pytest.raises(ValueError, match="lag_fill"):
            gates.build_gate(report, band_key="forecast_lead3")

    def test_matching_band_key_proceeds(self):
        report = {"band_key": "lag_fill", "by_target": {
            "tmax": {"Cfb": _metrics(True)}, "tmin": {},
        }}
        gate = gates.build_gate(report, band_key="lag_fill")
        assert gate["tmax"] == {"Cfb": True}

    def test_band_key_omitted_skips_the_check(self):
        report = {"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        gate = gates.build_gate(report)
        assert gate["tmax"] == {"Cfb": True}


class TestValidateGate:
    def test_well_formed_gate_passes(self):
        gate = gates.build_gate({"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}})
        gates.validate_gate(gate)  # must not raise

    def test_malformed_gate_raises(self):
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            gates.validate_gate({"tmax": "not an object"})


class TestValidateBlendGate:
    def test_well_formed_blend_gate_passes(self):
        blend_gate = {
            "tmax": {"Cfb": True}, "tmin": {"Cfb": True},
            "params": {
                "tmax": {"C": {"L_km": 10.0, "R_km": 25.0, "tau": 2.0}},
                "tmin": {"C": {"L_km": 10.0, "R_km": 25.0, "tau": 2.0}},
            },
        }
        gates.validate_blend_gate(blend_gate)  # must not raise

    def test_missing_params_key_raises(self):
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            gates.validate_blend_gate({"tmax": {}, "tmin": {}})

    def test_malformed_param_group_raises(self):
        import jsonschema
        blend_gate = {
            "tmax": {}, "tmin": {},
            "params": {"tmax": {"C": {"L_km": 10.0}}, "tmin": {}},  # missing R_km/tau
        }
        with pytest.raises(jsonschema.ValidationError):
            gates.validate_blend_gate(blend_gate)
