"""Unit tests for heatready_downscaling.gates -- ported from
crisisready/heat-risk-data-api's tests/test_publish_band_gate.py. See
PROVENANCE.md."""

import pytest

from heatready_downscaling import gates


def _metrics(qrf_beats_grid_with_margin, rmse_improvement_pct_debiased_cv=None, bias_correction_c=None,
             qrf_beats_grid=False, qrf_beats_grid_with_margin_affine=None, delta_scale_c=None):
    return {
        "qrf_beats_grid_with_margin": qrf_beats_grid_with_margin,
        "rmse_improvement_pct_debiased_cv": rmse_improvement_pct_debiased_cv,
        "bias_correction_c": bias_correction_c,
        "qrf_beats_grid": qrf_beats_grid,
        "qrf_beats_grid_with_margin_affine": qrf_beats_grid_with_margin_affine,
        "delta_scale_c": delta_scale_c,
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
            "delta_scale": {"tmax": {}, "tmin": {}},
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

    def test_old_style_metrics_dict_without_affine_keys_behaves_as_before(self):
        """A metrics dict from before delta_scale existed (no
        qrf_beats_grid_with_margin_affine/delta_scale_c keys at all, not
        even set to None) must not raise and must behave exactly as it did
        pre-extension -- .get() defaults, not direct key access."""
        report = {"by_target": {
            "tmax": {"Cfb": {
                "qrf_beats_grid_with_margin": True, "rmse_improvement_pct_debiased_cv": 0.05,
                "bias_correction_c": 0.412, "qrf_beats_grid": True,
            }},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"] == {"Cfb": True}
        assert gate["bias_correction"]["tmax"]["Cfb"] == 0.412
        assert "Cfb" not in gate["delta_scale"]["tmax"]


class TestDeltaScale:
    """The Cfb case this feature exists for: the offset-only correction
    fails its own margin (qrf_beats_grid_with_margin False) but its affine
    generalization clears it (qrf_beats_grid_with_margin_affine True) --
    the zone must still get enabled, via delta_scale instead of
    bias_correction."""

    def test_zone_passing_only_via_affine_is_enabled_via_delta_scale(self):
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(
                False, rmse_improvement_pct_debiased_cv=-0.0584, bias_correction_c=-0.333,
                qrf_beats_grid_with_margin_affine=True,
                delta_scale_c={"scale": 0.229, "offset": 0.537},
            )},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"]["Cfb"] is True
        assert gate["delta_scale"]["tmax"]["Cfb"] == {"scale": 0.229, "offset": 0.537}
        # bias_correction_c was never itself validated (offset-only failed
        # its own margin) -- must NOT be published as if it had been.
        assert "Cfb" not in gate["bias_correction"]["tmax"]

    def test_zone_passing_neither_path_stays_excluded(self):
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(False, qrf_beats_grid_with_margin_affine=False)},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert "Cfb" not in gate["tmax"]
        assert "Cfb" not in gate["delta_scale"]["tmax"]
        assert "Cfb" not in gate["bias_correction"]["tmax"]

    def test_zone_passing_both_paths_publishes_both_corrections(self):
        """Not Cfb's case, but a zone where the offset-only correction
        already clears margin AND the affine generalization is even
        better -- both get published; delta_scale alongside bias_correction,
        exactly as the plan specifies, not one replacing the other in the
        published JSON (serving decides priority, gates.py just publishes
        both when both validated)."""
        report = {"by_target": {
            "tmax": {"BSh": _metrics(
                True, rmse_improvement_pct_debiased_cv=0.117, bias_correction_c=0.6,
                qrf_beats_grid_with_margin_affine=True,
                delta_scale_c={"scale": 0.75, "offset": 0.4},
            )},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"]["BSh"] is True
        assert gate["bias_correction"]["tmax"]["BSh"] == 0.6
        assert gate["delta_scale"]["tmax"]["BSh"] == {"scale": 0.75, "offset": 0.4}

    def test_affine_passes_margin_but_no_delta_scale_c_publishes_nothing_new(self):
        """Defensive: qrf_beats_grid_with_margin_affine True but
        delta_scale_c None should not happen in practice (score.score_band
        only sets qrf_beats_grid_with_margin_affine meaningfully alongside
        delta_scale_c), but build_gate must not crash or publish a
        placeholder if it does."""
        report = {"by_target": {
            "tmax": {"Cfb": _metrics(
                False, qrf_beats_grid_with_margin_affine=True, delta_scale_c=None,
            )},
            "tmin": {},
        }}
        gate = gates.build_gate(report)
        assert gate["tmax"]["Cfb"] is True  # still enabled -- the margin check passed
        assert "Cfb" not in gate["delta_scale"]["tmax"]  # but nothing to publish


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


class TestBuildGateVariantMismatch:
    """2026-08-03, gate-variant scoping. UNLIKE band_key, this check is
    NOT skippable by omission -- both directions of mismatch are checked
    unconditionally (see build_gate's own docstring for why the
    variant-fitted-report-to-default-key direction is the dangerous one)."""

    def test_matching_variant_proceeds(self):
        report = {"base_variant": "native_noelev", "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        gate = gates.build_gate(report, variant="native_noelev")
        assert gate["tmax"] == {"Cfb": True}

    def test_default_report_with_no_variant_arg_proceeds(self):
        report = {"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        gate = gates.build_gate(report, variant=None)
        assert gate["tmax"] == {"Cfb": True}

    def test_variant_fitted_report_published_without_variant_arg_raises(self):
        """The dangerous direction: a native_noelev-fitted report must NOT
        silently land at the default (no-variant) key, which every other
        project in the same zone(s) reads under their unchanged, default
        Open-Meteo config."""
        report = {"base_variant": "native_noelev", "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        with pytest.raises(ValueError, match="native_noelev"):
            gates.build_gate(report, variant=None)

    def test_default_report_published_with_variant_arg_raises(self):
        report = {"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        with pytest.raises(ValueError, match="native_noelev"):
            gates.build_gate(report, variant="native_noelev")

    def test_mismatched_variant_names_raise(self):
        report = {"base_variant": "native_noelev", "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        with pytest.raises(ValueError):
            gates.build_gate(report, variant="some_other_variant")


class TestBuildGateZonesScopedReportRefusesDefaultKey:
    """2026-08-03, adversarial review finding: a --zones-scoped validation
    run with no --elevation-nan produces a report with base_variant=None,
    which the variant check above passes cleanly (None matches --variant
    omitted). Without this separate check, that report would silently
    replace the ENTIRE default gate with only the scoped zone(s)' data --
    build_gate has no merge step, it constructs the gate fresh from the
    report alone every time."""

    def test_zones_scoped_report_without_variant_raises(self):
        report = {"zones": ["Cfb"], "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        with pytest.raises(ValueError, match="Cfb"):
            gates.build_gate(report, variant=None)

    def test_zones_scoped_report_with_matching_variant_proceeds(self):
        """A zones-scoped report IS safe to publish, as long as it's going
        to a variant-suffixed key, never the shared default -- variant
        being non-None is the actual safety property, not the absence of
        a zones scope."""
        report = {
            "zones": ["Cfb"], "base_variant": "native_noelev",
            "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}},
        }
        gate = gates.build_gate(report, variant="native_noelev")
        assert gate["tmax"] == {"Cfb": True}

    def test_unscoped_report_without_variant_still_proceeds(self):
        """The new check must not affect a normal, unscoped, default-key
        publish -- only reports carrying a real zones scope."""
        report = {"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        gate = gates.build_gate(report, variant=None)
        assert gate["tmax"] == {"Cfb": True}

    def test_empty_zones_list_does_not_trigger_the_check(self):
        report = {"zones": [], "by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}}
        gate = gates.build_gate(report, variant=None)
        assert gate["tmax"] == {"Cfb": True}


class TestValidateGate:
    def test_well_formed_gate_passes(self):
        gate = gates.build_gate({"by_target": {"tmax": {"Cfb": _metrics(True)}, "tmin": {}}})
        gates.validate_gate(gate)  # must not raise

    def test_well_formed_gate_with_delta_scale_passes(self):
        """The empty-shape case above never exercises delta_scale's own
        per-zone {scale, offset} schema -- build a gate through the real
        affine-enable path so the schema is actually checked against a
        populated entry, not just its empty default."""
        gate = gates.build_gate({"by_target": {
            "tmax": {"Cfb": _metrics(
                False, qrf_beats_grid_with_margin_affine=True,
                delta_scale_c={"scale": 0.229, "offset": 0.537},
            )},
            "tmin": {},
        }})
        assert gate["delta_scale"]["tmax"]["Cfb"] == {"scale": 0.229, "offset": 0.537}
        gates.validate_gate(gate)  # must not raise

    def test_malformed_delta_scale_entry_raises(self):
        import jsonschema
        gate = {
            "tmax": {"Cfb": True}, "tmin": {},
            "bias_correction": {"tmax": {}, "tmin": {}},
            "delta_scale": {"tmax": {"Cfb": {"scale": 0.229}}, "tmin": {}},  # missing required "offset"
            "spatial_skill": {"tmax": {"Cfb": False}, "tmin": {}},
        }
        with pytest.raises(jsonschema.ValidationError):
            gates.validate_gate(gate)

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
