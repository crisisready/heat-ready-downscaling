"""Unit tests for heatready_downscaling.report."""

import jsonschema
import pytest

from heatready_downscaling import report


def _metrics(rmse_qrf_c=1.5, qrf_beats_grid=True):
    return {
        "n_grid": 40, "n_qrf_applied": 40, "rmse_grid_c": 2.0, "rmse_qrf_c": rmse_qrf_c,
        "bias_qrf_c": 0.1, "se_bias_qrf_c": 0.05, "bias_correction_c": None,
        "rmse_debiased_cv_c": None, "bias_debiased_cv_c": None, "bias_bounded_uncorrected": True,
        "rmse_improvement_pct": 0.1, "rmse_improvement_pct_debiased_cv": None,
        "qrf_beats_grid": qrf_beats_grid, "qrf_beats_grid_with_margin": qrf_beats_grid,
        "gated_insufficient_n": False,
    }


class TestBuildReport:
    def test_core_envelope_fields(self):
        r = report.build_report(
            model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
            sample_requested=10, rows_sampled=10, rows_paired=10,
            fidelity_check={"n": 5}, by_target={"tmax": {"Cfb": _metrics()}, "tmin": {}},
        )
        assert r["report_schema_version"] == report.REPORT_SCHEMA_VERSION
        assert r["model_version"] == "ds-test"
        assert r["band_key"] == "lag_fill"
        assert r["snapshot_version"] == "v2026.08"
        assert r["rows_paired"] == 10
        report.validate_report(r)  # must not raise

    def test_generated_by_omitted_when_not_given(self):
        r = report.build_report(
            model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
            sample_requested=1, rows_sampled=1, rows_paired=1,
            fidelity_check={"n": 0}, by_target={"tmax": {}, "tmin": {}},
        )
        assert "generated_by" not in r

    def test_extra_fields_merged_into_top_level(self):
        r = report.build_report(
            model_version="ds-test", band_key="forecast_lead3", snapshot_version="v2026.08",
            sample_requested=1, rows_sampled=1, rows_paired=1,
            fidelity_check={"n": 0}, by_target={"tmax": {}, "tmin": {}},
            extra={"lead_days": 3},
        )
        assert r["lead_days"] == 3

    def test_extra_field_colliding_with_core_envelope_raises(self):
        with pytest.raises(ValueError, match="rows_paired"):
            report.build_report(
                model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
                sample_requested=1, rows_sampled=1, rows_paired=1,
                fidelity_check={"n": 0}, by_target={"tmax": {}, "tmin": {}},
                extra={"rows_paired": 999},
            )


class TestValidateReport:
    def test_missing_required_field_raises(self):
        with pytest.raises(jsonschema.ValidationError):
            report.validate_report({"report_schema_version": 1})

    def test_malformed_by_target_metrics_raises(self):
        r = report.build_report(
            model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
            sample_requested=1, rows_sampled=1, rows_paired=1,
            fidelity_check={"n": 0}, by_target={"tmax": {"Cfb": {"rmse_qrf_c": 1.5}}, "tmin": {}},
        )
        with pytest.raises(jsonschema.ValidationError):
            report.validate_report(r)


class TestCompareReports:
    def _report(self, rmse_qrf_c):
        return report.build_report(
            model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
            sample_requested=10, rows_sampled=10, rows_paired=10,
            fidelity_check={"n": 5}, by_target={"tmax": {"Cfb": _metrics(rmse_qrf_c=rmse_qrf_c)}, "tmin": {}},
        )

    def test_within_tolerance_passes(self):
        result = report.compare_reports(self._report(1.5), self._report(1.501), {"rmse_qrf_c": 0.005})
        assert result.passed
        assert result.violations == []

    def test_out_of_tolerance_fails_with_violation_detail(self):
        result = report.compare_reports(self._report(1.5), self._report(1.6), {"rmse_qrf_c": 0.005})
        assert not result.passed
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v["metric"] == "rmse_qrf_c"
        assert v["target"] == "tmax"
        assert v["zone"] == "Cfb"
        assert v["claimed"] == 1.5
        assert v["reproduced"] == 1.6

    def test_zone_present_only_in_claimed_is_skipped_not_a_violation(self):
        claimed = self._report(1.5)
        reproduced = report.build_report(
            model_version="ds-test", band_key="lag_fill", snapshot_version="v2026.08",
            sample_requested=10, rows_sampled=10, rows_paired=10,
            fidelity_check={"n": 5}, by_target={"tmax": {}, "tmin": {}},
        )
        result = report.compare_reports(claimed, reproduced, {"rmse_qrf_c": 0.005})
        assert result.passed  # nothing to compare -- not treated as a violation

    def test_max_abs_deviation_reported_even_when_passing(self):
        result = report.compare_reports(self._report(1.5), self._report(1.503), {"rmse_qrf_c": 0.01})
        assert result.max_abs_deviation["rmse_qrf_c"] == pytest.approx(0.003, abs=1e-9)


class TestBooleanMetricsCompare:
    """Regression, PR #34. The non-scalar guard added in #31 excluded bool to
    catch nested blocks, but bool is a subclass of int and compares perfectly
    well -- so identical booleans became a hard violation. Any manifest naming
    a boolean metric in its tolerance block (qrf_beats_grid,
    gated_insufficient_n, proposed_correction_beats_grid_with_margin -- none
    ceiling-restricted, all legal keys) was rejected even when it reproduced
    exactly."""

    def _report(self, value):
        return {"by_target": {"tmax": {"Cfb": {"qrf_beats_grid": value}}}}

    def test_identical_booleans_reproduce(self):
        from heatready_downscaling import report

        r = self._report(True)
        assert report.compare_reports(r, r, {"qrf_beats_grid": 0.001}).passed

    def test_mismatched_booleans_are_still_a_violation(self):
        from heatready_downscaling import report

        result = report.compare_reports(
            self._report(True), self._report(False), {"qrf_beats_grid": 0.001},
        )
        assert not result.passed
        assert len(result.violations) == 1

    def test_a_nested_block_is_still_reported_as_non_scalar(self):
        """The case the guard was actually for."""
        from heatready_downscaling import report

        r = {"by_target": {"tmax": {"Cfb": {"blk": {"all": {}}}}}}
        result = report.compare_reports(r, r, {"blk": 0.01})
        assert not result.passed
        assert "not a scalar" in (result.violations[0].get("reason") or "")


class TestBooleanTolerancesCannotBypassTheReferee:
    """PR #34 round 1, HIGH. Two wrongs in a row: #31 excluded bool and made
    identical booleans a false REJECTION; the first version of #34's fix put
    them on the numeric path and opened a false ACCEPTANCE. abs(True - False)
    is 1, no boolean key has a _TOLERANCE_MAXIMA ceiling, and validate_manifest
    accepts any positive number for an unlisted key -- so a manifest could
    declare tolerance {qrf_beats_grid: 1.5} and have a claimed True
    'reproduce' against an independently derived False."""

    def _r(self, value):
        return {"by_target": {"tmax": {"Cfb": {"qrf_beats_grid": value}}}}

    def test_a_large_tolerance_cannot_make_true_reproduce_as_false(self):
        from heatready_downscaling import report

        result = report.compare_reports(self._r(True), self._r(False), {"qrf_beats_grid": 1.5})
        assert not result.passed, "a tolerance must never launder a boolean mismatch"
        assert "must match exactly" in (result.violations[0].get("reason") or "")

    @pytest.mark.parametrize("tol", [0.001, 1.0, 1.5, 999.0])
    def test_no_tolerance_at_all_launders_a_mismatch(self, tol):
        from heatready_downscaling import report

        assert not report.compare_reports(self._r(True), self._r(False), {"qrf_beats_grid": tol}).passed

    @pytest.mark.parametrize("value", [True, False])
    def test_identical_booleans_still_reproduce(self, value):
        from heatready_downscaling import report

        assert report.compare_reports(self._r(value), self._r(value), {"qrf_beats_grid": 0.001}).passed


def test_a_flipped_boolean_records_a_real_deviation_not_zero():
    """PR #34 round 2: the mismatch branch left abs_diff None and
    max_abs_deviation at 0.0, so a flipped boolean claim was written into the
    APPEND-ONLY ledger and the referee comment as a 0.0 deviation -- a
    permanent audit record saying a claim that inverted reproduced exactly.
    The base commit recorded 1.0; the bypass fix regressed it."""
    from heatready_downscaling import report

    t = {"by_target": {"tmax": {"Cfb": {"qrf_beats_grid": True}}}}
    f = {"by_target": {"tmax": {"Cfb": {"qrf_beats_grid": False}}}}
    result = report.compare_reports(t, f, {"qrf_beats_grid": 0.001})
    assert not result.passed
    assert result.violations[0]["abs_diff"] == 1.0
    assert result.max_abs_deviation["qrf_beats_grid"] == 1.0
