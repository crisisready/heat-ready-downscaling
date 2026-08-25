"""Unit tests for heatready_downscaling.score -- ported from
crisisready/heat-risk-data-api's tests/test_validate_lagfill_downscaling.py,
adapted to score_band's new (adapter, fold_salt) signature. See
PROVENANCE.md and score.py's own module docstring for what changed during
extraction."""

import math

import pytest

from heatready_downscaling import score


class _FakeAdapter:
    """Stands in for a real ModelAdapter -- returns pre-computed deltas in
    row order, ignoring extra_zone_gate/bias_correction (score_band never
    passes those; they're QRFModelAdapter.predict's own concern, already
    covered by test_contract.py)."""

    def __init__(self, deltas: list[float]):
        self._deltas = deltas

    def predict(self, rows, target, extra_zone_gate=None, bias_correction=None):
        return [{"applied": True, "delta_c": d} for d in self._deltas]


def _bias_test_rows_and_adapter(bias_c: float, n: int = 40, target: str = "tmax"):
    """n rows in one zone, all deterministic: raw grid error alternates
    +-5.0 (rmse_grid_c == 5.0 exactly, mean 0), the adapter's delta_c is
    chosen per-row so the CORRECTED residual is bias_c +- a tiny 0.01
    alternation (rmse_qrf_c ~= |bias_c|, se_bias_qrf_c ~= 0.01/sqrt(n), tiny
    regardless of bias_c) -- isolates the practical-floor bias check from
    the statistical (3*SE) one."""
    grid_val = 20.0
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    rows, deltas = [], []
    for i in range(n):
        raw_err = 5.0 if i % 2 == 0 else -5.0
        target_residual = bias_c + (0.01 if i % 2 == 0 else -0.01)
        truth = grid_val + raw_err
        delta_c = raw_err - target_residual
        rows.append({"climate_zone": "Cfa", grid_col: grid_val, truth_col: truth})
        deltas.append(delta_c)
    return rows, _FakeAdapter(deltas)


def _cv_test_rows_and_adapter(
    bias_c: float, n_stations: int = 12, rows_per_station: int = 20,
    target: str = "tmax", per_station_offsets: list[float] | None = None,
):
    """n_stations distinct, real station_id values x rows_per_station rows
    each -- enough distinct stations (>= score.BIAS_CV_MIN_STATIONS) for
    score_band's cross-validated debiasing path to engage.
    per_station_offsets, if given, is a length-n_stations list of PER-
    STATION idiosyncratic deviations added on top of bias_c -- the key
    leakage-guard case (a real leave-one-station-out CV must not be able
    to predict station A's own idiosyncratic offset from stations B..Z,
    which don't share it)."""
    grid_val = 20.0
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    offsets = per_station_offsets or [0.0] * n_stations
    rows, deltas = [], []
    for s in range(n_stations):
        sid = f"STN{s:03d}"
        for i in range(rows_per_station):
            raw_err = 5.0 if i % 2 == 0 else -5.0
            target_residual = bias_c + offsets[s] + (0.01 if i % 2 == 0 else -0.01)
            truth = grid_val + raw_err
            delta_c = raw_err - target_residual
            rows.append({"climate_zone": "Cfa", grid_col: grid_val, truth_col: truth, "station_id": sid})
            deltas.append(delta_c)
    return rows, _FakeAdapter(deltas)


class TestScoreBandBiasGate:
    """qrf_beats_grid_with_margin must fail a large, uncorrected systematic
    bias even when RMSE alone looks excellent -- these fixtures never set a
    real, distinct station_id per row, so score_band's CV path never has
    enough distinct stations to engage, exercising the
    bias_bounded_uncorrected FALLBACK specifically. See
    TestScoreBandBiasCorrectionCv below for the CV-engaged path."""

    def test_small_bias_still_passes_margin_gate(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.1)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert m["bias_qrf_c"] == pytest.approx(0.1, abs=0.02)
        assert m["bias_bounded_uncorrected"] is True
        assert m["qrf_beats_grid_with_margin"] is True

    def test_large_bias_fails_margin_gate_even_with_excellent_rmse_improvement(self):
        """rmse_qrf (~0.8C) is dramatically better than rmse_grid (5.0C, an
        84%+ improvement, far past the 3% AUTO_ENABLE_MARGIN bar) -- an
        RMSE-only gate would auto-enable the zone despite an 0.8C
        systematic bias; the bias check must catch it."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.8)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert m["bias_qrf_c"] == pytest.approx(0.8, abs=0.02)
        assert m["rmse_improvement_pct"] > 0.8  # the RMSE side alone looks great
        assert m["bias_bounded_uncorrected"] is False
        assert m["qrf_beats_grid_with_margin"] is False

    def test_qrf_beats_grid_plain_gate_unaffected_by_bias_check(self):
        """The PLAIN gate (qrf_beats_grid, used for reporting/diagnostics,
        not auto-enable) must stay pure-RMSE -- only the auto-enable gate
        has a bias requirement."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.8)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        assert result["Cfa"]["qrf_beats_grid"] is True

    def test_bias_exactly_at_practical_floor_passes(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.25)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        assert result["Cfa"]["bias_bounded_uncorrected"] is True

    def test_bias_just_past_practical_floor_fails(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.30)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        assert result["Cfa"]["bias_bounded_uncorrected"] is False

    def test_too_few_distinct_stations_skips_cv_entirely(self):
        """This fixture's rows never carry a station_id at all -- confirms
        the CV fields stay None and bias_correction_c is correctly
        withheld (never published unvalidated)."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert m["rmse_debiased_cv_c"] is None
        assert m["bias_debiased_cv_c"] is None
        assert m["bias_correction_c"] is None


class TestScoreBandBiasCorrectionCv:
    """The cross-validated debiasing path itself -- in-sample debiasing
    alone (fitting AND evaluating a correction on the same rows) is not
    valid: it trivially "fixes" whatever sample it's given, which is not
    evidence the correction generalizes."""

    def test_cv_engages_with_enough_distinct_stations(self):
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.8, n_stations=12)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert m["rmse_debiased_cv_c"] is not None
        assert m["bias_correction_c"] is not None

    def test_cv_recovers_a_stable_population_bias_shared_by_every_station(self):
        """Every station carries the IDENTICAL 0.8C bias -- a real,
        population-level effect. Leave-one-station-out CV must recover it:
        bias_debiased_cv_c near 0 (confirms no leakage), and the debiased
        margin clears the auto-enable bar."""
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.8, n_stations=15, rows_per_station=30)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert m["bias_qrf_c"] == pytest.approx(0.8, abs=0.02)
        # Sign: bias_correction_c is ADDED to delta_c at serving time -- it
        # must be +bias_qrf_c, not negated (a wrong sign would DOUBLE the
        # error instead of removing it -- see
        # test_bias_correction_c_has_the_sign_that_actually_reduces_error
        # below for the end-to-end proof).
        assert m["bias_correction_c"] == pytest.approx(0.8, abs=0.02)
        assert abs(m["bias_debiased_cv_c"]) < 0.05  # generalized cleanly -- no leakage
        assert m["rmse_debiased_cv_c"] < m["rmse_qrf_c"]  # debiasing genuinely helped out-of-fold too
        assert m["qrf_beats_grid_with_margin"] is True

    def test_bias_correction_c_has_the_sign_that_actually_reduces_error(self):
        """THE critical regression test: proves the sign end-to-end by
        actually APPLYING bias_correction_c the way
        contract.QRFModelAdapter.predict does (delta_c + correction) and
        confirming the corrected error is smaller than the uncorrected one
        -- not just asserting a specific numeric sign convention (a test
        that only checks a hardcoded sign can stay green while the
        convention itself is backwards)."""
        bias_c = 0.8
        rows, adapter = _cv_test_rows_and_adapter(bias_c=bias_c, n_stations=15, rows_per_station=30)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        correction = m["bias_correction_c"]

        uncorrected_abs_bias = abs(m["bias_qrf_c"])
        # Applying the correction shifts every corrected prediction by
        # +correction, which shifts bias_qrf_c by -correction (the same
        # additive relationship contract.QRFModelAdapter.predict/score_band
        # both use).
        corrected_bias = m["bias_qrf_c"] - correction
        assert abs(corrected_bias) < uncorrected_abs_bias, (
            f"applying bias_correction_c={correction} to a raw bias of {m['bias_qrf_c']} "
            f"must REDUCE the bias (got {corrected_bias}) -- a wrong-signed correction doubles it instead"
        )
        assert abs(corrected_bias) < 0.05  # should land very close to exactly zero for a stable population bias

    def test_per_station_idiosyncratic_deviation_is_not_overfit_by_cv(self):
        """THE key leakage-guard case: bias_c=0 (no real population bias),
        but each station carries its OWN large, idiosyncratic +-2C offset
        (their mean across stations is exactly 0 by construction). A valid
        leave-one-station-out CV must NOT be able to predict it --
        rmse_debiased_cv_c should stay close to rmse_qrf_c (no fake
        improvement)."""
        offsets = [2.0 if s % 2 == 0 else -2.0 for s in range(12)]
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.0, n_stations=12, per_station_offsets=offsets)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        m = result["Cfa"]
        assert abs(m["bias_qrf_c"]) < 0.1
        assert m["rmse_debiased_cv_c"] == pytest.approx(m["rmse_qrf_c"], rel=0.15)

    def test_bias_correction_c_refit_on_full_data_not_fold_restricted(self):
        """The PUBLISHED correction (bias_correction_c) is the full-sample
        estimate (+bias_qrf_c), not an average of the per-fold corrections
        -- CV's job is only to VALIDATE the approach generalizes, not to
        weaken the final shipped correction by holding out data
        unnecessarily."""
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.5, n_stations=12, rows_per_station=25)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        assert result["Cfa"]["bias_correction_c"] == pytest.approx(0.5, abs=0.02)


class TestScoreBandProposedCorrection:
    """Rung B (2026-08-25): a contributor-DECLARED, fixed correction, never
    fit from `rows` -- see score_band's own docstring for why no CV loop is
    needed here."""

    def test_zone_with_no_entry_gets_all_none(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        result = score.score_band(
            adapter, rows, "tmax", fold_salt="v-test",
            proposed_correction={"SomeOtherZone": {"bias_correction_c": 0.5}},
        )
        m = result["Cfa"]
        assert m["proposed_correction_kind"] is None
        assert m["proposed_correction_rmse_c"] is None
        assert m["proposed_correction_bias_c"] is None
        assert m["proposed_correction_beats_grid"] is None
        assert m["proposed_correction_beats_grid_with_margin"] is None
        assert m["proposed_vs_best_fit_gap_c"] is None

    def test_no_proposed_correction_at_all_gets_all_none(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        result = score.score_band(adapter, rows, "tmax", fold_salt="v-test")
        assert result["Cfa"]["proposed_correction_kind"] is None

    def test_accurate_declared_bias_beats_grid_with_near_zero_gap(self):
        """bias_c=0.5 fixture leaves qrf_err ~= 0.5 +- 0.01 (see
        _bias_test_rows_and_adapter's own docstring) -- bias_qrf is
        therefore ~0.5 exactly (the alternation cancels in the mean).
        Declaring bias_correction_c=0.5 should nearly zero out the
        residual and match bias_qrf almost exactly."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        result = score.score_band(
            adapter, rows, "tmax", fold_salt="v-test",
            proposed_correction={"Cfa": {"bias_correction_c": 0.5}},
        )
        m = result["Cfa"]
        assert m["proposed_correction_kind"] == "bias"
        assert m["proposed_correction_rmse_c"] == pytest.approx(0.01, abs=0.002)
        assert m["proposed_correction_bias_c"] == pytest.approx(0.0, abs=0.002)
        assert m["proposed_correction_beats_grid"] is True
        assert m["proposed_correction_beats_grid_with_margin"] is True
        assert m["proposed_vs_best_fit_gap_c"] == pytest.approx(0.0, abs=0.002)

    def test_wildly_wrong_declared_bias_fails_to_beat_grid(self):
        """Overcorrecting by 10C on top of the real ~0.5C bias should end
        up WORSE than the raw 5.0C grid RMSE, not better."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        result = score.score_band(
            adapter, rows, "tmax", fold_salt="v-test",
            proposed_correction={"Cfa": {"bias_correction_c": -10.0}},
        )
        m = result["Cfa"]
        assert m["proposed_correction_rmse_c"] == pytest.approx(10.5, abs=0.02)
        assert m["proposed_correction_beats_grid"] is False
        assert m["proposed_correction_beats_grid_with_margin"] is False
        # the gap from what score_band would itself have fit (~0.5) must be large
        assert m["proposed_vs_best_fit_gap_c"] == pytest.approx(10.5, abs=0.02)

    def test_affine_scale_actually_applied_not_ignored(self):
        """bias_c=0.0 fixture makes delta_c == true_err == raw_err exactly
        (+-5.0). Proposing scale=0.5, offset=0.0 should halve the residual
        to +-2.5 -- if the scale term were silently ignored (e.g. a bug
        that only ever read 'offset'), this would come out as 5.0, not
        2.5."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.0)
        result = score.score_band(
            adapter, rows, "tmax", fold_salt="v-test",
            proposed_correction={"Cfa": {"scale": 0.5, "offset": 0.0}},
        )
        m = result["Cfa"]
        assert m["proposed_correction_kind"] == "affine"
        assert m["proposed_correction_rmse_c"] == pytest.approx(2.5, abs=0.01)

    def test_malformed_entry_raises(self):
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        with pytest.raises(ValueError, match="proposed_correction"):
            score.score_band(
                adapter, rows, "tmax", fold_salt="v-test",
                proposed_correction={"Cfa": {"bogus_key": 1.0}},
            )

    def test_entry_with_both_shapes_raises(self):
        """Defense in depth (code-review finding, PR #24): submission.py's
        jsonschema oneOf is the primary guard, but score_forward_eval.py
        reads a merged manifest straight off disk every monthly cycle
        without re-validating it -- score_band itself must never silently
        pick a branch (e.g. "affine wins") for an ambiguous entry."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        # The message generalized from "has BOTH shapes" to "matches multiple
        # shapes" when the third (covariate_linear) shape landed -- naming two
        # would now be wrong. The guarantee under test is unchanged: an
        # ambiguous entry raises rather than picking a branch, and the message
        # still names the offending zone.
        with pytest.raises(ValueError, match="matches multiple shapes"):
            score.score_band(
                adapter, rows, "tmax", fold_salt="v-test",
                proposed_correction={"Cfa": {"bias_correction_c": 0.8, "scale": 0.9, "offset": 0.2}},
            )

    def test_result_is_identical_across_fold_salts(self):
        """The whole point of not needing a CV loop here: nothing is fit
        from `rows`, so the declared value's score can never depend on how
        rows are split into folds."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5)
        proposed = {"Cfa": {"bias_correction_c": 0.3}}
        results = [
            score.score_band(adapter, rows, "tmax", fold_salt=salt, proposed_correction=proposed)["Cfa"]
            for salt in ("v2026.08", "v2026.09", "v2026.10")
        ]
        rmses = {r["proposed_correction_rmse_c"] for r in results}
        assert len(rmses) == 1

    def test_insufficient_n_still_reports_but_never_auto_enables(self):
        """Never silently hide a thin-zone result -- rmse/bias/beats_grid
        must still be computed below MIN_ZONE_N, but the WITH_MARGIN
        (auto-enable-shaped) verdict must stay None, mirroring
        qrf_beats_grid_with_margin's own gated_insufficient_n discipline."""
        rows, adapter = _bias_test_rows_and_adapter(bias_c=0.5, n=10)  # n_qrf=10 < MIN_ZONE_N=30
        result = score.score_band(
            adapter, rows, "tmax", fold_salt="v-test",
            proposed_correction={"Cfa": {"bias_correction_c": 0.5}},
        )
        m = result["Cfa"]
        assert m["gated_insufficient_n"] is True
        assert m["proposed_correction_rmse_c"] is not None
        assert m["proposed_correction_beats_grid"] is not None  # not gated by MIN_ZONE_N
        assert m["proposed_correction_beats_grid_with_margin"] is None  # gated


class TestScoreBandFoldSalt:
    """New behavior (not in the private repo's original score_band): fold
    assignment is salted by snapshot_version, so a station->fold split
    can't be gamed across snapshot versions by re-submitting."""

    def test_fold_salt_is_required(self):
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.5, n_stations=12)
        with pytest.raises(TypeError):
            score.score_band(adapter, rows, "tmax")  # no fold_salt kwarg -- must raise, not default

    def test_different_fold_salts_can_change_fold_assignment(self):
        """Not a strict inequality test (a specific salt pair COULD hash to
        the same assignment by chance) -- just confirms fold_salt actually
        participates in the hash rather than being silently ignored, by
        checking at least one of several salts differs from the others'
        debiased result for a construction where fold membership matters
        (idiosyncratic per-station offsets, so a different fold split
        genuinely changes which offsets land in-fold vs out-of-fold)."""
        offsets = [2.0 if s % 2 == 0 else -2.0 for s in range(12)]
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.0, n_stations=12, per_station_offsets=offsets)
        results = [
            score.score_band(adapter, rows, "tmax", fold_salt=salt)["Cfa"]["bias_debiased_cv_c"]
            for salt in ("v2026.08", "v2026.09", "v2026.10", "v2026.11", "v2026.12")
        ]
        assert len(set(results)) > 1, "fold_salt should change fold membership across snapshot versions"

    def test_same_fold_salt_is_deterministic(self):
        rows, adapter = _cv_test_rows_and_adapter(bias_c=0.5, n_stations=12)
        r1 = score.score_band(adapter, rows, "tmax", fold_salt="v2026.08")
        r2 = score.score_band(adapter, rows, "tmax", fold_salt="v2026.08")
        assert r1 == r2


class TestFidelityReport:
    def _row(self, era5_tmax, era5_tmin, nrt_tmax, nrt_tmin):
        return {"era5_tmax": era5_tmax, "era5_tmin": era5_tmin, "nrt_tmax": nrt_tmax, "nrt_tmin": nrt_tmin}

    def test_clean_rows_produce_real_finite_stats(self):
        rows = [
            self._row(20.0, 10.0, 20.5, 10.2),
            self._row(22.0, 12.0, 21.5, 12.3),
        ]
        report = score.fidelity_report(rows)
        assert report["n"] == 2
        assert math.isfinite(report["tmax_diff_mean_c"])
        assert math.isfinite(report["tmin_diff_mean_c"])

    def test_single_nan_era5_value_does_not_poison_the_whole_result(self):
        rows = [
            self._row(20.0, 10.0, 20.5, 10.2),
            self._row(float("nan"), float("nan"), 21.5, 12.3),
            self._row(22.0, 12.0, 21.5, 12.3),
        ]
        report = score.fidelity_report(rows)
        assert report["n"] == 2  # the NaN row is dropped, not counted
        assert math.isfinite(report["tmax_diff_mean_c"])

    def test_all_nan_rows_returns_n_zero_not_a_nan_report(self):
        rows = [self._row(float("nan"), float("nan"), 20.0, 10.0)]
        assert score.fidelity_report(rows) == {"n": 0}

    def test_empty_input_returns_n_zero(self):
        assert score.fidelity_report([]) == {"n": 0}
