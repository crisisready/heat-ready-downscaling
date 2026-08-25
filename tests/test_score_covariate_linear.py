"""Unit tests for score_band's covariate_linear proposed_correction shape --
the third shape (2026-08-25), the one Valencia's real coast-distance result
needs. See score.score_band's own docstring for the design, and
score.STATIC_COVARIATE_ALLOWLIST / MAX_COVARIATE_TERMS for why the two hard
limits exist.

The headline behavioural change under test here is NOT "another number":
it's that a raw_grid-basis correction is scoreable in a zone where the model
applied to nothing at all, which is the situation in both zones the
grounding cases actually live in.
"""

import pytest

from heatready_downscaling import score


class _Adapter:
    """Applies a delta on the rows whose index is in `applied_idx` (all rows
    by default), and reports not-applied for the rest -- the shape
    score_band's own `applied` branch keys off."""

    def __init__(self, deltas, applied_idx=None):
        self._deltas = deltas
        self._applied_idx = applied_idx

    def predict(self, rows, target, extra_zone_gate=None, bias_correction=None, delta_scale=None):
        out = []
        for i, d in enumerate(self._deltas):
            applied = self._applied_idx is None or i in self._applied_idx
            out.append(
                {"applied": True, "delta_c": d} if applied
                else {"applied": False, "delta_c": None},
            )
        return out


def _rows(n=60, *, zone="BSh", slope=-0.04, intercept=0.8, cov_name="coast_dist_km",
          noise=0.0, hot_from=None, cov_values=None, target="tmax"):
    """n rows across n//6 stations, where the raw grid error is EXACTLY
    intercept + slope*covariate (plus optional alternating noise), so a
    correction declaring those same coefficients drives the residual to zero.

    hot_from: if given, rows from that index on get an observed tmax above
    score.DEFAULT_HOT_DAY_THRESHOLD_C and the rest below, so the two strata
    cover disjoint row sets.
    """
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"
    grid_val = 20.0
    rows, deltas = [], []
    for i in range(n):
        cov = (i % 30) if cov_values is None else cov_values[i]
        raw_err = intercept + slope * (cov if cov is not None else 0.0)
        if noise:
            raw_err += noise if i % 2 == 0 else -noise
        row = {
            "climate_zone": zone,
            "station_id": f"STN{i % max(1, n // 6):03d}",
            grid_col: grid_val,
            truth_col: grid_val + raw_err,
            cov_name: cov,
        }
        if target != "tmax":
            # Stratification is on observed tmax regardless of target, so a
            # tmin test has to supply it explicitly.
            row["station_tmax_c"] = 35.0
        if hot_from is not None:
            row["station_tmax_c"] = 35.0 if i >= hot_from else 25.0
        rows.append(row)
        deltas.append(0.0)
    return rows, deltas


def _entry(slope=-0.04, intercept=0.8, cov="coast_dist_km", basis="raw_grid", **kw):
    e = {"basis": basis, "intercept": intercept,
         "terms": [{"covariate": cov, "slope": slope}]}
    e.update(kw)
    return e


def _score(rows, deltas, entry, *, zone="BSh", target="tmax", applied_idx=None):
    return score.score_band(
        _Adapter(deltas, applied_idx), rows, target,
        fold_salt="v2026.08", proposed_correction={zone: entry},
    )[zone]


# --- the headline fix -------------------------------------------------------

def test_raw_grid_basis_is_scored_when_the_model_applied_to_nothing():
    """The whole point. In BSh lag_fill (zero measured stations) and Cwa tmin
    (fails its own CV gate) adapter.predict applies to no rows, and the old
    proposed_correction block sat inside `if n_qrf:` -- so exactly the cells
    this mechanism exists for silently produced no result."""
    rows, deltas = _rows()
    res = _score(rows, deltas, _entry(), applied_idx=set())

    assert res["n_qrf_applied"] == 0
    assert res["rmse_qrf_c"] is None, "sanity: the model really did apply to nothing"

    block = res["proposed_correction_by_stratum"]["all"]
    assert block["n_scored"] == len(rows)
    assert block["basis"] == "raw_grid"
    assert block["rmse_c"] == pytest.approx(0.0, abs=1e-9)
    assert block["beats_grid"] is True


def test_model_delta_basis_is_not_scoreable_when_the_model_applied_to_nothing():
    """The mirror case, and the reason `basis` has to be declared rather than
    inferred: a model_delta correction genuinely has nothing to attach to
    here, and that must read as "not scored", never as a pass."""
    rows, deltas = _rows()
    res = _score(rows, deltas, _entry(basis="model_delta"), applied_idx=set())

    block = res["proposed_correction_by_stratum"]["all"]
    assert block["n_scored"] == 0
    assert block["rmse_c"] is None
    assert block["beats_grid"] is None
    assert block["verdict"] is None


# --- correctness of the arithmetic ------------------------------------------

def test_recovers_an_exact_linear_relationship():
    rows, deltas = _rows(slope=-0.04, intercept=0.8)
    res = _score(rows, deltas, _entry(slope=-0.04, intercept=0.8))
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["grid_rmse_c"] > 0.3, "the uncorrected grid really is wrong"
    assert block["rmse_c"] == pytest.approx(0.0, abs=1e-9)
    assert block["rmse_improvement_pct"] == pytest.approx(1.0, abs=1e-9)
    assert block["verdict"] == "pass"


def test_wrong_sign_slope_is_scored_as_a_failure_not_silently_dropped():
    rows, deltas = _rows(slope=-0.04, intercept=0.8)
    res = _score(rows, deltas, _entry(slope=+0.04, intercept=0.8))
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["rmse_c"] > block["grid_rmse_c"]
    assert block["beats_grid"] is False
    assert block["verdict"] == "fail"


def test_model_delta_basis_scores_against_the_grid_on_matched_rows():
    """A model_delta correction is compared against the raw grid over the
    rows the model applied to -- not against the all-rows rmse_grid_c, which
    would mix two different row sets."""
    rows, deltas = _rows(n=60)
    # Model applies on half the rows and is perfect there; the proposal then
    # has nothing left to improve on those rows.
    applied = set(range(0, 60, 2))
    deltas = [
        (rows[i]["station_tmax_c"] - rows[i]["grid_tmax_c"]) for i in range(60)
    ]
    res = _score(rows, deltas, _entry(basis="model_delta", slope=0.0, intercept=0.0),
                 applied_idx=applied)
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["n_scored"] == len(applied)
    assert block["basis"] == "model_delta"
    # basis_rmse_c is the model's own residual on those rows (perfect here),
    # grid_rmse_c is the raw grid's on the SAME rows (not perfect).
    assert block["basis_rmse_c"] == pytest.approx(0.0, abs=1e-9)
    assert block["grid_rmse_c"] > 0.0


# --- fail-closed handling of covariate gaps ---------------------------------

def test_missing_covariate_rows_are_excluded_not_treated_as_zero():
    cov = [(i % 30) for i in range(60)]
    for i in range(0, 60, 3):
        cov[i] = None
    rows, deltas = _rows(cov_values=cov)
    res = _score(rows, deltas, _entry())
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["n_covariate_missing"] == 20
    assert block["n_scored"] == 40
    # The 40 scoreable rows are still corrected exactly; a zero-filled
    # covariate would have left a large residual on the excluded ones.
    assert block["rmse_c"] == pytest.approx(0.0, abs=1e-9)


def test_rows_outside_valid_range_are_excluded():
    rows, deltas = _rows()
    res = _score(rows, deltas, _entry(valid_range=[[0.0, 9.0]]))
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["n_covariate_out_of_range"] > 0
    assert block["n_scored"] < len(rows)
    assert block["rmse_c"] == pytest.approx(0.0, abs=1e-9)


def test_a_covariate_absent_from_every_row_scores_nothing():
    rows, deltas = _rows(cov_name="elevation_mean_m")
    res = _score(rows, deltas, _entry(cov="coast_dist_km"))
    block = res["proposed_correction_by_stratum"]["all"]

    assert block["n_covariate_missing"] == len(rows)
    assert block["n_scored"] == 0
    assert block["verdict"] is None


# --- stratification ---------------------------------------------------------

def test_hot_day_stratum_scores_a_disjoint_row_set():
    rows, deltas = _rows(n=120, hot_from=60)
    res = _score(rows, deltas, _entry())
    strata = res["proposed_correction_by_stratum"]

    assert set(strata) == set(score.STRATA)
    assert strata["all"]["n_scored"] == 120
    assert strata["hot_day"]["n_scored"] == 60


def test_a_correction_that_only_helps_on_hot_days_reads_that_way():
    """Valencia's actual pattern: the effect concentrates on hot days and is
    diluted by whole-year averaging. A single whole-sample number cannot say
    this, which is why strata exist."""
    grid_val = 20.0
    rows, deltas = [], []
    for i in range(120):
        hot = i >= 60
        cov = i % 30
        # On hot days the grid error really is intercept + slope*cov; on mild
        # days it is pure alternating noise the correction cannot help with.
        raw_err = (0.8 - 0.04 * cov) if hot else (1.5 if i % 2 == 0 else -1.5)
        rows.append({
            "climate_zone": "BSh", "station_id": f"STN{i % 20:03d}",
            "grid_tmax_c": grid_val, "station_tmax_c": grid_val + raw_err,
            "coast_dist_km": cov,
        })
        rows[-1]["station_tmax_c"] = grid_val + raw_err
        rows[-1]["station_tmax_c"] = grid_val + raw_err
        deltas.append(0.0)
    # Re-stamp the stratum key: observed tmax has to straddle the threshold.
    for i, r in enumerate(rows):
        r["station_tmax_c"] = (
            score.DEFAULT_HOT_DAY_THRESHOLD_C + 5.0 if i >= 60
            else score.DEFAULT_HOT_DAY_THRESHOLD_C - 5.0
        )
        # keep the error structure intact relative to the restamped truth
        r["grid_tmax_c"] = r["station_tmax_c"] - (
            (0.8 - 0.04 * (i % 30)) if i >= 60 else (1.5 if i % 2 == 0 else -1.5)
        )

    res = score.score_band(
        _Adapter(deltas), rows, "tmax", fold_salt="v2026.08",
        proposed_correction={"BSh": _entry()},
    )["BSh"]
    strata = res["proposed_correction_by_stratum"]

    assert strata["hot_day"]["rmse_improvement_pct"] > strata["all"]["rmse_improvement_pct"]
    assert strata["hot_day"]["verdict"] == "pass"


# --- interval and verdict --------------------------------------------------

def test_bootstrap_ci_is_reported_and_reproducible():
    rows, deltas = _rows(noise=0.2)
    first = _score(rows, deltas, _entry())["proposed_correction_by_stratum"]["all"]
    second = _score(rows, deltas, _entry())["proposed_correction_by_stratum"]["all"]

    ci = first["rmse_improvement_ci95_pct"]
    assert ci is not None and len(ci) == 2 and ci[0] <= ci[1]
    assert ci == second["rmse_improvement_ci95_pct"], "same snapshot salt, same interval"


def test_ci_depends_on_the_fold_salt():
    rows, deltas = _rows(noise=0.2)
    a = score.score_band(
        _Adapter(deltas), rows, "tmax", fold_salt="v2026.08",
        proposed_correction={"BSh": _entry()},
    )["BSh"]["proposed_correction_by_stratum"]["all"]["rmse_improvement_ci95_pct"]
    b = score.score_band(
        _Adapter(deltas), rows, "tmax", fold_salt="v2026.09",
        proposed_correction={"BSh": _entry()},
    )["BSh"]["proposed_correction_by_stratum"]["all"]["rmse_improvement_ci95_pct"]
    assert a != b


def test_verdict_is_candidate_when_the_interval_includes_zero():
    """A real positive point estimate whose CI includes zero is exactly
    Valencia's whole-year tmax result -- neither a pass nor a fail, and the
    state the old single boolean could not express."""
    grid_val = 20.0
    rows, deltas = [], []
    # One station carries a large opposite-signed error, so resampling whole
    # stations swings the reduction across zero.
    for i in range(60):
        cov = i % 30
        sid = i % 10
        # STN000's error is the exact MIRROR of the pattern, so applying the
        # correction doubles its error instead of removing it. One station in
        # ten: the point estimate stays clearly positive, but resampling whole
        # stations sometimes draws that station several times, which is what
        # pushes the lower bound of the interval below zero.
        raw_err = (0.8 - 0.04 * cov) if sid != 0 else -(0.8 - 0.04 * cov)
        rows.append({
            "climate_zone": "BSh", "station_id": f"STN{sid:03d}",
            "grid_tmax_c": grid_val, "station_tmax_c": grid_val + raw_err,
            "coast_dist_km": cov,
        })
        deltas.append(0.0)
    block = _score(rows, deltas, _entry())["proposed_correction_by_stratum"]["all"]

    assert block["rmse_improvement_pct"] > 0
    ci = block["rmse_improvement_ci95_pct"]
    assert ci[0] <= 0 <= ci[1], f"expected an interval spanning zero, got {ci}"
    assert block["verdict"] == "candidate"


def test_verdict_is_none_below_min_zone_n():
    rows, deltas = _rows(n=score.MIN_ZONE_N - 1)
    block = _score(rows, deltas, _entry())["proposed_correction_by_stratum"]["all"]
    assert block["n_scored"] < score.MIN_ZONE_N
    assert block["verdict"] is None


# --- does the covariate earn its keep -------------------------------------

def test_covariate_earning_its_keep_is_reported_both_ways():
    """PHASE6 ran this comparison by hand for Valencia: coast-distance beat
    the flat constant decisively on tmin and essentially tied it on tmax. The
    old vocabulary could not report it at all."""
    rows, deltas = _rows(slope=-0.04, intercept=0.8)
    earns = _score(rows, deltas, _entry(slope=-0.04, intercept=0.8))
    block = earns["proposed_correction_by_stratum"]["all"]
    assert block["covariate_earns_keep"] is True
    assert block["rmse_intercept_only_c"] > block["rmse_c"]

    # Now a case where the real structure is a flat offset, so the covariate
    # term is dead weight.
    flat_rows, flat_deltas = _rows(slope=0.0, intercept=0.8)
    doesnt = _score(flat_rows, flat_deltas, _entry(slope=-0.04, intercept=0.8))
    flat_block = doesnt["proposed_correction_by_stratum"]["all"]
    assert flat_block["covariate_earns_keep"] is False


def test_flat_shapes_report_no_covariate_verdict():
    rows, deltas = _rows()
    res = score.score_band(
        _Adapter(deltas), rows, "tmax", fold_salt="v2026.08",
        proposed_correction={"BSh": {"bias_correction_c": 0.5}},
    )["BSh"]
    block = res["proposed_correction_by_stratum"]["all"]
    assert res["proposed_correction_kind"] == "bias"
    assert res["proposed_correction_basis"] == "model_delta"
    assert block["rmse_intercept_only_c"] is None
    assert block["covariate_earns_keep"] is None


# --- rejected entries ------------------------------------------------------

@pytest.mark.parametrize("entry, match", [
    (_entry(basis="nonsense"), "not one of"),
    ({"basis": "raw_grid", "intercept": 0.0, "terms": []}, "no terms"),
    ({"basis": "raw_grid", "intercept": 0.0, "terms": [
        {"covariate": "coast_dist_km", "slope": 1.0},
        {"covariate": "elevation_mean_m", "slope": 1.0},
        {"covariate": "slope_deg", "slope": 1.0},
    ]}, "MAX_COVARIATE_TERMS"),
    ({"basis": "raw_grid", "intercept": 0.0, "terms": [
        {"covariate": "coast_dist_km", "slope": 1.0},
        {"covariate": "coast_dist_km", "slope": 2.0},
    ]}, "more than once"),
    (_entry(cov="grid_diurnal_range_c"), "STATIC_COVARIATE_ALLOWLIST"),
    (_entry(cov="nighttime_wind_ms"), "STATIC_COVARIATE_ALLOWLIST"),
    (_entry(valid_range=[[0.0, 1.0], [0.0, 1.0]]), "one \\[min, max\\] per term"),
    (_entry(valid_range=[[9.0, 1.0]]), "inverted"),
])
def test_invalid_covariate_linear_entries_raise(entry, match):
    rows, deltas = _rows()
    with pytest.raises(ValueError, match=match):
        _score(rows, deltas, entry)


def test_hot_day_threshold_must_come_from_the_fixed_enum():
    rows, deltas = _rows()
    with pytest.raises(ValueError, match="hot_day_threshold_c"):
        _score(rows, deltas, _entry(hot_day_threshold_c=28.5))


def test_an_entry_matching_two_shapes_is_ambiguous():
    rows, deltas = _rows()
    with pytest.raises(ValueError, match="multiple shapes"):
        _score(rows, deltas, {
            "basis": "raw_grid", "intercept": 0.0,
            "terms": [{"covariate": "coast_dist_km", "slope": 1.0}],
            "bias_correction_c": 0.5,
        })


def test_an_entry_matching_no_shape_is_rejected():
    rows, deltas = _rows()
    with pytest.raises(ValueError, match="matches none"):
        _score(rows, deltas, {"something_else": 1.0})
