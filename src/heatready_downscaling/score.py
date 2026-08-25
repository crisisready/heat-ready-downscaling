"""
score_band + fidelity_report -- the byte-identical scoring code both
contributors and the referee run. Extracted from
crisisready/heat-risk-data-api's scripts/validate_lagfill_downscaling.py
(lines ~93-117, 368-606 as of 2026-07-27, commit 57479e5). See
PROVENANCE.md.

validate_forecast_downscaling.py imported this exact function from
validate_lagfill_downscaling.py rather than cloning it -- there was never
a second implementation to reconcile, just two script entry points
reaching into each other's module (an anti-pattern for a public repo,
fixed by this extraction: both scripts now import from here instead).

Two changes made during extraction, both required by the crowdsourced
model-improvement program's design (not present in the private repo's
original code):

1. `MIN_ZONE_N`/`AUTO_ENABLE_MARGIN`/`PRACTICAL_BIAS_FLOOR_C`/
   `BIAS_CV_FOLDS`/`BIAS_CV_MIN_STATIONS` are public (no leading
   underscore) -- this program deliberately publishes these as visible
   contributor incentives (see CONTRIBUTING.md), not gatekeeping to hide.
2. `score_band` takes a REQUIRED `fold_salt` keyword-only argument, mixed
   into the station->fold assignment (`md5(f"{fold_salt}:{station_id}")`
   instead of `md5(str(station_id))`). Deterministic within one snapshot
   version, not gameable across snapshot versions by re-submitting until a
   favorable station split appears. Every real caller has a
   snapshot_version in hand (it's in the manifest a submission must pin) --
   passing the ACTUAL snapshot_version as fold_salt is not optional.
3. `score_band` takes a `heatready_downscaling.contract.ModelAdapter`
   (`adapter.predict(rows, target)`) instead of a raw joblib bundle dict
   passed to a free function (`downscaling.predict_downscaled(bundle, ...)`)
   -- see contract.py's own docstring for why.
"""

import hashlib
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# The two shapes a Rung B proposed_correction zone entry may take -- exactly
# one, never both. Public (code-review finding, PR #24: this shape was
# hand-duplicated between score_band's own runtime dispatch and
# submission.py's independent jsonschema oneOf, a real risk of the two
# drifting apart on a future third shape) so submission.py's
# _CANDIDATE_ZONE_SCHEMA can build its schema FROM these instead of
# re-listing the key names itself.
PROPOSED_CORRECTION_BIAS_KEYS = ("bias_correction_c",)
PROPOSED_CORRECTION_AFFINE_KEYS = ("scale", "offset")
# The third shape (2026-08-25): a correction that varies continuously with a
# static per-location covariate, rather than being one constant per zone.
# Valencia's real result -- corrected = base + intercept + slope*coast_dist_km
# -- is this shape, and no bucket size collapses it into a flat constant
# without losing the finding. See score_band's own docstring, section
# "covariate_linear".
PROPOSED_CORRECTION_COVARIATE_LINEAR_KEYS = ("basis", "intercept", "terms")

# Which series the correction is added to. `raw_grid` corrections are scored
# against rmse_grid over every row in the zone; `model_delta` corrections are
# scored against rmse_qrf over the rows the model actually applied to. This is
# not a cosmetic distinction: in the two zones the grounding cases live in
# (BSh lag_fill, with zero measured stations, and Cwa tmin, which fails its own
# CV gate) adapter.predict applies to few or no rows, so a model_delta-basis
# correction has nothing to attach to and only a raw_grid-basis one is
# scoreable at all.
PROPOSED_CORRECTION_BASES = ("raw_grid", "model_delta")

# Hard cap on free covariate terms. Deliberately small: Valencia's own
# 2-covariate fit was WORSE out-of-fold than the 1-covariate one (real
# overfitting, 8 training stations per fold against 3 free parameters), so the
# evidence in hand argues for one term and admits two only grudgingly. Raising
# this needs a real case that earned it out-of-fold, not a wish for
# generality.
MAX_COVARIATE_TERMS = 2

# Which covariates a covariate_linear correction may name. Two independent
# constraints, both load-bearing.
#
# STATIC per location: a promotion compiles this correction into one
# precomputed value per served polygon, which is only possible if the
# covariate does not vary by day. A day-varying covariate would force
# covariate evaluation into the serving path at inference time -- a
# serving-code change rather than a published-artifact change, and a new
# runtime dependency in the one place this system is most conservative about.
# Deliberately excluded on these grounds, though the model uses them:
# grid_daily_value_c, grid_diurnal_range_c, grid_specific_humidity_kgkg,
# nighttime_wind_ms, doy_sin/doy_cos.
#
# A REAL SNAPSHOT COLUMN: every name here must be a column in
# snapshot._pa_schema(), because score_band reads the covariate off the paired
# row and that schema is the authority on what a row's keys are.
# tests/test_score_covariate_linear.py asserts this exactly, so the two cannot
# drift.
#
# That second constraint exists because the first version of this list (PR
# #27) was built from features.FEATURE_ORDER -- the MODEL's derived feature
# names -- and four entries named keys no row has: aspect_sin, aspect_cos,
# log1p_pop_density, latitude. The snapshot's columns are aspect_deg,
# pop_density_per_km2 and lat. A contributor naming one of the four got every
# row silently excluded via the fail-closed missing-covariate path and a
# "not scored" result with no error, indistinguishable from an empty cell.
# The derived forms are the model's business; the raw columns are this
# allowlist's.
#
# aspect is dropped entirely rather than remapped to aspect_deg: it is a
# compass bearing, and a LINEAR slope on a circular variable is not
# interpretable (359 and 1 degrees are physically adjacent but maximally
# distant linearly). It never belonged on a covariate_LINEAR allowlist.
#
# coast_dist_km is absent because it is not a snapshot column yet. It is
# added here and to the snapshot schema together, so this list never again
# promises a covariate the data cannot supply.
STATIC_COVARIATE_ALLOWLIST = (
    "elevation_mean_m",
    "elevation_rel_to_gridcell_m",
    "elevation_m",
    "slope_deg",
    "canopy_height_mean_m",
    "canopy_frac_over_3m",
    "wc_built_frac",
    "wc_tree_frac",
    "wc_water_frac",
    "ghsl_urban_fraction",
    "pop_density_per_km2",
    "lst_warm_season_anomaly_c",
    "lat",
)

# Stratum thresholds are a FIXED enum, never a per-submission choice: a
# contributor free to pick the cutoff can shop for the one that flatters a
# result, which is the same gaming surface fold_salt exists to close. 30C is
# the hot-day cutoff Valencia's own analysis used.
HOT_DAY_THRESHOLD_C_CHOICES = (30.0, 35.0)
DEFAULT_HOT_DAY_THRESHOLD_C = 30.0
STRATA = ("all", "hot_day")

# Station-cluster bootstrap on the RMSE-reduction estimate: whole stations
# resampled with replacement, matching the {"bootstrap": {"n": 2000,
# "resample": "cluster"}} convention this codebase's spatial_ranking protocol
# already uses (a station's own days are not independent draws). The seed is
# derived from fold_salt so the interval is reproducible for a given snapshot
# version and not re-rollable by resubmitting.
BOOTSTRAP_N = 2000
BOOTSTRAP_CI_PCT = 95.0

MIN_ZONE_N = 30  # zones with fewer paired samples than this are reported but not gated
# A transferred (RAN-style) correction is theoretically disadvantaged relative
# to a same-source-trained one -- a razor-thin rmse_qrf_c < rmse_grid_c pass
# is exactly the noise-driven result that should not be treated as
# sufficient. Auto-enable decisions must clear this margin, not just the
# plain "<" the ERA5 band's own same-source gate uses; the plain result is
# still reported in full (qrf_beats_grid) for transparency, this only gates
# auto-enable.
AUTO_ENABLE_MARGIN = 0.03  # require >=3% relative RMSE improvement
# 0.25C is small next to a typical zone's ~1.5-3C RMSE but excludes the
# genuinely large-bias tail -- a correction that reduces variance while
# shifting the mean by up to ~2C would otherwise auto-enable with nothing
# to catch it.
PRACTICAL_BIAS_FLOOR_C = 0.25
BIAS_CV_FOLDS = 5  # station-grouped folds for validating the bias correction (see score_band)
BIAS_CV_MIN_STATIONS = 2 * BIAS_CV_FOLDS  # need enough distinct stations for folds to mean anything


def _flat_stratum_mirrors(by_stratum: dict | None) -> dict:
    """Flat, tolerance-checkable mirrors of the per-stratum metrics.

    report.compare_reports iterates FLAT keys in a zone's metric dict and
    subtracts them, so anything nested is invisible to the independent
    re-derivation promote_from_public.py performs. Mirroring the gating
    numbers as scalars is what makes the stratified evidence verifiable
    rather than merely reported. Every key here has a ceiling in
    submission._TOLERANCE_MAXIMA -- a metric with no ceiling accepts any
    tolerance a manifest declares, which would let an incorrect claim pass as
    reproduced.
    """
    out: dict = {
        "proposed_correction_ci95_lo_pct": None,
        "proposed_correction_ci95_hi_pct": None,
        "proposed_correction_hot_day_rmse_c": None,
        "proposed_correction_hot_day_bias_c": None,
        "proposed_correction_hot_day_margin_pct": None,
        "proposed_correction_hot_day_ci95_lo_pct": None,
        "proposed_correction_hot_day_ci95_hi_pct": None,
    }
    if not by_stratum:
        return out
    all_ci = (by_stratum.get("all") or {}).get("rmse_improvement_ci95_pct")
    if all_ci:
        out["proposed_correction_ci95_lo_pct"] = all_ci[0]
        out["proposed_correction_ci95_hi_pct"] = all_ci[1]
    hot = by_stratum.get("hot_day") or {}
    out["proposed_correction_hot_day_rmse_c"] = hot.get("rmse_c")
    out["proposed_correction_hot_day_bias_c"] = hot.get("bias_c")
    out["proposed_correction_hot_day_margin_pct"] = hot.get("rmse_improvement_pct")
    hot_ci = hot.get("rmse_improvement_ci95_pct")
    if hot_ci:
        out["proposed_correction_hot_day_ci95_lo_pct"] = hot_ci[0]
        out["proposed_correction_hot_day_ci95_hi_pct"] = hot_ci[1]
    return out


def classify_proposed_correction(entry: dict, zone: str | None = None) -> str:
    """Which of the three proposed_correction shapes `entry` is, or raise.

    `zone` is only used to name the offending zone in the error message --
    without it a multi-zone proposed_correction reports an ambiguous entry
    with no indication of which zone to go fix.

    Shape detection is by key presence, the same way the two original shapes
    were detected, and an entry matching MORE than one shape is a hard error
    rather than a precedence rule. submission.py's jsonschema oneOf is the
    primary guard, but score_forward_eval.py deliberately re-scores manifests
    read straight off disk without re-running the full schema check every
    cycle, so this function must not silently pick a branch for an ambiguous
    entry (code-review finding on PR #24, kept and extended to the third
    shape)."""
    matches = []
    if all(k in entry for k in PROPOSED_CORRECTION_COVARIATE_LINEAR_KEYS):
        matches.append("covariate_linear")
    if all(k in entry for k in PROPOSED_CORRECTION_AFFINE_KEYS):
        matches.append("affine")
    if all(k in entry for k in PROPOSED_CORRECTION_BIAS_KEYS):
        matches.append("bias")
    where = f"proposed_correction[{zone!r}]" if zone is not None else "proposed_correction entry"
    if len(matches) > 1:
        raise ValueError(
            f"{where} matches multiple shapes {matches!r} -- ambiguous, exactly one shape is "
            f"required; got keys {sorted(entry)!r}",
        )
    if not matches:
        raise ValueError(
            f"{where} matches none of the three shapes (covariate_linear/affine/bias) -- "
            f"got keys {sorted(entry)!r}",
        )
    return matches[0]


def validate_covariate_linear_entry(entry: dict, zone: str | None = None) -> None:
    """Raise on a covariate_linear entry this module cannot score honestly.

    Checked here rather than trusted from the schema for the same reason
    classify_proposed_correction re-checks shape: the monthly re-scoring path
    reads manifests off disk without re-validating them."""
    where = f"proposed_correction[{zone!r}]" if zone is not None else "covariate_linear entry"
    basis = entry["basis"]
    if basis not in PROPOSED_CORRECTION_BASES:
        raise ValueError(
            f"{where} basis {basis!r} is not one of {PROPOSED_CORRECTION_BASES!r}",
        )
    terms = entry["terms"]
    if not terms:
        raise ValueError(
            f"{where} has no terms -- an intercept-only correction is the "
            "existing bias_correction_c shape, submit that instead",
        )
    if len(terms) > MAX_COVARIATE_TERMS:
        raise ValueError(
            f"{where} has {len(terms)} terms, over the MAX_COVARIATE_TERMS "
            f"cap of {MAX_COVARIATE_TERMS} -- see that constant's own comment for why the cap "
            "is this small (a 2-covariate Valencia fit was measurably worse out-of-fold than "
            "the 1-covariate one)",
        )
    names = [t["covariate"] for t in terms]
    off_allowlist = [n for n in names if n not in STATIC_COVARIATE_ALLOWLIST]
    if off_allowlist:
        raise ValueError(
            f"{where} names covariate(s) {off_allowlist!r} that are not on "
            "STATIC_COVARIATE_ALLOWLIST -- see that constant's own comment: only static "
            "per-location covariates can be compiled to per-polygon values at promotion time, "
            "and a day-varying one would require covariate evaluation inside the serving path",
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"{where} names the same covariate more than once: {names!r} -- "
            "two slopes on one covariate are not separately identifiable",
        )
    valid_range = entry.get("valid_range")
    if valid_range is not None:
        if len(valid_range) != len(terms):
            raise ValueError(
                f"{where} valid_range has {len(valid_range)} entries for "
                f"{len(terms)} terms -- one [min, max] per term, in term order",
            )
        for name, bounds in zip(names, valid_range):
            if bounds is None:
                continue
            lo, hi = bounds
            if lo > hi:
                raise ValueError(
                    f"{where} valid_range for {name!r} is inverted: [{lo}, {hi}]",
                )


def _covariate_linear_effect(entry: dict, covariates: dict, n: int):
    """The per-row correction this entry adds, plus a mask of the rows it can
    honestly be scored on.

    A row is excluded (mask False) when a named covariate is missing for it, or
    when the covariate value falls outside the entry's declared valid_range.
    Both are fail-closed by design: a fit is only evidence over the covariate
    range it was fit on -- Valencia's own wider-cluster test (out to 115km and
    1,515m, versus the ~20km city cluster the fit came from) showed every
    correction degrade badly outside its fitted range -- so extrapolating a
    slope past that range and calling the result scored would overstate what
    was measured.

    Returns (effect, scoreable_mask, missing_mask, out_of_range_mask,
    absent_covariates) -- masks rather than counts, so the caller can report
    each stratum's own gap counts instead of the zone-wide ones, plus the
    names of any covariate that resolved on no row at all.
    """
    import numpy as np

    effect = np.full(n, float(entry["intercept"]))
    scoreable = np.ones(n, dtype=bool)
    absent_covariates: list[str] = []
    missing = np.zeros(n, dtype=bool)
    out_of_range = np.zeros(n, dtype=bool)
    valid_range = entry.get("valid_range") or [None] * len(entry["terms"])

    for term, bounds in zip(entry["terms"], valid_range):
        raw = covariates.get(term["covariate"])
        if raw is None or all(v is None for v in raw):
            # The covariate resolves on NO row. Almost always a name error
            # rather than genuine data sparsity, so it is logged loudly and
            # reported as its own flag: the fail-closed path alone made this
            # indistinguishable from an empty cell, which is how the PR #27
            # allowlist defect stayed invisible.
            logger.warning(
                "covariate %r resolved on 0 of %d rows -- check the name against "
                "snapshot._pa_schema()'s columns; the correction will not be scored",
                term["covariate"], n,
            )
            absent_covariates.append(term["covariate"])
            missing[:] = True
            continue
        values = np.array([np.nan if v is None else float(v) for v in raw])
        missing |= np.isnan(values)
        if bounds is not None:
            lo, hi = bounds
            out_of_range |= (~np.isnan(values)) & ((values < lo) | (values > hi))
        effect = effect + float(term["slope"]) * np.nan_to_num(values, nan=0.0)

    scoreable &= ~missing & ~out_of_range
    return effect, scoreable, missing, out_of_range, absent_covariates


def _bootstrap_reduction_ci(baseline_err, corrected_err, station_ids, *, fold_salt: str, stream: str = ""):
    """95% cluster-bootstrap CI on the RMSE reduction (as a fraction), whole
    stations resampled with replacement.

    Returns (lo, hi) or None when there aren't at least two distinct stations
    to resample -- a one-station bootstrap resamples the same cluster every
    draw and reports a spuriously tight interval, which is worse than
    reporting nothing.
    """
    import numpy as np

    unique = np.unique(station_ids)
    if len(unique) < 2:
        return None
    idx_by_station = {sid: np.flatnonzero(station_ids == sid) for sid in unique}
    # Seeded from fold_salt (the snapshot version) so the interval is
    # reproducible for a given snapshot and cannot be re-rolled by
    # resubmitting -- same reasoning as the station->fold hash.
    # `stream` (zone:target:stratum) is mixed in so each reported interval is
    # drawn from its own station-index sequence. Salted on fold_salt alone,
    # every interval in a report with the same station count reused the
    # identical resampling draw, making the all/hot_day intervals for a zone
    # (and its tmax/tmin intervals) perfectly correlated rather than
    # independent -- and nothing downstream should be able to mistake them for
    # independent evidence. Reproducibility is preserved either way, since the
    # stream label is itself deterministic. (code-review finding, PR #27)
    seed = int(hashlib.md5(f"{fold_salt}:{stream}:bootstrap".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    reductions = np.empty(BOOTSTRAP_N)
    n_stations = len(unique)
    for i in range(BOOTSTRAP_N):
        picks = rng.integers(0, n_stations, size=n_stations)
        rows = np.concatenate([idx_by_station[unique[j]] for j in picks])
        base = np.sqrt(np.mean(baseline_err[rows] ** 2))
        corr = np.sqrt(np.mean(corrected_err[rows] ** 2))
        reductions[i] = ((base - corr) / base) if base > 0 else np.nan
    reductions = reductions[~np.isnan(reductions)]
    if not len(reductions):
        return None
    tail = (100.0 - BOOTSTRAP_CI_PCT) / 2.0
    return (
        float(np.percentile(reductions, tail)),
        float(np.percentile(reductions, 100.0 - tail)),
    )


def _verdict(margin_pct, ci, n_rows: int) -> str | None:
    """pass / candidate / fail for a proposed correction in one stratum.

    Three states rather than one boolean, because the honest per-target result
    this program keeps producing is neither a pass nor a fail: Valencia's tmax
    reduction is a real positive point estimate whose CI includes zero, and
    collapsing that to False loses a genuine finding while collapsing it to
    True would ship an unproven correction. `pass` additionally requires the
    interval to exclude zero, which is a strictly stronger bar than the
    point-estimate margin alone -- and the bar the Valencia work already held
    itself to voluntarily.
    """
    if margin_pct is None or n_rows < MIN_ZONE_N:
        return None
    if margin_pct <= 0:
        return "fail"
    if margin_pct >= AUTO_ENABLE_MARGIN and ci is not None and ci[0] > 0:
        return "pass"
    return "candidate"


def score_band(
    adapter, rows: list[dict], target: str, *, fold_salt: str,
    proposed_correction: dict[str, dict] | None = None,
) -> dict:
    """Per-zone (plus overall) rmse_grid_c (raw base vs station) /
    rmse_qrf_c (corrected vs station) -- scored via adapter.predict
    directly against the SHIPPED model (no fold refitting), matching this
    program's explicit "do not retrain to score" design choice. Rows where
    adapter.predict did not apply a correction (missing covariate or the
    zone already fails its own CV gate) are still counted toward
    rmse_grid_c (the raw base baseline always exists) but excluded from
    rmse_qrf_c -- a zone can't be scored as "beats the grid" on predictions
    the production gate would never have served.

    Bias correction (MOS-style recalibration): the QRF's own delta_c can
    carry a real, per-zone systematic bias when applied to a base
    distribution (lag-fill/forecast) it was never trained on -- confirmed
    live, up to ~1.2C on some zones. bias_qrf_c is measured here and a
    correction (bias_correction_c = +bias_qrf_c) is published in the gate
    for the serving side to ADD to delta_c, recentering the correction
    rather than just gating zones out for having one.

    Sign, derived explicitly because getting this backwards would make
    production WORSE, not better: qrf_err = truth - (grid + delta_c), so
    bias_qrf_c = mean(qrf_err) = mean(truth) - mean(grid + delta_c) --
    POSITIVE bias_qrf_c means truth exceeds the corrected value on average
    (delta_c under-corrects). The fix must therefore ADD bias_qrf_c to
    delta_c (making the corrected value bigger, closer to truth), not
    subtract it -- verify with one concrete number: grid=20, truth=23,
    delta_c=2 (under-correcting by 1) -> corrected=22, qrf_err=1,
    bias_qrf_c=1; the correction needed is +1 (corrected'=20+2+1=23=truth),
    not -1.

    Critically, this correction is validated by K-FOLD CROSS-VALIDATION
    GROUPED BY STATION (BIAS_CV_FOLDS), not by measuring and evaluating the
    bias on the SAME rows -- computing a sample's own mean and subtracting
    it trivially "fixes" that exact sample (that's not validation, it's
    recentering an average), and would let genuine noise look like a real,
    generalizable correction. Each fold's bias is fit from the OTHER
    folds' stations only, then applied to that fold's own held-out rows --
    out-of-fold rmse_debiased_cv_c/bias_debiased_cv_c is the honest,
    leakage-free estimate of whether debiasing actually generalizes, and is
    what the auto-enable gate is scored against. The published
    bias_correction_c itself is refit on the FULL zone's data (standard
    practice once CV has validated the approach: CV proves it generalizes,
    the shipped correction uses every available point for its final, most
    precise estimate). Grouped by station rather than by row -- a station's
    multiple observation-days must never split across train/test, or the
    "held-out" fold would just be re-recognizing the same station's own
    already-seen bias.

    fold_salt: mixed into the station->fold hash
    (md5(f"{fold_salt}:{station_id}")) so fold assignment is deterministic
    within one snapshot version but not gameable across versions by
    re-submitting until a favorable split appears. Callers must pass the
    real snapshot_version they're scoring against -- there is no default,
    deliberately (see this module's own docstring).

    delta_scale_c (2026-07-28, plan-2026-07-28-lagfill-base-mismatch-fix.md
    section 3.1, "Option R"): generalizes bias_correction_c from a pure
    offset (e ~= d + b) to a scale+offset pair (e ~= alpha*d + b), validated
    out-of-fold via the SAME station-grouped folds used above. Indicated
    when a model is applied to a base distribution (grid product) it was
    not trained on: the learned delta can be systematically too large or
    too small for that base's actual error scale, which no additive
    constant alone can fix. alpha=1 recovers bias_correction_c's offset-only
    behavior exactly, so this is a strict superset, never a narrower
    correction. Computed only when it out-of-fold beats the offset-only fit
    (rmse_affine_cv_c < rmse_debiased_cv_c) -- see rmse_affine_cv_c/
    bias_affine_cv_c/qrf_beats_grid_with_margin_affine for the raw numbers
    either way. Whether to actually PUBLISH delta_scale_c for a zone (does
    it also clear AUTO_ENABLE_MARGIN) is gates.build_gate's decision, not
    this function's -- same division of responsibility as bias_correction_c
    already has.

    proposed_correction (2026-08-25, Rung B scoring extension --
    docs/plan-2026-08-25-crowdsourced-model-improvement-p0.md): a
    contributor-DECLARED, fixed correction to score, as opposed to the
    bias_correction_c/delta_scale_c above, which this function derives
    itself. Shape: {zone: {"bias_correction_c": float}} (offset-only) OR
    {zone: {"scale": float, "offset": float}} (affine) -- one shape per
    zone, never both. Unlike bias_correction_c/delta_scale_c, this value is
    never fit from `rows` -- it's applied as given and scored for whether
    it actually reduces error on THESE rows, which is why no station-fold
    CV loop is needed here: CV protects against overfitting a value FIT
    from the same data it's tested on, and this value was never fit from
    `rows` at all (its whole premise, per Seoul/Valencia, is that it came
    from data outside this snapshot). Applying it once across every
    qrf-applied row in the zone and applying it fold-by-fold to each fold's
    held-out rows are mathematically identical here (nothing is refit
    per-fold), so this deliberately does the simpler, single-pass
    computation. Reports proposed_correction_rmse_c/bias_c/beats_grid/
    beats_grid_with_margin (None, not 0/False, for a zone with no
    proposed_correction entry or zero qrf-applied rows -- absence is never
    silently treated as "the correction is fine") and
    proposed_vs_best_fit_gap_c (the RMS difference, over the zone's own
    observed delta_c range, between the proposed correction's effect and
    the mechanically-fit one's -- comparing raw parameters directly would
    be misleading for the affine shape, since scale and offset don't live
    on comparable scales and a right-scale-wrong-offset proposal must not
    report a near-zero gap). gates.build_gate/build_subzone_patch are
    unaffected by this parameter -- they still only ever publish the
    mechanically-derived bias_correction_c/delta_scale_c, never a
    submission's raw declared value; see GOVERNANCE.md's "a submission's
    own reported numbers are never trusted directly" language.

    covariate_linear (2026-08-25, the third proposed_correction shape): a
    correction that varies continuously with a static per-location covariate,
    rather than being one constant per zone. Shape: {"basis":
    "raw_grid"|"model_delta", "intercept": float, "terms": [{"covariate":
    str, "slope": float}], "valid_range": [[min, max], ...] | None,
    "hot_day_threshold_c": float | absent}. Valencia's real result --
    corrected = base + intercept + slope*coast_dist_km, a 31.0% hot-day tmin
    RMSE reduction with a bootstrap CI of [20.4%, 39.2%] -- is exactly this
    shape, and there is no bucket size at which it collapses into a flat
    constant without losing the finding.

    Three things about it are load-bearing rather than incidental:

    1. `basis` decides which series the correction is added to, and therefore
       which rows it can be scored on at all. A model_delta correction (the
       only kind the two original shapes could express) is meaningless where
       adapter.predict did not apply, and the zones the grounding cases live
       in are precisely the ones where it applies to few or no rows -- BSh's
       lag_fill gate has zero measured stations, and Cwa's tmin correction
       fails its own CV gate. A raw_grid correction is scoreable there. This
       is why the whole proposed_correction block sits OUTSIDE `if n_qrf:`,
       unlike its original PR #24 placement.
    2. The covariate is read off each row and the correction is evaluated per
       row. A row missing the covariate, or carrying a value outside the
       entry's declared valid_range, is EXCLUDED from scoring and counted in
       n_covariate_missing/n_covariate_out_of_range -- never silently treated
       as zero. Refusing to extrapolate past the fitted range is a real
       safety property, not pedantry: Valencia's own wider-cluster test (out
       to 115km and 1,515m against a ~20km city fit) showed every correction
       degrade badly outside the range it came from.
    3. Only STATIC per-location covariates belong here. That restriction is
       what lets a promotion compile this correction down to one precomputed
       value per polygon instead of requiring covariate evaluation inside the
       serving path -- a published-artifact change rather than a serving-code
       change. This function does not enforce staticness (it cannot see a
       covariate's semantics); the manifest schema and its CI allowlist do.

    Reporting: proposed_correction_by_stratum[stratum] for each of STRATA
    ("all", "hot_day"), each block carrying the raw-grid and basis RMSEs on
    the same rows, the improvement against each, a station-cluster bootstrap
    CI on the improvement, a pass/candidate/fail verdict, and -- for
    covariate_linear -- rmse_intercept_only_c and covariate_earns_keep, which
    ask whether the covariate term actually beats the flat constant this
    program could already express. Stratification exists because whole-year
    averaging is what hid Valencia's real result: its hot-day tmin reduction
    (31.0%) is materially larger than its whole-year one (19.4%), and its
    whole-year tmax reduction is a positive point estimate whose CI includes
    zero -- three genuinely different facts that one boolean against one
    whole-sample RMSE cannot distinguish."""
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"

    preds = adapter.predict(rows, target)

    # Every covariate any zone's proposed correction names, collected once so
    # the row loop below doesn't need to re-inspect proposed_correction per row.
    # A raw_grid-basis correction is scored over EVERY row in the zone, not just
    # the model-applied ones, so these have to be accumulated on both sides of
    # the `applied` branch.
    covariate_names: set[str] = set()
    for _entry in (proposed_correction or {}).values():
        for _term in _entry.get("terms") or ():
            covariate_names.add(_term["covariate"])

    def _new_bucket() -> dict:
        return {
            "grid_err": [], "qrf_err": [], "station_ids": [], "delta_c": [], "true_err": [],
            # All-row parallel series, for raw_grid-basis scoring and for
            # stratification. Row-aligned with grid_err.
            "all_station_ids": [], "all_obs_tmax": [],
            "all_cov": {name: [] for name in covariate_names},
            # Applied-row parallel series, row-aligned with qrf_err.
            "applied_obs_tmax": [],
            "applied_cov": {name: [] for name in covariate_names},
        }

    by_zone: dict[str, dict] = {}
    for r, p in zip(rows, preds):
        zone = r.get("climate_zone")
        if zone is None:
            continue
        b = by_zone.setdefault(zone, _new_bucket())
        truth = r[truth_col]
        grid_val = r[grid_col]
        b["grid_err"].append(truth - grid_val)
        b["all_station_ids"].append(r.get("station_id"))
        # Stratification is on OBSERVED tmax regardless of which target is
        # being scored -- "a hot day" is a property of the day, not of the
        # variable under test, and Valencia's own hot-day analysis stratified
        # tmin results on observed tmax exactly this way.
        b["all_obs_tmax"].append(r.get("station_tmax_c"))
        for name in covariate_names:
            b["all_cov"][name].append(r.get(name))
        if p["applied"]:
            b["applied_obs_tmax"].append(r.get("station_tmax_c"))
            for name in covariate_names:
                b["applied_cov"][name].append(r.get(name))
            corrected = grid_val + p["delta_c"]
            b["qrf_err"].append(truth - corrected)
            b["station_ids"].append(r.get("station_id"))
            # true_err/delta_c: the two operands of the affine fit below
            # (e ~= alpha*d + b). true_err duplicates grid_err's per-row
            # value but restricted to applied rows, kept row-aligned with
            # delta_c/station_ids -- grid_err itself can't be reused
            # directly since it also holds non-applied rows.
            b["delta_c"].append(p["delta_c"])
            b["true_err"].append(truth - grid_val)

    result: dict[str, dict] = {}
    for zone, b in by_zone.items():
        grid_err = np.array(b["grid_err"])
        rmse_grid = float(np.sqrt(np.mean(grid_err ** 2))) if len(grid_err) else None
        n_qrf = len(b["qrf_err"])
        rmse_debiased_cv = bias_debiased_cv = None
        rmse_affine_cv = bias_affine_cv = None
        delta_scale_c = None
        if n_qrf:
            qrf_err = np.array(b["qrf_err"])
            # dtype=str (not the default object dtype) -- np.unique/np.sort
            # raise TypeError comparing None to None in an all-None object
            # array. str(None) == "None" is still one consistent, sortable
            # value, so missing IDs just collapse into a single (harmless,
            # unmatched-across-zones) pseudo-station.
            station_ids = np.array([str(sid) for sid in b["station_ids"]])
            rmse_qrf = float(np.sqrt(np.mean(qrf_err ** 2)))
            bias_qrf = float(np.mean(qrf_err))
            se_bias_qrf = float(np.std(qrf_err, ddof=1) / np.sqrt(n_qrf)) if n_qrf > 1 else None

            unique_stations = np.unique(station_ids)
            if len(unique_stations) >= BIAS_CV_MIN_STATIONS:
                # Deterministic, snapshot-salted station->fold assignment
                # (stable across runs/processes, and not gameable across
                # snapshot versions -- see this function's own docstring).
                fold_of_station = {
                    sid: int(hashlib.md5(f"{fold_salt}:{sid}".encode()).hexdigest(), 16) % BIAS_CV_FOLDS
                    for sid in unique_stations
                }
                row_folds = np.array([fold_of_station[sid] for sid in station_ids])
                oof_err = np.empty(n_qrf)
                for k in range(BIAS_CV_FOLDS):
                    train_mask, test_mask = row_folds != k, row_folds == k
                    if not test_mask.any():
                        continue
                    fold_bias = float(np.mean(qrf_err[train_mask])) if train_mask.any() else 0.0
                    oof_err[test_mask] = qrf_err[test_mask] - fold_bias
                rmse_debiased_cv = float(np.sqrt(np.mean(oof_err ** 2)))
                bias_debiased_cv = float(np.mean(oof_err))  # should land near 0 -- confirms no leakage

                # Affine generalization of the offset-only correction above:
                # e ~= alpha*d + b (scale + offset applied to the model's own
                # predicted delta) instead of e ~= d + b (offset only).
                # alpha=1 recovers the offset-only correction exactly -- see
                # docs/plan-2026-07-28-lagfill-base-mismatch-fix.md section 3.1
                # ("Option R") for why this is indicated: a base-distribution
                # mismatch (model trained on one grid product, applied to a
                # different, more/less accurate one) manifests as a delta
                # that is systematically too large or too small, not merely
                # offset -- an overshoot/undershoot no single additive
                # constant can correct. Same station-grouped folds as above,
                # fit on the OTHER folds' stations only, applied out-of-fold.
                d = np.array(b["delta_c"])
                e = np.array(b["true_err"])
                oof_affine = np.empty(n_qrf)
                for k in range(BIAS_CV_FOLDS):
                    train_mask, test_mask = row_folds != k, row_folds == k
                    if not test_mask.any():
                        continue
                    X_train = np.column_stack([d[train_mask], np.ones(int(train_mask.sum()))])
                    coef, *_ = np.linalg.lstsq(X_train, e[train_mask], rcond=None)
                    alpha_k, offset_k = float(coef[0]), float(coef[1])
                    oof_affine[test_mask] = e[test_mask] - (alpha_k * d[test_mask] + offset_k)
                rmse_affine_cv = float(np.sqrt(np.mean(oof_affine ** 2)))
                bias_affine_cv = float(np.mean(oof_affine))

                # Published scale+offset (delta_scale_c) is only meaningful,
                # and only computed, when the affine fit generalizes BETTER
                # than the offset-only fit above -- publishing a scale term
                # that doesn't out-of-fold beat the simpler correction would
                # make delta_scale strictly riskier than today's behavior for
                # no benefit. gates.build_gate decides, per zone, whether
                # delta_scale actually clears the auto-enable margin (this
                # function only reports whether affine generalizes better,
                # not whether either correction is good enough to ship).
                if rmse_affine_cv < rmse_debiased_cv:
                    X_full = np.column_stack([d, np.ones(len(d))])
                    coef_full, *_ = np.linalg.lstsq(X_full, e, rcond=None)
                    delta_scale_c = {"scale": float(coef_full[0]), "offset": float(coef_full[1])}
        else:
            rmse_qrf = bias_qrf = se_bias_qrf = None

        # Rung B: score a contributor-DECLARED, fixed correction, if one was
        # given for this zone. Deliberately OUTSIDE the `if n_qrf:` block
        # above, unlike the original PR #24 placement: a raw_grid-basis
        # correction is scored against the raw grid over every row in the
        # zone and does not need the model to have applied to anything, and
        # the zones the grounding cases live in are exactly the ones where
        # the model applies to few or no rows (BSh lag_fill has zero measured
        # stations; Cwa tmin fails its own CV gate). Scoring only inside
        # `if n_qrf:` silently returned "no result" for precisely the cases
        # this mechanism exists to serve.
        #
        # No station-fold CV loop here, for the reason this function's
        # docstring already gives: nothing in this block is FIT from `rows`,
        # so a single pass over the scoreable rows is mathematically
        # identical to a fold-by-fold one. The bootstrap below is a
        # different thing -- an interval on the estimate, not a guard
        # against fitting leakage.
        proposed_entry = (proposed_correction or {}).get(zone)
        proposed_correction_kind = None
        proposed_correction_basis = None
        proposed_by_stratum: dict[str, dict] | None = None
        if proposed_entry is not None:
            proposed_correction_kind = classify_proposed_correction(proposed_entry, zone)
            hot_day_threshold_c = proposed_entry.get(
                "hot_day_threshold_c", DEFAULT_HOT_DAY_THRESHOLD_C,
            )
            if hot_day_threshold_c not in HOT_DAY_THRESHOLD_C_CHOICES:
                raise ValueError(
                    f"proposed_correction[{zone!r}] hot_day_threshold_c="
                    f"{hot_day_threshold_c!r} is not one of "
                    f"{HOT_DAY_THRESHOLD_C_CHOICES!r} -- the threshold is a fixed enum, not a "
                    "free choice, so a submission cannot shop for the cutoff that flatters its "
                    "result (see HOT_DAY_THRESHOLD_C_CHOICES' own comment)",
                )

            if proposed_correction_kind == "covariate_linear":
                validate_covariate_linear_entry(proposed_entry, zone)
                proposed_correction_basis = proposed_entry["basis"]
            else:
                # The two original shapes both act on the model's own delta,
                # which is what makes them meaningless where the model did
                # not apply. Recorded explicitly rather than left implicit so
                # every stratum block below reports which baseline it beat.
                proposed_correction_basis = "model_delta"

            if proposed_correction_basis == "raw_grid":
                baseline_err_full = np.array(b["grid_err"])
                # The raw grid IS the basis here, so the two series coincide.
                grid_err_full = baseline_err_full
                stratum_ids_full = np.array([str(sid) for sid in b["all_station_ids"]])
                obs_tmax_full = b["all_obs_tmax"]
                cov_full = b["all_cov"]
            else:
                baseline_err_full = np.array(b["qrf_err"])
                # true_err is grid_err restricted to the model-applied rows and
                # kept row-aligned with qrf_err (see the row-build loop), which
                # is what makes an honest same-rows comparison against the raw
                # grid possible for a model_delta-basis correction.
                grid_err_full = np.array(b["true_err"])
                stratum_ids_full = np.array([str(sid) for sid in b["station_ids"]])
                obs_tmax_full = b["applied_obs_tmax"]
                cov_full = b["applied_cov"]

            n_full = len(baseline_err_full)
            if proposed_correction_kind == "covariate_linear":
                (
                    effect_full, scoreable_full, missing_full, out_of_range_full,
                    absent_covariates,
                ) = _covariate_linear_effect(proposed_entry, cov_full, n_full)
                intercept_only_full = np.full(n_full, float(proposed_entry["intercept"]))
            elif proposed_correction_kind == "bias":
                effect_full = np.full(n_full, float(proposed_entry["bias_correction_c"]))
                scoreable_full = np.ones(n_full, dtype=bool)
                missing_full = out_of_range_full = np.zeros(n_full, dtype=bool)
                intercept_only_full = None
                absent_covariates = []
            else:  # affine -- the effect is on delta_c, so it varies per row
                delta_arr = np.array(b["delta_c"])
                effect_full = (
                    delta_arr * float(proposed_entry["scale"])
                    + float(proposed_entry["offset"])
                    - delta_arr
                )
                scoreable_full = np.ones(n_full, dtype=bool)
                missing_full = out_of_range_full = np.zeros(n_full, dtype=bool)
                intercept_only_full = None
                absent_covariates = []

            hot_mask_full = np.array(
                [(t is not None and t >= hot_day_threshold_c) for t in obs_tmax_full],
            ) if n_full else np.zeros(0, dtype=bool)
            # Rows where the STRATIFIER itself is missing. station_tmax_c is
            # nullable, and a tmin claim on rows with no paired tmax
            # observation would otherwise produce a hot_day block that is
            # all-None and indistinguishable from "no hot days occurred" --
            # a real difference, and the hot-day number is the one this whole
            # shape exists to surface. (code-review finding, PR #27)
            stratifier_missing_full = np.array(
                [t is None for t in obs_tmax_full],
            ) if n_full else np.zeros(0, dtype=bool)

            proposed_by_stratum = {}
            for stratum in STRATA:
                in_stratum = (
                    np.ones(n_full, dtype=bool) if stratum == "all" else hot_mask_full
                )
                mask = in_stratum & scoreable_full
                n_scored = int(mask.sum())
                block: dict = {
                    "n_scored": n_scored,
                    # Scoped to THIS stratum, not to the zone: reporting the
                    # zone-wide gap count inside the hot_day block would
                    # overstate how much of the hot-day sample was dropped.
                    "n_covariate_missing": int((in_stratum & missing_full).sum()),
                    "n_covariate_out_of_range": int((in_stratum & out_of_range_full).sum()),
                    # Only meaningful for a stratum that USES the stratifier.
                    "n_stratifier_missing": (
                        0 if stratum == "all" else int(stratifier_missing_full.sum())
                    ),
                    # Non-empty means a named covariate resolved on no row at
                    # all -- almost certainly a name error, and the one signal
                    # that separates "wrong covariate name" from "empty cell".
                    "covariates_absent_from_every_row": list(absent_covariates),
                    # Which series the correction was ADDED to. Reported
                    # explicitly because a raw_grid correction and a
                    # model_delta correction are scored over different row
                    # sets, and a reader comparing two submissions has to
                    # know which is which.
                    "basis": proposed_correction_basis,
                }
                if not n_scored:
                    # Absence is reported as absence, never as a zero or a
                    # passing boolean -- the same posture every other metric
                    # in this function takes.
                    block.update({
                        "grid_rmse_c": None, "basis_rmse_c": None, "rmse_c": None,
                        "bias_c": None, "rmse_improvement_pct": None,
                        "rmse_improvement_vs_basis_pct": None, "beats_grid": None,
                        "beats_grid_with_margin": None,
                        "rmse_improvement_ci95_pct": None, "verdict": None,
                        "rmse_intercept_only_c": None, "rmse_best_constant_c": None,
                        "covariate_earns_keep": None,
                    })
                    proposed_by_stratum[stratum] = block
                    continue

                base_err = baseline_err_full[mask]
                corr_err = base_err - effect_full[mask]
                # The headline comparison is against the RAW GRID, on the same
                # rows -- that is what "beats the grid" means everywhere else
                # in this module and what the auto-enable gate is asking. The
                # basis comparison is reported alongside it as context ("did
                # the correction improve on the thing it modifies"), not as
                # the verdict.
                #
                # Note this is deliberately stricter and more honest than the
                # flat proposed_correction_beats_grid field below, which
                # compares an applied-rows-only RMSE against the all-rows
                # rmse_grid_c. That mismatch is pre-existing behaviour with
                # live consumers, so it is left exactly as it was rather than
                # silently changed; the per-stratum block does it on matched
                # rows.
                grid_err_stratum = grid_err_full[mask]
                grid_rmse = float(np.sqrt(np.mean(grid_err_stratum ** 2)))
                base_rmse = float(np.sqrt(np.mean(base_err ** 2)))
                corr_rmse = float(np.sqrt(np.mean(corr_err ** 2)))
                margin = ((grid_rmse - corr_rmse) / grid_rmse) if grid_rmse > 0 else None
                margin_vs_basis = (
                    ((base_rmse - corr_rmse) / base_rmse) if base_rmse > 0 else None
                )
                ci = _bootstrap_reduction_ci(
                    grid_err_stratum, corr_err, stratum_ids_full[mask],
                    fold_salt=fold_salt, stream=f"{zone}:{target}:{stratum}",
                )
                # Does the covariate term actually earn its keep, or would the
                # intercept alone (i.e. a flat constant, the shape we already
                # had) have done as well? This is exactly the comparison
                # PHASE6 ran by hand for Valencia, where coast-distance beat
                # the flat constant decisively on tmin and essentially tied it
                # on tmax -- a difference the old vocabulary could not report.
                if intercept_only_full is not None:
                    io_err = base_err - intercept_only_full[mask]
                    io_rmse = float(np.sqrt(np.mean(io_err ** 2)))
                    # The verdict is against the BEST flat constant on these
                    # rows (mean(base_err), which is what the mechanical
                    # bias_correction_c fit would produce), NOT against the
                    # submitted intercept. The declared intercept comes from a
                    # JOINT fit with the slope and is not the optimal constant,
                    # so comparing against it is biased toward keeping the
                    # covariate: a proposal with a badly-placed intercept and a
                    # near-zero slope would report earns_keep=True purely
                    # because its own intercept-only variant is bad. The
                    # documented question is "would the flat constant we
                    # already had have done as well", and that constant is the
                    # best one. (code-review finding, PR #27; this is also the
                    # comparison PHASE6 actually ran, against the zone mean.)
                    best_const_rmse = float(
                        np.sqrt(np.mean((base_err - float(np.mean(base_err))) ** 2)),
                    )
                    earns_keep = corr_rmse < best_const_rmse
                else:
                    io_rmse = None
                    best_const_rmse = None
                    earns_keep = None
                block.update({
                    "grid_rmse_c": grid_rmse,
                    "basis_rmse_c": base_rmse,
                    "rmse_c": corr_rmse,
                    "bias_c": float(np.mean(corr_err)),
                    "rmse_improvement_pct": margin,
                    "rmse_improvement_vs_basis_pct": margin_vs_basis,
                    "beats_grid": corr_rmse < grid_rmse,
                    "beats_grid_with_margin": (
                        (margin >= AUTO_ENABLE_MARGIN)
                        if (margin is not None and n_scored >= MIN_ZONE_N) else None
                    ),
                    "rmse_improvement_ci95_pct": list(ci) if ci is not None else None,
                    "verdict": _verdict(margin, ci, n_scored),
                    "rmse_intercept_only_c": io_rmse,
                    "rmse_best_constant_c": best_const_rmse,
                    "covariate_earns_keep": earns_keep,
                })
                proposed_by_stratum[stratum] = block

        # The flat proposed_correction_* fields are the "all" stratum, kept at
        # the top level unchanged so every existing consumer (gates.build_gate,
        # report.compare_reports, promote_from_public.py's evidence check)
        # keeps reading exactly what it read before this change.
        _all = (proposed_by_stratum or {}).get("all") or {}
        proposed_correction_rmse_c = _all.get("rmse_c")
        proposed_correction_bias_c = _all.get("bias_c")
        # How many rows the PROPOSAL was actually scored on. For a
        # model_delta-basis proposal this equals n_qrf, so nothing changes.
        # For a raw_grid-basis one it is the count of scoreable rows in the
        # zone, which is the whole point: n_qrf is 0 in exactly the cells this
        # shape exists to unlock (BSh lag_fill, Cwa tmin), and every gate
        # below that keyed off n_qrf would have reported "insufficient data"
        # forever for a proposal with hundreds of perfectly good grid rows.
        # Caught by all three round-1 reviewers independently.
        proposed_correction_n_scored = _all.get("n_scored")
        # The n that actually governs THIS cell's gating: the proposal's own
        # sample when there is a proposal, the model-applied count otherwise
        # (Rung A, unchanged byte-for-byte).
        gating_n = (
            proposed_correction_n_scored if proposed_entry is not None else n_qrf
        ) or 0
        has_both = rmse_qrf is not None and rmse_grid is not None
        beats_grid = (rmse_qrf < rmse_grid) if has_both else None
        margin_pct = ((rmse_grid - rmse_qrf) / rmse_grid) if (has_both and rmse_grid > 0) else None
        margin_pct_debiased_cv = (
            (rmse_grid - rmse_debiased_cv) / rmse_grid
            if (rmse_debiased_cv is not None and rmse_grid and rmse_grid > 0) else None
        )
        margin_pct_affine_cv = (
            (rmse_grid - rmse_affine_cv) / rmse_grid
            if (rmse_affine_cv is not None and rmse_grid and rmse_grid > 0) else None
        )
        # Which grid baseline the FLAT fields compare against.
        #
        # For bias/affine: rmse_grid_c, the zone-wide value, exactly as before
        # this change -- these numbers have live consumers and are left
        # untouched, mismatch and all (see the per-stratum block's own comment).
        #
        # For covariate_linear: the "all" stratum's own matched-row
        # grid_rmse_c. This is not tidiness, it closes a live gaming surface
        # (code-review finding, PR #27): a covariate_linear proposal can
        # EXCLUDE rows, via a narrow valid_range or a sparsely-populated
        # covariate, so its RMSE may cover far fewer rows than rmse_grid_c
        # does. Comparing a hand-picked 40-row RMSE against a 120-row
        # rmse_grid_c would let a contributor declare a range covering only
        # favourable stations and bank a "win" in the monthly cycle. There is
        # no back-compatibility cost here because this shape is new.
        if proposed_correction_kind == "covariate_linear":
            proposed_grid_baseline = _all.get("grid_rmse_c")
        else:
            proposed_grid_baseline = rmse_grid
        proposed_correction_beats_grid = (
            (proposed_correction_rmse_c < proposed_grid_baseline)
            if (proposed_correction_rmse_c is not None and proposed_grid_baseline is not None)
            else None
        )
        proposed_correction_margin_pct = (
            (proposed_grid_baseline - proposed_correction_rmse_c) / proposed_grid_baseline
            if (
                proposed_correction_rmse_c is not None
                and proposed_grid_baseline and proposed_grid_baseline > 0
            ) else None
        )
        # Same MIN_ZONE_N/AUTO_ENABLE_MARGIN bar the mechanically-derived
        # with_margin fields use -- a contributor's declared value gets no
        # easier a bar than the maintainer's own fitted one.
        proposed_correction_beats_grid_with_margin = (
            (proposed_correction_margin_pct >= AUTO_ENABLE_MARGIN)
            if (proposed_correction_margin_pct is not None and gating_n >= MIN_ZONE_N) else None
        )
        # proposed_vs_best_fit_gap_c: how far the DECLARED value is from
        # what this function would have fit itself, evaluated on the
        # EFFECT (over the zone's own observed delta_c range), not raw
        # parameters -- comparing scale/offset directly would be
        # misleading (they don't live on comparable scales, and a
        # right-scale-wrong-offset proposal would report a near-zero gap
        # on scale alone). Referee-transparency only -- never gates
        # anything; a large gap does not fail a proposal that otherwise
        # clears the bar above, it's context for the human reading the
        # report.
        #
        # covariate_linear has no entry here on purpose: there is no
        # mechanically-fitted covariate correction for it to be compared
        # against (this function never fits a slope), so a gap number would
        # have nothing to be a gap FROM. The per-stratum
        # rmse_intercept_only_c/covariate_earns_keep pair is the honest
        # analogue for that shape -- it asks whether the covariate term beats
        # the flat constant we already had.
        #
        # Both branches below are guarded on n_qrf: proposed_correction_kind
        # is now set for every zone with a proposal (it moved out of the
        # `if n_qrf:` block so raw_grid corrections could be scored at all),
        # so without the guard a zone where the model applied to nothing would
        # take np.mean of an empty array and report NaN as a gap.
        proposed_vs_best_fit_gap_c = None
        if proposed_correction_kind == "bias" and bias_qrf is not None:
            proposed_vs_best_fit_gap_c = abs(proposed_entry["bias_correction_c"] - bias_qrf)
        elif proposed_correction_kind == "affine" and n_qrf:
            d_arr = np.array(b["delta_c"])
            fit_scale, fit_offset = (
                (delta_scale_c["scale"], delta_scale_c["offset"]) if delta_scale_c is not None
                else (1.0, bias_qrf if bias_qrf is not None else 0.0)
            )
            proposed_effect = d_arr * proposed_entry["scale"] + proposed_entry["offset"]
            fit_effect = d_arr * fit_scale + fit_offset
            proposed_vs_best_fit_gap_c = float(np.sqrt(np.mean((proposed_effect - fit_effect) ** 2)))
        # Fallback ONLY for zones with too few distinct stations to run the
        # CV above (rmse_debiased_cv is None): the correction can't be
        # validated, so it is NOT applied or trusted -- but the raw bias
        # still has to clear the same statistical-OR-practical-floor bound.
        # This must never silently degrade to "ignore bias" just because
        # there wasn't enough data to correct for it.
        bias_bounded_uncorrected = (
            (abs(bias_qrf) <= 3 * se_bias_qrf or abs(bias_qrf) <= PRACTICAL_BIAS_FLOOR_C)
            # `se_bias_qrf is not None`, NOT a truthy check: se_bias_qrf ==
            # 0.0 is a legitimate, valid value (zero-variance residuals)
            # and must still be evaluated against PRACTICAL_BIAS_FLOOR_C --
            # a bare truthy check treats 0.0 the same as "never computed"
            # and silently fails closed (None) instead of correctly
            # passing/failing the practical-floor half of the OR.
            if (bias_qrf is not None and se_bias_qrf is not None) else None
        )
        result[zone] = {
            "n_grid": len(grid_err),
            "n_qrf_applied": n_qrf,
            "rmse_grid_c": rmse_grid,
            "rmse_qrf_c": rmse_qrf,
            "bias_qrf_c": bias_qrf,
            "se_bias_qrf_c": se_bias_qrf,
            # Published for serving to ADD to delta_c -- SIGN IS +bias_qrf,
            # not negated (see this function's own docstring). Refit on the
            # full zone's data (not fold-restricted), the standard "CV
            # validates, final fit uses everything" pattern. None (not
            # 0.0) when there weren't enough distinct stations to validate
            # it via CV at all -- absence must never be silently treated
            # as "no correction needed."
            "bias_correction_c": (
                bias_qrf if (bias_qrf is not None and rmse_debiased_cv is not None) else None
            ),
            "rmse_debiased_cv_c": rmse_debiased_cv,
            "bias_debiased_cv_c": bias_debiased_cv,
            "bias_bounded_uncorrected": bias_bounded_uncorrected,
            "rmse_improvement_pct": margin_pct,
            "rmse_improvement_pct_debiased_cv": margin_pct_debiased_cv,
            "qrf_beats_grid": beats_grid if n_qrf >= MIN_ZONE_N else None,
            # Stricter bar for auto-enable (see AUTO_ENABLE_MARGIN comment)
            # -- None when there isn't even enough data for the plain gate.
            # Scored against the DEBIASED, cross-validated RMSE when CV was
            # possible. When CV wasn't possible (too few distinct
            # stations), falls back to the RAW margin check PLUS
            # bias_bounded_uncorrected -- never just the raw margin alone,
            # which would silently reopen the same gap for small-station
            # zones specifically.
            "qrf_beats_grid_with_margin": (
                (
                    margin_pct_debiased_cv >= AUTO_ENABLE_MARGIN
                    if margin_pct_debiased_cv is not None
                    else (
                        margin_pct is not None and margin_pct >= AUTO_ENABLE_MARGIN
                        and bool(bias_bounded_uncorrected)
                    )
                )
                if n_qrf >= MIN_ZONE_N else None
            ),
            "rmse_affine_cv_c": rmse_affine_cv,
            "bias_affine_cv_c": bias_affine_cv,
            "rmse_improvement_pct_affine_cv": margin_pct_affine_cv,
            # delta_scale_c generalizes bias_correction_c from an offset to a
            # (scale, offset) pair -- see docs/plan-2026-07-28-lagfill-base-
            # mismatch-fix.md section 3.1 ("Option R"). None unless the
            # affine fit out-of-fold beats the offset-only fit above (set
            # alongside delta_scale_c, never independently -- see the affine
            # block above). gates.build_gate is where "does it also clear
            # AUTO_ENABLE_MARGIN" gets decided; this field only reports that
            # the shape generalizes better, not that it's good enough to
            # publish.
            "delta_scale_c": delta_scale_c,
            # Mirrors qrf_beats_grid_with_margin's stricter auto-enable bar,
            # but scored against the affine CV instead of the debiased-only
            # CV -- lets gates.build_gate enable a zone via EITHER a
            # generalized offset-only correction OR a generalized affine
            # one, whichever validates. None when affine wasn't computed at
            # all (too few distinct stations, or n_qrf < MIN_ZONE_N).
            "qrf_beats_grid_with_margin_affine": (
                (margin_pct_affine_cv >= AUTO_ENABLE_MARGIN if margin_pct_affine_cv is not None else None)
                if n_qrf >= MIN_ZONE_N else None
            ),
            # Reflects the n that actually governs this cell (see gating_n
            # above): identical to n_qrf < MIN_ZONE_N for every Rung A cell
            # and every model_delta-basis proposal, and correct rather than
            # permanently-insufficient for a raw_grid-basis one.
            # score_forward_eval.py reads this to decide
            # win/loss/insufficient_n, so fixing it here fixes that consumer
            # too rather than needing a parallel change there.
            "gated_insufficient_n": gating_n < MIN_ZONE_N,
            "proposed_correction_n_scored": proposed_correction_n_scored,
            # Rung B (2026-08-25): scoring a contributor-DECLARED, fixed
            # correction, never the mechanically-derived one above -- see
            # this function's own docstring. None (not 0/False) whenever no
            # proposed_correction entry exists for this zone or n_qrf==0 --
            # absence must never be silently treated as "the proposal is
            # fine" (same fail-closed convention this module already uses
            # for bias_correction_c/delta_scale_c).
            "proposed_correction_kind": proposed_correction_kind,
            "proposed_correction_rmse_c": proposed_correction_rmse_c,
            "proposed_correction_bias_c": proposed_correction_bias_c,
            "proposed_correction_beats_grid": proposed_correction_beats_grid,
            "proposed_correction_margin_pct": proposed_correction_margin_pct,
            "proposed_correction_beats_grid_with_margin": proposed_correction_beats_grid_with_margin,
            "proposed_vs_best_fit_gap_c": proposed_vs_best_fit_gap_c,
            # Which series the proposal is added to ("raw_grid" or
            # "model_delta"), and the per-stratum metric blocks. Both are
            # None when no proposal was given for this zone. The flat
            # proposed_correction_* fields above are the "all" stratum and
            # are unchanged, so nothing that read this report before this
            # change has to be updated to keep working.
            "proposed_correction_basis": proposed_correction_basis,
            "proposed_correction_by_stratum": proposed_by_stratum,
            # Flat mirrors of the stratified metrics that actually gate a
            # decision. These exist because report.compare_reports -- the
            # function promote_from_public.py re-derives against -- walks FLAT
            # metric keys inside each zone's dict and does arithmetic on them;
            # a nested block is unreachable to it. Without these mirrors the
            # new evidence would be reported but never independently verified,
            # which is precisely the thing GOVERNANCE.md says never to do.
            # The "all" stratum's rmse/bias/margin already have flat fields
            # above, so only its interval is mirrored here; the hot_day
            # stratum is mirrored in full.
            **_flat_stratum_mirrors(proposed_by_stratum),
        }
    return result


def fidelity_report(fidelity_rows: list[dict]) -> dict:
    """How closely a band's near-real-time reconstruction tracks the
    already-stored ERA5-Land grid value -- a sanity check on the
    day-bucketing design choice, not a validation metric in its own right.
    A large systematic bias here (beyond ordinary NRT-vs-ERA5 disagreement,
    which IS expected) would suggest a day-boundary bug rather than genuine
    model-input drift.

    Filters out any row with a non-finite (NaN/inf) value in any of the 4
    fields before computing -- defense in depth: a station-day's stored
    grid_tmax_c/grid_tmin_c can genuinely hold a literal float('nan') for
    some rows, and np.mean() silently returns NaN for the WHOLE array if
    even one element is NaN, disabling this entire sanity-check net rather
    than just skipping the bad row."""
    if not fidelity_rows:
        return {"n": 0}
    fidelity_rows = [
        r for r in fidelity_rows
        if all(math.isfinite(r[k]) for k in ("era5_tmax", "era5_tmin", "nrt_tmax", "nrt_tmin"))
    ]
    if not fidelity_rows:
        return {"n": 0}
    tmax_diff = np.array([r["nrt_tmax"] - r["era5_tmax"] for r in fidelity_rows])
    tmin_diff = np.array([r["nrt_tmin"] - r["era5_tmin"] for r in fidelity_rows])
    return {
        "n": len(fidelity_rows),
        "tmax_diff_mean_c": float(tmax_diff.mean()),
        "tmax_diff_std_c": float(tmax_diff.std()),
        "tmin_diff_mean_c": float(tmin_diff.mean()),
        "tmin_diff_std_c": float(tmin_diff.std()),
    }
