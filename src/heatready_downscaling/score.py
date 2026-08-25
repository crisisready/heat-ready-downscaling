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
import math

import numpy as np

# The two shapes a Rung B proposed_correction zone entry may take -- exactly
# one, never both. Public (code-review finding, PR #24: this shape was
# hand-duplicated between score_band's own runtime dispatch and
# submission.py's independent jsonschema oneOf, a real risk of the two
# drifting apart on a future third shape) so submission.py's
# _CANDIDATE_ZONE_SCHEMA can build its schema FROM these instead of
# re-listing the key names itself.
PROPOSED_CORRECTION_BIAS_KEYS = ("bias_correction_c",)
PROPOSED_CORRECTION_AFFINE_KEYS = ("scale", "offset")

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
    own reported numbers are never trusted directly" language."""
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"

    preds = adapter.predict(rows, target)

    by_zone: dict[str, dict] = {}
    for r, p in zip(rows, preds):
        zone = r.get("climate_zone")
        if zone is None:
            continue
        b = by_zone.setdefault(
            zone, {"grid_err": [], "qrf_err": [], "station_ids": [], "delta_c": [], "true_err": []},
        )
        truth = r[truth_col]
        grid_val = r[grid_col]
        b["grid_err"].append(truth - grid_val)
        if p["applied"]:
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

            # Rung B: score a contributor-DECLARED, fixed correction, if one
            # was given for this zone -- see this function's own docstring
            # for why no station-fold CV loop is needed (nothing here is fit
            # from `rows`, so a single-pass computation over every
            # qrf-applied row is mathematically identical to a fold-by-fold
            # one). Kept independent of BIAS_CV_MIN_STATIONS -- that gate
            # exists to make FITTING honest, and nothing here is fit.
            proposed_entry = (proposed_correction or {}).get(zone)
            if proposed_entry is not None:
                delta_c_arr = np.array(b["delta_c"])
                true_err_arr = np.array(b["true_err"])
                is_affine = all(k in proposed_entry for k in PROPOSED_CORRECTION_AFFINE_KEYS)
                is_bias = all(k in proposed_entry for k in PROPOSED_CORRECTION_BIAS_KEYS)
                # Defense in depth (code-review finding, PR #24): submission.py's
                # own jsonschema oneOf is the primary guard against a single
                # entry carrying BOTH shapes, but a caller reading a manifest
                # straight off disk without re-validating it (as
                # score_forward_eval.py's monthly re-scoring does, by design --
                # re-running the full jsonschema check every cycle would be
                # redundant with the one-time check at merge time) must not
                # silently pick a branch here. An ambiguous entry is a hard
                # error, never "affine wins."
                if is_affine and is_bias:
                    raise ValueError(
                        f"proposed_correction[{zone!r}] has BOTH ('scale','offset') and "
                        f"'bias_correction_c' -- ambiguous, exactly one shape is required",
                    )
                if is_affine:
                    proposed_correction_kind = "affine"
                    proposed_new_err = true_err_arr - (
                        delta_c_arr * proposed_entry["scale"] + proposed_entry["offset"]
                    )
                elif is_bias:
                    proposed_correction_kind = "bias"
                    # qrf_err already IS true_err - delta_c (see the row-build
                    # loop above); adding a constant offset to delta_c
                    # subtracts that same constant from the residual.
                    proposed_new_err = qrf_err - proposed_entry["bias_correction_c"]
                else:
                    raise ValueError(
                        f"proposed_correction[{zone!r}] has neither ('scale','offset') nor "
                        f"'bias_correction_c' -- got keys {sorted(proposed_entry)!r}",
                    )
                proposed_correction_rmse_c = float(np.sqrt(np.mean(proposed_new_err ** 2)))
                proposed_correction_bias_c = float(np.mean(proposed_new_err))
            else:
                proposed_correction_kind = None
                proposed_correction_rmse_c = None
                proposed_correction_bias_c = None

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
            proposed_correction_kind = None
            proposed_correction_rmse_c = None
            proposed_correction_bias_c = None
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
        proposed_correction_beats_grid = (
            (proposed_correction_rmse_c < rmse_grid)
            if (proposed_correction_rmse_c is not None and rmse_grid is not None) else None
        )
        proposed_correction_margin_pct = (
            (rmse_grid - proposed_correction_rmse_c) / rmse_grid
            if (proposed_correction_rmse_c is not None and rmse_grid and rmse_grid > 0) else None
        )
        # Same MIN_ZONE_N/AUTO_ENABLE_MARGIN bar the mechanically-derived
        # with_margin fields use -- a contributor's declared value gets no
        # easier a bar than the maintainer's own fitted one.
        proposed_correction_beats_grid_with_margin = (
            (proposed_correction_margin_pct >= AUTO_ENABLE_MARGIN)
            if (proposed_correction_margin_pct is not None and n_qrf >= MIN_ZONE_N) else None
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
        proposed_vs_best_fit_gap_c = None
        if proposed_correction_kind == "bias" and bias_qrf is not None:
            proposed_vs_best_fit_gap_c = abs(proposed_entry["bias_correction_c"] - bias_qrf)
        elif proposed_correction_kind == "affine":
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
            "gated_insufficient_n": n_qrf < MIN_ZONE_N,
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
