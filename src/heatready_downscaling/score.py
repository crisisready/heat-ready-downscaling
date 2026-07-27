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


def score_band(adapter, rows: list[dict], target: str, *, fold_salt: str) -> dict:
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
    deliberately (see this module's own docstring)."""
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"

    preds = adapter.predict(rows, target)

    by_zone: dict[str, dict] = {}
    for r, p in zip(rows, preds):
        zone = r.get("climate_zone")
        if zone is None:
            continue
        b = by_zone.setdefault(zone, {"grid_err": [], "qrf_err": [], "station_ids": []})
        truth = r[truth_col]
        grid_val = r[grid_col]
        b["grid_err"].append(truth - grid_val)
        if p["applied"]:
            corrected = grid_val + p["delta_c"]
            b["qrf_err"].append(truth - corrected)
            b["station_ids"].append(r.get("station_id"))

    result: dict[str, dict] = {}
    for zone, b in by_zone.items():
        grid_err = np.array(b["grid_err"])
        rmse_grid = float(np.sqrt(np.mean(grid_err ** 2))) if len(grid_err) else None
        n_qrf = len(b["qrf_err"])
        rmse_debiased_cv = bias_debiased_cv = None
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
        else:
            rmse_qrf = bias_qrf = se_bias_qrf = None
        has_both = rmse_qrf is not None and rmse_grid is not None
        beats_grid = (rmse_qrf < rmse_grid) if has_both else None
        margin_pct = ((rmse_grid - rmse_qrf) / rmse_grid) if (has_both and rmse_grid > 0) else None
        margin_pct_debiased_cv = (
            (rmse_grid - rmse_debiased_cv) / rmse_grid
            if (rmse_debiased_cv is not None and rmse_grid and rmse_grid > 0) else None
        )
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
            "gated_insufficient_n": n_qrf < MIN_ZONE_N,
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
