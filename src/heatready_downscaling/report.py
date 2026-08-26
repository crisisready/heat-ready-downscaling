"""
The report envelope every submission's claimed_report.json and every
referee re-derivation must share exactly, plus a tolerance comparator.
Extracted/adapted from crisisready/heat-risk-data-api's
scripts/validate_lagfill_downscaling.py's `_build_report` (lines ~753-783
as of 2026-07-27, commit 57479e5). See PROVENANCE.md.

Canonicalization decision (2026-07-27, see this repo's Phase 1 design
consult): the private repo's two validation scripts did NOT actually agree
on a report shape -- validate_lagfill_downscaling.py used
"rows_nrt_paired", validate_forecast_downscaling.py used "rows_lead_paired"
plus its own "lead_days" field. build_report below canonicalizes on
"rows_paired" (band-agnostic) and carries any band-specific field (like
lead_days) through the `extra` kwarg, merged into the envelope's top level
-- so the CORE envelope (report_schema_version, generated_by, snapshot_version,
band_key, model_version, sample_requested, rows_sampled, rows_paired,
fidelity_check, by_target) is identical across every band, and only truly
band-specific extras vary.
"""

REPORT_SCHEMA_VERSION = 1

_SCORE_BAND_METRICS_SCHEMA: dict = {
    "type": "object",
    "required": [
        "n_grid", "n_qrf_applied", "rmse_grid_c", "rmse_qrf_c",
        "qrf_beats_grid", "qrf_beats_grid_with_margin", "gated_insufficient_n",
    ],
    "properties": {
        "n_grid": {"type": "integer"},
        "n_qrf_applied": {"type": "integer"},
        "rmse_grid_c": {"type": ["number", "null"]},
        "rmse_qrf_c": {"type": ["number", "null"]},
        "bias_qrf_c": {"type": ["number", "null"]},
        "se_bias_qrf_c": {"type": ["number", "null"]},
        "bias_correction_c": {"type": ["number", "null"]},
        "rmse_debiased_cv_c": {"type": ["number", "null"]},
        "bias_debiased_cv_c": {"type": ["number", "null"]},
        "bias_bounded_uncorrected": {"type": ["boolean", "null"]},
        "rmse_improvement_pct": {"type": ["number", "null"]},
        "rmse_improvement_pct_debiased_cv": {"type": ["number", "null"]},
        "qrf_beats_grid": {"type": ["boolean", "null"]},
        "qrf_beats_grid_with_margin": {"type": ["boolean", "null"]},
        # 2026-07-28, score.score_band's delta_scale extension (plan-2026-
        # 07-28-lagfill-base-mismatch-fix.md section 3.1/8.5) -- no
        # additionalProperties restriction above means these were always
        # accepted even before being listed here, but listing them keeps
        # this schema an accurate description of score_band's real output.
        "rmse_affine_cv_c": {"type": ["number", "null"]},
        "bias_affine_cv_c": {"type": ["number", "null"]},
        "rmse_improvement_pct_affine_cv": {"type": ["number", "null"]},
        "qrf_beats_grid_with_margin_affine": {"type": ["boolean", "null"]},
        "delta_scale_c": {
            "type": ["object", "null"],
            "properties": {"scale": {"type": "number"}, "offset": {"type": "number"}},
        },
        "gated_insufficient_n": {"type": "boolean"},
        # 2026-08-25, score.score_band's Rung B proposed_correction extension --
        # no additionalProperties restriction above means these were always
        # accepted even before being listed here (same non-event delta_scale_c's
        # own addition already was), listed for an accurate schema.
        "proposed_correction_kind": {"enum": ["bias", "affine", "covariate_linear", None]},
        "proposed_correction_basis": {"enum": ["raw_grid", "model_delta", None]},
        "proposed_correction_n_scored": {"type": ["integer", "null"]},
        # The per-stratum block is nested and deliberately not constrained in
        # detail here -- compare_reports cannot compare it (see its own
        # guard), and the numbers that gate a decision are mirrored as the
        # flat scalars below, which CAN be tolerance-checked.
        "proposed_correction_by_stratum": {"type": ["object", "null"]},
        "proposed_correction_ci95_lo_pct": {"type": ["number", "null"]},
        "proposed_correction_ci95_hi_pct": {"type": ["number", "null"]},
        "proposed_correction_hot_day_rmse_c": {"type": ["number", "null"]},
        "proposed_correction_hot_day_bias_c": {"type": ["number", "null"]},
        "proposed_correction_hot_day_margin_pct": {"type": ["number", "null"]},
        "proposed_correction_hot_day_ci95_lo_pct": {"type": ["number", "null"]},
        "proposed_correction_hot_day_ci95_hi_pct": {"type": ["number", "null"]},
        "proposed_correction_rmse_c": {"type": ["number", "null"]},
        "proposed_correction_bias_c": {"type": ["number", "null"]},
        "proposed_correction_beats_grid": {"type": ["boolean", "null"]},
        "proposed_correction_margin_pct": {"type": ["number", "null"]},
        "proposed_correction_beats_grid_with_margin": {"type": ["boolean", "null"]},
        "proposed_vs_best_fit_gap_c": {"type": ["number", "null"]},
    },
}

REPORT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "report_schema_version", "model_version", "band_key", "snapshot_version",
        "sample_requested", "rows_sampled", "rows_paired", "fidelity_check", "by_target",
    ],
    "properties": {
        "report_schema_version": {"const": REPORT_SCHEMA_VERSION},
        "generated_by": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"}, "version": {"type": "string"}, "git_commit": {"type": ["string", "null"]},
            },
        },
        "snapshot_version": {"type": "string"},
        "band_key": {"type": "string"},
        "model_version": {"type": "string"},
        "sample_requested": {"type": "integer"},
        "rows_sampled": {"type": "integer"},
        "rows_paired": {"type": "integer"},
        "fidelity_check": {
            "type": "object",
            "required": ["n"],
            "properties": {"n": {"type": "integer"}},
        },
        "by_target": {
            "type": "object",
            "properties": {
                "tmax": {"type": "object", "additionalProperties": _SCORE_BAND_METRICS_SCHEMA},
                "tmin": {"type": "object", "additionalProperties": _SCORE_BAND_METRICS_SCHEMA},
            },
        },
    },
}


def build_report(
    *,
    model_version: str,
    band_key: str,
    snapshot_version: str,
    sample_requested: int,
    rows_sampled: int,
    rows_paired: int,
    fidelity_check: dict,
    by_target: dict[str, dict],
    generated_by: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the shared report envelope. `by_target` is expected to
    already be {"tmax": score.score_band(...), "tmin": score.score_band(...)}
    -- this function does not call score_band itself, so a caller (a
    validation CLI, run_submission.py, or score_forward_eval.py) controls
    exactly which adapter/fold_salt was used to produce it.

    extra: band-specific fields (e.g. {"lead_days": 3} for a forecast
    band) merged into the envelope's top level. Must not collide with any
    of this function's own required keys -- raises ValueError if it does,
    rather than silently overwriting the canonical envelope."""
    report: dict = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "model_version": model_version,
        "band_key": band_key,
        "snapshot_version": snapshot_version,
        "sample_requested": sample_requested,
        "rows_sampled": rows_sampled,
        "rows_paired": rows_paired,
        "fidelity_check": fidelity_check,
        "by_target": by_target,
    }
    if generated_by is not None:
        report["generated_by"] = generated_by
    if extra:
        collisions = set(extra) & set(report)
        if collisions:
            raise ValueError(f"extra fields collide with the core report envelope: {sorted(collisions)}")
        report.update(extra)
    return report


def validate_report(report: dict) -> None:
    """Raises jsonschema.ValidationError if `report` doesn't match
    REPORT_SCHEMA."""
    import jsonschema
    jsonschema.validate(report, REPORT_SCHEMA)


class ToleranceResult:
    """Result of comparing a claimed report against an independently
    reproduced one. `passed` is True only if every metric named in
    `tolerance` is within its allowed absolute deviation for every
    (target, zone) both reports have in common."""

    def __init__(self, passed: bool, max_abs_deviation: dict[str, float], violations: list[dict]):
        self.passed = passed
        self.max_abs_deviation = max_abs_deviation
        self.violations = violations

    def __repr__(self) -> str:
        return f"ToleranceResult(passed={self.passed}, max_abs_deviation={self.max_abs_deviation}, violations={self.violations})"


def compare_reports(claimed: dict, reproduced: dict, tolerance: dict[str, float]) -> ToleranceResult:
    """Compare `claimed["by_target"]` against `reproduced["by_target"]` for
    every metric named as a key in `tolerance` (e.g.
    {"rmse_improvement_pct_debiased_cv": 0.002, "bias_correction_c": 0.01,
    "rmse_qrf_c": 0.005}, matching a submission manifest's own `tolerance:`
    block). A metric missing from either side for a given (target, zone)
    is skipped, not treated as a violation -- a contributor is never
    penalized for a zone the reproduction couldn't score (e.g. insufficient
    n on the independent snapshot copy); that mismatch is caught separately
    by comparing zone COVERAGE, which is the caller's responsibility, not
    this function's.

    `promote_from_public.py`-style callers should treat `passed=False` as a
    hard failure: never trust a submission's own claimed numbers, always
    require them to reproduce within tolerance against an independently
    re-derived report before anything is published."""
    violations: list[dict] = []
    max_abs_deviation: dict[str, float] = {metric: 0.0 for metric in tolerance}

    claimed_by_target = claimed.get("by_target", {})
    reproduced_by_target = reproduced.get("by_target", {})

    for target in ("tmax", "tmin"):
        claimed_zones = claimed_by_target.get(target, {})
        reproduced_zones = reproduced_by_target.get(target, {})
        for zone, claimed_metrics in claimed_zones.items():
            reproduced_metrics = reproduced_zones.get(zone)
            if reproduced_metrics is None:
                continue
            for metric, allowed in tolerance.items():
                claimed_val = claimed_metrics.get(metric)
                reproduced_val = reproduced_metrics.get(metric)
                if claimed_val is None or reproduced_val is None:
                    continue
                # Skip anything that isn't a plain number. A metric dict can
                # now legitimately hold a nested block
                # (proposed_correction_by_stratum, 2026-08-25), and a manifest
                # naming it in its own tolerance block would otherwise reach
                # abs(dict - dict) and raise TypeError. Reported as a
                # violation rather than a crash: a non-numeric metric can
                # never be shown to reproduce within a numeric tolerance, and
                # silently skipping it would let it pass unchecked.
                # Booleans compare by EQUALITY, never by tolerance.
                #
                # Two wrongs in a row here, worth recording. #31 excluded bool
                # from the numeric path to catch nested blocks, which made
                # identical booleans a hard violation -- a false rejection.
                # The first version of THIS fix put them back on the numeric
                # path, which opened a false ACCEPTANCE: abs(True - False) is
                # 1, no boolean key has a _TOLERANCE_MAXIMA ceiling, and
                # validate_manifest accepts any positive number for an
                # unlisted key -- so `tolerance: {qrf_beats_grid: 1.5}` made a
                # claimed True "reproduce" against an independently derived
                # False, and run_submission reported status: pass. Verified
                # live before this rewrite. A referee bypass is strictly worse
                # than the rejection it replaced.
                #
                # The comment I wrote then cited "none of which are
                # ceiling-restricted" as evidence those keys were safe. That
                # is exactly what made them exploitable; I had the fact
                # backwards.
                #
                # A tolerance on a boolean is meaningless anyway: it either
                # reproduces or it does not.
                if isinstance(claimed_val, bool) or isinstance(reproduced_val, bool):
                    if claimed_val is reproduced_val:
                        max_abs_deviation[metric] = max(max_abs_deviation[metric], 0.0)
                        continue
                    violations.append({
                        "target": target, "zone": zone, "metric": metric,
                        "claimed": claimed_val, "reproduced": reproduced_val,
                        "abs_diff": None, "allowed": allowed,
                        "reason": "boolean metrics must match exactly -- a tolerance cannot "
                                  "make a claimed True reproduce against a derived False",
                    })
                    continue
                # Non-scalars (regression context, PR #34).
                # Excluding them to catch non-scalars was too broad: bool is a
                # subclass of int and compares perfectly well, so
                # abs(True - True) == 0 passed before that guard and became a
                # hard violation after it. Verified: compare_reports(r, r,
                # {"qrf_beats_grid": 0.001}) returned passed=False on
                # byte-identical inputs, so any manifest naming a boolean
                # metric in its tolerance block -- qrf_beats_grid,
                # gated_insufficient_n, proposed_correction_beats_grid_with_margin,
                # none of which are ceiling-restricted and all of which are
                # legal keys -- was rejected even when it reproduced exactly.
                # A mismatched pair still fails: abs(True - False) == 1
                # exceeds any sane tolerance, which is the correct verdict.
                if not isinstance(claimed_val, (int, float)) \
                        or not isinstance(reproduced_val, (int, float)):
                    violations.append({
                        "target": target, "zone": zone, "metric": metric,
                        "claimed": claimed_val, "reproduced": reproduced_val,
                        "abs_diff": None, "allowed": allowed,
                        "reason": "metric is not a scalar -- cannot be compared against a numeric tolerance",
                    })
                    continue
                abs_diff = abs(claimed_val - reproduced_val)
                max_abs_deviation[metric] = max(max_abs_deviation[metric], abs_diff)
                if abs_diff > allowed:
                    violations.append({
                        "target": target, "zone": zone, "metric": metric,
                        "claimed": claimed_val, "reproduced": reproduced_val,
                        "abs_diff": abs_diff, "allowed": allowed,
                    })

    return ToleranceResult(passed=not violations, max_abs_deviation=max_abs_deviation, violations=violations)
