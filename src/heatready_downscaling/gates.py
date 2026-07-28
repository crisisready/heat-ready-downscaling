"""
build_gate + the jsonschemas for both gate shapes this program publishes.
Extracted from crisisready/heat-risk-data-api's scripts/publish_band_gate.py
(lines ~73-107 as of 2026-07-27, commit 57479e5); the blend-gate schema is
new here (that repo's scripts/publish_blend_gate.py only ever did an ad-hoc
`for required_key in (...)` presence check, no real schema) -- added for
parity, so this module genuinely owns "the gate shape" for both gate kinds.
See PROVENANCE.md.
"""

BAND_GATE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["tmax", "tmin", "bias_correction", "spatial_skill"],
    "properties": {
        "tmax": {"type": "object", "additionalProperties": {"const": True}},
        "tmin": {"type": "object", "additionalProperties": {"const": True}},
        "bias_correction": {
            "type": "object",
            "required": ["tmax", "tmin"],
            "properties": {
                "tmax": {"type": "object", "additionalProperties": {"type": "number"}},
                "tmin": {"type": "object", "additionalProperties": {"type": "number"}},
            },
        },
        # delta_scale (2026-07-28, plan-2026-07-28-lagfill-base-mismatch-fix.md
        # section 3.1/8.5, "Option R"): a per-zone {scale, offset} pair, ALONGSIDE
        # bias_correction rather than replacing it -- NOT required at the
        # top level, so a gate dict built before this field existed still
        # validates. A zone present here takes priority over the same
        # zone's bias_correction entry at serving time (see
        # downscaling.predict_downscaled); a zone ABSENT here (including
        # every zone in every gate published before this field existed)
        # falls back to bias_correction with an implicit scale of 1.0 --
        # today's exact behavior, unchanged.
        "delta_scale": {
            "type": "object",
            "properties": {
                "tmax": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["scale", "offset"],
                        "properties": {"scale": {"type": "number"}, "offset": {"type": "number"}},
                    },
                },
                "tmin": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["scale", "offset"],
                        "properties": {"scale": {"type": "number"}, "offset": {"type": "number"}},
                    },
                },
            },
        },
        "spatial_skill": {
            "type": "object",
            "required": ["tmax", "tmin"],
            "properties": {
                "tmax": {"type": "object", "additionalProperties": {"type": "boolean"}},
                "tmin": {"type": "object", "additionalProperties": {"type": "boolean"}},
            },
        },
    },
}

BLEND_GATE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["tmax", "tmin", "params"],
    "properties": {
        "tmax": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "tmin": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "params": {
            "type": "object",
            "properties": {
                "tmax": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["L_km", "R_km", "tau"],
                        "properties": {
                            "L_km": {"type": "number"}, "R_km": {"type": "number"}, "tau": {"type": "number"},
                        },
                    },
                },
                "tmin": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["L_km", "R_km", "tau"],
                        "properties": {
                            "L_km": {"type": "number"}, "R_km": {"type": "number"}, "tau": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}


def build_gate(report: dict, band_key: str | None = None) -> dict:
    """band_key, when given, is asserted against report["band_key"] --
    the validation scripts both stamp their own band into every report
    they write, so a report generated for one band (e.g. lag_fill) can
    never be silently published under a different --band-key (e.g.
    forecast_lead3) without this raising first. Optional (default None,
    skipping the check) only so this stays callable the same way against a
    hand-built report dict that omits "band_key" -- every real invocation
    from a publish CLI always passes it.

    bias_correction (MOS-style recalibration): for every zone that passes
    ONLY because of the debiased, cross-validated check
    (rmse_improvement_pct_debiased_cv, not the raw rmse_improvement_pct),
    the zone's bias_correction_c is published too -- QRFModelAdapter.predict
    adds this to delta_c at serving time, so production actually applies
    the recentered correction the validation proved out, not just a bare
    pass/fail flag. A zone that passes via the raw (non-debiased) fallback
    path (bias_bounded_uncorrected -- too few distinct stations for CV)
    publishes NO bias_correction entry: its raw bias was already confirmed
    small enough not to need one.

    spatial_skill (honest-labeling): True when the zone's RAW (uncorrected)
    QRF delta already beats grid (qrf_beats_grid is True) -- genuine
    neighborhood-resolution spatial downscaling skill, the per-polygon
    signal itself does the work. False when a zone passes the published
    gate ONLY via the debiased-CV margin (qrf_beats_grid is False but
    qrf_beats_grid_with_margin is True) -- the entire improvement is
    carried by a flat per-Köppen-zone MOS bias constant; the QRF's spatial
    delta contributes nothing (or is net-negative) for that zone. Both are
    legitimate, CV-validated, more-accurate-than-grid corrections worth
    serving -- this field exists so nothing downstream represents a
    spatial_skill=False zone as evidence of neighborhood-resolution
    downscaling working there. Does NOT change serving behavior in any
    way; purely for honest disclosure/display.

    Only True verdicts are written explicitly; a zone with
    qrf_beats_grid_with_margin (or, since 2026-07-28,
    qrf_beats_grid_with_margin_affine -- see delta_scale below) False OR
    None (insufficient n, or gate-failed already at the ERA5 level) is
    simply omitted -- a serving-side .get(zone, False) lookup treats
    "absent" identically to "explicitly False", so omitting is equivalent
    to False and keeps the published file small and readable.

    delta_scale (2026-07-28, plan-2026-07-28-lagfill-base-mismatch-fix.md
    section 3.1/8.5, "Option R"): a zone is enabled (gate[target][zone] =
    True) if EITHER the offset-only correction (qrf_beats_grid_with_margin)
    OR its affine generalization (qrf_beats_grid_with_margin_affine)
    clears AUTO_ENABLE_MARGIN -- this is how a base-distribution mismatch
    (Cfb on the lag_fill band, where the model's delta systematically
    overshoots) gets a real fix instead of staying permanently ungated.
    delta_scale[target][zone] is published, ALONGSIDE bias_correction (not
    replacing it), only when the affine fit both clears its own margin AND
    out-of-fold beats the offset-only fit (score.score_band already only
    computes delta_scale_c under that second condition; this function adds
    the margin check). bias_correction is still published whenever the
    offset-only fit itself validates, regardless of which one wins -- so a
    server that has not yet been updated to read delta_scale keeps serving
    the offset-only correction exactly as before, and a zone that passes
    ONLY via affine (offset-only fails its own margin, e.g. Cfb) publishes
    delta_scale with NO bias_correction entry, since bias_correction alone
    was never validated to help there."""
    if band_key is not None and report.get("band_key") != band_key:
        raise ValueError(
            f"report band_key {report.get('band_key')!r} does not match --band-key {band_key!r} "
            "-- refusing to publish a report generated for a different band",
        )
    gate: dict = {
        "tmax": {}, "tmin": {},
        "bias_correction": {"tmax": {}, "tmin": {}},
        "delta_scale": {"tmax": {}, "tmin": {}},
        "spatial_skill": {"tmax": {}, "tmin": {}},
    }
    for target in ("tmax", "tmin"):
        for zone, metrics in report.get("by_target", {}).get(target, {}).items():
            debias_passes = metrics.get("qrf_beats_grid_with_margin") is True
            affine_passes = metrics.get("qrf_beats_grid_with_margin_affine") is True
            if not (debias_passes or affine_passes):
                continue
            gate[target][zone] = True
            # Only publish a correction for zones validated via the CV path
            # (rmse_improvement_pct_debiased_cv not None) -- the raw-fallback
            # path (too few distinct stations for CV) already required the
            # UNCORRECTED bias to be small on its own, so no correction is
            # needed or was ever validated for those zones.
            if debias_passes and metrics.get("bias_correction_c") is not None:
                gate["bias_correction"][target][zone] = metrics["bias_correction_c"]
            # delta_scale_c is only non-None when it already out-of-fold
            # beats the offset-only fit (see score.score_band) -- so an
            # affine fit that merely clears its own margin without beating
            # offset-only never gets here with a value to publish.
            if affine_passes and metrics.get("delta_scale_c") is not None:
                gate["delta_scale"][target][zone] = metrics["delta_scale_c"]
            # spatial_skill: True iff the RAW (uncorrected) QRF delta
            # already beats grid -- see this function's own docstring.
            gate["spatial_skill"][target][zone] = bool(metrics.get("qrf_beats_grid"))
    return gate


def validate_gate(gate: dict) -> None:
    """Raises jsonschema.ValidationError if `gate` doesn't match
    BAND_GATE_SCHEMA. Clearer failure than the private repo's original
    publish_band_gate.py, which had no schema check at all before
    uploading."""
    import jsonschema
    jsonschema.validate(gate, BAND_GATE_SCHEMA)


def validate_blend_gate(gate: dict) -> None:
    """Raises jsonschema.ValidationError if `gate` doesn't match
    BLEND_GATE_SCHEMA. Replaces the private repo's ad-hoc
    `for required_key in ("tmax", "tmin", "params")` presence check with
    real schema validation, including the per-group L_km/R_km/tau shape."""
    import jsonschema
    jsonschema.validate(gate, BLEND_GATE_SCHEMA)
