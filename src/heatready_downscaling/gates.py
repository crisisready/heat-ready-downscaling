"""
build_gate + the jsonschemas for both gate shapes this program publishes.
Extracted from crisisready/heat-risk-data-api's scripts/publish_band_gate.py
(lines ~73-107 as of 2026-07-27, commit 57479e5); the blend-gate schema is
new here (that repo's scripts/publish_blend_gate.py only ever did an ad-hoc
`for required_key in (...)` presence check, no real schema) -- added for
parity, so this module genuinely owns "the gate shape" for both gate kinds.
See PROVENANCE.md.
"""

import copy

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
        # delta_scale_subzone / bias_correction_subzone (2026-08-08,
        # PARIS_CONFIDENCE_ROADMAP.md's "next round plan" in
        # crisisready/heat-risk-data-api -- the France-within-Cfb shippable
        # candidate): one level finer than delta_scale/bias_correction,
        # NOT required, so every gate published before this field existed
        # still validates, and a server that hasn't been updated to read it
        # keeps serving the zone-level correction exactly as before -- same
        # backward-compat shape delta_scale itself used when it was added
        # alongside bias_correction.
        #
        # Keyed {target: {zone: {subzone_code: {scale, offset}}}} -- one
        # extra nesting level under the existing zone key, not a parallel
        # top-level-by-subzone structure, so a zone with no subzone data at
        # all is simply absent here (matches every other field's "absent
        # means not published" convention).
        #
        # subzone_code is a GHCN station_id COUNTRY PREFIX (e.g. "FR"),
        # the SAME convention validate_lagfill_downscaling.py's own
        # --regions flag uses -- deliberately not a free-text region name,
        # so gates.py never needs its own separate concept of what a
        # subzone "is."
        #
        # WHY THIS IS A SEPARATE FIELD, NOT JUST A FINER delta_scale: a
        # subzone-scoped correction must NEVER silently reach a project
        # outside that subzone the way a flat delta_scale[target][zone]
        # entry would (every project in that zone reads the same value).
        # build_gate() refuses to build this field from a whole-zone
        # report (see its own docstring) -- ONLY build_subzone_patch(),
        # called on a --regions-scoped report, ever populates it, and only
        # as a MERGE into the currently-published gate (never a fresh
        # build+overwrite the way the rest of this schema's fields are) --
        # see publish_band_gate.py's own --regions handling. Serving-side
        # resolution is likewise opt-in per project (crisisready/
        # heat-risk-data-api's manager.py, an explicit override dict
        # mirroring _LAG_FILL_FORECAST_NATIVE_RESOLUTION_OVERRIDE's own
        # precedent) -- a project not explicitly opted into a subzone code
        # never reads this field at all, regardless of what's published
        # here, so publishing a subzone entry can never silently change
        # what any EXISTING project serves.
        "delta_scale_subzone": {
            "type": "object",
            "properties": {
                "tmax": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["scale", "offset"],
                            "properties": {"scale": {"type": "number"}, "offset": {"type": "number"}},
                        },
                    },
                },
                "tmin": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["scale", "offset"],
                            "properties": {"scale": {"type": "number"}, "offset": {"type": "number"}},
                        },
                    },
                },
            },
        },
        "bias_correction_subzone": {
            "type": "object",
            "properties": {
                "tmax": {
                    "type": "object",
                    "additionalProperties": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "tmin": {
                    "type": "object",
                    "additionalProperties": {"type": "object", "additionalProperties": {"type": "number"}},
                },
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


SPATIAL_RANKING_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["model_version", "protocol", "by_zone"],
    "properties": {
        "model_version": {"type": "string"},
        "computed_at": {"type": "string"},
        "protocol": {"type": "object"},
        "by_zone": {
            "type": "object",
            "required": ["tmax", "tmin"],
            "properties": {
                target_key: {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["class", "n_clusters_used", "n_stations_used"],
                        "properties": {
                            "class": {"enum": ["strong", "moderate", "weak", "unproven"]},
                            "ranking_spearman": {"type": ["number", "null"]},
                            "ranking_spearman_ci95": {
                                "type": ["array", "null"],
                                "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                            },
                            "amplitude_slope": {"type": ["number", "null"]},
                            "n_clusters_used": {"type": "integer"},
                            "n_stations_used": {"type": "integer"},
                            "n_regions": {"type": "integer"},
                        },
                    },
                }
                for target_key in ("tmax", "tmin")
            },
        },
        "region_scale_secondary": {"type": "object"},
    },
}


def validate_spatial_ranking(artifact: dict) -> None:
    """Raises jsonschema.ValidationError if `artifact` doesn't match
    SPATIAL_RANKING_SCHEMA -- same real-schema-check idiom validate_gate/
    validate_blend_gate already use, not an ad-hoc presence check."""
    import jsonschema
    jsonschema.validate(artifact, SPATIAL_RANKING_SCHEMA)


def build_gate(report: dict, band_key: str | None = None, variant: str | None = None) -> dict:
    """band_key, when given, is asserted against report["band_key"] --
    the validation scripts both stamp their own band into every report
    they write, so a report generated for one band (e.g. lag_fill) can
    never be silently published under a different --band-key (e.g.
    forecast_lead3) without this raising first. Optional (default None,
    skipping the check) only so this stays callable the same way against a
    hand-built report dict that omits "band_key" -- every real invocation
    from a publish CLI always passes it.

    variant (2026-08-03, gate-variant scoping -- see downscaling.
    load_band_gate's own docstring in heat-risk-data-api): same idea as
    band_key, checked in BOTH directions. A report fitted under an
    alternate base distribution (e.g. --elevation-nan, stamped
    report["base_variant"]="native_noelev") must be published under the
    matching --variant, and a DEFAULT report (no base_variant stamped)
    must be published with no --variant at all -- either mismatch raises.
    The dangerous direction is the second one: publishing a variant-fitted
    report to the DEFAULT (no-variant) S3 key would silently corrupt the
    gate every OTHER project (in the same zone(s), under the DEFAULT
    request config) reads, which is exactly the cross-project corruption
    this whole variant mechanism exists to prevent -- so this check is not
    optional/skippable the way band_key's is, and there is no bypass for a
    hand-built report dict here.

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
    downscaling working there. Originally purely informational (never
    read by any serving path) -- **no longer true as of 2026-07-31**:
    crisisready/heat-risk-data-api's downscaling.spatial_ranking_for_band
    now reads this field directly to veto a different, separately-shipped
    confidence signal (`spatial_ranking`) on non-ERA5 bands whose own gate
    doesn't show real spatial skill (see that repo's PARIS_CONFIDENCE_
    ROADMAP.md, "R3 scoping" section, for the full reasoning). This field
    is now load-bearing for that serving path -- a future change to how
    qrf_beats_grid is computed here will silently move a served field in
    that repo, not just a disclosure string in this one.

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
    report_variant = report.get("base_variant")
    if report_variant != variant:
        raise ValueError(
            f"report base_variant {report_variant!r} does not match --variant {variant!r} "
            "-- refusing to publish a report fitted under one base distribution to a gate "
            "key for a different one (this would corrupt every other project reading that key)",
        )
    # 2026-08-03, adversarial review finding: a --zones-scoped validation run
    # with NO --elevation-nan produces a report with base_variant=None, which
    # passes the check above cleanly (None matches --variant omitted) -- but
    # this function builds the gate FRESH from the report alone, no merge
    # with whatever's currently published, so publishing it to the DEFAULT
    # key would silently replace every zone NOT in this report's scope with
    # nothing (wiped tmax/tmin pass bits, bias_correction, delta_scale,
    # spatial_skill) for every other project reading that key. variant is
    # the escape hatch: a zones-scoped report is only safe to publish when
    # it's also going to a variant-suffixed key, never the shared default.
    if report.get("zones") and variant is None:
        raise ValueError(
            f"report is scoped to zones={report['zones']!r} but --variant was not given -- "
            "refusing to publish a zone-scoped report to the shared default key, which would "
            "silently wipe every zone NOT in this report for every other project reading it "
            "(this function builds the gate fresh from the report, it does not merge with what's "
            "currently published). Either publish under a --variant, or rerun the validation "
            "without --zones for a report that's safe to publish to the default key.",
        )
    # 2026-08-08, sub-zone (country) scoping: a regions-scoped report's own
    # by_target metrics reflect ONLY that subset of a zone's stations (e.g.
    # French Cfb stations only) -- feeding it through this function's normal
    # "build a fresh gate from the report" path would write those
    # France-only numbers to the SHARED, whole-zone gate[target][zone]
    # entry, exactly the cross-country regression this program's own
    # research found real (Cfb's raw tmin bias flips sign between France
    # and the rest of the zone). Refuse unconditionally here, the same way
    # a zones-scoped/no-variant report is refused above -- a regions-scoped
    # report must go through build_subzone_patch() instead, which never
    # touches the flat zone-level fields at all.
    if report.get("regions") or report.get("exclude_regions"):
        raise ValueError(
            f"report is scoped to regions={report.get('regions')!r} "
            f"exclude_regions={report.get('exclude_regions')!r} -- refusing to publish via build_gate, "
            "which would write this subset's own numbers to the shared, whole-zone gate entry every "
            "OTHER country in that zone also reads. Use build_subzone_patch() instead (regions-scoped "
            "reports only -- exclude_regions-scoped reports, e.g. 'the rest of Cfb', are a regression "
            "CHECK, not something this program ever publishes on their own).",
        )
    gate: dict = {
        "tmax": {}, "tmin": {},
        "bias_correction": {"tmax": {}, "tmin": {}},
        "delta_scale": {"tmax": {}, "tmin": {}},
        "spatial_skill": {"tmax": {}, "tmin": {}},
    }
    for target in ("tmax", "tmin"):
        for zone, metrics in report.get("by_target", {}).get(target, {}).items():
            debias_passes, affine_passes = _evidence_bar_verdict(metrics)
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


def _evidence_bar_verdict(metrics: dict) -> tuple[bool, bool]:
    """(debias_passes, affine_passes) -- the SAME evidence bar build_gate
    and build_subzone_patch both hold every zone/subzone entry to (both
    docstrings explicitly claim this; 2026-08-08 review finding: they had
    silently duplicated this exact block instead of sharing it, a real
    risk of the two drifting apart on a future edit to one but not the
    other). Factored out here so there is exactly one place this decision
    is made."""
    debias_passes = metrics.get("qrf_beats_grid_with_margin") is True
    affine_passes = metrics.get("qrf_beats_grid_with_margin_affine") is True
    return debias_passes, affine_passes


def build_subzone_patch(report: dict, band_key: str | None = None, variant: str | None = None) -> dict:
    """The regions-scoped counterpart to build_gate() -- builds a small
    PATCH fragment ({"delta_scale_subzone": {...}, "bias_correction_subzone":
    {...}}), never a full gate, and never anything a caller could
    accidentally PUT as a whole gate (it deliberately omits "tmax"/"tmin"/
    "bias_correction"/"delta_scale"/"spatial_skill" entirely) -- the
    caller (publish_band_gate.py's --regions path) is required to GET the
    currently-published gate and MERGE this patch into it, never overwrite.

    Same band_key/variant checks as build_gate, for the same reasons (see
    its own docstring) -- a regions-scoped report is still subject to both.

    Requires report["regions"] to be a single-element list -- a subzone
    patch is keyed by exactly one subzone code per publish, so a report
    scoped to multiple regions at once (which validate_lagfill_downscaling.py's
    --regions flag technically allows, e.g. --regions FR,DE) has no single
    subzone code to file its numbers under and must be split into separate
    single-region validation runs before publishing, one patch per region.
    This is a deliberate restriction on what's PUBLISHABLE, not on what the
    validation harness can score -- --regions FR,DE remains a legitimate
    thing to score (e.g. as an ad-hoc combined check) even though the
    result can't go through this function."""
    if band_key is not None and report.get("band_key") != band_key:
        raise ValueError(
            f"report band_key {report.get('band_key')!r} does not match --band-key {band_key!r} "
            "-- refusing to publish a report generated for a different band",
        )
    report_variant = report.get("base_variant")
    if report_variant != variant:
        raise ValueError(
            f"report base_variant {report_variant!r} does not match --variant {variant!r} -- "
            "refusing to publish a subzone patch fitted under one base distribution against a "
            "gate key for a different one",
        )
    regions = report.get("regions")
    if not regions:
        raise ValueError(
            "report has no regions -- build_subzone_patch is only for --regions-scoped reports; "
            "use build_gate for an unscoped or --zones-scoped (whole-zone) report",
        )
    if len(regions) != 1:
        raise ValueError(
            f"report is scoped to regions={regions!r} -- build_subzone_patch requires exactly one "
            "region per publish (a subzone patch is keyed by a single subzone code); rerun the "
            "validation once per region if a combined multi-region report was used to score",
        )
    subzone = regions[0]

    patch: dict = {"delta_scale_subzone": {"tmax": {}, "tmin": {}}, "bias_correction_subzone": {"tmax": {}, "tmin": {}}}
    for target in ("tmax", "tmin"):
        for zone, metrics in report.get("by_target", {}).get(target, {}).items():
            debias_passes, affine_passes = _evidence_bar_verdict(metrics)
            if not (debias_passes or affine_passes):
                continue
            # Same publish conditions as build_gate's own bias_correction/delta_scale
            # blocks (see that function's docstring) -- a subzone entry is held to
            # the SAME evidence bar as a zone entry, not a looser one, because
            # a subzone claim is exactly as capable of corrupting what gets served
            # as a zone claim is (just to a narrower project set).
            if debias_passes and metrics.get("bias_correction_c") is not None:
                patch["bias_correction_subzone"][target].setdefault(zone, {})[subzone] = metrics["bias_correction_c"]
            if affine_passes and metrics.get("delta_scale_c") is not None:
                patch["delta_scale_subzone"][target].setdefault(zone, {})[subzone] = metrics["delta_scale_c"]
    return patch


def merge_subzone_patch(current_gate: dict, patch: dict) -> dict:
    """Merges a build_subzone_patch() fragment into an already-published
    gate dict (as returned by a real GET, or {} on a first-ever subzone
    publish to a gate with no subzone data yet) -- returns a NEW dict,
    does not mutate current_gate in place, so a caller can still compare
    before/after for logging without the comparison being trivially equal.

    Deliberately additive at the (target, zone, subzone) leaf only -- every
    OTHER key in current_gate (tmax, tmin, bias_correction, delta_scale,
    spatial_skill, and any OTHER zone's or OTHER subzone's entry already
    present under delta_scale_subzone/bias_correction_subzone) is carried
    through byte-for-byte unchanged. This is the actual mechanism that
    makes a subzone publish safe where a build_gate publish of the same
    data would not be -- see build_gate's own regions refusal."""
    merged = copy.deepcopy(current_gate) if current_gate else {}
    for field in ("delta_scale_subzone", "bias_correction_subzone"):
        merged.setdefault(field, {"tmax": {}, "tmin": {}})
        for target in ("tmax", "tmin"):
            merged[field].setdefault(target, {})
            for zone, by_subzone in patch.get(field, {}).get(target, {}).items():
                merged[field][target].setdefault(zone, {})
                merged[field][target][zone].update(by_subzone)
    return merged


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
