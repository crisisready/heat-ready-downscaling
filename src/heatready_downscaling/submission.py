"""
The submission manifest schema every contributor's `manifest.yaml` must
satisfy, and every referee (CI lint, `run_submission.py`, a future
`promote_from_public.py`) validates against -- single source of truth, per
the Phase 3 design consult (2026-07-27): "the lint and the referee share
it," never two independent implementations that could silently drift
apart, the same discipline this package already applies to score_band's
extraction (score.py's own docstring).

New in this package (Phase 3, no private-repo equivalent) -- CONTRIBUTING.md
already documents the intended manifest.yaml shape; this module is that
documentation made enforceable. See PROVENANCE.md.
"""

from heatready_downscaling import score as _score
from heatready_downscaling import licensing as _licensing
from heatready_downscaling import snapshot as _snapshot

MANIFEST_SCHEMA_VERSION = 1

# Bounded tolerance values (design-consult finding, 2026-07-27): an
# unbounded tolerance block lets a submission set e.g. rmse_qrf_c: 999 and
# always "reproduce" regardless of how wrong its claimed numbers are. These
# ceilings are deliberately generous relative to the metrics' own typical
# magnitudes (score.py's own docstrings: RMSEs are usually 1-3C, bias
# corrections up to ~1.2C observed live) -- loose enough to tolerate real
# floating-point/platform variation across independent re-derivations,
# tight enough that "reproduces within tolerance" still means something.
_TOLERANCE_MAXIMA = {
    "rmse_improvement_pct_debiased_cv": 0.01,
    "bias_correction_c": 0.05,
    "rmse_qrf_c": 0.02,
    "rmse_grid_c": 0.02,
    "rmse_debiased_cv_c": 0.02,
    # 2026-08-25, Rung B: same treatment as the mechanically-derived metrics
    # above -- without a listed ceiling, a Rung B manifest's tolerance block
    # could set e.g. proposed_correction_rmse_c: 999 and always "reproduce."
    "proposed_correction_rmse_c": 0.02,
    "proposed_correction_bias_c": 0.05,
    "proposed_correction_margin_pct": 0.01,
    "proposed_vs_best_fit_gap_c": 0.05,
    # 2026-08-25, the covariate_linear shape: ceilings for score_band's flat
    # stratum mirrors. Every mirrored metric needs one -- validate_manifest
    # only enforces a ceiling for metrics listed HERE, so an unlisted metric
    # silently accepts any tolerance a manifest declares.
    "proposed_correction_ci95_lo_pct": 0.01,
    "proposed_correction_ci95_hi_pct": 0.01,
    "proposed_correction_hot_day_rmse_c": 0.02,
    "proposed_correction_hot_day_bias_c": 0.05,
    "proposed_correction_hot_day_margin_pct": 0.01,
    "proposed_correction_hot_day_ci95_lo_pct": 0.01,
    "proposed_correction_hot_day_ci95_hi_pct": 0.01,
}

# The covariate_linear shape (2026-08-25, the third one). Kept as an explicit
# schema rather than generated from a key tuple like the two flat shapes
# below, because its value shape is nested (a term list, an optional
# valid_range parallel to it) rather than a flat set of numbers.
#
# The `covariate` enum is score.STATIC_COVARIATE_ALLOWLIST, referenced rather
# than re-listed -- see that constant's own comment for why the list is
# closed. This is the CI-enforced half of the staticness rule; score_band's
# validate_covariate_linear_entry is the runtime half, because the monthly
# re-scoring path reads manifests off disk without re-validating them.
_COVARIATE_TERM_SCHEMA: dict = {
    "type": "object",
    "required": ["covariate", "slope"],
    "additionalProperties": False,
    "properties": {
        "covariate": {"enum": list(_score.STATIC_COVARIATE_ALLOWLIST)},
        "slope": {"type": "number"},
    },
}

_COVARIATE_LINEAR_SCHEMA: dict = {
    "type": "object",
    "required": list(_score.PROPOSED_CORRECTION_COVARIATE_LINEAR_KEYS),
    "additionalProperties": False,
    "properties": {
        "basis": {"enum": list(_score.PROPOSED_CORRECTION_BASES)},
        "intercept": {"type": "number"},
        # maxItems is score.MAX_COVARIATE_TERMS, not a hand-written number --
        # the cap exists because Valencia's own 2-covariate fit was worse
        # out-of-fold than its 1-covariate one, and the schema and the runtime
        # check must not be able to drift apart on it.
        "terms": {
            "type": "array", "minItems": 1,
            "maxItems": _score.MAX_COVARIATE_TERMS,
            "items": _COVARIATE_TERM_SCHEMA,
        },
        # One [min, max] per term, in term order; null for a term that
        # deliberately declares no range. Absent means no range check at all,
        # which score_band permits but which a reviewer should push back on --
        # a fit is only evidence over the range it was fit on.
        "valid_range": {
            "type": ["array", "null"],
            "minItems": 1, "maxItems": _score.MAX_COVARIATE_TERMS,
            "items": {
                "type": ["array", "null"],
                "minItems": 2, "maxItems": 2,
                "items": {"type": "number"},
            },
        },
        # A fixed enum, never a free number -- a contributor able to pick the
        # hot-day cutoff can shop for the one that flatters the result, the
        # same gaming surface fold_salt exists to close.
        "hot_day_threshold_c": {"enum": list(_score.HOT_DAY_THRESHOLD_C_CHOICES)},
    },
}

# One zone's declared Rung B value -- exactly one of the three shapes, never
# more than one. The two flat shapes are built FROM
# score.PROPOSED_CORRECTION_BIAS_KEYS/AFFINE_KEYS (code-review finding, PR
# #24) rather than re-listing the key names here independently -- score_band's
# own runtime dispatch and this schema are now guaranteed to agree on what the
# valid shapes are, which is exactly the property that made adding the third
# shape here a required, visible step rather than something that could slip
# into one side only.
_CANDIDATE_ZONE_SCHEMA: dict = {
    "type": "object",
    "oneOf": [
        {
            "type": "object", "required": list(keys), "additionalProperties": False,
            "properties": {k: {"type": "number"} for k in keys},
        }
        for keys in (_score.PROPOSED_CORRECTION_BIAS_KEYS, _score.PROPOSED_CORRECTION_AFFINE_KEYS)
    ] + [_COVARIATE_LINEAR_SCHEMA],
}

MANIFEST_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version", "submission_id", "author", "track", "rung", "snapshot",
        "claims", "method", "claimed_report", "tolerance",
    ],
    "properties": {
        "schema_version": {"const": MANIFEST_SCHEMA_VERSION},
        # {YYYY-MM}-{NNN}, matching submissions/{YYYY-MM}/{NNN}-{github}-{slug}/ --
        # not proof of uniqueness (that's a ledger-time check, see
        # scripts/run_submission.py), just the required shape.
        "submission_id": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{3}$"},
        "author": {
            "type": "object",
            "required": ["github"],
            "properties": {
                "github": {"type": "string", "minLength": 1},
                "name": {"type": ["string", "null"]},
                "orcid": {"type": ["string", "null"]},
                "affiliation": {"type": ["string", "null"]},
            },
        },
        "track": {"enum": ["serving-ready", "research"]},
        # Rung C listed for forward-compatibility with the schema shape --
        # not yet open; scripts/run_submission.py rejects it explicitly
        # rather than relying on this schema alone (see this repo's own
        # GOVERNANCE.md "Rung C (new model code): not yet open").
        "rung": {"enum": ["A", "B", "C"]},
        "snapshot": {
            "type": "object",
            "required": ["version", "manifest_sha256"],
            "properties": {
                "version": {"type": "string", "minLength": 1},
                "manifest_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            },
        },
        "claims": {
            # maxItems: 1 (Codex adversarial review finding, PR #24 round 2):
            # "exactly one claims[] entry per submission" was already a
            # stated v1 restriction (run_submission.py's own module
            # docstring), but was previously enforced only by
            # run_submission.py's cross_check, a SEPARATE runtime check --
            # not by this schema. That meant validate_manifest (and
            # anything built on it, like this schema's own rung-B coverage
            # check just below, and score_forward_eval.py's
            # load_active_candidates) could see a multi-claim manifest as
            # "structurally valid" while only ever reading claims[0] --
            # a second claim's cells would silently never be coverage-
            # checked or scored. Making it a schema-level invariant means
            # every caller of validate_manifest gets the same guarantee,
            # not just the one script that happened to add its own check.
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": {
                "type": "object",
                "required": ["model_version", "band_key", "targets", "zones"],
                "properties": {
                    "model_version": {"type": "string", "minLength": 1},
                    # Reuses snapshot._BANDS -- the single "all recognized bands"
                    # list, never redefined here, so this schema can't silently
                    # drift from what the snapshot itself actually partitions by.
                    "band_key": {"enum": list(_snapshot._BANDS)},
                    "targets": {
                        "type": "array", "minItems": 1,
                        "items": {"enum": ["tmax", "tmin"]}, "uniqueItems": True,
                    },
                    "zones": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True},
                },
            },
        },
        "method": {
            "type": "object",
            "required": ["kind", "entrypoint", "package_version"],
            "properties": {
                "kind": {"enum": ["rerun-validator", "parameters", "model"]},
                "entrypoint": {"type": "string", "minLength": 1},
                "args": {"type": "array", "items": {"type": "string"}},
                "package_version": {"type": "string", "minLength": 1},
                "code_ref": {"type": ["object", "null"]},
                "extra_covariates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "source", "license", "global", "reproducible_fetch"],
                        "properties": {
                            "name": {"type": "string"},
                            "source": {"type": "string"},
                            "url": {"type": "string"},
                            # SPDX identifier, or the "proprietary-licensed" escape
                            # hatch requiring a named licensor. The rules live in
                            # heatready_downscaling.licensing and are enforced by
                            # validate_manifest below, plus a reachability check in
                            # the Data licensing workflow. This comment previously
                            # claimed CI already enforced it; that was untrue from
                            # July until 2026-08-25 -- the allowlist did not exist and
                            # any string passed. Kept as a bare string type here
                            # because the allowlist is a Python set, not something
                            # jsonschema should duplicate.
                            "license": {"type": "string"},
                            "global": {"type": "boolean"},
                            "cadence": {"type": "string"},
                            "reproducible_fetch": {"type": "string"},
                            # Required by licensing.check_license_id when
                            # license == "proprietary-licensed"; that rule is
                            # cross-field, so it lives in Python rather than
                            # here (same convention as the tolerance ceilings).
                            "licensor": {"type": "string"},
                        },
                    },
                },
                # Any data input beyond the published snapshot, per the
                # roadmap's standing data-sourcing policy. Distinct from
                # extra_covariates, which is specifically the research
                # track's new-model-covariate path: a local model's training
                # data, or a contributed sensor source, is a data_source
                # without being a model covariate.
                #
                # Shape-checked here; the licensing RULES (SPDX allowlist,
                # the named-licensor escape hatch, tier/attribution
                # consistency) live in heatready_downscaling.licensing and
                # are enforced by validate_manifest below, so the monthly
                # re-scoring path that reads merged manifests off disk
                # without re-running jsonschema is covered too.
                "data_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "license", "reproducible_fetch", "redistribution_tier"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "version": {"type": "string"},
                            "license": {"type": "string", "minLength": 1},
                            "license_note": {"type": "string"},
                            "licensor": {"type": "string"},
                            "attribution_required": {"type": "boolean"},
                            "redistribution_tier": {
                                "enum": list(_licensing.REDISTRIBUTION_TIERS),
                            },
                            "reproducible_fetch": {"type": "string", "minLength": 1},
                            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                        },
                    },
                },
                # 2026-08-25, Rung B scoring extension (docs/plan-2026-08-25-
                # crowdsourced-model-improvement-p0.md): the actual DECLARED
                # value a Rung B submission proposes -- same shape
                # score.score_band's own proposed_correction param takes per
                # target, one level up (keyed by target here since a single
                # manifest's claims[0].targets can list both). Required when
                # rung=="B", disallowed otherwise (checked in validate_manifest
                # below, not in this schema -- same "cross-field checks live in
                # Python" convention _TOLERANCE_MAXIMA's own ceiling check
                # already uses here, not a jsonschema if/then).
                "candidate": {
                    "type": "object",
                    # additionalProperties: False (code-review finding, PR #24) --
                    # without this, a mistyped target key (e.g. "Tmax") passes
                    # schema validation silently and is simply never read by
                    # reproduce()'s per-target .get(target) lookup, so the
                    # contributor's declared correction goes unscored with no
                    # error surfaced anywhere. validate_manifest below adds a
                    # SECOND, stronger check: not just "no unknown keys" but
                    # "every claimed (target, zone) actually has an entry."
                    "additionalProperties": False,
                    "properties": {
                        target_key: {"type": "object", "additionalProperties": _CANDIDATE_ZONE_SCHEMA}
                        for target_key in ("tmax", "tmin")
                    },
                },
            },
        },
        "claimed_report": {"type": "string", "minLength": 1},
        # Every value must be > 0 (a zero-or-negative tolerance can never
        # pass, which is never the intent) and <= its documented ceiling.
        "tolerance": {
            "type": "object",
            "additionalProperties": {"type": "number", "exclusiveMinimum": 0},
        },
        "reproducibility": {
            "type": "object",
            "properties": {
                "seed": {"type": ["integer", "null"]},
                "runtime_notes": {"type": ["string", "null"]},
            },
        },
    },
}


def validate_manifest(manifest: dict, *, check_licensing: bool = True) -> None:
    """Raises jsonschema.ValidationError on a structurally invalid
    manifest, or ValueError if a `tolerance` value exceeds its documented
    ceiling (a check jsonschema's own vocabulary can't express per-key
    against a table like _TOLERANCE_MAXIMA), or if `method.candidate`'s
    presence doesn't match `rung` (required for Rung B, disallowed
    otherwise -- see MANIFEST_SCHEMA's own comment on `candidate`), or
    licensing.LicensingError if any declared data source or extra covariate
    fails the SPDX allowlist / named-licensor rules (see
    heatready_downscaling.licensing, which implements a control CONTRIBUTING.md
    documented long before anything enforced it)."""
    import jsonschema

    jsonschema.validate(manifest, MANIFEST_SCHEMA)

    rung = manifest.get("rung")
    candidate = manifest.get("method", {}).get("candidate")
    if rung == "B" and not candidate:
        raise ValueError(
            "rung 'B' requires a non-empty method.candidate (the actual bias_correction_c/"
            "scale+offset value(s) being proposed) -- a Rung B submission with no declared "
            "value has nothing for the referee to score",
        )
    if rung != "B" and candidate:
        raise ValueError(
            f"method.candidate is only meaningful for rung 'B', got rung {rung!r} -- "
            "a Rung A submission (evaluation coverage only) proposes no correction of its own",
        )
    if rung == "B":
        # Run score_band's own runtime validator here too, so a manifest that
        # is schema-valid but semantically impossible becomes a readable hard
        # reject at lint time instead of an uncaught ValueError deep inside
        # run_submission.py's reproduce() call (code-review finding, PR #27).
        # JSON Schema cannot express "valid_range has one entry per term",
        # "valid_range bounds are ordered", or "no covariate appears twice",
        # so all three passed validation and then crashed the referee.
        for target_key, by_zone in (candidate or {}).items():
            for zone_key, entry in (by_zone or {}).items():
                if all(k in entry for k in _score.PROPOSED_CORRECTION_COVARIATE_LINEAR_KEYS):
                    _score.validate_covariate_linear_entry(entry, f"{target_key}/{zone_key}")
        # Coverage check (Codex adversarial review finding, PR #24): every
        # (target, zone) this manifest CLAIMS must have its own candidate
        # entry -- not just "candidate is non-empty." Without this, a
        # submission could declare a value for one claimed cell and leave
        # others uncovered; score_forward_eval.py treats an uncovered cell's
        # proposed_correction_entry as None, silently falling back to a
        # Rung-A-style mechanical fit -- letting a contributor win/get
        # credit on a cell they never actually proposed a value for.
        claim = manifest["claims"][0]
        missing_cells = [
            (target, zone) for target in claim["targets"] for zone in claim["zones"]
            if zone not in (candidate.get(target) or {})
        ]
        if missing_cells:
            raise ValueError(
                f"method.candidate is missing an entry for claimed cell(s) {missing_cells!r} -- "
                "a Rung B submission must declare a value for EVERY (target, zone) it claims",
            )

    tolerance = manifest.get("tolerance", {})
    too_loose = {
        metric: (value, _TOLERANCE_MAXIMA[metric])
        for metric, value in tolerance.items()
        if metric in _TOLERANCE_MAXIMA and value > _TOLERANCE_MAXIMA[metric]
    }
    if too_loose:
        raise ValueError(
            "manifest tolerance exceeds the maximum allowed for one or more metrics "
            f"(metric: (claimed, max_allowed)): {too_loose} -- a tolerance this loose would let "
            "an incorrect claim pass as 'reproduced'"
        )

    # Licensing, for every data input beyond the published snapshot. Raises
    # licensing.LicensingError (a ValueError) on a violation.
    #
    # check_licensing=False exists for ONE caller, score_forward_eval.py, and
    # the reason is a code-review finding on PR #31 that corrected my own
    # original justification for putting this here at all. I argued the check
    # had to run in validate_manifest because the monthly cycle reads merged
    # manifests off disk without re-running jsonschema, so a merge-time-only
    # rule would not bind the official cycle. That over-reached. Licensing is
    # an ADMISSION decision: it belongs at the door, where the referee can
    # turn a violation into a readable rejection on the contributor's own PR.
    # Re-litigating it at scoring time is actively harmful, because
    # score_forward_eval wraps this call in `except Exception: continue` (by
    # design -- one bad manifest must not take down a cron run for every
    # other cell). So tightening SPDX_ALLOWLIST later, or a manifest merged
    # under the old nonexistent gate, would SILENTLY drop an already-admitted
    # candidate from the monthly cycle with nothing but a log warning: the
    # cell loses its active candidate and nobody is rejected or notified.
    # Admission stays strict; scoring does not re-open it.
    if check_licensing:
        _licensing.check_manifest_licensing(manifest)


def parse_submission_id(submission_id: str) -> tuple[str, int]:
    """Split "2026-08-001" into ("2026-08", 1) -- the (cycle_month, seq)
    pair used both for the submissions/{YYYY-MM}/{NNN}-... directory
    convention and for uniqueness checks against the ledger."""
    year_month, seq = submission_id.rsplit("-", 1)
    return year_month, int(seq)
