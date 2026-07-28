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
            "type": "array", "minItems": 1,
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
                            # SPDX identifier, or the "proprietary-licensed" escape hatch
                            # (CONTRIBUTING.md's own research-track section) requiring a
                            # named licensor -- enforced by CI (allowlist + HEAD check),
                            # not by this schema alone.
                            "license": {"type": "string"},
                            "global": {"type": "boolean"},
                            "cadence": {"type": "string"},
                            "reproducible_fetch": {"type": "string"},
                        },
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


def validate_manifest(manifest: dict) -> None:
    """Raises jsonschema.ValidationError on a structurally invalid
    manifest, or ValueError if a `tolerance` value exceeds its documented
    ceiling (a check jsonschema's own vocabulary can't express per-key
    against a table like _TOLERANCE_MAXIMA)."""
    import jsonschema

    jsonschema.validate(manifest, MANIFEST_SCHEMA)

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


def parse_submission_id(submission_id: str) -> tuple[str, int]:
    """Split "2026-08-001" into ("2026-08", 1) -- the (cycle_month, seq)
    pair used both for the submissions/{YYYY-MM}/{NNN}-... directory
    convention and for uniqueness checks against the ledger."""
    year_month, seq = submission_id.rsplit("-", 1)
    return year_month, int(seq)
