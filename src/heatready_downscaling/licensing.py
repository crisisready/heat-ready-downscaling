"""
Licensing checks for any data a submission uses beyond the published snapshot.

WHY THIS FILE EXISTS, stated plainly because it is a correction rather than a
new feature: CONTRIBUTING.md has told contributors since July that "CI enforces
`extra_covariates[].license` against an allowlist of SPDX identifiers plus a
`proprietary-licensed` escape hatch that requires a named licensor and flags
the submission for manual review", and submission.py's own schema comment
repeated it ("enforced by CI (allowlist + HEAD check), not by this schema
alone"). Neither was true. There was no allowlist, no escape-hatch check, no
HEAD check, and no license logic in any workflow -- the field was typed as a
bare string, so any license text at all passed, including a proprietary or
non-redistributable one. Nothing wrong was actually ingested (the only merged
submission declares no extra covariates), but a documented control that does
not exist is worse than an admitted gap, particularly in the externally-facing
repo of a program whose whole premise is that its gates are real and publicly
inspectable. This module is that promise, implemented.

Two checkable surfaces, one shared rule set:

  method.extra_covariates[]  -- the research track's new-model-covariate path,
                                which CONTRIBUTING.md already documents.
  method.data_sources[]      -- any data input beyond the published snapshot,
                                per the roadmap's standing data-sourcing
                                policy. heatready_downscaling.coastline's
                                NATURAL_EARTH_COASTLINE_DATA_SOURCE is the
                                first real entry, written before this checker
                                existed; its key names are what this module
                                now formalises.

Split by what needs the network, deliberately. Everything here is OFFLINE and
pure, so validate_manifest can call it and so score_forward_eval.py's monthly
re-scoring -- which reads merged manifests straight off disk without
re-running jsonschema -- is covered by the same rules. The
`reproducible_fetch` reachability check lives in scripts/check_data_licensing.py
because it makes real HTTP requests, and a validator that silently depends on
the network is a validator that fails for the wrong reasons.
"""

# SPDX identifiers a submission may declare for data it brings. Deliberately
# short and deliberately restricted to licenses that permit redistribution
# without a copyleft obligation we cannot honour for a public snapshot: this
# program republishes contributed data as part of a Zenodo-DOI'd artifact
# under CC BY 4.0 (see DATA_LICENSE), so a license that forbids
# redistribution, or requires downstream share-alike we do not apply, cannot
# be accepted silently.
#
# NC (non-commercial) variants are excluded on purpose. HeatReady's own data
# license carries no such restriction, so accepting NC data would make the
# published snapshot's stated terms wrong for part of its own contents.
SPDX_ALLOWLIST = frozenset({
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-3.0",
    "ODbL-1.0",
    "ODC-BY-1.0",
    "PDDL-1.0",
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "OGL-UK-3.0",
})

# The escape hatch CONTRIBUTING.md promises. It is NOT a way around the
# allowlist -- it is a way to declare data we hold a real license for that has
# no SPDX identifier, which is common for municipal and national-agency
# agreements. It requires a named licensor so the claim is attributable, and
# it always flags the submission for human review rather than passing
# silently.
PROPRIETARY_LICENSE_ID = "proprietary-licensed"

# What a data_sources entry must carry. redistribution_tier is the field that
# actually decides whether contributed data can enter a published snapshot at
# all, so it is required rather than inferred from the license string.
REDISTRIBUTION_TIERS = ("unrestricted", "attribution-required", "no-redistribution")

_REQUIRED_DATA_SOURCE_KEYS = ("name", "license", "reproducible_fetch", "redistribution_tier")


class LicensingError(ValueError):
    """A licensing rule a submission fails. Raised, not returned, so a caller
    that forgets to check a return value cannot accidentally admit data."""


def check_license_id(license_id, *, where: str, licensor=None) -> bool:
    """Validate one license declaration. Returns True when the entry needs
    human review (the proprietary path), False when it clears automatically.

    Raises LicensingError otherwise -- including for a license that merely
    looks plausible. Case is significant: SPDX identifiers are
    case-sensitive, and quietly accepting "cc-by-4.0" for "CC-BY-4.0" would
    make the allowlist advisory rather than a gate.
    """
    if not isinstance(license_id, str) or not license_id.strip():
        raise LicensingError(f"{where}: license is missing or empty")
    # Compared stripped (code-review finding, PR #31): a trailing space inside
    # YAML quotes is an easy typo, and comparing the RAW value meant
    # "CC-BY-4.0 " missed both the allowlist AND the case-insensitive
    # near-miss hint, producing the full allowed-list dump for the most
    # likely mistake there is.
    license_id = license_id.strip()

    if license_id == PROPRIETARY_LICENSE_ID:
        if not (isinstance(licensor, str) and licensor.strip()):
            raise LicensingError(
                f"{where}: license '{PROPRIETARY_LICENSE_ID}' requires a non-empty "
                "'licensor' naming who granted it. The escape hatch exists for real "
                "agreements without an SPDX identifier, not as a way past the allowlist, "
                "so the claim has to be attributable to somebody.",
            )
        return True

    if license_id not in SPDX_ALLOWLIST:
        near = sorted(
            candidate for candidate in SPDX_ALLOWLIST
            if candidate.lower() == license_id.lower()
        )
        hint = (
            f" Did you mean {near[0]!r}? SPDX identifiers are case-sensitive."
            if near else
            f" Allowed: {', '.join(sorted(SPDX_ALLOWLIST))}, or "
            f"'{PROPRIETARY_LICENSE_ID}' with a named licensor."
        )
        raise LicensingError(f"{where}: license {license_id!r} is not on the allowlist.{hint}")
    return False


def check_data_source(entry: dict, *, where: str) -> bool:
    """Validate one method.data_sources entry. Returns True when it needs
    human review."""
    if not isinstance(entry, dict):
        raise LicensingError(f"{where}: data_sources entry must be an object, got {type(entry).__name__}")

    missing = [key for key in _REQUIRED_DATA_SOURCE_KEYS if not entry.get(key)]
    if missing:
        raise LicensingError(f"{where}: data_sources entry is missing {missing}")

    tier = entry["redistribution_tier"]
    if tier not in REDISTRIBUTION_TIERS:
        raise LicensingError(
            f"{where}: redistribution_tier {tier!r} is not one of {REDISTRIBUTION_TIERS}",
        )
    # no-redistribution data cannot enter a published snapshot, which is the
    # whole point of asking. It is not rejected outright -- a local model may
    # legitimately train on data we cannot republish -- but it can never pass
    # without a human deciding, so it is always flagged.
    needs_review = check_license_id(
        entry["license"], where=where, licensor=entry.get("licensor"),
    )
    if tier == "no-redistribution":
        needs_review = True

    if entry.get("attribution_required") is True and tier == "unrestricted":
        raise LicensingError(
            f"{where}: attribution_required is true but redistribution_tier is "
            "'unrestricted' -- these contradict each other, and the snapshot's own "
            "attribution notices are generated from the tier",
        )
    return needs_review


def check_extra_covariate(entry: dict, *, where: str) -> bool:
    """Validate one method.extra_covariates entry's licensing. The research
    track's own additional rules (global coverage, a documented fetch) are
    already enforced by the manifest schema; this covers the licensing half
    CONTRIBUTING.md promised and nothing enforced."""
    if not isinstance(entry, dict):
        raise LicensingError(f"{where}: extra_covariates entry must be an object")
    return check_license_id(
        entry.get("license"), where=where, licensor=entry.get("licensor"),
    )


def _entries(method: dict, key: str) -> list:
    """The list under method[key], or [] -- never something that iterates
    surprisingly. A mapping here would otherwise iterate its KEYS and check
    strings as if they were entries (code-review finding, PR #31), and
    check_manifest_licensing is reachable from two paths that never ran
    jsonschema first."""
    value = method.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise LicensingError(
            f"method.{key} must be a list, got {type(value).__name__}",
        )
    return value


def audit_manifest_licensing(manifest: dict) -> tuple[list[str], list[str]]:
    """(violations, needs_review) for every licensing-relevant entry.

    Collects rather than raising, for two reasons the first version got wrong
    (code-review findings, PR #31). Raising on the FIRST violation discarded
    the needs-review list entirely, so a manifest with one bad SPDX id plus a
    proprietary-licensed entry never printed the entry that is supposed to
    "always reach a human" -- and a contributor with three bad licences
    learned about them one CI run at a time.

    Structural garbage still raises: a non-dict `method`, or a mapping where a
    list belongs, is not a licensing verdict and must not be reported as one.
    """
    if not isinstance(manifest, dict):
        raise LicensingError(
            f"manifest must be an object, got {type(manifest).__name__}",
        )
    method = manifest.get("method")
    if method is None:
        return [], []
    if not isinstance(method, dict):
        raise LicensingError(
            f"manifest.method must be an object, got {type(method).__name__}",
        )

    violations: list[str] = []
    flagged: list[str] = []

    checks = (
        ("data_sources", check_data_source),
        ("extra_covariates", check_extra_covariate),
    )
    for key, check in checks:
        for i, entry in enumerate(_entries(method, key)):
            where = f"method.{key}[{i}]"
            try:
                needs_review = check(entry, where=where)
            except LicensingError as exc:
                violations.append(str(exc))
                continue
            if needs_review:
                name = entry.get("name") if isinstance(entry, dict) else None
                license_id = entry.get("license") if isinstance(entry, dict) else None
                flagged.append(f"{where} ({name}): {license_id}")

    return violations, flagged


def check_manifest_licensing(manifest: dict) -> list[str]:
    """audit_manifest_licensing, but raising on any violation -- the form
    validate_manifest wants. Reports EVERY violation at once rather than the
    first, so one round trip tells a contributor everything that is wrong.
    Returns the needs-review list when nothing is violated."""
    violations, flagged = audit_manifest_licensing(manifest)
    if violations:
        raise LicensingError(
            "licensing violation(s):\n  - " + "\n  - ".join(violations),
        )
    return flagged
