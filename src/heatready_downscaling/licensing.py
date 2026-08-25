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
pure, so validate_manifest can call it. The `reproducible_fetch` reachability
check lives in scripts/check_data_licensing.py because it makes real HTTP
requests, and a validator that silently depends on the network is a validator
that fails for the wrong reasons.

WHERE THESE RULES DO AND DO NOT APPLY, stated precisely because getting this
wrong is the very thing this module exists to correct. They are an ADMISSION
gate: enforced by validate_manifest, which the referee runs on a submission's
own PR, and by the Data licensing workflow. They are deliberately NOT applied
by score_forward_eval.py's monthly cycle, which passes check_licensing=False --
see validate_manifest's own comment for why re-litigating admission at scoring
time would silently drop already-admitted candidates.

The practical consequence, so nobody has to infer it: a licence hand-edited
into a manifest AFTER merge is not caught by the official cycle. It is caught
by the review of the PR making that edit, which is the only place a human sees
it. An earlier draft of this docstring claimed the monthly cycle was covered.
It was true when written and false after the round-2 fix, and leaving it would
have documented a control this module no longer applies -- exactly the failure
described above.
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
    "ODC-BY-1.0",
    "PDDL-1.0",
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "OGL-UK-3.0",
})

# Licenses that are real and usable but cannot AUTO-pass, because honouring
# them requires a decision this gate cannot make on its own. They route to a
# maintainer exactly like the proprietary escape hatch does.
#
# ODbL-1.0 is the motivating case (code-review finding, PR #31 round 2): it is
# share-alike for databases -- a derivative database must itself be offered
# under ODbL -- while DATA_LICENSE publishes contributed data as CC BY 4.0
# with an explicit "No additional restrictions" clause. Auto-passing ODbL
# would admit data and then republish it on terms that conflict with its own
# licence. That is a licensing decision with legal weight, not a checkbox, and
# the allowlist comment three lines up already said the list excludes copyleft
# we cannot honour -- ODbL was on it anyway.
SPDX_NEEDS_REVIEW = frozenset({
    "ODbL-1.0",
    "CC-BY-SA-4.0",
    "CC-BY-SA-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
})

# SPDX ids that carry an attribution obligation. A source under one of these
# cannot be declared redistribution_tier: unrestricted, because the snapshot
# would then republish it with no attribution notice.
SPDX_ATTRIBUTION_REQUIRED = frozenset({
    "CC-BY-4.0",
    "CC-BY-3.0",
    "ODC-BY-1.0",
    "OGL-UK-3.0",
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
})

# The escape hatch CONTRIBUTING.md promises. It is NOT a way around the
# allowlist -- it is a way to declare data we hold a real license for that has
# no SPDX identifier, which is common for municipal and national-agency
# agreements. It requires a named licensor so the claim is attributable, and
# it always flags the submission for human review rather than passing
# silently.
PROPRIETARY_LICENSE_ID = "proprietary-licensed"

# What a data_sources entry must carry. redistribution_tier is required rather
# than inferred from the licence string because it records a DECISION about
# what may be republished, and the licence alone does not determine that for
# dual-licensed or partially-restricted sources.
#
# Note what does NOT yet exist, since an earlier version of this module
# asserted it: nothing reads redistribution_tier or attribution_required
# outside this module and coastline.py. There is no code generating the
# snapshot's attribution notices from the tier -- that is a real follow-up, and
# claiming it already worked was the same documented-but-absent-mechanism
# mistake this module was written to correct (code-review finding, PR #31
# round 2, the third instance in this PR alone).
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

    if license_id in SPDX_NEEDS_REVIEW:
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

    # The HARMFUL direction, which the first version left unguarded
    # (code-review finding, PR #31 round 2): a CC-BY-4.0 source declared
    # `unrestricted` with attribution_required omitted passed silently, so an
    # attribution-obligated dataset was admitted as needing none. The
    # previously-checked direction (declaring MORE attribution than the tier
    # implies) is the harmless one.
    license_id = (entry.get("license") or "").strip()
    if license_id in SPDX_ATTRIBUTION_REQUIRED and tier == "unrestricted":
        raise LicensingError(
            f"{where}: license {license_id!r} carries an attribution obligation, so "
            "redistribution_tier cannot be 'unrestricted' -- use "
            "'attribution-required' (see DATA_LICENSE, which carries LandScan/ORNL's "
            "attribution forward for exactly this reason)",
        )
    if entry.get("attribution_required") is True and tier == "unrestricted":
        raise LicensingError(
            f"{where}: attribution_required is true but redistribution_tier is "
            "'unrestricted' -- these contradict each other",
        )
    if entry.get("attribution_required") is False and license_id in SPDX_ATTRIBUTION_REQUIRED:
        raise LicensingError(
            f"{where}: attribution_required is false but license {license_id!r} requires "
            "attribution",
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
    # Every ITEM must be a mapping too, and this raises rather than being
    # collected as a per-entry violation, so the two categories stay crisp:
    # a SHAPE problem is "your file is malformed" (the caller reports it as
    # unreadable, exit 4), and a violation is "your licence is not
    # acceptable" (exit 1). A list of strings under data_sources is a YAML
    # mistake, not a licensing claim, and telling a contributor they have a
    # licensing violation when they have a mis-indentation is the mis-report
    # this whole exit-code split exists to prevent (code-review finding, PR
    # #31 round 2).
    bad = [i for i, entry in enumerate(value) if not isinstance(entry, dict)]
    if bad:
        raise LicensingError(
            f"method.{key} entries must be objects; entr{'ies' if len(bad) > 1 else 'y'} "
            f"{bad} {'are' if len(bad) > 1 else 'is'} not",
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
