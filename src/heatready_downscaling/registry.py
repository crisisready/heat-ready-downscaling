"""
The model registry: one manifest per model, git-versioned, in this repository.

Roadmap Phase 1 (`research/crowdsourced-modeling/ROADMAP.md` in the private
`crisisready/heat-risk-data-api` repo). A "model" here is a REGISTRY ENTRY,
not a file: an entry declares identity and lineage, the exact cells it claims,
the evidence backing each claimed cell, the artifacts it needs, the data
sources it used beyond the published snapshot, and how it compiles into
something the serving side can actually apply.

WHY A REGISTRY AT ALL. Today the 342-cell evidence matrix exists only as gate
JSON on S3 plus prose scattered across two repositories' research directories.
Nothing machine-readable says "this model claims these cells, on this
evidence". That is why a question as basic as "what does Seoul's model
actually claim, and how well" cannot be answered without reading a research
thread. The registry makes that one file per model, reviewed as a PR, with git
history as the audit trail.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

It does not execute anything. A manifest is a declaration; `GOVERNANCE.md`'s
"no contributor Python executes anywhere in v1" is untouched.

It does not decide promotion. `status` is a record of where an entry has got
to, and moving it is a reviewed PR like anything else -- the registry is the
place a promotion is written DOWN, never the place one is granted.

STATUS IS APPEND-ONLY, and the history is the point rather than the current
value: a reader asking "was this ever serving, and when did it stop"
should get an answer from the file, not from `git log`. The same reasoning and
the same check shape as `ledger.py`'s append-only credit lines.
"""

import re

REGISTRY_SCHEMA_VERSION = 1

# `local/valencia-coast-v1`, `global/ds-2026.07-rf5`. The prefix is not
# decoration: it separates the one model that must have a defined fail-closed
# behaviour EVERYWHERE (global) from models whose claims are geographically
# bounded (local), and the serving-side menu treats those differently.
MODEL_ID_PATTERN = r"^(global|local)/[a-z0-9][a-z0-9._-]{2,63}$"

# Where an entry has got to. Ordered, and the order is meaningful: an entry
# moves forward through these, and `retired` is reachable from any of them.
STATUSES = ("registered", "validated", "serving", "retired")

# How an entry turns into something the serving side applies. The vocabulary
# of METHODS is deliberately open (a manifest's method.kind is free text --
# the record already holds bias constants, zone affines, covariate-linear
# corrections, blend kernels, base-source variants, local retrains); the
# vocabulary of SERVING PRIMITIVES is deliberately closed, because adding one
# is the only thing here that touches core serving code.
SERVING_PRIMITIVES = (
    "zone_params",
    "polygon_params",
    "artifact_route",
    "base_variant",
    "output_postprocess",
)

# How a claimed cell's evidence was held out. A first-class enumerated field
# rather than prose, because the three designs are not interchangeable and a
# reader comparing two entries needs to know which one produced the number.
#
# station_salted_fold  -- the referee's own salted station-grouped CV.
# spatial_holdout      -- held-out LOCATIONS, never seen in training. Seoul's
#                         63 held-out dong. The right design for a dense local
#                         network, and NOT expressible as a salted fold.
# forward_only_cycle   -- scored against station-days that did not exist at
#                         submission time. The only uncheatable holdout, and
#                         only available where the underlying source is public
#                         and growing (GHCN-Daily). A dense private sensor
#                         network has no equivalent.
HOLDOUT_DESIGNS = ("station_salted_fold", "spatial_holdout", "forward_only_cycle")

# What kind of ground truth a claimed cell's evidence rests on. From the Rung D
# design (docs/design-2026-08-25-rung-d-contributed-data.md): the leaderboard's
# value comes from these categories staying distinct, so a reader can tell
# "anyone can check this" from "we checked it" without reading the manifest.
#
# Recorded here at schema-definition time rather than bolted on when Rung D is
# implemented, so entries registered now do not need a migration later.
EVIDENCE_PROVENANCE = (
    "publicly_reproducible",   # a third party can obtain the data and recompute
    "maintainer_attested",     # re-derived by us against data they cannot obtain
    "consumer_derived",        # rests on consumer-network data (see the Rung D bar)
    "mixed",
)


def _cell_schema() -> dict:
    return {
        "type": "object",
        "required": ["target", "zone", "evidence"],
        "additionalProperties": False,
        "properties": {
            "target": {"enum": ["tmax", "tmin"]},
            "zone": {"type": "string", "minLength": 1},
            # Nullable, and this was found by doing the retroactive
            # registrations rather than reasoned in advance -- which is what
            # the roadmap says that exercise is for. A band-keyed CORRECTION
            # (zone_params, polygon_params) is meaningless without a band: it
            # recalibrates one base distribution. An artifact-routed MODEL is
            # not a per-band correction at all; it produces rows in a
            # project's daily update across whatever bands that update
            # covers, so forcing a band onto it would have made me invent
            # one. validate_manifest requires it for the former and permits
            # null for the latter.
            "band": {"type": ["string", "null"], "minLength": 1},
            # Where this claim applies, when it is narrower than the zone.
            # Free text by design -- a country prefix, a project id, a city
            # name -- because the serving side resolves geography per row and
            # this field is documentation for a human reviewer, not a lookup
            # key. Making it a lookup key is what broke the subzone mechanism
            # (see promote_from_public.py's own docstring).
            "geography": {"type": ["string", "null"]},
            "evidence": {
                "type": "object",
                "required": ["metric", "value", "holdout_design", "provenance"],
                "additionalProperties": False,
                "properties": {
                    "metric": {"type": "string", "minLength": 1},
                    "value": {"type": "number"},
                    # [lo, hi]. Null when no interval was computed, which is
                    # itself informative -- an entry claiming a cell with no
                    # interval is making a weaker claim than one that does,
                    # and the reader should see which.
                    "ci95": {
                        "type": ["array", "null"],
                        "minItems": 2, "maxItems": 2,
                        "items": {"type": "number"},
                    },
                    "n_stations": {"type": ["integer", "null"], "minimum": 0},
                    "n_clusters": {"type": ["integer", "null"], "minimum": 0},
                    "stratum": {"type": ["string", "null"]},
                    "holdout_design": {"enum": list(HOLDOUT_DESIGNS)},
                    "provenance": {"enum": list(EVIDENCE_PROVENANCE)},
                    # Path to the report this number came from, REPO-RELATIVE,
                    # plus its sha256. Repo-relative rather than
                    # entry-relative so an entry can cite evidence that
                    # already exists -- the merged submission's own
                    # claimed_report.json, say -- instead of duplicating it
                    # into registry/ where the copy can silently drift from
                    # the original. A claim whose evidence cannot be located
                    # is a claim nobody can check.
                    "report": {"type": ["string", "null"]},
                    "report_sha256": {"type": ["string", "null"], "pattern": "^[a-f0-9]{64}$"},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
    }


def manifest_schema() -> dict:
    """JSON Schema for a registry manifest. A function rather than a module
    constant so `heatready_downscaling.licensing`'s vocabularies are read at
    call time and cannot drift from a copy frozen at import."""
    from heatready_downscaling import licensing

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "registry_schema_version", "model_id", "version", "authors",
            "method", "claims", "status_history",
        ],
        "additionalProperties": False,
        "properties": {
            "registry_schema_version": {"const": REGISTRY_SCHEMA_VERSION},
            "model_id": {"type": "string", "pattern": MODEL_ID_PATTERN},
            "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "authors": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "github": {"type": ["string", "null"]},
                        "orcid": {"type": ["string", "null"]},
                        "affiliation": {"type": ["string", "null"]},
                    },
                },
            },
            # What this entry descends from. A local model retrained from the
            # global one, or a v2 superseding a v1, should say so -- otherwise
            # the registry records a set of unrelated models rather than a
            # lineage.
            "lineage": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "derived_from": {"type": ["string", "null"]},
                    "supersedes": {"type": ["string", "null"]},
                    "note": {"type": ["string", "null"]},
                },
            },
            "method": {
                "type": "object",
                "required": ["kind", "compile_to"],
                "additionalProperties": False,
                "properties": {
                    # OPEN vocabulary, deliberately. See this module's own
                    # docstring: the record already holds at least eight
                    # distinct improvement shapes and the system must admit
                    # ones nobody has invented yet.
                    "kind": {"type": "string", "minLength": 1},
                    "compile_to": {"enum": list(SERVING_PRIMITIVES)},
                    "summary": {"type": ["string", "null"]},
                    # Reuses the licensing gate's own block (#31) rather than
                    # inventing a second provenance vocabulary -- and so a
                    # registry entry's data sources are held to exactly the
                    # rules a submission's are.
                    "data_sources": {"type": "array"},
                },
            },
            # Required for compile_to == artifact_route, checked in
            # validate_manifest rather than here: an artifact-routed model
            # that does not pin its feature order cannot be safely loaded,
            # and contract.validate_feature_order is the existing precedent.
            "feature_contract": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "sha256"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                        "size_bytes": {"type": ["integer", "null"], "minimum": 0},
                        "release_url": {"type": ["string", "null"]},
                        "doi": {"type": ["string", "null"]},
                        "s3_uri": {"type": ["string", "null"]},
                    },
                },
            },
            "claims": {"type": "array", "minItems": 1, "items": _cell_schema()},
            "status_history": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["status", "at"],
                    "additionalProperties": False,
                    "properties": {
                        "status": {"enum": list(STATUSES)},
                        "at": {"type": "string", "minLength": 4},
                        "by": {"type": ["string", "null"]},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


class RegistryError(ValueError):
    """A registry rule a manifest fails."""


def current_status(manifest: dict) -> str:
    """The entry's status now: the last line of its append-only history."""
    return manifest["status_history"][-1]["status"]


def validate_manifest(
    manifest: dict, *, model_dir: str | None = None, repo_root: str | None = None,
) -> None:
    """Raise on a manifest that is structurally invalid, internally
    inconsistent, or that breaks a rule JSON Schema cannot express.

    `model_dir` enables the path/id consistency check; `repo_root` enables the
    evidence-file checks (every claimed cell's report exists and matches its
    recorded sha256). Both are skipped rather than faked when absent, so a
    caller validating a manifest in memory gets the structural half without a
    misleading pass on the half it cannot check.
    """
    import jsonschema

    from heatready_downscaling import licensing

    jsonschema.validate(manifest, manifest_schema())

    # The directory name has to be the model_id, or `registry/<model_id>/`
    # stops being addressable and two entries can silently claim one id.
    if model_dir is not None:
        import os

        expected = manifest["model_id"].replace("/", os.sep)
        if not os.path.normpath(model_dir).endswith(expected):
            raise RegistryError(
                f"manifest declares model_id {manifest['model_id']!r} but lives at "
                f"{model_dir!r} -- the path must be registry/<model_id>/",
            )

    # Licensing, through the gate that already exists rather than a second
    # copy of the rules (#31). A registry entry declaring a data source is
    # held to exactly what a submission declaring one is held to.
    licensing.check_manifest_licensing({"method": manifest.get("method") or {}})

    # An artifact-routed entry must pin its feature order and name a real
    # artifact -- but only once it claims to be more than a record.
    #
    # This requirement is scaled to status because of what happened when it
    # was not. Registering Seoul retroactively, its artifact's sha256 and
    # feature order are simply not recorded anywhere in either repository, so
    # an unconditional requirement forced me to write a PLACEHOLDER feature
    # name and an all-zero hash -- fabricated data satisfying a check, which
    # is worse than an absent check because it looks like evidence. An entry
    # at `registered` is a record of something that exists; the artifact
    # requirements bind at `validated` and beyond, where the entry starts
    # asserting it could be loaded.
    status = current_status(manifest)
    if manifest["method"]["compile_to"] == "artifact_route" and status != "registered":
        if not manifest.get("feature_contract"):
            raise RegistryError(
                f"{manifest['model_id']}: compile_to 'artifact_route' at status {status!r} "
                "requires a feature_contract -- an artifact whose feature order is unrecorded "
                "cannot be safely loaded (see contract.validate_feature_order)",
            )
        if not manifest.get("artifacts"):
            raise RegistryError(
                f"{manifest['model_id']}: compile_to 'artifact_route' at status {status!r} "
                "requires at least one artifact -- there is nothing to route to",
            )

    # A sentinel hash is not a hash. Catching this matters more than it looks:
    # the failure mode is not a missing checksum, it is a checksum-shaped
    # string that passes a pattern check and then verifies against nothing.
    for artifact in manifest.get("artifacts") or []:
        if set(artifact["sha256"]) == {"0"}:
            raise RegistryError(
                f"{manifest['model_id']}: artifact {artifact['name']!r} has an all-zero "
                "sha256, which is a placeholder rather than a checksum. Omit the artifact "
                "until its real hash is known -- an entry at status 'registered' does not "
                "need one.",
            )

    _check_status_history(manifest)
    _check_claims(manifest)

    # A band-keyed correction recalibrates ONE base distribution, so a claim
    # without a band does not identify what it corrects.
    if manifest["method"]["compile_to"] in ("zone_params", "polygon_params"):
        missing = [
            f"{c['target']}/{c['zone']}" for c in manifest["claims"] if not c.get("band")
        ]
        if missing:
            raise RegistryError(
                f"{manifest['model_id']}: compile_to "
                f"{manifest['method']['compile_to']!r} requires a band on every claim -- a "
                f"band-keyed correction recalibrates one base distribution, so {missing} do "
                "not identify what they correct",
            )

    if repo_root is not None:
        _check_evidence_files(manifest, repo_root)


def _check_status_history(manifest: dict) -> None:
    """Append-only, chronological, and no repeats of the current status."""
    history = manifest["status_history"]
    seen_at = None
    for i, entry in enumerate(history):
        if seen_at is not None and entry["at"] < seen_at:
            raise RegistryError(
                f"{manifest['model_id']}: status_history is out of order at index {i} "
                f"({entry['at']} follows {seen_at}) -- the history is the audit trail, so its "
                "order has to be real",
            )
        seen_at = entry["at"]
    statuses = [e["status"] for e in history]
    if statuses[0] != "registered":
        raise RegistryError(
            f"{manifest['model_id']}: status_history must begin with 'registered', got "
            f"{statuses[0]!r}",
        )
    for i in range(1, len(statuses)):
        if statuses[i] == statuses[i - 1]:
            raise RegistryError(
                f"{manifest['model_id']}: status_history repeats {statuses[i]!r} at index {i} "
                "-- a status line records a TRANSITION, so a repeat says nothing",
            )
    if "retired" in statuses[:-1]:
        raise RegistryError(
            f"{manifest['model_id']}: 'retired' appears before the end of status_history -- "
            "an entry does not come back from retired under the same id; register a new one",
        )


def _check_claims(manifest: dict) -> None:
    """No duplicate cells, and provenance consistent with the entry's data."""
    seen: set[tuple] = set()
    for claim in manifest["claims"]:
        key = (claim["target"], claim["zone"], claim["band"], claim.get("geography"))
        if key in seen:
            raise RegistryError(
                f"{manifest['model_id']}: duplicate claim for {key} -- two evidence blocks for "
                "one cell means the entry does not say which number it stands behind",
            )
        seen.add(key)


def _check_evidence_files(manifest: dict, repo_root: str) -> None:
    """Every recorded evidence report exists and matches its sha256."""
    import hashlib
    import os

    for claim in manifest["claims"]:
        evidence = claim["evidence"]
        rel = evidence.get("report")
        if not rel:
            continue
        path = os.path.join(repo_root, rel)
        if not os.path.exists(path):
            raise RegistryError(
                f"{manifest['model_id']}: claim {claim['target']}/{claim['zone']}/"
                f"{claim['band']} cites evidence at {rel!r}, which does not exist",
            )
        recorded = evidence.get("report_sha256")
        if not recorded:
            continue
        with open(path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != recorded:
            raise RegistryError(
                f"{manifest['model_id']}: evidence file {rel!r} has sha256 {actual}, manifest "
                f"records {recorded} -- the evidence changed after the claim was written",
            )


def load_manifest(model_dir: str, *, repo_root: str | None = None) -> dict:
    """Read and validate `<model_dir>/manifest.yaml`."""
    import os

    import yaml

    with open(os.path.join(model_dir, "manifest.yaml")) as fh:
        manifest = yaml.safe_load(fh)
    validate_manifest(manifest, model_dir=model_dir, repo_root=repo_root)
    return manifest


def iter_registry(registry_dir: str = "registry"):
    """Every entry under `registry/`, as (model_dir, manifest), validated."""
    import glob
    import os

    for path in sorted(glob.glob(os.path.join(registry_dir, "*", "*", "manifest.yaml"))):
        model_dir = os.path.dirname(path)
        yield model_dir, load_manifest(model_dir, repo_root=os.path.dirname(os.path.abspath(registry_dir)))
