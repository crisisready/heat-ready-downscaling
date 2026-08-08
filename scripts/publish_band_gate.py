"""
PROVENANCE: moved from crisisready/heat-risk-data-api's scripts/publish_band_gate.py at commit a31ec14863273904ae6a9de7a6c34cb77f84f4f4 (2026-07-27, Phase 1.4, plan section 5.4). See this repository's own PROVENANCE.md.

REFACTORED during the move: `build_gate` (previously defined in this file)
is now `heatready_downscaling.gates.build_gate` -- see gates.py's own
docstring. This CLI also now calls `heatready_downscaling.gates.
validate_gate` before printing/uploading, a real schema check the private
repo's original version never had.

Publish a validated band-specific zone gate to S3 for production to load
via downscaling.load_band_gate (crisisready/heat-risk-data-api).

Reads a scripts/validate_lagfill_downscaling.py / validate_forecast_
downscaling.py JSON report and extracts each (target, zone)'s
qrf_beats_grid_with_margin verdict -- the STRICTER, margin-gated bar (see
heatready_downscaling.score's own comment on AUTO_ENABLE_MARGIN), not the
plain qrf_beats_grid inequality -- into the {"tmax": {zone: bool}, "tmin":
{zone: bool}, "bias_correction": {"tmax": {zone: float}, "tmin": {zone:
float}}, "spatial_skill": {"tmax": {zone: bool}, "tmin": {zone: bool}}}
shape downscaling.load_band_gate expects at
s3://{bucket}/downscaling/band_gates/{model_version}/{band_key}.json.

bias_correction (MOS-style recalibration -- see score_band's own
docstring): for every zone that passes ONLY because of the debiased,
cross-validated check (rmse_improvement_pct_debiased_cv, not the raw
rmse_improvement_pct), the zone's bias_correction_c is published too --
downscaling.predict_downscaled adds this to delta_c at serving time, so
production actually applies the recentered correction the validation
proved out, not just a bare pass/fail flag. A zone that passes via the
raw (non-debiased) fallback path (bias_bounded_uncorrected -- too few
distinct stations for CV) publishes NO bias_correction entry: its
raw bias was already confirmed small enough not to need one.

spatial_skill (honest-labeling): True when the zone's RAW (uncorrected)
QRF delta already beats grid (qrf_beats_grid is True) -- genuine
neighborhood-resolution spatial downscaling skill, the per-polygon
signal itself does the work. False when a zone passes the published gate
ONLY via the debiased-CV margin (qrf_beats_grid is False but
qrf_beats_grid_with_margin is True) -- the entire improvement is
carried by a flat per-Köppen-zone MOS bias constant; the QRF's spatial
delta contributes nothing (or is net-negative) for that zone. Both are
legitimate, CV-validated, more-accurate-than-grid corrections worth
serving -- this field exists so nothing downstream (docs, a model-
performance page, a public claim) represents a spatial_skill=False zone
as evidence of neighborhood-resolution downscaling working there.
Originally did not change serving behavior at all (downscaling.
predict_downscaled never read it) -- **no longer true as of 2026-07-31**:
crisisready/heat-risk-data-api's downscaling.spatial_ranking_for_band now
reads this field to veto a separate confidence signal (spatial_ranking)
on non-ERA5 bands whose own gate doesn't show real spatial skill (see
that repo's PARIS_CONFIDENCE_ROADMAP.md, "R3 scoping" section). This
field is now load-bearing for that serving path, not purely disclosure.

Deliberately a separate, explicit publish step rather than the validation
script uploading its own result automatically -- a human (or a review-gated
CI step) reads the validation report and decides whether to publish before
anything in production can start reading it: a wrong-but-confident
correction should never ship without a human decision.

Fail-closed on zone drops (2026-08-03): before uploading, this script GETs
whatever gate is CURRENTLY published at the target key and refuses to
proceed (no S3 write, non-zero exit) if the new gate would drop any zone
(per target, tmax/tmin separately) present in that current gate but absent
from the new one. --variant scoping already prevents the *cross-key*
version of this landmine (a variant-fitted report can never silently
overwrite the default key -- see build_gate's own docstring), but says
nothing about a same-key publish that simply omits a zone the current gate
has -- e.g. a validation run scoped to fewer zones, or a zone that failed
this time. --confirm-drops opts out of the refusal for an intentional
drop, matching this script's own "human decision required" idiom above
rather than a silent auto-publish.

Only True verdicts are written explicitly; a zone with qrf_beats_grid_with_
margin False OR None (insufficient n, or gate-failed already at the ERA5
level) is simply omitted -- downscaling.load_band_gate's own .get(zone,
False) lookup treats "absent" identically to "explicitly False", so omitting
is equivalent to False and keeps the published file small and readable (a
quick glance shows exactly which zones are live, not a wall of False
entries).

Usage:
    python scripts/publish_band_gate.py \\
        --report /tmp/lagfill_full_report.json \\
        --model-version ds-2026.07-rf5 --band-key lag_fill \\
        --profile nish-climateverse
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from heatready_downscaling.gates import build_gate, build_subzone_patch, merge_subzone_patch, validate_gate


def _get_current_gate(client, bucket, key):
    """GET the gate currently published at (bucket, key), or {} if nothing
    is published there yet (NoSuchKey/404 -- a real, expected first-publish
    case, not an error). Any other S3 error is re-raised rather than
    silently treated as "nothing published" -- fail-closed applies to
    every caller of this helper, not just the zone-drop check it was
    originally written for."""
    from botocore.exceptions import ClientError

    try:
        existing_obj = client.get_object(Bucket=bucket, Key=key)
        return json.loads(existing_obj["Body"].read())
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {}
        raise


def _refuse_if_zones_would_be_dropped(client, bucket, key, new_gate, confirm_drops):
    """Compares the gate currently published at (bucket, key) against
    new_gate's tmax/tmin zone sets. Raises SystemExit(1) (no caller side
    effect yet -- this must run before put_object) if any zone present in
    the current gate is absent from new_gate, unless confirm_drops is True.
    No prior gate means nothing to drop, so this always proceeds on a
    first-ever publish to a key."""
    existing_gate = _get_current_gate(client, bucket, key)
    if not existing_gate:
        return

    dropped = {}
    for target in ("tmax", "tmin"):
        removed = sorted(set(existing_gate.get(target, {})) - set(new_gate.get(target, {})))
        if removed:
            dropped[target] = removed
    if not dropped:
        return

    if confirm_drops:
        for target, zones in dropped.items():
            print(f"--confirm-drops set: publishing anyway, dropping {target} zone(s) {zones} "
                  f"that are currently published at s3://{bucket}/{key}")
        return

    for target, zones in dropped.items():
        print(f"REFUSING to publish: would drop {target} zone(s) {zones} that are currently "
              f"published at s3://{bucket}/{key} but absent from the new gate. "
              f"Pass --confirm-drops if this is intentional.", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, help="path to a validate_*_downscaling.py JSON report")
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--band-key", required=True,
        choices=["lag_fill"] + [f"forecast_lead{n}" for n in range(1, 8)],
    )
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--profile", default=None, help="named AWS profile (omit on EC2 with an attached IAM role)")
    parser.add_argument("--dry-run", action="store_true", help="print the gate that would be published, don't upload")
    # 2026-08-03, gate-variant scoping (see downscaling.load_band_gate's own
    # docstring in heat-risk-data-api): must match the --report's own
    # stamped base_variant exactly (build_gate enforces this in BOTH
    # directions, unconditionally -- see its own docstring) -- there is no
    # way to bypass the check by omitting this flag.
    parser.add_argument("--variant", default=None,
                         help="publish to downscaling/band_gates/{model_version}/{band_key}__{variant}.json "
                              "instead of the default (no-variant) key -- for a gate fitted under an "
                              "alternate base distribution (e.g. --elevation-nan on the validation harness). "
                              "Must match the --report's own stamped base_variant exactly.")
    parser.add_argument("--confirm-drops", action="store_true",
                         help="required to proceed if the new gate would drop a zone (tmax or tmin) "
                              "that is present in the gate currently published at the target key. "
                              "Omit unless you specifically intend to remove a zone.")
    args = parser.parse_args()

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile

    with open(args.report) as f:
        report = json.load(f)

    bucket = args.bucket or os.environ.get("VULNERABILITY_DATA_BUCKET")
    key_band = f"{args.band_key}__{args.variant}" if args.variant else args.band_key
    key = f"downscaling/band_gates/{args.model_version}/{key_band}.json"

    # regions/exclude_regions are read from the report itself (stamped by
    # validate_lagfill_downscaling.py's --regions/--exclude-regions), not a
    # separate CLI flag here -- same reason --variant IS a separate flag
    # while zones is not: variant changes WHERE this publishes (a
    # different S3 key), so it needs an explicit, checkable-against-the-
    # report flag; regions changes WHAT KIND of publish this is (patch vs.
    # full gate), which the report already unambiguously says on its own.
    #
    # exclude_regions-WITHOUT-regions gets its own explicit, clean refusal
    # here (2026-08-08 review finding) rather than falling through to
    # _publish_subzone_patch, which would hit build_subzone_patch's own
    # "report has no regions" ValueError uncaught -- a real, live-reproduced
    # crash (raw traceback, not this CLI's usual SystemExit-with-message
    # idiom) for exactly the "score the rest of the zone as a regression
    # check" workflow this program's own regions/exclude_regions design is
    # meant to support. An exclude_regions-only report is legitimate to
    # SCORE (validate_lagfill_downscaling.py's own regression-check use
    # case) but was never meant to be PUBLISHABLE on its own -- see
    # build_gate's own refusal message for the same point stated from the
    # gates.py side.
    if report.get("exclude_regions") and not report.get("regions"):
        raise SystemExit(
            "report is scoped to exclude_regions="
            f"{report['exclude_regions']!r} with no regions -- this shape is a regression CHECK "
            "(e.g. 'score the rest of Cfb'), not something this program ever publishes on its own. "
            "Nothing to publish here.",
        )
    if report.get("regions"):
        _publish_subzone_patch(report, args, bucket, key)
    else:
        _publish_full_gate(report, args, bucket, key)


def _publish_full_gate(report: dict, args, bucket: str | None, key: str) -> None:
    gate = build_gate(report, band_key=args.band_key, variant=args.variant)
    validate_gate(gate)
    print(f"Band gate for model={args.model_version} band={args.band_key}"
          f"{f' variant={args.variant}' if args.variant else ''}:")
    print(json.dumps(gate, indent=2))
    for target in ("tmax", "tmin"):
        skill = gate["spatial_skill"][target]
        genuine = sorted(z for z, s in skill.items() if s)
        bias_only = sorted(z for z, s in skill.items() if not s)
        print(f"  {target}: {len(gate[target])} zone(s) passing -> {sorted(gate[target])}")
        print(f"  {target} genuine spatial skill (raw QRF beats grid): {genuine}")
        if bias_only:
            print(f"  {target} BIAS-CORRECTION-ONLY (raw QRF does NOT beat grid, only the debiased/CV-corrected value does): {bias_only}")
        corrections = gate["bias_correction"][target]
        if corrections:
            print(f"  {target} bias corrections applied: " + ", ".join(f"{z}={v:+.3f}C" for z, v in sorted(corrections.items())))

    if args.dry_run:
        print("--dry-run: not uploading")
        return

    import boto3

    if not bucket:
        raise SystemExit("--bucket or VULNERABILITY_DATA_BUCKET is required to publish")
    client = boto3.client("s3")

    _refuse_if_zones_would_be_dropped(client, bucket, key, gate, args.confirm_drops)

    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(gate, indent=2).encode(), ContentType="application/json",
    )
    print(f"Published to s3://{bucket}/{key}")


def _publish_subzone_patch(report: dict, args, bucket: str | None, key: str) -> None:
    """A regions-scoped report never goes through build_gate (see its own
    refusal + gates.py's module docstring) -- this builds a small
    additive patch and MERGES it into whatever's currently published at
    `key`, rather than building a fresh gate and overwriting. Requires a
    real GET even in --dry-run mode (to show what the merge would
    actually produce against the real current state, not an assumed
    empty one) -- the one meaningful difference from the full-gate path,
    which only needs S3 access once it's actually about to publish."""
    patch = build_subzone_patch(report, band_key=args.band_key, variant=args.variant)
    regions = report["regions"]
    print(f"Sub-zone patch for model={args.model_version} band={args.band_key} region={regions[0]}"
          f"{f' variant={args.variant}' if args.variant else ''}:")
    print(json.dumps(patch, indent=2))

    if not bucket:
        raise SystemExit("--bucket or VULNERABILITY_DATA_BUCKET is required to publish (needed even for "
                          "--dry-run here, to show the merge against the real currently-published gate)")

    import boto3
    _do_subzone_publish(boto3.client("s3"), patch, bucket, key, dry_run=args.dry_run)


def _do_subzone_publish(client, patch: dict, bucket: str, key: str, dry_run: bool) -> dict:
    """The actual GET-merge-[validate]-PUT sequence, factored out from
    _publish_subzone_patch so tests can inject a mocked S3 client directly
    (matching _refuse_if_zones_would_be_dropped's own client-as-parameter
    shape) instead of going through subprocess + real AWS credentials.
    Returns the merged gate (whether or not it was actually uploaded) so a
    caller/test can inspect it without re-parsing stdout."""
    current_gate = _get_current_gate(client, bucket, key)
    # A subzone patch is a REFINEMENT of an already-published zone-level
    # gate, not a standalone artifact -- refuse to create one out of thin
    # air. Two real reasons, not just schema pedantry: (1) BAND_GATE_SCHEMA
    # requires tmax/tmin/bias_correction/spatial_skill at the top level, so
    # a merge into a genuinely empty/missing current gate would produce an
    # object that fails validate_gate anyway (caught this exact case via a
    # real test failure before adding this check, not guessed at); (2) even
    # if the schema allowed it, publishing subzone data with no zone-level
    # entry is operationally meaningless -- the serving side's sub-zone
    # lookup is only ever consulted AFTER the zone itself already resolves
    # (see the round's serving-side design), so a subzone-only gate would
    # be dead data nothing ever reads.
    if not current_gate:
        raise SystemExit(
            f"No gate currently published at s3://{bucket}/{key} -- refusing to publish a sub-zone "
            "patch with no zone-level gate to refine. Publish the whole-zone gate first (build_gate, "
            "the normal, unscoped publish path), then publish this sub-zone patch as a follow-up.",
        )
    # 2026-08-08 review finding: the check above catches "no gate at all"
    # but not "gate exists, but not for THIS specific zone" -- e.g. the
    # current gate has tmax={'Cfb': True} only, and this patch carries a
    # Csa entry (a zone that's never cleared the zone-level gate). That
    # would publish delta_scale_subzone data for a zone with no top-level
    # tmax['Csa']/tmin['Csa'] entry at all -- exactly the "dead data
    # nothing ever reads" case the no-current-gate check above already
    # exists to prevent, just one level finer. Check every zone referenced
    # in the patch (either subzone field, either target) is actually
    # enabled (True) in current_gate's own top-level tmax/tmin first.
    unpublished_zones = []
    for field in ("delta_scale_subzone", "bias_correction_subzone"):
        for target, by_zone in patch.get(field, {}).items():
            for zone in by_zone:
                if not current_gate.get(target, {}).get(zone):
                    unpublished_zones.append((target, zone))
    if unpublished_zones:
        detail = ", ".join(f"{t}/{z}" for t, z in sorted(set(unpublished_zones)))
        raise SystemExit(
            f"Refusing to publish a sub-zone patch for zone(s) not yet enabled at the zone level in "
            f"the currently-published gate: {detail}. A sub-zone entry for a zone with no top-level "
            "tmax/tmin=True entry would be dead data nothing ever reads -- get that zone passing the "
            "whole-zone gate first.",
        )
    merged_gate = merge_subzone_patch(current_gate, patch)
    validate_gate(merged_gate)

    # 2026-08-08 review finding: this loop only inspected delta_scale_subzone
    # -- a patch entry that exists ONLY in bias_correction_subzone (the
    # offset-only/debias path passes but the affine path doesn't) was being
    # merged and uploaded with zero mention in this human-readable summary,
    # even though the whole point of a separate, explicit publish step is a
    # human seeing what's about to change before it does. Report both
    # fields, and union their zones per target so a zone appearing in only
    # one of the two still gets a line.
    for target in ("tmax", "tmin"):
        delta_zones = patch["delta_scale_subzone"][target]
        bias_zones = patch["bias_correction_subzone"][target]
        for zone in sorted(set(delta_zones) | set(bias_zones)):
            parts = []
            if zone in delta_zones:
                parts.append(f"delta_scale for {sorted(delta_zones[zone])}")
            if zone in bias_zones:
                parts.append(f"bias_correction for {sorted(bias_zones[zone])}")
            print(f"  {target}/{zone}: publishing subzone " + " and ".join(parts) +
                  f" (zone-level default at s3://{bucket}/{key} is untouched)")

    if dry_run:
        print(f"--dry-run: not uploading. Merged gate that WOULD be published to s3://{bucket}/{key}:")
        print(json.dumps(merged_gate, indent=2))
        return merged_gate

    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(merged_gate, indent=2).encode(), ContentType="application/json",
    )
    print(f"Published sub-zone patch (merged) to s3://{bucket}/{key}")
    return merged_gate


if __name__ == "__main__":
    main()
