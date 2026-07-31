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

from heatready_downscaling.gates import build_gate, validate_gate


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
    args = parser.parse_args()

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile

    with open(args.report) as f:
        report = json.load(f)

    gate = build_gate(report, band_key=args.band_key)
    validate_gate(gate)
    print(f"Band gate for model={args.model_version} band={args.band_key}:")
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

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    key = f"downscaling/band_gates/{args.model_version}/{args.band_key}.json"
    client = boto3.client("s3")
    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(gate, indent=2).encode(), ContentType="application/json",
    )
    print(f"Published to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
