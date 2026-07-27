"""
PROVENANCE: moved from crisisready/heat-risk-data-api's scripts/publish_blend_gate.py at commit e6cc1d2a338fd9c87ef51cbc1daf6cb1ed5e2b3b (2026-07-27, Phase 1.4, plan section 5.4). See this repository's own PROVENANCE.md.

REFACTORED during the move: the private repo's original ad-hoc
`for required_key in ("tmax", "tmin", "params")` presence check is replaced
by `heatready_downscaling.gates.validate_blend_gate` -- a real jsonschema
check, including the per-group L_km/R_km/tau shape (see gates.py's own
docstring).

Publish a validated station-blend gate to S3 for production to load via
downscaling.load_blend_gate (crisisready/heat-risk-data-api).

Reads a scripts/validate_station_blend.py `--gate-out` JSON file (already
shaped {"tmax": {zone: bool}, "tmin": {zone: bool}, "params": {"tmax":
{group_letter: {"L_km":.., "R_km":.., "tau":..}, ...}, "tmin": {...}}} --
params are per broad Koppen group A-E, not one single global triple, see
validate_station_blend.py's module docstring) and uploads it verbatim to
s3://{bucket}/downscaling/blend_gates/{model_version}/{band_key}.json.

Deliberately a separate, explicit publish step rather than
validate_station_blend.py uploading its own result automatically -- a human
(or a review-gated CI step) decides whether to publish before anything in
production can start reading it, matching the identical discipline
scripts/publish_band_gate.py already establishes for the QRF's own gates.

--band-key is hard-restricted to "lag_fill" (not an open choice like
publish_band_gate.py's era5/forecast_lead1..7 list): downscaling.
load_blend_gate's own docstring is explicit that only the lag-fill band can
ever produce a nonempty blend result at serving time today (METAR's ~72h
fetch window means no real observation exists yet for the ERA5 band's
always-5-6-days-old dates, or for any forecast lead, which is always a
future date) -- publishing under "era5" or "forecast_lead{N}" would be
harmless at serving time (blend_deltas would just never find a matching
nearby-station anomaly) but would falsely imply a distribution nobody has
actually validated. The CLI itself refuses those keys rather than relying
on operator discipline alone.

Usage:
    python scripts/publish_blend_gate.py \\
        --gate /tmp/station_blend_gate.json \\
        --model-version ds-2026.07-rf5 \\
        --profile nish-climateverse
"""

from __future__ import annotations

import argparse
import json
import os

from heatready_downscaling.gates import validate_blend_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", required=True, help="path to a validate_station_blend.py --gate-out JSON file")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--band-key", default="lag_fill", choices=["lag_fill"])
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--profile", default=None, help="named AWS profile (omit on EC2 with an attached IAM role)")
    parser.add_argument("--dry-run", action="store_true", help="print the gate that would be published, don't upload")
    args = parser.parse_args()

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile

    with open(args.gate) as f:
        gate = json.load(f)

    validate_blend_gate(gate)

    print(f"Blend gate for model={args.model_version} band={args.band_key}:")
    print(json.dumps(gate, indent=2))
    for target in ("tmax", "tmin"):
        zones = sorted(z for z, v in gate[target].items() if v)
        print(f"  {target}: {len(zones)} zone(s) passing -> {zones}")
        group_params = gate["params"].get(target) or {}
        for group, params in sorted(group_params.items()):
            print(f"  {target} group {group}: L_km={params['L_km']}, R_km={params['R_km']}, tau={params['tau']}")

    if args.dry_run:
        print("--dry-run: not uploading")
        return

    import boto3

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    key = f"downscaling/blend_gates/{args.model_version}/{args.band_key}.json"
    client = boto3.client("s3")
    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(gate, indent=2).encode(), ContentType="application/json",
    )
    print(f"Published to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
