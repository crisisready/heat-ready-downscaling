"""
Publish a validated spatial-ranking evidence artifact to S3 for production
to load via downscaling.load_spatial_ranking (crisisready/heat-risk-data-
api). Companion to publish_band_gate.py -- same dry-run/human-review/
fail-closed-if-absent contract -- but a DIFFERENT claim: this answers
whether the model demonstrably ranks nearby locations correctly (held-out
cross-validation), not whether a per-zone correction beats the raw grid.

Reads a research/downscaling-confidence/eval_spatial_ranking_clusters.py
JSON result (crisisready/heat-risk-data-api's own research repo -- that
script computes the real leave-region-out CV, station-demeaned, cluster-
bootstrapped Spearman per (target, zone); see its own module docstring and
the 2026-07-31 Opus design consult recorded in PARIS_CONFIDENCE_ROADMAP.md's
"Round 5" section for the full methodology and why CLUSTER-scale contrast
on LEAVE-REGION-OUT folds was the chosen design over the two simpler
alternatives -- region-scale-everything (too coarse/upward-biased) and
leave-cluster-out folds (a novel, unvalidated protocol)).

Reshapes that script's {"by_target": {"tmax": {"by_zone": {...}}, "tmin":
{...}}} report into the published {"model_version", "computed_at",
"protocol", "by_zone": {"tmax": {...}, "tmin": {...}}, "region_scale_
secondary"} shape downscaling.load_spatial_ranking expects, at
s3://{bucket}/downscaling/spatial_ranking/{model_version}.json.

Deliberately a separate, explicit publish step (build_gate/publish_band_
gate.py's own precedent) -- a human reads the eval result and decides to
publish before anything in production can start reading it. NO pooled/
"_default" fallback key is ever written -- a zone absent from "by_zone"
serves 'unproven' at read time (downscaling.load_spatial_ranking's own
fail-closed default), never a number borrowed from an easier population.

Usage:
    python scripts/publish_spatial_ranking.py \\
        --result /path/to/spatial_ranking_eval_result.json \\
        --model-version ds-2026.07-rf5 --computed-at 2026-07-31T04:50:52Z \\
        --profile nish-climateverse
"""

from __future__ import annotations

import argparse
import json
import os

from heatready_downscaling.gates import validate_spatial_ranking


def build_artifact(result: dict, model_version: str, computed_at: str) -> dict:
    by_zone = {}
    region_scale_secondary = {}
    for target in ("tmax", "tmin"):
        target_report = result["by_target"][target]["by_zone"]
        by_zone[target] = {}
        region_scale_secondary[target] = {}
        for zone, r in target_report.items():
            by_zone[target][zone] = {
                "class": r["class"],
                "ranking_spearman": r["ranking_spearman"],
                "ranking_spearman_ci95": r["ranking_spearman_ci95"],
                "amplitude_slope": r["amplitude_slope"],
                "n_clusters_used": r["n_clusters_used"],
                "n_stations_used": r["n_stations_used"],
                "n_regions": r["n_regions"],
            }
            if r.get("region_scale_secondary_spearman") is not None:
                region_scale_secondary[target][zone] = r["region_scale_secondary_spearman"]

    return {
        "model_version": model_version,
        "computed_at": computed_at,
        "protocol": result["protocol"],
        "by_zone": by_zone,
        "region_scale_secondary": region_scale_secondary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, help="path to eval_spatial_ranking_clusters.py's result.json")
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--computed-at", required=True,
        help="UTC timestamp the eval actually ran (real value, not this script's own run time -- "
             "pass the eval job's own status.json 'updated_at' at its 'done' stage)",
    )
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--profile", default=None, help="named AWS profile (omit on EC2 with an attached IAM role)")
    parser.add_argument("--dry-run", action="store_true", help="print the artifact that would be published, don't upload")
    args = parser.parse_args()

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile

    with open(args.result) as f:
        result = json.load(f)

    artifact = build_artifact(result, args.model_version, args.computed_at)
    validate_spatial_ranking(artifact)

    print(f"Spatial-ranking artifact for model={args.model_version}:")
    print(json.dumps(artifact, indent=2))
    for target in ("tmax", "tmin"):
        by_class: dict[str, list[str]] = {}
        for zone, r in artifact["by_zone"][target].items():
            by_class.setdefault(r["class"], []).append(zone)
        print(f"  {target}:")
        for cls in ("strong", "moderate", "weak", "unproven"):
            zones = sorted(by_class.get(cls, []))
            if zones:
                print(f"    {cls}: {zones}")

    if args.dry_run:
        print("--dry-run: not uploading")
        return

    import boto3

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    key = f"downscaling/spatial_ranking/{args.model_version}.json"
    client = boto3.client("s3")
    client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(artifact, indent=2).encode(), ContentType="application/json",
    )
    print(f"Published to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
