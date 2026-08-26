"""
Worked Rung A example (roadmap Phase 2's "contributor front door" -- see
examples/README.md): re-run the same evaluation the referee runs, against a
real dark cell, using nothing but this public repo and the published
snapshot. Produces a real `claimed_report.json` you could open a Rung A
submission PR with (see CONTRIBUTING.md's "Submission format").

WHAT "DARK" MEANS HERE, stated plainly: no submission has ever claimed this
(model_version, band_key, target, zone) combination (`ledger/credit.jsonl` is
still empty as of this writing -- see `docs/leaderboard.md`), not that the
model is known to fail here. The named cell below happens to independently
beat the grid baseline with real margin as of snapshot v2026.07 -- a genuine
result, not a guaranteed one: re-running this script against a newer snapshot
can come back differently, which is the whole point of scoring against real,
changing ground truth rather than a fixed fixture.

THE MECHANISM, exactly what the referee does (`scripts/run_submission.py`'s
`reproduce()`, imported directly below rather than reimplemented): download
the public snapshot release, build a `contract.FrozenPredictionAdapter` over
its frozen predictions partition, and call
`heatready_downscaling.score.score_band` -- the same call for a real
submission PR and for this local dry run. No AWS credentials, no private
modules: this is why `validate_lagfill_downscaling.py`/
`validate_forecast_downscaling.py` are NOT what a Rung A submission runs
against the snapshot (see CONTRIBUTING.md's Rung A section) -- they need live
Aurora/Open-Meteo access this public repo has no path to.

Usage:
    python examples/rung_a_evaluate_dark_cell.py
    python examples/rung_a_evaluate_dark_cell.py --target tmin --zone BSk --out my_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import run_submission  # noqa: E402

from heatready_downscaling import report  # noqa: E402

MODEL_VERSION = "ds-2026.07-rf5"
BAND_KEY = "lag_fill"
SNAPSHOT_VERSION = "v2026.07"

# The named dark cell this example highlights -- CLI-overridable so a reader
# can point this at any zone the snapshot's lag_fill band covers (see
# examples/README.md for how to list them).
DEFAULT_TARGET = "tmax"
DEFAULT_ZONE = "BWh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=["tmax", "tmin"])
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--snapshot-version", default=SNAPSHOT_VERSION)
    parser.add_argument("--cache-root", default=".cache/snapshots")
    parser.add_argument("--out", default="examples/output/rung_a_claimed_report.json")
    parser.add_argument(
        "--require-beats-grid", action=argparse.BooleanOptionalAction, default=True,
        help="fail (exit 1) if the named cell does not beat the grid baseline -- keeps this "
        "example honest about what it claims when run in CI. Pass --no-require-beats-grid to "
        "just look at the numbers for a zone you're exploring (default: on)",
    )
    args = parser.parse_args()

    print(f"Downloading public snapshot {args.snapshot_version} (cached under {args.cache_root})...")
    snapshot_dir = run_submission.download_snapshot(args.snapshot_version, args.cache_root)

    print(f"Reproducing {MODEL_VERSION} / {BAND_KEY} against the snapshot (same call the referee makes)...")
    reproduced = run_submission.reproduce(snapshot_dir, MODEL_VERSION, BAND_KEY, args.snapshot_version)
    report.validate_report(reproduced)

    zone_result = reproduced["by_target"][args.target].get(args.zone)
    if zone_result is None:
        raise SystemExit(
            f"{args.target}/{args.zone}/{BAND_KEY} has no rows in snapshot {args.snapshot_version} "
            "-- pick a zone this band actually covers (print reproduced['by_target'] to see them all)",
        )

    print(f"\n{args.target}/{args.zone}/{BAND_KEY}, model {MODEL_VERSION}, snapshot {args.snapshot_version}:")
    print(f"  rmse_grid_c              = {zone_result['rmse_grid_c']:.4f}")
    print(f"  rmse_qrf_c               = {zone_result['rmse_qrf_c']:.4f}")
    print(f"  n_grid / n_qrf_applied   = {zone_result['n_grid']} / {zone_result['n_qrf_applied']}")
    print(f"  qrf_beats_grid_with_margin = {zone_result['qrf_beats_grid_with_margin']}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reproduced, f, indent=2, default=str)
    print(f"\nWrote {args.out} -- this IS a real claimed_report.json (see CONTRIBUTING.md's "
          "\"Submission format\" for how to turn this into a submission PR).")

    if args.require_beats_grid and not zone_result["qrf_beats_grid_with_margin"]:
        raise SystemExit(
            f"{args.target}/{args.zone} no longer beats the grid baseline with margin as of "
            f"snapshot {args.snapshot_version} -- this example's claim is stale and needs a new "
            "cell (pass --target/--zone to explore, --no-require-beats-grid to just see the "
            "numbers, or update DEFAULT_TARGET/DEFAULT_ZONE once you've found a new one)",
        )


if __name__ == "__main__":
    main()
