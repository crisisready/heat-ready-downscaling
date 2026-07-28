"""
Lightweight era5-band marginal-pass scan (addition #2 to
plan-2026-07-28-lagfill-base-mismatch-fix.md, from Nishant 2026-07-28).

Note on scope: the plan's brief assumed era5-band per-zone CV numbers
already existed on disk from the training/gate-publish process. A search
(2026-07-28) found no such file -- _summary.json is rf4-vintage and has no
`era5` key under band_gates; the one claimed_report.json is lag_fill-only;
metadata.json with the embedded leave-region-out CV lives only in the
private S3 bucket, not locally. What *is* true and cheap: score.score_band
is band-agnostic arithmetic over already-frozen predictions (no model
fit, no training) -- paired/band=era5 joined to
predictions/model=ds-2026.07-rf5/band=era5 on (station_id, date), grouped
by climate_zone, exactly the same join the lag_fill investigation used.
Running it here is the cheap-audit equivalent of reading a pre-computed
file, not a retrain.

Caveat that must travel with these numbers (plan section 2.3): rf5 was
TRAINED on these era5 rows, so this is an in-sample scoring pass, not a
held-out test. Report it as such -- these numbers are useful for spotting
which zones are already riding close to AUTO_ENABLE_MARGIN in production's
own training band, not as a clean generalization estimate.

Run: /root/projects/crisisready/heat-ready-downscaling/.venv/bin/python \
     scripts/experiments/era5_marginal_scan.py
Writes: /tmp/lagfill_exp/era5_marginal_scan.json
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heatready_downscaling import contract, snapshot  # noqa: E402
from heatready_downscaling.score import AUTO_ENABLE_MARGIN, score_band  # noqa: E402

SNAPSHOT_DIR = "/tmp/release_recheck"
MODEL_VERSION = "ds-2026.07-rf5"
BAND = "era5"
FOLD_SALT = "v2026.07"
OUTPUT_PATH = Path("/tmp/lagfill_exp/era5_marginal_scan.json")

# "Marginal" = clears the gate but by less than this much headroom above
# AUTO_ENABLE_MARGIN. Cfa/tmin's known case clears by +0.6pp (0.006) over
# the 3% margin -- use a slightly wider net (2x that) to catch neighbors.
MARGIN_HEADROOM_FLAG_PP = 0.012


def main():
    rows = snapshot.read_band_partitions(SNAPSHOT_DIR, BAND)
    adapter = contract.FrozenPredictionAdapter.from_snapshot(SNAPSHOT_DIR, MODEL_VERSION, BAND)

    output = {
        "snapshot_dir": SNAPSHOT_DIR,
        "model_version": MODEL_VERSION,
        "band": BAND,
        "fold_salt": FOLD_SALT,
        "caveat": "in-sample: ds-2026.07-rf5 was trained on these era5 rows -- not a held-out generalization estimate",
        "cells": {},
        "marginal_cases": [],
    }

    for target in ("tmax", "tmin"):
        result = score_band(adapter, rows, target, fold_salt=FOLD_SALT)
        output["cells"][target] = result
        for zone, r in result.items():
            margin = r.get("rmse_improvement_pct_debiased_cv")
            passes = r.get("qrf_beats_grid_with_margin")
            if margin is None:
                continue
            headroom = margin - AUTO_ENABLE_MARGIN
            if passes and 0 <= headroom <= MARGIN_HEADROOM_FLAG_PP:
                output["marginal_cases"].append({
                    "target": target,
                    "zone": zone,
                    "rmse_improvement_pct_debiased_cv": margin,
                    "headroom_over_margin_pp": headroom,
                    "n_qrf_applied": r["n_qrf_applied"],
                })
            elif not passes and r.get("n_qrf_applied", 0) >= 1:
                # also flag outright failures -- cheap to report, useful context
                output["marginal_cases"].append({
                    "target": target,
                    "zone": zone,
                    "rmse_improvement_pct_debiased_cv": margin,
                    "headroom_over_margin_pp": headroom,
                    "n_qrf_applied": r["n_qrf_applied"],
                    "note": "FAILS gate outright (margin below AUTO_ENABLE_MARGIN or insufficient n/bias)",
                })

    output["marginal_cases"].sort(key=lambda c: c["headroom_over_margin_pp"] if c["headroom_over_margin_pp"] is not None else 999)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {OUTPUT_PATH}")

    print("\n=== marginal / failing cases (era5 band, in-sample) ===")
    for c in output["marginal_cases"]:
        print(c)


if __name__ == "__main__":
    main()
