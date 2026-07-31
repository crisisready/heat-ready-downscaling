"""
Phase 2 (plan-2026-07-28-lagfill-base-mismatch-fix.md, section 6), extended
per Nishant's 2026-07-31 direct go-ahead to cover forecast_lead1 alongside
lag_fill -- see heat-risk-data-api's PARIS_CONFIDENCE_ROADMAP.md "Round 6"
section for the full context and the verification+extension consult this
script's design comes from.

Trains a QRF model DIRECTLY on a given band's own historical grid values
(instead of transferring an era5-trained model to it), reusing
train_downscaling.py's own functions UNMODIFIED (build_training_feature_
matrix, leave_region_out_cv, cv_metrics_by_zone, conformal calibration,
_build_aoa_index) -- never forked, so training and inference can never
silently diverge on how a feature is computed, and so this experiment is
byte-identical in method to every model this repo ships for real.

WHY THIS EXISTS: ds-2026.07-rf5 is trained only on era5_land_cds rows. Its
learned correction is sized for that base's error pattern and overshoots
when applied unchanged to a base with a different (usually smaller) error
-- confirmed for lag_fill (docs/investigation-2026-07-28-cfb-lagfill-gate-
failure.md): Cfb tmax RMSE is LOWER on the untouched open_meteo_hfa grid
(1.557C) than on era5_land_cds (which the model was trained to correct),
so the model's delta actively overshoots there. The already-shipped fix
for that (delta_scale, an out-of-fold-validated per-zone affine rescale)
recovers real, gate-clearing improvement WITHOUT a retrain -- that fix is
the honest baseline this retrain must beat, not a naive "no correction"
or "grid_debias_only" comparison. See score_band's own bias_correction_c/
delta_scale_c fields for that baseline; this script does not compute it
(scripts/experiments/affine_recalibration.py already does, for lag_fill).

CONTROL ARM: one era5-band model, trained on the SAME public snapshot,
serves as the control for every band arm -- not one control per candidate.
It isolates "trained on the right base" from "trained on the public
snapshot's slightly smaller corpus than rf5 saw" (rf5 trained on the full
private ghcn_training; this snapshot excludes ~15% of stations, per
CONTRIBUTING.md's holdout-station exclusion for the public release).

DELIBERATELY DOES NOT run regression_kriging_cv (plan section 2.4 -- a
literature-comparison baseline that contributes nothing to the question
being asked here, and OrdinaryKriging.execute over the full held-out set
per fold is a large, pointless cost for this experiment specifically).

DELIBERATELY WRITES TO LOCAL DISK ONLY -- never calls save_model_artifacts
(which uploads to S3). This is a research artifact, not a deployable one;
per the plan's own escalation points (section 9), promoting any model
change is a hard stop reserved for a human decision, and creating a
production-shaped bundle at all needs explicit go-ahead (given, for this
run, by Nishant on 2026-07-31 -- see the roadmap entry above).

Usage (bastion, /opt/ghcn-build/.venv/bin/python3 or this repo's own .venv):
    python scripts/experiments/train_on_band.py --band lag_fill \
        --model-version ds-2026.07-omhfa-exp1 --n-jobs 5 \
        --snapshot-dir /tmp/release_recheck --output-dir /tmp/lagfill_exp
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import joblib  # noqa: E402

from heatready_downscaling import snapshot  # noqa: E402
from heatready_downscaling.features import FEATURE_ORDER  # noqa: E402

import train_downscaling as td  # noqa: E402


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class StatusWriter:
    """Local-file progress checkpoint (no S3 -- this script never touches
    AWS). Flushed on every write so a `tail -f`/re-read from another shell
    sees real progress, matching this whole project's own pre-flight rule
    that any run over ~2 minutes needs observable, flushed progress."""

    def __init__(self, path):
        self.path = Path(path)
        self.state = {"started_at": _now(), "stage": "starting", "log": []}

    def log(self, msg):
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        self.state["log"].append(line)
        self.state["log"] = self.state["log"][-500:]
        self._write()

    def set_stage(self, stage):
        self.state["stage"] = stage
        self.log(f"STAGE: {stage}")

    def _write(self):
        self.state["updated_at"] = _now()
        self.path.write_text(json.dumps(self.state, indent=2, default=str))


def add_delta_columns(rows: list[dict], status: "StatusWriter") -> list[dict]:
    """Plan section 2.2: delta_tmax_c/delta_tmin_c for a non-era5 band is
    ONE subtraction, computed in-place from columns every snapshot row
    already carries -- station_tmax_c/station_tmin_c (real truth) and
    grid_tmax_c/grid_tmin_c (this band's own real base value). Every row
    gets the key set (never left absent), even when a value is missing --
    build_training_feature_matrix indexes rows[i][delta_col] directly, not
    via .get(), so an absent key would KeyError deep inside a joblib
    worker far from this, the real cause. None (not NaN) for a row that
    can't compute a delta -- build_training_feature_matrix's own
    _target_is_finite guard already treats None as "drop this row",
    matching the real SQL guard's own semantics for ghcn_training."""
    n_missing = {"tmax": 0, "tmin": 0}
    for r in rows:
        for target, truth_col, grid_col in (
            ("tmax", "station_tmax_c", "grid_tmax_c"),
            ("tmin", "station_tmin_c", "grid_tmin_c"),
        ):
            truth, grid = r.get(truth_col), r.get(grid_col)
            delta_col = f"delta_{target}_c"
            if truth is None or grid is None:
                r[delta_col] = None
                n_missing[target] += 1
                continue
            d = truth - grid
            r[delta_col] = d if math.isfinite(d) else None
            if not math.isfinite(d):
                n_missing[target] += 1
    status.log(f"delta columns computed in-place: {n_missing['tmax']} row(s) missing/non-finite "
               f"delta_tmax_c, {n_missing['tmin']} missing/non-finite delta_tmin_c, out of {len(rows)}")
    return rows


def train_one_target(status, X, y, regions, zones, target, n_jobs, seed):
    """Mirrors train_downscaling.main()'s own per-target loop exactly
    (leave_region_out_cv -> cv_metrics_by_zone -> conformal calibration ->
    final fit -> AOA index), MINUS regression_kriging_cv (see module
    docstring) and MINUS the S3 upload."""
    status.log(f"[{target}] {len(y)} usable row(s) across {len(set(regions))} region(s), "
               f"{len(set(zones))} climate zone(s) -- running leave_region_out_cv(n_jobs={n_jobs})...")
    t0 = time.monotonic()
    cv = td.leave_region_out_cv(X, y, regions, n_jobs=n_jobs)
    status.log(f"[{target}] leave_region_out_cv done in {time.monotonic() - t0:.1f}s")

    zone_metrics = td.cv_metrics_by_zone(y, zones, cv, kriging_oof_median=None)
    status.log(f"[{target}] overall: {zone_metrics['overall']}")
    for zone, m in sorted(zone_metrics["by_zone"].items()):
        gate = "PASS" if m["qrf_beats_grid"] else "FAIL"
        status.log(f"[{target}]   zone {zone}: n={m.get('n')} rmse_grid={m['rmse_grid_c']:.3f} "
                   f"rmse_qrf={m['rmse_qrf_c']:.3f} [{gate}]")

    q95_by_zone = td.conformal_q95_by_zone(y, zones, cv)
    coverage = td.conformal_empirical_coverage(y, zones, cv, q95_by_zone)
    status.log(f"[{target}] empirical conformal coverage: {coverage:.4f} (target [0.93, 0.97])")

    valid_di = cv["oof_di"][cv["valid"] & ~np.isnan(cv["oof_di"])]
    ood_threshold = float(np.mean(valid_di)) if len(valid_di) else None

    status.log(f"[{target}] final fit on all {len(y)} row(s)...")
    from quantile_forest import RandomForestQuantileRegressor
    t0 = time.monotonic()
    final_model = RandomForestQuantileRegressor(**td._QRF_PARAMS).fit(X, y)
    status.log(f"[{target}] final fit done in {time.monotonic() - t0:.1f}s")

    aoa = td._build_aoa_index(final_model, X, target, seed=seed)
    importances = td.feature_importance_weights(final_model)
    status.log(f"[{target}] feature importances (FEATURE_ORDER order): "
               f"{dict(zip(FEATURE_ORDER, [round(float(w), 4) for w in importances]))}")

    return {
        "model": final_model,
        "aoa": aoa,
        "zone_metrics": zone_metrics,
        "q95_by_zone": q95_by_zone,
        "coverage": coverage,
        "ood_threshold": ood_threshold,
        "feature_importances": {name: float(w) for name, w in zip(FEATURE_ORDER, importances)},
        "n_rows": int(len(y)),
        "elapsed_seconds": time.monotonic() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", required=True, choices=["era5", "lag_fill"] + [f"forecast_lead{n}" for n in range(1, 8)])
    ap.add_argument("--model-version", required=True,
                     help="Local-only label, e.g. ds-2026.07-omhfa-exp1 -- never a ds-2026.07-rfN name.")
    ap.add_argument("--snapshot-dir", default="/tmp/release_recheck")
    ap.add_argument("--output-dir", default="/tmp/lagfill_exp")
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="Passed to leave_region_out_cv. 1 on gu-dev (2 core/8GB -- OOM risk at "
                          "anything higher, see plan section 2.4). 5+ is safe on the bastion "
                          "(r6i.4xlarge, 16 vCPU/128GB).")
    ap.add_argument("--months", default=None,
                     help="Comma-separated YYYY-MM list, e.g. 2023-01 -- restricts to a subset of "
                          "the snapshot for a cheap smoke test. Omit for the real, full run.")
    args = ap.parse_args()
    months = args.months.split(",") if args.months else None

    out_dir = Path(args.output_dir) / f"model-{args.band}-{args.model_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(out_dir / "status.json")
    t_start = time.monotonic()

    status.set_stage("load_snapshot")
    rows = snapshot.read_band_partitions(args.snapshot_dir, args.band, months=months)
    status.log(f"loaded {len(rows)} real row(s) for band={args.band} from {args.snapshot_dir}")
    if not rows:
        status.log("no rows found for this band -- aborting")
        return

    status.set_stage("add_delta_columns")
    rows = add_delta_columns(rows, status)

    results_by_target = {}
    for target in ("tmax", "tmin"):
        status.set_stage(f"build_feature_matrix_{target}")
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, target)

        status.set_stage(f"train_{target}")
        result = train_one_target(status, X, y, regions, zones, target, args.n_jobs,
                                   seed=td._QRF_PARAMS["random_state"])
        results_by_target[target] = result

    status.set_stage("write_bundle")
    artifact_bundle = {}
    metadata_cv, metadata_conformal, feature_importances = {}, {}, {}
    ood_thresholds = []
    for target, r in results_by_target.items():
        artifact_bundle[f"model_{target}"] = r["model"]
        artifact_bundle.update(r["aoa"])
        metadata_cv[target] = r["zone_metrics"]
        metadata_conformal[target] = r["q95_by_zone"]
        feature_importances[target] = r["feature_importances"]
        if r["ood_threshold"] is not None:
            ood_thresholds.append(r["ood_threshold"])

    metadata = {
        "model_version": args.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "band": args.band,
        "snapshot_dir": args.snapshot_dir,
        "feature_order": list(FEATURE_ORDER),
        "targets": ["delta_tmax_c", "delta_tmin_c"],
        "conformal_q95_by_zone": metadata_conformal.get("tmax", {}),
        "conformal_q95_by_zone_tmin": metadata_conformal.get("tmin", {}),
        "ood_aoa_threshold": (sum(ood_thresholds) / len(ood_thresholds)) if ood_thresholds else None,
        "cv": {"leave_region_out": metadata_cv},
        "feature_importances": feature_importances,
        "n_rows_by_target": {t: r["n_rows"] for t, r in results_by_target.items()},
        "conformal_empirical_coverage": {t: r["coverage"] for t, r in results_by_target.items()},
        "note": "LOCAL RESEARCH ARTIFACT ONLY -- never uploaded to S3, never a ds-2026.07-rfN name, "
                "never promoted. See train_on_band.py's own module docstring.",
    }

    joblib_path = out_dir / "model.joblib"
    with open(joblib_path, "wb") as f:
        joblib.dump(artifact_bundle, f)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    status.log(f"wrote local bundle to {joblib_path} ({joblib_path.stat().st_size / 1e6:.1f} MB) "
               f"+ {out_dir / 'metadata.json'}")

    status.set_stage("done")
    status.log(f"TOTAL elapsed: {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
