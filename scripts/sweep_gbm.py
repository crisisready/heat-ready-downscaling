"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.


Gradient-boosted quantile trees vs. the shipped QRF -- Fable consult roadmap
experiment #2 (docs/pipeline-fable-consult-2026-07-19-downscaling-diagnosis.md
section 9), design-reviewed and revised per
docs/pipeline-fable-consult-2026-07-19-gbm-experiment-design.md before this
ever ran against real data.

Reuses build_training_feature_matrix/cv_metrics_by_zone from
train_downscaling.py so a GBM result is scored by the exact same code path
and directly comparable to ds-2026.07-rf2's own CV numbers. LightGBM's
`objective="quantile"` fits ONE quantile per model (unlike quantile_forest,
which gives all three from one forest), so each fold needs 3 separate fits
(0.025/0.5/0.975) -- flattened into the same (fold x quantile) flat task
list / single joblib.Parallel pool pattern as sweep_qrf_hyperparams.py, for
the same oversubscription-avoidance reason.

Two design-review findings baked in, not optional:
1. Hyperparameters are a literature-grounded 6-config grid
   (_GBM_COMMON/_GBM_GRID), not a single guessed config -- n_estimators=300
   matched to QRF's own tree count was a category error (learning_rate and
   n_estimators are a joint budget, Probst et al. 2019), and running GBM
   WITHOUT stochastic feature subsampling would ignore the one thing this
   dataset has already taught this build (max_features="sqrt" was THE fix
   for QRF's own CV-gate failure; feature_fraction is its direct GBM analog).
2. Quantile crossing (independently fitting 3 quantiles gives no monotonicity
   guarantee) is fixed via Chernozhukov, Fernandez-Val & Galichon (2010)'s
   monotone rearrangement -- a per-row sort of the three predictions, proven
   weakly closer to the true quantile function in finite samples than the
   raw (possibly crossed) estimates. Applied inside leave_region_out_cv_gbm
   before the OOF arrays reach cv_metrics_by_zone/conformal_q95_by_zone --
   a crossed pair there produces a negative interval half-width, which
   silently drops that row from conformal calibration rather than just
   "looking wrong".

Ship rule (pre-declared before the first real run, per the design review --
written here AND in the devlog so model selection can't drift post-hoc):
replace QRF with GBM only if, post-rearrangement, GBM beats QRF's own overall
leave-region-out RMSE by >= 0.05 C on BOTH targets, passes the beat-the-grid
gate in at least as many zones per target, and conformal coverage stays
sane (crossing rate small, calibrated coverage in [0.93, 0.97]). A tie goes
to QRF -- it ships as one artifact with structurally monotone quantiles;
GBM would ship as three boosters plus a rearrangement step.

Usage:
    # dry run: 1 config, 1 target, to get real per-config timing first
    python scripts/sweep_gbm.py --targets tmax --limit 1 --n-jobs 15 \
        --output /tmp/sweep_gbm_dryrun.json

    # full grid
    python scripts/sweep_gbm.py --targets tmax,tmin --n-jobs 15 \
        --output /tmp/sweep_gbm.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # train_downscaling.py is a sibling script

import numpy as np
from joblib import Parallel, delayed

import train_downscaling as td

_QUANTILES = (0.025, 0.5, 0.975)

# num_leaves=31 ~ interaction depth 5, inside ESL Sec 10.12.2's 4<=J<=8 range
# for boosting -- the one value from the original guess that survives
# scrutiny (design review Sec 1.1), kept fixed rather than swept.
_GBM_COMMON = {
    "num_leaves": 31,
    "feature_fraction": 0.25,   # ~ sqrt(20)/20 -- the GBM analog of the QRF max_features="sqrt" fix
    "bagging_fraction": 0.5,    # Friedman 2002 / ESL Sec 10.12 stochastic-boosting default
    "bagging_freq": 1,
    "random_state": 20260718,   # matches td._QRF_PARAMS's fixed seed
    "n_jobs": 1,                # forced -- parallelism comes from the flat task pool, not per-fit
    "verbose": -1,
}

# 3 (learning_rate, n_estimators) pairs at ~constant budget (lr*M ~= 50,
# Probst et al. 2019: the two are coupled, vary the pair not one alone) x
# 2 min_child_samples values (LightGBM's own large-data guidance is
# "hundreds to thousands"; the QRF sweep found leaf size barely mattered
# once features were subsampled, so this is a cheap check, not an
# assumption). No early stopping (design review Sec 5.2): the held-out
# region can't be used as a validation set without leaking the fold, and a
# random row split is optimistic under this dataset's per-station
# pseudo-replication.
_GBM_GRID = [
    {"config_id": f"lr{lr}_n{n}_mcs{mcs}", "learning_rate": lr, "n_estimators": n, "min_child_samples": mcs}
    for lr, n in [(0.10, 500), (0.05, 1000), (0.02, 2500)]
    for mcs in (50, 200)
]


def _fit_one_gbm_quantile(
    held_region: str, quantile: float, X: np.ndarray, y: np.ndarray,
    regions_arr: np.ndarray, gbm_params: dict, min_fold_train_rows: int,
) -> dict | None:
    """Fit + predict ONE (fold, quantile) pair. Mirrors
    train_downscaling._fit_one_qrf_fold's shape (standalone, explicit
    params, returns None on skip) but LightGBM needs one model per
    quantile rather than one model giving all three."""
    from lightgbm import LGBMRegressor

    test_mask = regions_arr == held_region
    train_mask = ~test_mask
    if train_mask.sum() < min_fold_train_rows or test_mask.sum() == 0:
        return None

    model = LGBMRegressor(objective="quantile", alpha=quantile, **gbm_params)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[test_mask])

    return {"region": held_region, "quantile": quantile, "test_mask": test_mask, "preds": preds}


def leave_region_out_cv_gbm(
    X: np.ndarray, y: np.ndarray, regions: list[str], gbm_params: dict, n_jobs: int,
) -> dict:
    """GBM analogue of train_downscaling.leave_region_out_cv -- same fold
    structure (leave-region-out, never random), same output shape
    (oof_lo/oof_median/oof_hi/valid/crossing_rate), so cv_metrics_by_zone
    scores it identically to the QRF CV. Flattens (fold x quantile) into
    one flat task list run through a single joblib.Parallel pool, rather
    than looping quantiles inside a per-fold parallel call -- avoids
    nested parallelism the same way sweep_qrf_hyperparams.py's flattening
    does.

    Applies Chernozhukov/Fernandez-Val/Galichon (2010) monotone
    rearrangement (a per-row sort of the 3 independently fitted quantiles)
    before returning -- see the module docstring for why this must happen
    here, not downstream. crossing_rate (fraction of held-out rows where
    the raw predictions were non-monotone before sorting) is returned
    alongside the CV arrays so the caller can log/persist it."""
    regions_arr = np.array(regions)
    unique_regions = sorted(set(regions))

    tasks = [(region, q) for region in unique_regions for q in _QUANTILES]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit_one_gbm_quantile)(
            region, q, X, y, regions_arr, gbm_params, td._MIN_FOLD_TRAIN_ROWS,
        )
        for region, q in tasks
    )

    n = len(y)
    oof_lo = np.full(n, np.nan)
    oof_median = np.full(n, np.nan)
    oof_hi = np.full(n, np.nan)

    by_region: dict[str, dict[float, dict]] = {}
    for (region, q), result in zip(tasks, results):
        if result is None:
            continue
        by_region.setdefault(region, {})[q] = result

    for region, by_q in by_region.items():
        if set(by_q.keys()) != set(_QUANTILES):
            print(f"  fold '{region}': incomplete quantile set, skipping")
            continue
        m = by_q[0.5]["test_mask"]
        oof_lo[m] = by_q[0.025]["preds"]
        oof_median[m] = by_q[0.5]["preds"]
        oof_hi[m] = by_q[0.975]["preds"]
        print(f"  fold '{region}': {int(m.sum())} held-out row(s)")

    valid = ~np.isnan(oof_median)

    # Chernozhukov et al. 2010 monotone rearrangement: per-row sort of the
    # three independently fitted quantiles. Weakly improves each quantile
    # estimate in finite samples (any Lp norm); a no-op wherever no
    # crossing occurred. Must happen before cv_metrics_by_zone/
    # conformal_q95_by_zone see these arrays -- a crossed (hi < lo) pair
    # produces a negative interval half-width there, which silently drops
    # that row from conformal calibration rather than just scoring oddly.
    stacked = np.vstack([oof_lo, oof_median, oof_hi])  # (3, n)
    crossed = np.any(np.diff(stacked, axis=0) < 0, axis=0)
    crossing_rate = float(np.mean(crossed[valid])) if valid.any() else 0.0
    stacked.sort(axis=0)
    oof_lo, oof_median, oof_hi = stacked[0], stacked[1], stacked[2]

    return {
        "valid": valid, "oof_lo": oof_lo, "oof_median": oof_median, "oof_hi": oof_hi,
        "crossing_rate": crossing_rate,
    }


def run(targets: list[str], n_jobs: int, output_path: str | None, limit: int | None) -> list[dict]:
    grid = _GBM_GRID[:limit] if limit else _GBM_GRID
    print(f"Sweep: {len(grid)} GBM config(s) x {len(targets)} target(s)")

    t0 = time.monotonic()
    rows = td.load_training_rows()
    print(f"Loaded {len(rows)} row(s) in {time.monotonic() - t0:.1f}s")

    data_by_target = {}
    for target in targets:
        t0 = time.monotonic()
        X, y, regions, zones, _lons, _lats = td.build_training_feature_matrix(rows, target)
        data_by_target[target] = {"X": X, "y": y, "regions": regions, "zones": zones}
        print(f"[{target}] feature matrix: {X.shape[0]} usable row(s) in {time.monotonic() - t0:.1f}s")

    all_results = []
    for config in grid:
        gbm_params = {**_GBM_COMMON, **{k: v for k, v in config.items() if k != "config_id"}}
        for target in targets:
            d = data_by_target[target]
            t_cv = time.monotonic()
            cv = leave_region_out_cv_gbm(d["X"], d["y"], d["regions"], gbm_params, n_jobs=n_jobs)
            elapsed = time.monotonic() - t_cv
            print(f"[{target}] {config['config_id']}: GBM CV done in {elapsed:.1f}s, crossing_rate={cv['crossing_rate']:.4f}")

            metrics = td.cv_metrics_by_zone(d["y"], d["zones"], cv)
            o = metrics["overall"]
            print(
                f"[{target}] {config['config_id']}: rmse_grid={o['rmse_grid_c']:.4f} "
                f"rmse_gbm={o['rmse_qrf_c']:.4f} beats_grid={o['qrf_beats_grid']} bias={o['bias_qrf_c']:.4f}"
            )
            for zone, zm in sorted(metrics["by_zone"].items()):
                print(
                    f"[{target}] {config['config_id']}   zone {zone}: rmse_grid={zm['rmse_grid_c']:.3f} "
                    f"rmse_gbm={zm['rmse_qrf_c']:.3f} beats_grid={zm['qrf_beats_grid']}"
                )
            all_results.append({
                "target": target, "config": config, "crossing_rate": cv["crossing_rate"],
                "metrics": metrics,
                "note": "rmse_qrf_c/bias_qrf_c/raw_qrf_interval_coverage keys hold GBM values -- scored by the shared harness, not renamed",
            })

            if output_path:
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    if output_path:
        print(f"Wrote {len(all_results)} result(s) to {output_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="tmax,tmin")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N grid configs (dry run)")
    args = parser.parse_args()
    run(args.targets.split(","), args.n_jobs, args.output, args.limit)


if __name__ == "__main__":
    main()
