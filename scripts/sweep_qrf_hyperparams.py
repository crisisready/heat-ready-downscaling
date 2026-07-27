"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.


Hyperparameter sweep for the downscaling QRF model -- Fable consult roadmap
experiment #1 (docs/pipeline-fable-consult-2026-07-19-downscaling-diagnosis.md
section 9): tests the top root-cause hypothesis (H1 -- min_samples_leaf=5
lets trees memorize per-station offsets that don't transfer under
leave-region-out CV) by sweeping min_samples_leaf/max_features/max_depth
under the SAME leave-region-out CV harness the shipped model uses.

Reuses build_training_feature_matrix/_fit_one_qrf_fold/cv_metrics_by_zone
from train_downscaling.py directly rather than reimplementing CV logic --
every combo is scored by the identical code path the real training run
uses, so a result here is directly comparable to ds-2026.07-rf1's numbers.
Regression kriging is deliberately NOT run per combo: it's broken (OOMs,
see the consult doc) and independent of QRF's hyperparameters, so refitting
it per combo would waste most of the wall-clock budget on something that
can't change combo-to-combo.

Parallelization: flattens (combo x target x fold) into ONE flat list of
single-threaded RF fits (qrf_params forces n_jobs=1 per fit) and runs the
whole list through a single joblib.Parallel(n_jobs=<cores>) pool. This is
deliberately NOT nested per-combo parallelism -- running C combos
concurrently, each internally re-parallelizing across F folds, would
oversubscribe the machine (C x F threads competing for however many cores
exist). Flattening to one pool of equal-cost, independent single-threaded
tasks lets joblib load-balance the whole grid across every core with no
manual accounting.

Usage:
    # dry run: 1 target, 2 leaf sizes, all folds -- verifies correctness and
    # gives a real per-task timing before committing to the full grid.
    python scripts/sweep_qrf_hyperparams.py --stage 1 --targets tmax \
        --limit 2 --n-jobs 15 --output /tmp/sweep_dryrun.json

    # stage 1: min_samples_leaf sweep only (tests H1 directly, cheapest)
    python scripts/sweep_qrf_hyperparams.py --stage 1 --n-jobs 15 \
        --output /tmp/sweep_stage1.json

    # stage 2: full min_samples_leaf x max_features x max_depth grid
    python scripts/sweep_qrf_hyperparams.py --stage 2 --n-jobs 15 \
        --output /tmp/sweep_stage2.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from joblib import Parallel, delayed

import train_downscaling as td

# Stage 1 tests the top hypothesis alone (leaf size), holding max_features at
# the literature-standard decorrelating default and max_depth unconstrained --
# cheapest, highest-expected-value experiment per the consult's own ordering.
STAGE1_GRID = {
    "min_samples_leaf": [5, 20, 50, 100, 200],
    "max_features": ["sqrt"],
    "max_depth": [None],
}

# Stage 2 is the full 3-parameter grid from the consult roadmap, for once
# stage 1 has narrowed which leaf sizes are worth cross-checking.
STAGE2_GRID = {
    "min_samples_leaf": [5, 20, 50, 100, 200],
    "max_features": ["sqrt", 0.33, 0.5, 1.0],
    "max_depth": [None, 10, 20],
}


def combo_grid(stage: int) -> list[dict]:
    grid = STAGE1_GRID if stage == 1 else STAGE2_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def combo_id(combo: dict) -> str:
    return "leaf{min_samples_leaf}_feat{max_features}_depth{max_depth}".format(**combo)


def qrf_params_for_combo(combo: dict) -> dict:
    return {
        "n_estimators": td._QRF_PARAMS["n_estimators"],
        "random_state": td._QRF_PARAMS["random_state"],
        "min_samples_leaf": combo["min_samples_leaf"],
        "max_features": combo["max_features"],
        "max_depth": combo["max_depth"],
        "n_jobs": 1,  # forced -- parallelism comes from the flat task pool, not per-fit
    }


def build_task_list(combos: list[dict], targets: list[str], data_by_target: dict) -> list[tuple]:
    """(combo_id, combo, target, held_region, qrf_params) per fold-fit."""
    tasks = []
    for combo in combos:
        cid = combo_id(combo)
        qrf_params = qrf_params_for_combo(combo)
        for target in targets:
            regions = data_by_target[target]["regions"]
            for region in sorted(set(regions)):
                tasks.append((cid, combo, target, region, qrf_params))
    return tasks


def regroup_and_score(tasks: list[tuple], fit_results: list, data_by_target: dict) -> list[dict]:
    """Reassembles per-(combo, target) out-of-fold arrays from the flat
    per-fold fit_results (same shape leave_region_out_cv itself builds
    internally) and scores each with cv_metrics_by_zone."""
    grouped: dict[tuple, dict] = {}
    for (cid, combo, target, region, _qrf_params), result in zip(tasks, fit_results):
        key = (cid, target)
        bundle = grouped.setdefault(key, {"combo": combo, "regions": [], "results": []})
        bundle["regions"].append(region)
        bundle["results"].append(result)

    scored = []
    for (cid, target), bundle in grouped.items():
        d = data_by_target[target]
        y = d["y"]
        n = len(y)
        oof_lo = np.full(n, np.nan)
        oof_median = np.full(n, np.nan)
        oof_hi = np.full(n, np.nan)
        oof_di = np.full(n, np.nan)
        for region, result in zip(bundle["regions"], bundle["results"]):
            if result is None:
                continue
            m = result["test_mask"]
            oof_lo[m], oof_median[m], oof_hi[m], oof_di[m] = (
                result["lo"], result["median"], result["hi"], result["di"],
            )
        valid = ~np.isnan(oof_median)
        cv = {"valid": valid, "oof_lo": oof_lo, "oof_median": oof_median, "oof_hi": oof_hi, "oof_di": oof_di}
        metrics = td.cv_metrics_by_zone(y, d["zones"], cv)
        scored.append({"combo_id": cid, "target": target, "combo": bundle["combo"], "metrics": metrics})
    return scored


def run_sweep(targets: list[str], stage: int, n_jobs: int, limit: int | None, output_path: str | None) -> list[dict]:
    combos = combo_grid(stage)
    if limit:
        combos = combos[:limit]
    print(f"Sweep: {len(combos)} hyperparameter combo(s) x {len(targets)} target(s)")

    t0 = time.monotonic()
    rows = td.load_training_rows()
    print(f"Loaded {len(rows)} row(s) in {time.monotonic() - t0:.1f}s")

    data_by_target = {}
    for target in targets:
        t0 = time.monotonic()
        X, y, regions, zones, _lons, _lats = td.build_training_feature_matrix(rows, target)
        data_by_target[target] = {"X": X, "y": y, "regions": regions, "zones": zones}
        print(f"[{target}] feature matrix: {X.shape[0]} usable row(s) in {time.monotonic() - t0:.1f}s")

    tasks = build_task_list(combos, targets, data_by_target)
    print(f"Flattened to {len(tasks)} single-threaded fold-fit task(s), running with n_jobs={n_jobs}")

    t_sweep = time.monotonic()
    fit_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(td._fit_one_qrf_fold)(
            region,
            data_by_target[target]["X"],
            data_by_target[target]["y"],
            np.array(data_by_target[target]["regions"]),
            qrf_params,
            td._MIN_FOLD_TRAIN_ROWS,
        )
        for (_cid, _combo, target, region, qrf_params) in tasks
    )
    elapsed = time.monotonic() - t_sweep
    print(f"All {len(tasks)} fit(s) done in {elapsed:.1f}s ({elapsed / len(tasks):.2f}s/task avg)")

    scored = regroup_and_score(tasks, fit_results, data_by_target)
    for entry in scored:
        o = entry["metrics"]["overall"]
        print(
            f"[{entry['target']}] {entry['combo_id']}: "
            f"rmse_grid={o['rmse_grid_c']:.4f} rmse_qrf={o['rmse_qrf_c']:.4f} "
            f"beats_grid={o['qrf_beats_grid']} bias={o['bias_qrf_c']:.4f}"
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(scored, f, indent=2)
        print(f"Wrote {len(scored)} result(s) to {output_path}")

    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--targets", default="tmax,tmin")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N combos (dry run)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_sweep(args.targets.split(","), args.stage, args.n_jobs, args.limit, args.output)


if __name__ == "__main__":
    main()
