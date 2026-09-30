"""
Out-of-time station evaluation: fit the QRF delta model once on rows dated on/before --cutoff and
score it on rows dated after --cutoff, against the raw ERA5-Land grid value.

Reuses train_downscaling's own row loading, merging, feature matrix and QRF params, so the model is
the trainer's model, not a proxy. No cross-validation, no artifacts published: per-row predictions go
to --out-dir/oot_{tmax,tmin}.csv.gz and a summary to --out-dir/oot_summary.json.

Test rows are split into two sets by whether the station has any training row:
  A  station has no row on/before the cutoff (unseen station, unseen period)
  B  station has training rows (seen station, unseen period)
Summaries are reported per set and never pooled. Error is (grid + predicted delta) - station, i.e.
predicted delta minus observed delta; the grid's error is -observed delta.

Example (candidate-only, reads ghcn_training read-only):
  python scripts/oot_eval.py --cutoff 2024-12-31 --extra-rows-json rows_none.json \\
      --extra-rows-json rows_break.json --exclude-station-since IN005010600:2023-01-01 \\
      --out-dir /opt/ghcn-build/oot --n-jobs 8
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_downscaling as td  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(err_model: np.ndarray, err_grid: np.ndarray) -> dict:
    return {
        "n_rows": int(len(err_model)),
        "rmse_model_c": float(np.sqrt(np.mean(err_model ** 2))) if len(err_model) else None,
        "rmse_grid_c": float(np.sqrt(np.mean(err_grid ** 2))) if len(err_grid) else None,
        "bias_model_c": float(np.mean(err_model)) if len(err_model) else None,
        "bias_grid_c": float(np.mean(err_grid)) if len(err_grid) else None,
    }


def summarize(test_set: list[str], zones: list[str], delta_obs: np.ndarray, delta_pred: np.ndarray,
              stations: list[str]) -> dict:
    err_model = delta_pred - delta_obs
    err_grid = -delta_obs
    ts, zn, st = np.array(test_set), np.array(zones), np.array(stations)
    out: dict = {}
    for name in ("A", "B"):
        m = ts == name
        block = {"overall": _metrics(err_model[m], err_grid[m]),
                 "n_stations": int(len(set(st[m])))}
        bsh = m & (zn == "BSh")
        block["BSh"] = {**_metrics(err_model[bsh], err_grid[bsh]), "n_stations": int(len(set(st[bsh])))}
        out[name] = block
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cutoff", required=True, help="YYYY-MM-DD; train on rows dated <= cutoff, test on later rows")
    p.add_argument("--extra-rows-json", action="append", default=[])
    p.add_argument("--exclude-station-since", action="append", default=[], metavar="STATION:YYYY-MM-DD")
    p.add_argument("--feature-set", choices=sorted(td.SUPPORTED_FEATURE_ORDERS), default="base")
    p.add_argument("--targets", default="tmax,tmin", help="comma list of tmax,tmin")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-jobs", type=int, default=-1, help="QRF fit/predict workers")
    args = p.parse_args()
    if args.feature_set != "base":
        raise SystemExit("regime feature sets need --regime-features-csv plumbing; this evaluation uses the base set")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = td.load_training_rows()
    print(f"Loaded {len(rows)} ghcn_training row(s)")
    extra_sources = []
    for path in args.extra_rows_json:
        with open(path, "rb") as f:
            raw = f.read()
        payload = json.loads(raw)
        if payload.get("complete") is not True:
            raise SystemExit(f"{path} is not marked complete")
        rows, counts = td.merge_extra_rows(rows, payload["rows"])
        extra_sources.append({"path": os.path.basename(path), "sha256": hashlib.sha256(raw).hexdigest(), **counts})
        print(f"Extra rows from {path}: {counts}")
    excluded = {}
    if args.exclude_station_since:
        rows, excluded = td.exclude_station_since(rows, args.exclude_station_since)
        print(f"Excluded station rows: {excluded}")

    from quantile_forest import RandomForestQuantileRegressor

    summary = {"cutoff": args.cutoff, "feature_set": args.feature_set, "extra_rows_sources": extra_sources,
               "excluded_station_rows": excluded, "trainer_file_sha256": _sha256(td.__file__),
               "qrf_params": {**td._QRF_PARAMS, "n_jobs": args.n_jobs}, "targets": {}}
    feature_order = td.SUPPORTED_FEATURE_ORDERS[args.feature_set]
    for target in args.targets.split(","):
        X, y, regions, zones, lons, lats, keep = td.build_training_feature_matrix(
            rows, target, return_keep=True, feature_order=feature_order)
        dates = np.array([str(rows[i]["date"])[:10] for i in keep])
        stations = np.array([rows[i]["station_id"] for i in keep])
        train = dates <= args.cutoff
        test = ~train
        train_stations = set(stations[train])
        print(f"[{target}] train rows {int(train.sum())}, test rows {int(test.sum())}, "
              f"train stations {len(train_stations)}")
        if train.sum() < td._MIN_FOLD_TRAIN_ROWS or test.sum() == 0:
            raise SystemExit(f"[{target}] empty train or test side")
        model = RandomForestQuantileRegressor(**{**td._QRF_PARAMS, "n_jobs": args.n_jobs}).fit(X[train], y[train])
        pred = model.predict(X[test], quantiles=[0.5])
        pred = np.asarray(pred).reshape(-1)
        t_idx = np.where(test)[0]
        test_set = ["B" if stations[j] in train_stations else "A" for j in t_idx]
        grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
        path = os.path.join(args.out_dir, f"oot_{target}.csv.gz")
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["station_id", "date", "zone", "region", "test_set", "grid_c", "delta_obs", "delta_pred"])
            for k, j in enumerate(t_idx):
                r = rows[keep[j]]
                w.writerow([stations[j], dates[j], zones[j], regions[j], test_set[k], r.get(grid_col),
                            f"{y[j]:.4f}", f"{pred[k]:.4f}"])
        summary["targets"][target] = {
            "train_rows": int(train.sum()), "test_rows": int(test.sum()),
            "train_stations": len(train_stations),
            "predictions_file": os.path.basename(path), "predictions_sha256": _sha256(path),
            "scores": summarize(test_set, [zones[j] for j in t_idx], y[t_idx], pred, [stations[j] for j in t_idx]),
        }
        print(f"[{target}] {json.dumps(summary['targets'][target]['scores'])}")

    out = os.path.join(args.out_dir, "oot_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out} sha256={_sha256(out)}")


if __name__ == "__main__":
    main()
