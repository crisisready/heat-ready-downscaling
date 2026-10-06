"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.


Offline QRF downscaling model trainer (downscaling build Phase 4 -- see
docs/plan-2026-07-18-neighborhood-resolution-downscaling.md sections 4, 5
items 11-13, and 4.7 for the AOA method's literature grounding).

Loads ghcn_training (populated by scripts/build_training_set.py), fits two
quantile_forest.RandomForestQuantileRegressor models (delta_tmax_c,
delta_tmin_c) on the 14-feature vector from FEATURE_ORDER, runs
leave-region-out cross-validation (fold key = region -- a country/admin1
code, never a random split; Roberts et al. 2017), performs split-conformal
calibration per Koppen climate zone (section 4.4), computes an AOA-style
(Meyer & Pebesma 2021) feature-importance-weighted dissimilarity index for
out-of-distribution detection (section 4.7's decided replacement for a plain
Mahalanobis distance), then fits a final pair of models on all data and
writes model.joblib + metadata.json to S3 (section 2.6).

Also runs a regression-kriging comparison baseline (regression_kriging_cv)
across the SAME leave-region-out folds -- a real, literature-matched
robustness check, not just a citation: Appelhans et al.'s Kilimanjaro study
(section 4.7) found a residual-kriging hybrid competitive with plain RF,
so per-zone RMSE is reported for both QRF and kriging, and which one wins,
rather than assuming the literature review settles it.

Gate (plan section 5 item 11): a climate zone's leave-region-out downscaled
RMSE must beat raw ERA5-Land's RMSE for that zone, or the zone is excluded
from zones_passing_cv_gate in metadata -- Phase 5's inference wiring must
fall back to the raw grid value for polygons in an excluded zone. This
script does not enforce the gate itself (it has no serving path to fall
back within); it computes and reports the per-zone pass/fail so a human (or
Phase 5's code) can act on it.

Usage:
    python scripts/train_downscaling.py --model-version ds-2026.07-rf1 \
        --profile nish-climateverse
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from datetime import date, datetime

import numpy as np

from heatready_downscaling.contract import aoa_dissimilarity, feature_importance_weights
from heatready_downscaling import grid_bias_prior
from heatready_downscaling import ssod_guard as ssod_guard_mod
from heatready_downscaling.features import (FEATURE_ORDER, GRID_BIAS_FEATURES, REGIME_FEATURES,
                                            SUPPORTED_FEATURE_ORDERS, build_feature_matrix)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(REPO_ROOT, "credentials.yaml")

_QRF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 5,
    # quantile_forest's own default (max_features=1.0, every split considers
    # all 20 features) was the real cause of ds-2026.07-rf1 losing to a bare
    # linear trend on identical folds -- all-feature splitting maximizes
    # correlation between trees, which is exactly the wrong thing for a
    # leave-region-out transfer task. Confirmed via a hyperparameter sweep
    # (docs/pipeline-fable-consult-2026-07-19-downscaling-diagnosis.md
    # experiment #1, scripts/sweep_qrf_hyperparams.py): "sqrt" alone, with
    # min_samples_leaf unchanged, flips tmax from failing every zone to
    # passing all 19/19, and tmin from failing most zones to passing 15/19
    # (the remaining 4 -- Am/As/BWh/Cwa, all tropical/arid -- are a distinct,
    # narrower gap, not a regularization problem; see the roadmap doc).
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 20260718,  # fixed seed -- reproducible CV/final fits, not wall-clock-derived
}
_MIN_FOLD_TRAIN_ROWS = 50  # a fold with fewer rows than this on the training side is skipped, not fit
_MIN_CONFORMAL_POINTS = 20  # a zone with fewer out-of-fold points than this falls back to "_default"
_AOA_TRAINING_SAMPLE_CAP = 20_000  # bounds the shipped NearestNeighbors index size/joblib artifact size
_KRIGING_VARIOGRAM_SAMPLE_CAP = 3_000  # bounds pykrige's O(n^2) pairwise-distance variogram fit


def _bucket_from_credentials() -> str:
    if not os.path.exists(CREDENTIALS_FILE):
        bucket = os.environ.get("VULNERABILITY_DATA_BUCKET")
        if bucket:
            return bucket
        raise FileNotFoundError(f"{CREDENTIALS_FILE} not found and VULNERABILITY_DATA_BUCKET is unset")
    import yaml
    with open(CREDENTIALS_FILE) as f:
        creds = yaml.safe_load(f)
    return creds["vulnerability"]["bucket"]


def load_training_rows() -> list[dict]:
    """All ghcn_training rows with the columns this script needs non-null --
    region/climate_zone (CV fold key + zone reporting) and both deltas/grid
    values (the regression targets). Individual covariate columns are
    allowed to be null here; build_training_feature_matrix drops rows
    missing a specific feature per-target, so a row missing only
    lst_warm_season_anomaly_c still contributes to whichever rows/targets
    it CAN support -- filtering it out entirely at the SQL level would
    silently waste otherwise-usable station-days."""
    import db
    return db.execute(
        """
        SELECT station_id, date, lon, lat, region, climate_zone,
               station_tmax_c, station_tmin_c, grid_tmax_c, grid_tmin_c,
               delta_tmax_c, delta_tmin_c,
               lst_warm_season_anomaly_c, canopy_height_mean_m, canopy_frac_over_3m,
               wc_built_frac, wc_tree_frac, wc_water_frac, ghsl_urban_fraction,
               pop_density_per_km2, elevation_rel_to_gridcell_m, elevation_mean_m,
               slope_deg, aspect_deg, grid_specific_humidity_kgkg, koppen_main_group_code,
               nighttime_wind_ms
        FROM ghcn_training
        WHERE region IS NOT NULL AND climate_zone IS NOT NULL
          AND grid_tmax_c IS NOT NULL AND grid_tmin_c IS NOT NULL
          AND delta_tmax_c IS NOT NULL AND delta_tmin_c IS NOT NULL
        """,
    )


EXTRA_ROWS_ZONE = "_extra_rows"
# Extra rows may carry only one target (e.g. build_gsod_bsh_rows.py --tmax-only): they need these
# plus at least one of delta_tmax_c/delta_tmin_c, and build_training_feature_matrix then drops a
# row only from the target it lacks.
_EXTRA_REQUIRED_NON_NULL = ("region", "climate_zone", "grid_tmax_c", "grid_tmin_c")


def _subset_cv(cv: dict, mask: np.ndarray) -> dict:
    """leave_region_out_cv's per-row arrays restricted to `mask` (non-array entries kept)."""
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v) for k, v in cv.items()}


def merge_extra_rows(rows: list[dict], extra_rows: list[dict]) -> tuple[list[dict], dict]:
    """Add rows from a builder's JSON output (e.g. build_gsod_bsh_rows.py) to the ghcn_training
    rows, without writing them to the database first -- so a candidate model can be trained on
    a corpus extension before anyone decides whether to upsert it.

    An extra row needs region, climate_zone, grid_tmax_c and grid_tmin_c plus at least one of
    delta_tmax_c/delta_tmin_c (a single-target row trains only that target). On a
    (station_id, date) key that already exists in ghcn_training, the database row wins: the
    extension adds days, it never silently replaces an ingested one. Across files, the first file
    wins, so pass a file with both targets before a tmax-only one that overlaps it.
    Returns (merged rows, counts)."""
    existing = {(r["station_id"], str(r["date"])) for r in rows if not r.get("_extra")}
    seen_extra = {(r["station_id"], str(r["date"])) for r in rows if r.get("_extra")}
    seen = seen_extra
    added, dup_db, dup_extra, dropped = [], 0, 0, 0
    for r in extra_rows:
        if (any(r.get(k) is None for k in _EXTRA_REQUIRED_NON_NULL)
                or (r.get("delta_tmax_c") is None and r.get("delta_tmin_c") is None)):
            dropped += 1
            continue
        key = (r["station_id"], str(r["date"]))
        if key in existing:
            dup_db += 1
            continue
        if key in seen:
            dup_extra += 1
            continue
        seen.add(key)
        added.append({**r, "_extra": True})
    return rows + added, {"added": len(added), "duplicate_of_db_row": dup_db,
                          "duplicate_within_extra": dup_extra, "dropped_null": dropped}


# Column names in a --regime-features-csv (Open-Meteo archive daily output) -> REGIME_FEATURES.
REGIME_CSV_COLUMNS = {
    "precipitation_sum": "era5_precip_sum_mm", "shortwave_radiation_sum": "era5_sw_rad_sum_mj",
    "wind_speed_10m_max": "era5_wind_max_kmh", "relative_humidity_2m_mean": "era5_rh_mean_pct",
}


def attach_regime_features(rows: list[dict], csv_paths: list[str]) -> dict:
    """Join daily ERA5 regime covariates onto every row by (station_id, date), in place. The CSVs
    carry station_id, date (YYYY-MM-DD) and REGIME_CSV_COLUMNS' keys. A row with no match keeps the
    features as None, so build_training_feature_matrix drops it for the regime feature order (no
    imputation). Returns counts."""
    import csv
    table = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            for rec in csv.DictReader(f):
                vals = {}
                for src, dst in REGIME_CSV_COLUMNS.items():
                    v = rec.get(src)
                    x = float(v) if v not in (None, "") else None
                    # a literal "nan"/"inf" must count as missing, or it passes the completeness mask
                    vals[dst] = x if x is not None and math.isfinite(x) else None
                table[(rec["station_id"], rec.get("date") or rec.get("time"))] = vals
    matched = 0
    for r in rows:
        vals = table.get((r["station_id"], str(r["date"])[:10]))
        if vals is not None:
            matched += 1
            r.update(vals)
        else:
            r.update({c: None for c in REGIME_FEATURES})
    return {"rows": len(rows), "matched": matched, "table_rows": len(table)}


def exclude_station_since(rows: list[dict], specs: list[str]) -> tuple[list[dict], dict]:
    """Drop every row of a station dated on or after a cut date, for records whose recent level is
    unresolved (e.g. Ahmedabad airport IN005010600 from 2023-01-01: its GSOD and METAR records diverge
    by 1-1.5 C since 2023). `specs` are "STATION_ID:YYYY-MM-DD". Applies to database and extra rows
    alike. Returns (kept rows, {spec: rows dropped})."""
    cuts = {}
    for spec in specs:
        sid, _, d = spec.partition(":")
        if not sid or len(d) != 10:
            raise SystemExit(f"--exclude-station-since expects STATION_ID:YYYY-MM-DD, got {spec!r}")
        if sid in cuts:
            raise SystemExit(f"--exclude-station-since given twice for {sid}")
        cuts[sid] = d
    dropped = {f"{k}:{v}": 0 for k, v in cuts.items()}
    kept = []
    for r in rows:
        cut = cuts.get(r["station_id"])
        if cut is not None and str(r["date"])[:10] >= cut:
            dropped[f"{r['station_id']}:{cut}"] += 1
            continue
        kept.append(r)
    present = {r["station_id"] for r in rows}
    unknown = sorted(set(cuts) - present)
    if unknown:  # a mistyped id would otherwise exclude nothing, silently
        raise SystemExit(f"--exclude-station-since: no rows for station(s) {unknown}")
    return kept, dropped


def read_holdout_stations(path: str) -> list[str]:
    """Station ids, one per line (# comments and blank lines ignored), from a scorecard holdout list
    (scorecard/v1/holdout_stations.txt): stations moved out of training so the out-of-time scorecard
    can score them as unseen."""
    with open(path) as f:
        ids = [line.split("#", 1)[0].strip() for line in f]
    ids = [i for i in ids if i]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate station ids")
    return ids


def exclude_stations(rows: list[dict], station_ids: list[str]) -> tuple[list[dict], dict]:
    """Drop every row of the given stations at every date (database and extra rows alike). A listed
    station with no rows is reported with 0, not an error: a holdout list spans corpus parts that a
    given run may not load. Returns (kept rows, {station_id: rows dropped})."""
    wanted = set(station_ids)
    dropped = {sid: 0 for sid in station_ids}
    kept = []
    for r in rows:
        if r["station_id"] in wanted:
            dropped[r["station_id"]] += 1
            continue
        kept.append(r)
    return kept, dropped


def write_oof_csv(path: str, rows: list[dict], keep: list[int], y: np.ndarray, zones: list[str],
                  cv: dict, grid_col: str) -> int:
    """Per-row leave-region-out out-of-fold predictions (station_id, date, zone, region, grid value,
    observed delta, OOF median delta), so centered/station-month skill and amplitude can be computed
    outside the trainer. gzip if the path ends in .gz. Returns rows written."""
    import csv, gzip
    opener = gzip.open if path.endswith(".gz") else open
    n = 0
    with opener(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "date", "zone", "region", "grid_c", "delta_obs", "delta_oof_median"])
        for j, i in enumerate(keep):
            if not cv["valid"][j]:
                continue
            r = rows[i]
            w.writerow([r["station_id"], str(r["date"])[:10], zones[j], r["region"], r.get(grid_col),
                        f"{y[j]:.4f}", f"{cv['oof_median'][j]:.4f}"])
            n += 1
    return n


def build_training_feature_matrix(
    rows: list[dict], target: str, return_keep: bool = False, feature_order: tuple = FEATURE_ORDER,
) -> tuple:
    """
    Build (X, y, regions, zones, lons, lats) for `target` in {"tmax", "tmin"}
    from ghcn_training rows, reusing build_feature_matrix's
    exact per-row feature construction (same feature order, same log1p/doy
    transforms) so training and inference can never silently diverge on how
    a feature is computed. Rows missing any of the 14 features are dropped
    (no imputation, matching the design's explicit non-goal) -- logged so a
    silent data-quality regression doesn't look like "training just worked
    with less data than expected." lons/lats (not part of the model's own
    feature vector) are returned alongside X/y for regression_kriging_cv,
    which needs real coordinates to krige spatially.
    """
    covariate_rows = [
        {
            "grid_tmax_c": r.get("grid_tmax_c"), "grid_tmin_c": r.get("grid_tmin_c"),
            "lat": r.get("lat"), "lon": r.get("lon"), "date": r.get("date"),
            "lst_warm_season_anomaly_c": r.get("lst_warm_season_anomaly_c"),
            "canopy_height_mean_m": r.get("canopy_height_mean_m"),
            "canopy_frac_over_3m": r.get("canopy_frac_over_3m"),
            "wc_built_frac": r.get("wc_built_frac"), "wc_tree_frac": r.get("wc_tree_frac"),
            "wc_water_frac": r.get("wc_water_frac"), "ghsl_urban_fraction": r.get("ghsl_urban_fraction"),
            "pop_density_per_km2": r.get("pop_density_per_km2"),
            "elevation_rel_to_gridcell_m": r.get("elevation_rel_to_gridcell_m"),
            "elevation_mean_m": r.get("elevation_mean_m"),
            "slope_deg": r.get("slope_deg"), "aspect_deg": r.get("aspect_deg"),
            "grid_specific_humidity_kgkg": r.get("grid_specific_humidity_kgkg"),
            "koppen_main_group_code": r.get("koppen_main_group_code"),
            "nighttime_wind_ms": r.get("nighttime_wind_ms"),
            **{c: r.get(c) for c in REGIME_FEATURES},
            **{c: r.get(c) for c in GRID_BIAS_FEATURES},
        }
        for r in rows
    ]
    X_all, complete_mask, _ = build_feature_matrix(covariate_rows, target, feature_order=feature_order)
    delta_col = "delta_tmax_c" if target == "tmax" else "delta_tmin_c"

    def _target_is_finite(i: int) -> bool:
        # The SQL guard (delta_tmax_c/delta_tmin_c IS NOT NULL) does not
        # catch this: a grid cell fully masked out (e.g. a tiny island whose
        # ERA5-Land cell is entirely ocean) can store an actual IEEE NaN in
        # a DOUBLE PRECISION column -- Postgres treats NaN as a valid float,
        # not a null, so it sails through "IS NOT NULL" and would otherwise
        # crash the QRF fit deep inside a joblib worker with an opaque
        # "Input y contains NaN" far from this, the real cause.
        v = rows[i].get(delta_col)
        return v is not None and math.isfinite(v)

    keep = [i for i, ok in enumerate(complete_mask) if ok and _target_is_finite(i)]
    dropped = len(rows) - len(keep)
    if dropped:
        print(f"[{target}] dropping {dropped}/{len(rows)} row(s) missing a required feature or a non-finite target")

    X = X_all[keep]
    y = np.array([rows[i].get(delta_col) for i in keep], dtype=float)
    regions = [rows[i]["region"] for i in keep]
    zones = [rows[i]["climate_zone"] for i in keep]
    lons = np.array([rows[i]["lon"] for i in keep], dtype=float)
    lats = np.array([rows[i]["lat"] for i in keep], dtype=float)
    if return_keep:
        return X, y, regions, zones, lons, lats, keep
    return X, y, regions, zones, lons, lats


def attach_grid_bias(rows: list[dict], target: str) -> dict:
    """Set grid_bias_prior_c / grid_bias_support on every row for `target`, from a station-month table built
    from all of `rows` (a station never informs its own prior; see grid_bias_prior). Returns the table, which
    is what serving would compute the prior from."""
    delta_col = "delta_tmax_c" if target == "tmax" else "delta_tmin_c"
    table = grid_bias_prior.station_month_table([r["station_id"] for r in rows], [r["lat"] for r in rows],
                                                [r["lon"] for r in rows], [r["date"] for r in rows],
                                                [r.get(delta_col) for r in rows])
    prior, support = grid_bias_prior.priors_for_rows(table, [r["lat"] for r in rows], [r["lon"] for r in rows],
                                                     [r["date"] for r in rows])
    for r, p, w in zip(rows, prior, support):
        r["grid_bias_prior_c"], r["grid_bias_support"] = float(p), float(w)
    return table


def grid_bias_context(rows: list[dict], keep: list[int], feature_order: tuple) -> dict | None:
    """What a CV fold needs to rebuild the grid-bias columns from its own training rows, or None when the
    feature order has none."""
    if GRID_BIAS_FEATURES[0] not in feature_order:
        return None
    return {"cols": tuple(feature_order.index(c) for c in GRID_BIAS_FEATURES),
            "station_id": [rows[i]["station_id"] for i in keep], "lat": [rows[i]["lat"] for i in keep],
            "lon": [rows[i]["lon"] for i in keep], "date": [rows[i]["date"] for i in keep]}


def refit_grid_bias_columns(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, ctx: dict) -> np.ndarray:
    """X with the grid-bias columns recomputed from the rows in train_mask alone, so a held-out region's
    priors never see its own stations (training rows still skip their own site)."""
    idx = np.flatnonzero(train_mask)
    table = grid_bias_prior.station_month_table([ctx["station_id"][i] for i in idx], [ctx["lat"][i] for i in idx],
                                                [ctx["lon"][i] for i in idx], [ctx["date"][i] for i in idx],
                                                y[idx].tolist())
    prior, support = grid_bias_prior.priors_for_rows(table, ctx["lat"], ctx["lon"], ctx["date"])
    X = X.copy()
    X[:, ctx["cols"][0]], X[:, ctx["cols"][1]] = prior, support
    return X


def apply_cv_region_map(regions: list[str], path: str | None) -> tuple[list[str], dict | None]:
    """Group FIPS regions into one leave-region-out fold each ({"CH": "NE_ASIA", "KN": "NE_ASIA", ...}), so a
    whole region the model must generalise to is held out at once. Unlisted regions stay their own fold."""
    if not path:
        return regions, None
    with open(path) as f:
        mapping = json.load(f)
    return [mapping.get(r, r) for r in regions], mapping


def _fit_one_qrf_fold(
    held_region: str, X: np.ndarray, y: np.ndarray, regions_arr: np.ndarray,
    qrf_params: dict, min_fold_train_rows: int, grid_bias_ctx: dict | None = None,
) -> dict | None:
    """
    Fit + score ONE leave-region-out fold. Standalone (not a closure) and
    takes every threshold as an explicit argument, not a module-level
    lookup, so it works correctly when run in a joblib worker PROCESS (a
    subprocess re-imports this module fresh -- a test's
    patch.object(module, "_MIN_FOLD_TRAIN_ROWS", ...) would silently not
    apply inside that subprocess if this function read the module global
    directly instead of receiving it as a parameter).

    Returns None if the fold is skipped (too few training rows) -- the
    caller decides how to report that, since a skip is a normal outcome,
    not this function's job to print from a background worker.
    """
    from quantile_forest import RandomForestQuantileRegressor

    test_mask = regions_arr == held_region
    train_mask = ~test_mask
    if train_mask.sum() < min_fold_train_rows or test_mask.sum() == 0:
        return None
    if grid_bias_ctx is not None:
        X = refit_grid_bias_columns(X, y, train_mask, grid_bias_ctx)

    model = RandomForestQuantileRegressor(**qrf_params).fit(X[train_mask], y[train_mask])
    preds = model.predict(X[test_mask], quantiles=[0.025, 0.5, 0.975])

    weights = feature_importance_weights(model)
    train_mean = X[train_mask].mean(axis=0)
    train_std = X[train_mask].std(axis=0)
    di = aoa_dissimilarity(X[test_mask], X[train_mask], weights, train_mean, train_std)

    return {
        "region": held_region, "test_mask": test_mask,
        "lo": preds[:, 0], "median": preds[:, 1], "hi": preds[:, 2], "di": di,
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
    }


def leave_region_out_cv(X: np.ndarray, y: np.ndarray, regions: list[str], n_jobs: int = -1,
                        grid_bias_ctx: dict | None = None) -> dict:
    """
    Spatial leave-region-out CV (Roberts et al. 2017 -- never a random
    split, since neighboring station-days are spatially correlated and
    would leak across a random train/test boundary). For every distinct
    region, fit a fold model on every OTHER region's rows and predict the
    held-out region, so every row gets exactly one out-of-fold prediction.

    Folds are fit CONCURRENTLY (joblib, process-based) -- each fold's fit
    is completely independent of every other fold's, and as the training
    set grows to more countries/regions this is the dominant lever for
    keeping retrain wall-clock time down (confirmed 2026-07-18: with only
    2-5 folds the sequential cost is barely noticeable, but a future
    expansion to dozens of countries would make this the long pole).
    _QRF_PARAMS' own n_jobs is overridden to 1 for the per-fold fit here --
    letting each fold ALSO parallelize across all cores while multiple
    folds run concurrently would oversubscribe the machine (F folds x C
    threads each, competing for C cores, instead of the intended F-folds-
    across-C-cores split).

    Returns a dict with the out-of-fold arrays (median/lo/hi prediction,
    AOA dissimilarity, and which rows actually got a fold prediction) --
    the raw material both the per-zone RMSE gate (plan section 5 item 11)
    and the conformal calibration (item 12) are computed from.
    """
    from joblib import Parallel, delayed

    regions_arr = np.array(regions)
    unique_regions = sorted(set(regions))
    fold_qrf_params = {**_QRF_PARAMS, "n_jobs": 1}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_qrf_fold)(region, X, y, regions_arr, fold_qrf_params, _MIN_FOLD_TRAIN_ROWS, grid_bias_ctx)
        for region in unique_regions
    )

    n = len(y)
    oof_lo = np.full(n, np.nan)
    oof_median = np.full(n, np.nan)
    oof_hi = np.full(n, np.nan)
    oof_di = np.full(n, np.nan)

    for region, result in zip(unique_regions, results):
        if result is None:
            print(f"  fold '{region}': skipped (fewer than {_MIN_FOLD_TRAIN_ROWS} training rows)")
            continue
        m = result["test_mask"]
        oof_lo[m], oof_median[m], oof_hi[m], oof_di[m] = result["lo"], result["median"], result["hi"], result["di"]
        print(f"  fold '{region}': {result['n_train']} train / {result['n_test']} held-out rows")

    valid = ~np.isnan(oof_median)
    return {
        "valid": valid, "oof_lo": oof_lo, "oof_median": oof_median,
        "oof_hi": oof_hi, "oof_di": oof_di,
    }


def cv_metrics_by_zone(y: np.ndarray, zones: list[str], cv: dict, kriging_oof_median: np.ndarray | None = None) -> dict:
    """
    Per-zone (plus overall) RMSE-grid/RMSE-qrf/bias/raw-QRF-coverage -- the
    pass/fail gate (plan section 5 item 11): a zone must show
    rmse_qrf_c < rmse_grid_c or it doesn't ship for that regime. rmse_grid_c
    is RMS(y) directly -- "predicting delta=0" IS the raw grid value, so no
    separate raw-metrics computation is needed.

    kriging_oof_median (regression_kriging_cv's out-of-fold predictions, the
    SAME leave-region-out folds QRF was scored on): when given, also reports
    rmse_kriging_c/bias_kriging_c/qrf_beats_kriging per zone -- a real,
    literature-matched robustness comparison (plan section 4.7's Kilimanjaro
    precedent found a residual-kriging hybrid competitive with/better than
    plain RF on one metric; "cite the paper" isn't the same as "checked it
    against our own data"). Kriging has no quantile/interval output
    comparable to QRF's, so it's scored on point-prediction RMSE/bias only.
    """
    valid = cv["valid"]
    zones_arr = np.array(zones)
    y_err_qrf = y - cv["oof_median"]
    y_err_kriging = (y - kriging_oof_median) if kriging_oof_median is not None else None

    def _metrics(mask: np.ndarray) -> dict | None:
        y_m, err_m = y[mask], y_err_qrf[mask]
        if len(y_m) == 0:
            return None
        rmse_grid = float(np.sqrt(np.mean(y_m ** 2)))
        rmse_qrf = float(np.sqrt(np.mean(err_m ** 2)))
        in_interval = (y_m >= cv["oof_lo"][mask]) & (y_m <= cv["oof_hi"][mask])
        result = {
            "n": int(mask.sum()),
            "rmse_grid_c": rmse_grid,
            "rmse_qrf_c": rmse_qrf,
            "bias_qrf_c": float(np.mean(err_m)),
            "raw_qrf_interval_coverage": float(np.mean(in_interval)),
            "qrf_beats_grid": rmse_qrf < rmse_grid,
        }
        if y_err_kriging is not None:
            kriging_mask = mask & ~np.isnan(y_err_kriging)
            if kriging_mask.sum() > 0:
                rmse_kriging = float(np.sqrt(np.mean(y_err_kriging[kriging_mask] ** 2)))
                result["rmse_kriging_c"] = rmse_kriging
                result["bias_kriging_c"] = float(np.mean(y_err_kriging[kriging_mask]))
                result["qrf_beats_kriging"] = rmse_qrf < rmse_kriging
        return result

    by_zone = {}
    for zone in sorted(set(zones_arr[valid])):
        metrics = _metrics(valid & (zones_arr == zone))
        if metrics is not None:
            by_zone[zone] = metrics
    overall = _metrics(valid)
    return {"by_zone": by_zone, "overall": overall}


def _fit_one_kriging_fold(
    held_region: str, X: np.ndarray, y: np.ndarray, regions_arr: np.ndarray,
    lons: np.ndarray, lats: np.ndarray, n_closest_points: int, min_fold_train_rows: int,
    variogram_sample_cap: int = _KRIGING_VARIOGRAM_SAMPLE_CAP, seed: int = 20260718,
) -> dict | None:
    """Fit + score ONE regression-kriging fold. Standalone, explicit
    parameters (not module globals) for the same joblib-subprocess/test-
    patchability reason as _fit_one_qrf_fold. Returns None if skipped.

    pykrige's OrdinaryKriging builds the full O(n^2) pairwise-distance
    matrix over EVERY point passed to its constructor to fit the empirical
    variogram -- n_closest_points only bounds the neighbor search at
    .execute() (prediction) time, not the fit itself. Passing a full
    leave-region-out training fold (often 150k-250k rows) blew this up to
    a 100+ GiB allocation attempt (confirmed 2026-07-19: every fold OOM'd
    and silently fell back to the trend-only prediction, making every
    "QRF vs kriging" comparison up to that point actually "QRF vs a bare
    linear trend"). Fixed the same way _build_aoa_index already bounds a
    similar O(n) structure: fit the variogram (and serve as the kriging
    system's reference set for the .execute() neighbor search) from a
    random subsample capped at variogram_sample_cap -- n_closest_points is
    already small (<=30), so a few thousand reference points is more than
    enough density for that neighbor search, at a small, bounded cost."""
    from pykrige.ok import OrdinaryKriging
    from sklearn.linear_model import LinearRegression

    test_mask = regions_arr == held_region
    train_mask = ~test_mask
    if train_mask.sum() < min_fold_train_rows or test_mask.sum() == 0:
        return None

    trend = LinearRegression().fit(X[train_mask], y[train_mask])
    train_resid = y[train_mask] - trend.predict(X[train_mask])
    trend_pred_test = trend.predict(X[test_mask])

    train_lons, train_lats, train_resid_for_kriging = lons[train_mask], lats[train_mask], train_resid
    if train_mask.sum() > variogram_sample_cap:
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(int(train_mask.sum()), size=variogram_sample_cap, replace=False)
        train_lons = train_lons[sample_idx]
        train_lats = train_lats[sample_idx]
        train_resid_for_kriging = train_resid_for_kriging[sample_idx]

    kriging_error = None
    try:
        ok = OrdinaryKriging(
            train_lons, train_lats, train_resid_for_kriging,
            variogram_model="spherical", enable_plotting=False, verbose=False,
        )
        kriged_resid, _ = ok.execute(
            "points", lons[test_mask], lats[test_mask],
            n_closest_points=min(n_closest_points, len(train_resid_for_kriging)), backend="loop",
        )
    except Exception as exc:
        # A degenerate variogram (e.g. near-constant residuals in a region
        # with little spatial structure) must not crash the whole CV run --
        # fall back to the trend alone for this fold (kriged residual = 0).
        # The failure is returned (not printed here) so the parent process
        # reports it -- consistent ordering with the other fold summaries,
        # and testable without capturing subprocess stdout.
        kriging_error = str(exc)
        kriged_resid = np.zeros(int(test_mask.sum()))

    return {
        "region": held_region, "test_mask": test_mask,
        "pred": trend_pred_test + kriged_resid, "kriging_error": kriging_error,
    }


def regression_kriging_cv(
    X: np.ndarray, y: np.ndarray, regions: list[str], lons: np.ndarray, lats: np.ndarray,
    n_closest_points: int = 30, n_jobs: int = -1,
) -> np.ndarray:
    """
    Regression-kriging out-of-fold predictions across the SAME leave-region-
    out folds as leave_region_out_cv -- a real, literature-matched
    comparison baseline (plan section 4.7): Appelhans et al.'s Kilimanjaro
    study found a residual-kriging hybrid competitive with (better than, on
    one metric) plain RF, so "does QRF actually beat the closest documented
    alternative on OUR data" deserves a real answer, not just a citation.

    Per fold: fit a linear trend on the same covariates QRF uses, krige the
    TRAINING residuals spatially (ordinary kriging on lon/lat, local
    neighborhood via n_closest_points -- exact global kriging is an
    O(n^2)/O(n^3) covariance-matrix solve and intractable at this dataset's
    row count), and predict a held-out row as
    trend(X) + kriged_residual(lon, lat). Folds run CONCURRENTLY (joblib,
    process-based), same rationale as leave_region_out_cv -- each fold's
    trend fit + kriging solve is fully independent of every other fold's.

    No quantiles/uncertainty output -- kriging's variance is a spatial-
    covariance artifact, not a calibrated prediction interval comparable to
    QRF's; this baseline is scored on point-prediction RMSE/bias only (see
    cv_metrics_by_zone's kriging_oof_median parameter).

    Returns oof_median (np.nan where a fold was skipped, same shape/skip
    rule as leave_region_out_cv's oof_median).
    """
    from joblib import Parallel, delayed

    regions_arr = np.array(regions)
    unique_regions = sorted(set(regions))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_kriging_fold)(
            region, X, y, regions_arr, lons, lats, n_closest_points, _MIN_FOLD_TRAIN_ROWS,
        )
        for region in unique_regions
    )

    n = len(y)
    oof_median = np.full(n, np.nan)
    for region, result in zip(unique_regions, results):
        if result is None:
            continue
        if result["kriging_error"] is not None:
            print(f"  regression-kriging fold '{region}': kriging failed ({result['kriging_error']}), using trend only")
        oof_median[result["test_mask"]] = result["pred"]

    return oof_median


def conformal_q95_by_zone(y: np.ndarray, zones: list[str], cv: dict) -> dict[str, float]:
    """
    Split-conformal calibration (plan section 4.4): nonconformity score
    s = |delta_true - delta_pred_median| / (QRF interval half-width),
    computed out-of-fold so the calibration never sees a point the model
    was fit on. Takes the 95th percentile of s per Koppen zone; zones with
    too few calibration points fall back to '_default' (all valid points
    pooled) at inference (QRFModelAdapter.predict already does this
    zone.get(..., zone.get("_default", ...)) lookup).
    """
    valid = cv["valid"]
    half_width = (cv["oof_hi"] - cv["oof_lo"]) / 2.0
    half_width_safe = np.where(half_width > 0, half_width, np.nan)
    nonconformity = np.abs(y - cv["oof_median"]) / half_width_safe
    zones_arr = np.array(zones)

    result: dict[str, float] = {}
    for zone in sorted(set(zones_arr[valid])):
        mask = valid & (zones_arr == zone) & ~np.isnan(nonconformity)
        if mask.sum() < _MIN_CONFORMAL_POINTS:
            continue
        result[zone] = float(np.percentile(nonconformity[mask], 95))

    all_mask = valid & ~np.isnan(nonconformity)
    result["_default"] = float(np.percentile(nonconformity[all_mask], 95)) if all_mask.sum() else 1.0
    return result


def conformal_empirical_coverage(y: np.ndarray, zones: list[str], cv: dict, q95_by_zone: dict) -> float:
    """Empirical coverage of the CALIBRATED (not raw QRF) interval across
    all out-of-fold points -- plan section 5 item 12's check, target
    range [0.93, 0.97]."""
    valid = cv["valid"]
    half_width = (cv["oof_hi"] - cv["oof_lo"]) / 2.0
    zones_arr = np.array(zones)
    q95 = np.array([q95_by_zone.get(z, q95_by_zone["_default"]) for z in zones_arr])
    calibrated_half_width = half_width * q95
    lo = cv["oof_median"] - calibrated_half_width
    hi = cv["oof_median"] + calibrated_half_width
    in_interval = (y >= lo) & (y <= hi)
    return float(np.mean(in_interval[valid]))


def _build_aoa_index(model, X: np.ndarray, target: str, seed: int) -> dict:
    """Feature-importance weights + z-score scaling stats + a (possibly
    subsampled) copy of the training feature matrix, everything
    aoa_dissimilarity needs at inference to score a new
    polygon-day against the SAME reference distribution the shipped model
    was fit on. Subsampled (not the full multi-million-row table) to bound
    the shipped model.joblib's size -- a random sample is a fair
    representative of the training covariate distribution for a
    nearest-neighbor distance, which doesn't need every point, just enough
    density to not systematically miss a real nearby training analog.

    Keys are suffixed with `target` ("tmax"/"tmin") -- tmax and tmin fit
    separate models with separate feature importances and, critically,
    different distributions for feature 0 (grid_tmax_c vs grid_tmin_c), so
    storing one shared "aoa_train_features" key for both targets would let
    the second call in main()'s per-target loop silently overwrite the
    first target's reference distribution, making tmax predictions score
    their out-of-distribution-ness against tmin's training data."""
    rng = np.random.RandomState(seed)
    if len(X) > _AOA_TRAINING_SAMPLE_CAP:
        idx = rng.choice(len(X), size=_AOA_TRAINING_SAMPLE_CAP, replace=False)
        sample = X[idx]
    else:
        sample = X
    return {
        f"aoa_train_features_{target}": sample,
        f"aoa_feature_weights_{target}": feature_importance_weights(model),
        f"aoa_train_mean_{target}": X.mean(axis=0),
        f"aoa_train_std_{target}": X.std(axis=0),
    }


def save_model_artifacts(
    bucket: str, model_version: str, artifact_bundle: dict, metadata: dict,
    candidate_only: bool = False,
) -> None:
    """Writes to s3://{bucket}/downscaling/models/{model_version}/ -- the exact prefix
    src/downscaling.py's load_model()/load_model_metadata() read from, i.e. this is the
    live-serving publish path. `candidate_only=True` writes the SAME artifacts to a
    sibling prefix outside `downscaling/` entirely (`research/candidate-models/`) so a
    retrain's held-out results can be reported, and the fitted model kept around for a
    later publish, without ever touching the tree any serving code reads -- see
    docs/pipeline-fable-consult-2026-07-19-downscaling-diagnosis.md's own publish-gate
    discussion for why a retrain and a publish are deliberately two separate actions."""
    import tempfile

    import boto3
    import joblib
    from boto3.s3.transfer import TransferConfig

    client = boto3.client("s3")
    prefix = (
        f"research/candidate-models/{model_version}/" if candidate_only
        else f"downscaling/models/{model_version}/"
    )
    # Dump to disk and upload multipart. A single put_object caps at 5 GB, and rf6's bundle
    # was already 4.84 GB: the 2026-09-29 rf7 control (685k rows) failed there after the
    # whole fit, with the model only in memory. The local copy is kept (path printed) so an
    # upload failure never costs the fit; metadata goes first since it is small and the CV
    # record is the part most worth keeping.
    local_dir = os.environ.get("MODEL_ARTIFACT_DIR") or tempfile.mkdtemp(prefix=f"{model_version}-")
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, "model.joblib")
    with open(os.path.join(local_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(artifact_bundle, model_path)
    print(f"Wrote local artifacts to {local_dir}")
    client.put_object(
        Bucket=bucket, Key=f"{prefix}metadata.json",
        Body=json.dumps(metadata, indent=2).encode(), ContentType="application/json",
    )
    client.upload_file(model_path, bucket, f"{prefix}model.joblib",
                       Config=TransferConfig(multipart_threshold=256 * 1024 ** 2,
                                             multipart_chunksize=256 * 1024 ** 2))
    if candidate_only:
        print(f"CANDIDATE ONLY -- not published. Wrote to s3://{bucket}/{prefix} "
              f"(outside downscaling/, nothing live reads this). Re-run without "
              f"--candidate-only, with Nishant's own live go-ahead, to publish for real.")
    else:
        print(f"Wrote model artifacts to s3://{bucket}/{prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version", required=True, help="e.g. ds-2026.07-rf1")
    parser.add_argument("--bucket", default=None, help="S3 bucket for model artifacts; defaults to credentials.yaml/VULNERABILITY_DATA_BUCKET")
    parser.add_argument("--profile", default=None, help="Named AWS profile (omit on EC2 with an attached IAM role).")
    parser.add_argument("--candidate-only", action="store_true",
                         help="Write model.joblib/metadata.json to research/candidate-models/{version}/ "
                              "instead of the live-serving downscaling/models/{version}/ prefix. Use this "
                              "for every retrain until a human has explicitly approved publishing it.")
    parser.add_argument("--cv-n-jobs", type=int, default=-1,
                         help="worker processes for the leave-region-out QRF and kriging CV (default -1 = one "
                              "per core). Each worker holds its own copy of the feature matrix, so this, not "
                              "the final fit, sets peak memory: at ~670k rows 16 workers need ~150 GB. Results "
                              "don't depend on it (every fold fit is seeded).")
    parser.add_argument("--extra-rows-json", action="append", default=[],
                         help="Train on ghcn_training PLUS the rows in this builder JSON output "
                              "({\"rows\": [...]}, e.g. build_gsod_bsh_rows.py). Repeatable. The DB row "
                              "wins on a duplicate (station_id, date); across files, the first file wins. Only with --candidate-only. Path + sha256 recorded in metadata.json.")
    parser.add_argument("--feature-set", choices=sorted(SUPPORTED_FEATURE_ORDERS), default="base",
                        help="'regime' adds REGIME_FEATURES (needs --regime-features-csv); only with --candidate-only")
    parser.add_argument("--regime-features-csv", action="append", default=[],
                        help="Open-Meteo daily regime covariates per (station_id, date); repeatable; sha256 recorded")
    parser.add_argument("--exclude-station-since", action="append", default=[], metavar="STATION:YYYY-MM-DD",
                        help="drop a station's rows dated on/after the date (unresolved recent record); repeatable; "
                             "recorded in metadata; only with --candidate-only")
    parser.add_argument("--holdout-stations", default=None, metavar="PATH",
                        help="drop every row of the stations listed in PATH at every date (the scorecard's "
                             "holdout list, scorecard/v1/holdout_stations.txt); recorded in metadata")
    parser.add_argument("--cv-region-map", default=None, metavar="JSON",
                        help="group FIPS regions into single leave-region-out folds, e.g. {\"CH\": \"NE_ASIA\"}; "
                             "recorded in metadata; only with --candidate-only")
    parser.add_argument("--ssod-allowlist", default=None, metavar="CSV",
                        help="stations whose 2025+ values were checked against a GSOD twin (ssod_guard): every other "
                             "non-US GHCN-D row dated 2025-01-01 or later is dropped. Required whenever the corpus "
                             "has such rows; a header-only CSV drops them all. Recorded in metadata")
    parser.add_argument("--save-oof-dir", default=None,
                        help="also write per-row leave-region-out OOF predictions to DIR/oof_{tmax,tmin}.csv.gz")
    args = parser.parse_args()
    if args.exclude_station_since and not args.candidate_only:
        raise SystemExit("--exclude-station-since is only allowed with --candidate-only")
    if args.feature_set == "regime" and not (args.candidate_only and args.regime_features_csv):
        raise SystemExit("--feature-set regime needs --candidate-only and at least one --regime-features-csv")
    if args.feature_set == "gridbias" and not args.candidate_only:
        raise SystemExit("--feature-set gridbias is only allowed with --candidate-only")
    if args.cv_region_map and not args.candidate_only:
        raise SystemExit("--cv-region-map is only allowed with --candidate-only")
    feature_order = SUPPORTED_FEATURE_ORDERS[args.feature_set]
    if args.extra_rows_json and not args.candidate_only:
        raise SystemExit("--extra-rows-json trains on rows that aren't in ghcn_training; it is only allowed "
                         "with --candidate-only")

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
    bucket = args.bucket or _bucket_from_credentials()

    rows = load_training_rows()
    print(f"Loaded {len(rows)} ghcn_training row(s)")
    if not rows:
        print("No training rows available -- nothing to train. Run scripts/build_training_set.py first.")
        return
    extra_sources = []
    for path in args.extra_rows_json:
        import hashlib
        with open(path, "rb") as f:
            raw = f.read()
        payload = json.loads(raw)
        if payload.get("complete") is not True:
            raise SystemExit(f"{path} is not marked complete (a partial builder checkpoint?) -- refusing to train on it")
        extra = payload["rows"]
        rows, counts = merge_extra_rows(rows, extra)
        extra_sources.append({"path": os.path.basename(path), "sha256": hashlib.sha256(raw).hexdigest(),
                              "rows_in_file": len(extra), **counts})
        print(f"Extra rows from {path}: {counts}")

    ssod_guard = None
    n_ssod = ssod_guard_mod.count_affected(rows)
    if n_ssod and not args.ssod_allowlist:
        raise SystemExit(f"{n_ssod} non-US GHCN-D row(s) are dated on or after {ssod_guard_mod.SWITCH_DATE} (NOAA's "
                         "SSOD-v2 source switch: TMAX low, TMIN high). Pass --ssod-allowlist CSV (a header-only file "
                         "drops them all); see heatready_downscaling/ssod_guard.py")
    if args.ssod_allowlist:
        rows, ssod_guard = ssod_guard_mod.apply_guard(rows, ssod_guard_mod.read_allowlist(args.ssod_allowlist))
        ssod_guard.update({"allowlist": os.path.basename(args.ssod_allowlist),
                           "sha256": ssod_guard_mod.sha256_file(args.ssod_allowlist)})
        print(f"SSOD-v2 guard: {ssod_guard}")

    excluded_station_rows = {}
    if args.exclude_station_since:
        rows, excluded_station_rows = exclude_station_since(rows, args.exclude_station_since)
        print(f"Excluded station rows: {excluded_station_rows}")

    holdout = None
    if args.holdout_stations:
        import hashlib
        ids = read_holdout_stations(args.holdout_stations)
        rows, dropped = exclude_stations(rows, ids)
        with open(args.holdout_stations, "rb") as f:
            holdout = {"path": os.path.basename(args.holdout_stations), "sha256": hashlib.sha256(f.read()).hexdigest(),
                       "rows_dropped": dropped}
        print(f"Holdout stations dropped: {dropped}")

    regime_sources = []
    if args.regime_features_csv:
        import hashlib
        counts = attach_regime_features(rows, args.regime_features_csv)
        print(f"Regime features attached: {counts}")
        def _sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        regime_sources = [{"path": os.path.basename(p), "sha256": _sha256(p)} for p in args.regime_features_csv]

    artifact_bundle: dict = {}
    metadata_cv: dict = {}
    metadata_cv_db_rows: dict = {}
    metadata_conformal: dict = {}
    ood_thresholds: list[float] = []
    cv_region_map = None
    grid_bias_meta = {}

    for target in ("tmax", "tmin"):
        print(f"\n=== target: delta_{target}_c ===")
        if "grid_bias_prior_c" in feature_order:
            table = attach_grid_bias(rows, target)
            artifact_bundle[f"grid_bias_table_{target}"] = grid_bias_prior.to_json(table)
            grid_bias_meta[target] = {"station_months": int(sum(len(v[0]) for v in table.values()))}
        X, y, regions, zones, lons, lats, keep = build_training_feature_matrix(rows, target, return_keep=True,
                                                                               feature_order=feature_order)
        regions, cv_region_map = apply_cv_region_map(regions, args.cv_region_map)
        print(f"[{target}] {len(y)} usable row(s) across {len(set(regions))} region(s), {len(set(zones))} climate zone(s)")

        gb_ctx = grid_bias_context(rows, keep, feature_order)
        cv = leave_region_out_cv(X, y, regions, n_jobs=args.cv_n_jobs, grid_bias_ctx=gb_ctx)
        if gb_ctx is None:
            print(f"[{target}] running regression-kriging comparison baseline...")
            kriging_oof_median = regression_kriging_cv(X, y, regions, lons, lats, n_jobs=args.cv_n_jobs)
        else:  # its folds would see the full-data grid-bias columns; it is a comparison baseline only, skip it
            kriging_oof_median = None
        zone_metrics = cv_metrics_by_zone(y, zones, cv, kriging_oof_median=kriging_oof_median)
        if extra_sources:
            # The same out-of-fold predictions, scored on ghcn_training's own rows only (extra rows
            # pooled under EXTRA_ROWS_ZONE), so each zone's numbers compare like for like with a
            # model trained on ghcn_training alone.
            zones_db = [EXTRA_ROWS_ZONE if rows[i].get("_extra") else z for i, z in zip(keep, zones)]
            db_only = cv_metrics_by_zone(y, zones_db, cv, kriging_oof_median=kriging_oof_median)
            # cv_metrics_by_zone's "overall" pools every row regardless of zone label; recompute
            # it over the database rows alone.
            db_mask = np.array([not rows[i].get("_extra") for i in keep])
            db_only["overall"] = cv_metrics_by_zone(
                y[db_mask], [z for z, m in zip(zones, db_mask) if m], _subset_cv(cv, db_mask),
                kriging_oof_median=None if kriging_oof_median is None else kriging_oof_median[db_mask],
            ).get("overall")
            metadata_cv_db_rows[target] = db_only
        if args.save_oof_dir:
            os.makedirs(args.save_oof_dir, exist_ok=True)
            n_oof = write_oof_csv(os.path.join(args.save_oof_dir, f"oof_{target}.csv.gz"), rows, keep, y, zones, cv,
                                  "grid_tmax_c" if target == "tmax" else "grid_tmin_c")
            print(f"[{target}] wrote {n_oof} OOF rows to {args.save_oof_dir}")
        q95_by_zone = conformal_q95_by_zone(y, zones, cv)
        coverage = conformal_empirical_coverage(y, zones, cv, q95_by_zone)

        print(f"[{target}] overall: {zone_metrics['overall']}")
        print(f"[{target}] empirical conformal coverage: {coverage:.3f} (target [0.93, 0.97])")
        for zone, m in zone_metrics["by_zone"].items():
            gate = "PASS" if m["qrf_beats_grid"] else "FAIL"
            kriging_note = ""
            if "rmse_kriging_c" in m:
                vs_kriging = "beats" if m["qrf_beats_kriging"] else "LOSES TO"
                kriging_note = f" | rmse_kriging={m['rmse_kriging_c']:.3f} (QRF {vs_kriging} kriging)"
            print(
                f"[{target}]   zone {zone}: rmse_grid={m['rmse_grid_c']:.3f} "
                f"rmse_qrf={m['rmse_qrf_c']:.3f} [{gate}]{kriging_note}"
            )

        valid_di = cv["oof_di"][cv["valid"] & ~np.isnan(cv["oof_di"])]
        ood_threshold = float(np.mean(valid_di)) if len(valid_di) else None
        if ood_threshold is not None:
            ood_thresholds.append(ood_threshold)

        from quantile_forest import RandomForestQuantileRegressor
        final_model = RandomForestQuantileRegressor(**_QRF_PARAMS).fit(X, y)
        artifact_bundle[f"model_{target}"] = final_model
        artifact_bundle.update(_build_aoa_index(final_model, X, target, seed=_QRF_PARAMS["random_state"]))

        metadata_cv[target] = zone_metrics
        metadata_conformal[target] = q95_by_zone

    metadata = {
        "model_version": args.model_version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_order": list(feature_order),
        "regime_features_sources": regime_sources,
        **({"grid_bias_prior": {"length_km": grid_bias_prior.LENGTH_KM, "radius_km": grid_bias_prior.RADIUS_KM,
                                "same_site_km": grid_bias_prior.SAME_SITE_KM,
                                "shrink_weight": grid_bias_prior.SHRINK_WEIGHT,
                                "min_days": grid_bias_prior.MIN_DAYS, **grid_bias_meta}} if grid_bias_meta else {}),
        **({"cv_region_map": cv_region_map} if cv_region_map else {}),
        **({"ssod_guard": ssod_guard} if ssod_guard else {}),
        "excluded_station_rows": excluded_station_rows,
        **({"holdout_stations": holdout} if holdout else {}),
        "targets": ["delta_tmax_c", "delta_tmin_c"],
        # Conformal calibration is per-target (tmax/tmin fit their own QRF
        # interval widths); predict_downscaled reads whichever target's
        # metadata it needs via the SAME "conformal_q95_by_zone" key the
        # design doc's example uses for the tmax case -- both live here so
        # neither target's calibration is silently dropped.
        "conformal_q95_by_zone": metadata_conformal.get("tmax", {}),
        "conformal_q95_by_zone_tmin": metadata_conformal.get("tmin", {}),
        "ood_aoa_threshold": (sum(ood_thresholds) / len(ood_thresholds)) if ood_thresholds else None,
        "cv": {"leave_region_out": metadata_cv},
        "training_rows": len(rows),
        "extra_rows_sources": extra_sources,
        "cv_ghcn_training_rows_only": {"leave_region_out": metadata_cv_db_rows} if extra_sources else None,
    }

    save_model_artifacts(bucket, args.model_version, artifact_bundle, metadata, candidate_only=args.candidate_only)


if __name__ == "__main__":
    main()
