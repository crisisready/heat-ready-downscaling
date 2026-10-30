"""
Out-of-time scorecard runner (scorecard/v1/spec.yaml; scoring and the ship rule live in
heatready_downscaling.scorecard, which this script only feeds).

Two subcommands, both read-only against ghcn_training (the trainer's own SELECT):

  build-manifest  Freeze the station-day manifest: every (station, date, target) of a truth station
                  in a scored year with an observed delta and a grid value. Writes the manifest file
                  whose sha256 goes into the spec, plus each station's region and urban/rural tag.

  run             Score a candidate against the incumbent on one tier.
                  fast: cutoff 2024-12-31, score 2025. full: every cutoff in the spec, model refit at
                  each, paired deltas pooled across years.
                  For each cutoff, a recipe (scorecard/recipes/*.yaml: corpus + feature set) is refit
                  on rows dated on or before the cutoff with every truth station removed at every
                  date, then predicted on the manifest rows of the scored year THROUGH THE SERVING
                  CONTRACT (heatready_downscaling.contract.QRFModelAdapter.predict, the mirror of the
                  API's predict_downscaled): the zone's CV gate from the version's metadata.json, grid
                  fallback where the model does not apply. Served delta is 0 where it falls back.
                  Incumbent predictions are cached by a content key, so a layer candidate
                  (--candidate-predictions: served deltas its own code fit before each cutoff)
                  scores in minutes against the frozen incumbent.

Example (bastion, prod ghcn_training read-only):
  python scripts/scorecard.py run --spec scorecard/v1/spec.yaml --tier fast \\
      --declaration scorecard/v1/declarations/2026-10-rf8b-vs-rf6.yaml \\
      --incumbent-recipe scorecard/recipes/ds-2026.09-rf6.yaml --incumbent-metadata rf6_metadata.json \\
      --candidate-recipe scorecard/recipes/ds-2026.09-rf8b.yaml --candidate-metadata rf8b_metadata.json \\
      --data-dir /root/scorecard/inputs --cache-dir /root/scorecard/cache --out-dir /root/scorecard/out --n-jobs 8
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_path(_spec_path: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


# ---------------------------------------------------------------- inputs

def load_truth_stations(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_recipe(path: str) -> dict:
    import yaml
    with open(path) as f:
        recipe = yaml.safe_load(f)
    for key in ("model_version", "feature_set", "extra_rows"):
        if key not in recipe:
            raise SystemExit(f"recipe {path} missing {key!r}")
    if recipe["feature_set"] not in ("base", "gridbias"):
        raise SystemExit("the scorecard refits base and gridbias recipes only; a regime recipe needs its covariate "
                         "CSVs plumbed through first")
    recipe.setdefault("cv_region_map", None)
    # Optional keys, read with .get() so a recipe without them keeps its cache key:
    #   lapse_k_tmax, lapse_offset_cap_m  the trainer's --lapse-k-tmax / --lapse-offset-cap-m (tmin never gets a term)
    #   ssod_allowlist                    the trainer's --ssod-allowlist (repo-relative CSV)
    #   extra_rows_targets                targets whose fit adds extra_rows (default both)
    #   qrf_params, qrf_params_targets    overrides of the trainer's _QRF_PARAMS, for the named targets (default both)
    if recipe.get("lapse_k_tmax") and recipe["feature_set"] != "base":
        raise SystemExit(f"recipe {path}: a lapse term needs the base feature set, as in the trainer")
    if recipe.get("lapse_offset_cap_m") is not None and not recipe.get("lapse_k_tmax"):
        raise SystemExit(f"recipe {path}: lapse_offset_cap_m needs lapse_k_tmax")
    for key in ("extra_rows_targets", "qrf_params_targets"):
        if not set(recipe.get(key, ("tmax", "tmin"))) <= {"tmax", "tmin"}:
            raise SystemExit(f"recipe {path}: {key} must name tmax and/or tmin")
    unknown = set(recipe.get("qrf_params", {})) - {"n_estimators", "min_samples_leaf", "max_samples", "max_features"}
    if unknown:
        raise SystemExit(f"recipe {path}: qrf_params {sorted(unknown)} are not overridable")
    #   residual_layer_tmin               a station-grouped out-of-fold residual table added to the served median
    #                                     (residual_layer_table); keys below, oof_qrf_params optional. tmin only:
    #                                     its out-of-fold fits do not carry tmax's lapse term
    if "residual_layer_tmax" in recipe:
        raise SystemExit(f"recipe {path}: residual_layer_tmax is not supported (the lapse term is not in its fits)")
    for target in ("tmin",):
        layer = recipe.get(f"residual_layer_{target}")
        if layer is not None:
            need = {"length_km", "radius_km", "same_site_km", "shrink_weight", "folds", "min_days"}
            if recipe["feature_set"] != "base":
                raise SystemExit(f"recipe {path}: residual_layer_{target} needs the base feature set (a gridbias "
                                 "prior built from all stations would leak a held-out fold's targets)")
            if not need <= set(layer) or set(layer) - need - {"oof_qrf_params"} \
                    or not isinstance(layer["folds"], int) or not isinstance(layer["min_days"], int) or layer["folds"] < 2:
                raise SystemExit(f"recipe {path}: residual_layer_{target} needs exactly {sorted(need)} "
                                 "(+ oof_qrf_params), folds >= 2")
            if set(layer.get("oof_qrf_params", {})) - {"n_estimators", "min_samples_leaf", "max_samples",
                                                       "max_features"}:
                raise SystemExit(f"recipe {path}: residual_layer_{target}.oof_qrf_params not overridable")
    return recipe


def load_extra_file(data_dir: str, name: str, sha256: str) -> list[dict]:
    path = os.path.join(data_dir, name)
    got = _sha256_file(path)
    if got != sha256:
        raise SystemExit(f"{path}: sha256 {got} does not match the frozen {sha256}")
    with open(path) as f:
        payload = json.load(f)
    if payload.get("complete") is not True:
        raise SystemExit(f"{path} is not marked complete")
    return payload["rows"]


def load_rows(spec: dict, recipes: list[dict], data_dir: str) -> tuple[list[dict], list[dict]]:
    """ghcn_training rows plus every extra file a recipe or the truth set needs, each extra row
    tagged with the file it came from (_src) so each recipe's own corpus can be selected."""
    import train_downscaling as td
    rows = td.load_training_rows()
    for r in rows:
        r["_src"] = "ghcn_training"
    sources = []
    wanted: dict[str, str] = {}
    for item in [*spec.get("truth_extra_rows", []), *(x for rc in recipes for x in rc["extra_rows"])]:
        if wanted.setdefault(item["name"], item["sha256"]) != item["sha256"]:
            raise SystemExit(f"two different sha256 values declared for {item['name']}")
    for name, sha in wanted.items():
        before = len(rows)
        rows, counts = td.merge_extra_rows(rows, load_extra_file(data_dir, name, sha))
        for r in rows[before:]:
            r["_src"] = name
        sources.append({"name": name, "sha256": sha, **counts})
    ov = spec.get("truth_overrides")
    if ov:
        counts = apply_truth_overrides(rows, load_truth_overrides(_repo_path("", ov["path"]), ov["sha256"]))
        sources.append({"name": os.path.basename(ov["path"]), "sha256": ov["sha256"], **counts})
    return rows, sources


def load_truth_overrides(path: str, sha256: str) -> dict:
    """{(station_id, date): tmax_c or None} from a truth-override CSV (station_id, date, tmax_c; empty = drop)."""
    got = _sha256_file(path)
    if got != sha256:
        raise SystemExit(f"{path}: sha256 {got} does not match the frozen {sha256}")
    with open(path, newline="") as f:
        return {(r["station_id"], r["date"]): (float(r["tmax_c"]) if r["tmax_c"] else None) for r in csv.DictReader(f)}


def apply_truth_overrides(rows: list[dict], overrides: dict) -> dict:
    """Scorecard v2b: NOAA's 2025 GHCN-D source switch (SSOD v2, MFLAG H) puts a biased value in a truth row's
    station tmax. A listed (station, date) takes the override as its station tmax, delta recomputed against the
    row's own grid value, or loses both targets when the override is empty; tmin is dropped on every listed day.
    Rows are changed in place; only truth stations are listed, and truth stations never train a recipe."""
    replaced = dropped = 0
    for r in rows:
        k = (r["station_id"], str(r["date"])[:10])
        if k not in overrides:
            continue
        v = overrides[k]
        if v is None or r.get("grid_tmax_c") is None:
            r["station_tmax_c"] = r["delta_tmax_c"] = None
            dropped += 1
        else:
            r["station_tmax_c"], r["delta_tmax_c"] = v, v - r["grid_tmax_c"]
            replaced += 1
        r["station_tmin_c"] = r["delta_tmin_c"] = None
    return {"overrides": len(overrides), "rows_replaced": replaced, "rows_dropped": dropped}


# ---------------------------------------------------------------- manifest

def manifest_keys(rows: list[dict], truth_ids: set[str], years: set[int]) -> list[tuple[str, str, str]]:
    keys = []
    for r in rows:
        if r["station_id"] not in truth_ids or int(str(r["date"])[:4]) not in years:
            continue
        for t in ("tmax", "tmin"):
            d, g = r.get(f"delta_{t}_c"), r.get(f"grid_{t}_c")
            if d is not None and g is not None and math.isfinite(d) and math.isfinite(g):
                keys.append((r["station_id"], str(r["date"])[:10], t))
    return keys


def write_manifest(path: str, keys) -> None:
    with gzip.open(path, "wt", newline="") as f:
        for s, d, t in sorted(keys):
            f.write(f"{s},{d},{t}\n")


def read_manifest(path: str) -> list[tuple[str, str, str]]:
    with gzip.open(path, "rt") as f:
        return [tuple(line.rstrip("\n").split(",")) for line in f if line.strip()]


def check_manifest(spec: dict, spec_path: str) -> list[tuple[str, str, str]]:
    from heatready_downscaling.scorecard import manifest_sha256
    m = spec["manifest"]
    if not m.get("sha256"):
        raise SystemExit("spec has no frozen manifest sha256; run build-manifest and freeze it first")
    keys = read_manifest(_repo_path(spec_path, m["path"]))
    got = manifest_sha256(keys)
    if got != m["sha256"]:
        raise SystemExit(f"manifest sha256 {got} does not match the spec's frozen {m['sha256']}")
    return keys


def cmd_build_manifest(args) -> None:
    from heatready_downscaling.scorecard import load_spec, manifest_sha256, score_year
    spec = load_spec(args.spec)
    stations = load_truth_stations(_repo_path(args.spec, spec["truth_stations"]))
    truth_ids = {s["station_id"] for s in stations}
    years = {score_year(c) for tier in spec["tiers"].values() for c in tier["cutoffs"]}
    rows, sources = load_rows(spec, [], args.data_dir)
    keys = manifest_keys(rows, truth_ids, years)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "manifest.csv.gz")
    write_manifest(path, keys)
    # per-station region (mode of the row label) and urban/rural from the mean GHSL urban fraction
    region, urban = defaultdict(Counter), defaultdict(list)
    for r in rows:
        if r["station_id"] in truth_ids:
            region[r["station_id"]][r.get("region")] += 1
            if r.get("ghsl_urban_fraction") is not None:
                urban[r["station_id"]].append(float(r["ghsl_urban_fraction"]))
    thr = float(spec["urban_ghsl_fraction_threshold"])
    for s in stations:
        sid = s["station_id"]
        s["region"] = region[sid].most_common(1)[0][0] if region[sid] else ""
        s["setting"] = ("urban" if np.mean(urban[sid]) >= thr else "rural") if urban[sid] else "unknown"
    out_csv = os.path.join(args.out_dir, "truth_stations.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stations[0].keys()))
        w.writeheader()
        w.writerows(stations)
    counts = Counter((k[2], k[1][:4]) for k in keys)
    no_rows = sorted(truth_ids - {k[0] for k in keys})
    summary = {"manifest_sha256": manifest_sha256(keys), "n_keys": len(keys),
               "keys_by_target_year": {f"{t}:{y}": n for (t, y), n in sorted(counts.items())},
               "stations_with_keys": len({k[0] for k in keys}), "truth_stations_without_rows": no_rows,
               "sources": sources}
    with open(os.path.join(args.out_dir, "manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- refit + served prediction

def _haversine_min_km(lat, lon, lats, lons) -> float:
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2, lo2 = np.radians(lats), np.radians(lons)
    a = np.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float((6371.0 * 2 * np.arcsin(np.sqrt(a))).min()) if len(lats) else float("inf")


def recipe_corpus(rows: list[dict], recipe: dict, cutoff: str, truth_ids: set[str],
                  target: str | None = None) -> list[dict]:
    """The recipe's training rows at a cutoff: its own sources (extra rows only for the targets the
    recipe names), dated on or before the cutoff, with every truth station removed at every date and
    the recipe's SSOD allowlist applied as the trainer applies it."""
    srcs = {"ghcn_training"}
    if target is None or target in recipe.get("extra_rows_targets", ("tmax", "tmin")):
        srcs |= {x["name"] for x in recipe["extra_rows"]}
    train = [r for r in rows if r["_src"] in srcs and str(r["date"])[:10] <= cutoff
             and r["station_id"] not in truth_ids]
    if recipe.get("ssod_allowlist"):
        from heatready_downscaling import ssod_guard
        train, _ = ssod_guard.apply_guard(train, ssod_guard.read_allowlist(_repo_path("", recipe["ssod_allowlist"])))
    return train


def extra_stations_near_truth(rows: list[dict], stations: list[dict], km: float) -> set[str]:
    """Extra-file stations (never in ghcn_training) within km of a truth station: twins under another id,
    which would leak the truth into a recipe that adds the file. Removed like truth stations."""
    skip = {r["station_id"] for r in rows if r["_src"] == "ghcn_training"} | {s["station_id"] for s in stations}
    coords = {}
    for r in rows:
        if r["_src"] != "ghcn_training" and r["station_id"] not in skip and r.get("lat") is not None \
                and r.get("lon") is not None:
            coords.setdefault(r["station_id"], (float(r["lat"]), float(r["lon"])))
    lats = np.array([float(s["lat"]) for s in stations], float)
    lons = np.array([float(s["lon"]) for s in stations], float)
    if not coords:
        return set()
    return {sid for sid, (la, lo) in coords.items() if _haversine_min_km(la, lo, lats, lons) <= km}


_FINGERPRINT_COLS = ("station_id", "date", "climate_zone", "region", "grid_tmax_c", "grid_tmin_c", "delta_tmax_c",
                     "delta_tmin_c", "lst_warm_season_anomaly_c", "canopy_height_mean_m", "canopy_frac_over_3m",
                     "wc_built_frac", "wc_tree_frac", "wc_water_frac", "ghsl_urban_fraction", "pop_density_per_km2",
                     "elevation_rel_to_gridcell_m", "elevation_mean_m", "slope_deg", "aspect_deg",
                     "grid_specific_humidity_kgkg", "koppen_main_group_code", "nighttime_wind_ms", "lat", "lon")


def rows_fingerprint(rows: list[dict]) -> str:
    """sha256 over the values a fit or a prediction depends on, so a cached prediction is reused only
    when the training corpus and the predicted rows are unchanged (a DB backfill invalidates it)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(tuple(r.get(c) for c in _FINGERPRINT_COLS)).encode())
    return h.hexdigest()


def _cache_key(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def attach_grid_bias_for_fit(train_rows: list[dict], test_rows: list[dict], target: str) -> tuple[list, list]:
    """Copies of train_rows and test_rows carrying grid_bias_prior_c / grid_bias_support built from
    train_rows alone (the cutoff corpus without truth stations), as the trainer does for its final fit.
    Copies, so the incumbent's rows (and its cache fingerprint) are never touched."""
    import train_downscaling as td
    from heatready_downscaling import grid_bias_prior as gbp
    train_rows = [dict(r) for r in train_rows]
    test_rows = [dict(r) for r in test_rows]
    table = td.attach_grid_bias(train_rows, target)
    prior, support = gbp.priors_for_rows(table, [r["lat"] for r in test_rows], [r["lon"] for r in test_rows],
                                         [r["date"] for r in test_rows])
    for r, p, w in zip(test_rows, prior, support):
        r["grid_bias_prior_c"], r["grid_bias_support"] = float(p), float(w)
    return train_rows, test_rows


def served_predictions(train_rows: list[dict], test_rows: list[dict], target: str, metadata: dict,
                       n_jobs: int, feature_set: str = "base", recipe: dict | None = None) -> list[dict]:
    """Fit the recipe's QRF on train_rows and return the SERVED result for each test row: the DS
    serving contract with this version's own CV gate (from its metadata.json). A lapse recipe fits
    delta + term as the trainer does, and the adapter subtracts it again."""
    import train_downscaling as td
    from quantile_forest import RandomForestQuantileRegressor

    from heatready_downscaling.contract import QRFModelAdapter, derive_zones_passing_cv_gate
    from heatready_downscaling.features import SUPPORTED_FEATURE_ORDERS, lapse_term

    FEATURE_ORDER = SUPPORTED_FEATURE_ORDERS[feature_set]
    if feature_set == "gridbias":
        train_rows, test_rows = attach_grid_bias_for_fit(train_rows, test_rows, target)
    X, y, *_ = td.build_training_feature_matrix(train_rows, target, feature_order=FEATURE_ORDER)
    if len(y) < td._MIN_FOLD_TRAIN_ROWS:
        return []
    recipe = recipe or {}
    lapse_k = {"tmax": float(recipe.get("lapse_k_tmax") or 0.0), "tmin": 0.0}
    cap = recipe.get("lapse_offset_cap_m")
    y_fit = np.asarray(y, dtype=float) + lapse_term(X, tuple(FEATURE_ORDER), lapse_k[target], cap)
    params = recipe.get("qrf_params", {}) if target in recipe.get("qrf_params_targets", ("tmax", "tmin")) else {}
    model = RandomForestQuantileRegressor(**{**td._QRF_PARAMS, **params, "n_jobs": n_jobs}).fit(X, y_fit)
    bundle = {
        "lapse_k": lapse_k, "lapse_offset_cap_m": cap,
        f"model_{target}": model,
        "metadata": {"model_version": metadata["model_version"], "feature_order": list(FEATURE_ORDER),
                     "conformal_q95_by_zone": metadata.get("conformal_q95_by_zone", {}),
                     "conformal_q95_by_zone_tmin": metadata.get("conformal_q95_by_zone_tmin", {}),
                     "ood_aoa_threshold": metadata.get("ood_aoa_threshold")},
        "zones_passing_cv_gate": derive_zones_passing_cv_gate(metadata),
        **td._build_aoa_index(model, X, target, td._QRF_PARAMS["random_state"]),
    }
    results = QRFModelAdapter(bundle).predict(test_rows, target)
    layer = recipe.get(f"residual_layer_{target}")
    if layer:
        table = residual_layer_table(train_rows, target, FEATURE_ORDER, {**td._QRF_PARAMS, **params}, layer, n_jobs)
        cache = {}
        for r, res in zip(test_rows, results):
            if res["applied"] and r.get("lat") is not None and r.get("lon") is not None:
                k = (float(r["lat"]), float(r["lon"]), int(str(r["date"])[5:7]))
                if k not in cache:
                    cache[k] = residual_prior_at(table, *k, layer)
                res["delta_c"] += cache[k]
        moved = [v for v in cache.values() if v]
        print(f"[{target}] residual layer: {sum(len(v[2]) for v in table.values())} station-months, "
              f"{len(moved)}/{len(cache)} test site-months moved, mean |prior| "
              f"{np.mean(np.abs(moved)) if moved else 0.0:.3f} C", flush=True)
    return results


def residual_layer_table(train_rows: list[dict], target: str, feature_order, qrf_params: dict, layer: dict,
                         n_jobs: int) -> dict:
    """Station-grouped out-of-fold residuals (observed minus the forest's median) of the training rows, as a
    month -> (lats, lons, mean residual) table of station-months with at least layer["min_days"] rows. A
    serving port would ship this table and add residual_prior_at() to the served median; no truth station is
    in train_rows, and every fold's forest is fit without the stations it predicts."""
    import train_downscaling as td
    from quantile_forest import RandomForestQuantileRegressor

    X, y, _r, _z, lons, lats, keep = td.build_training_feature_matrix(train_rows, target, return_keep=True,
                                                                     feature_order=tuple(feature_order))
    sids = np.array([train_rows[i]["station_id"] for i in keep])
    months = np.array([int(str(train_rows[i]["date"])[5:7]) for i in keep])
    folds = np.array([int(hashlib.sha256(s.encode()).hexdigest(), 16) % layer["folds"] for s in sids])
    resid = np.full(len(y), np.nan)
    if len(set(folds.tolist())) < 2:
        raise SystemExit(f"[{target}] residual layer: the training stations fall in fewer than 2 folds")
    for k in range(layer["folds"]):
        hold = folds == k
        if not hold.any():
            continue
        m = RandomForestQuantileRegressor(**{**qrf_params, **layer.get("oof_qrf_params", {}), "n_jobs": n_jobs})
        m.fit(X[~hold], y[~hold])
        # unweighted median: same values as the served (weighted) median on these forests, about 200x faster
        resid[hold] = y[hold] - m.predict(X[hold], quantiles=0.5, weighted_quantile=False)
        print(f"[{target}] residual layer fold {k + 1}/{layer['folds']}: {int(hold.sum())} rows, "
              f"rmse {float(np.sqrt(np.mean(resid[hold] ** 2))):.3f} C", flush=True)
    acc = {}
    for s, mo, la, lo, v in zip(sids, months, lats, lons, resid):
        a = acc.setdefault((s, mo), [la, lo, 0.0, 0])
        a[2] += v
        a[3] += 1
    table = {}
    for (s, mo), (la, lo, tot, n) in acc.items():
        if n >= layer["min_days"]:
            table.setdefault(mo, []).append((la, lo, tot / n))
    return {mo: tuple(np.array(c) for c in zip(*v)) for mo, v in table.items()}


def residual_prior_at(table: dict, lat: float, lon: float, month: int, layer: dict) -> float:
    """Gaussian-kernel mean of neighbouring station-months' residuals, shrunk toward 0: sum(w r) / (sum(w) +
    shrink_weight), neighbours within radius_km and farther than same_site_km (a site never informs itself)."""
    t = table.get(month)
    if t is None:
        return 0.0
    la, lo, mean = t
    p1, p2 = math.radians(lat), np.radians(la)
    h = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(np.radians(lo - lon) / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))
    use = (km > layer["same_site_km"]) & (km <= layer["radius_km"])
    if not use.any():
        return 0.0
    w = np.exp(-0.5 * (km[use] / layer["length_km"]) ** 2)
    return float((w * mean[use]).sum() / (w.sum() + layer["shrink_weight"]))


def recipe_predictions(rows, recipe, metadata, cutoff, target, truth_ids, test_rows, cache_dir, key_parts,
                       n_jobs) -> tuple[dict, dict]:
    """{(station_id, date): (served_delta, ci95)} for one recipe/cutoff/target, from the cache when the
    content key matches. Returns (predictions, provenance)."""
    train = recipe_corpus(rows, recipe, cutoff, truth_ids, target)
    served_meta = {"gate": metadata.get("cv", {}).get("leave_region_out", {}).get(target, {}).get("by_zone"),
                   "q95": metadata.get("conformal_q95_by_zone" if target == "tmax" else "conformal_q95_by_zone_tmin"),
                   "ood": metadata.get("ood_aoa_threshold")}
    allow = ({"ssod_allowlist_sha256": _sha256_file(_repo_path("", recipe["ssod_allowlist"]))}
             if recipe.get("ssod_allowlist") else {})   # the list's contents, not just its path, shape the corpus
    key = _cache_key({**key_parts, **allow, "recipe": recipe, "cutoff": cutoff, "target": target, "served_meta": served_meta,
                      "train_rows_sha256": rows_fingerprint(train), "test_rows_sha256": rows_fingerprint(test_rows)})
    path = os.path.join(cache_dir, f"{recipe['model_version']}__{cutoff}__{target}__{key}.csv.gz")
    if os.path.exists(path):
        preds = {}
        with gzip.open(path, "rt", newline="") as f:
            for rec in csv.DictReader(f):
                preds[(rec["station_id"], rec["date"])] = (float(rec["served_delta"]),
                                                          float(rec["ci95"]) if rec["ci95"] else float("nan"))
        return preds, {"cache": os.path.basename(path), "cache_hit": True, "sha256": _sha256_file(path)}
    results = served_predictions(train, test_rows, target, metadata, n_jobs, recipe["feature_set"], recipe)
    if not results:
        return {}, {"cache": None, "cache_hit": False, "train_rows": len(train),
                    "note": "fewer training rows than the trainer's minimum before this cutoff"}
    os.makedirs(cache_dir, exist_ok=True)
    preds = {}
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "date", "served_delta", "applied", "ci95"])
        for r, res in zip(test_rows, results):
            delta = round(res["delta_c"], 4) if res["applied"] else 0.0
            ci = round(res["ci95_c"], 4) if res["applied"] else None
            k = (r["station_id"], str(r["date"])[:10])
            preds[k] = (delta, float("nan") if ci is None else ci)
            w.writerow([k[0], k[1], f"{delta:.4f}", int(bool(res["applied"])), "" if ci is None else f"{ci:.4f}"])
    return preds, {"cache": os.path.basename(path), "cache_hit": False, "train_rows": len(train),
                   "train_stations": len({r["station_id"] for r in train}), "sha256": _sha256_file(path)}


def load_candidate_predictions(dir_: str, cutoff: str, target: str) -> dict:
    path = os.path.join(dir_, f"{cutoff}__{target}.csv.gz")
    if not os.path.exists(path):
        return {}
    preds = {}
    with gzip.open(path, "rt", newline="") as f:
        for rec in csv.DictReader(f):
            preds[(rec["station_id"], rec["date"])] = (float(rec["served_delta"]),
                                                      float(rec["ci95"]) if rec.get("ci95") else float("nan"))
    return preds


# ---------------------------------------------------------------- run

def cmd_run(args) -> None:
    import pandas as pd
    import yaml

    from heatready_downscaling import scorecard as sc
    spec = sc.load_spec(args.spec)
    manifest = check_manifest(spec, args.spec)
    with open(args.declaration) as f:
        decl = yaml.safe_load(f)
    sc.validate_declaration(decl)
    stations = load_truth_stations(_repo_path(args.spec, spec["truth_stations"]))
    truth_ids = {s["station_id"] for s in stations}
    if any(not s.get("setting") or not s.get("region") for s in stations):
        raise SystemExit("truth_stations.csv has no setting/region: commit build-manifest's enriched file first")
    inc = load_recipe(args.incumbent_recipe)
    cand = load_recipe(args.candidate_recipe) if args.candidate_recipe else None
    if (cand is None) == (args.candidate_predictions is None):
        raise SystemExit("give exactly one of --candidate-recipe or --candidate-predictions")
    if cand is not None and decl["candidate"] != cand["model_version"]:
        raise SystemExit(f"declaration names candidate {decl['candidate']!r}, recipe is {cand['model_version']!r}")
    if decl["incumbent"] != inc["model_version"]:
        raise SystemExit(f"declaration names incumbent {decl['incumbent']!r}, recipe is {inc['model_version']!r}")
    with open(args.incumbent_metadata) as f:
        inc_meta = json.load(f)
    cand_meta = None
    if cand is not None:
        with open(args.candidate_metadata) as f:
            cand_meta = json.load(f)
    for meta, rc in ((inc_meta, inc), (cand_meta, cand)):
        if meta is not None and meta.get("model_version") != rc["model_version"]:
            raise SystemExit(f"metadata model_version {meta.get('model_version')!r} != recipe {rc['model_version']!r}")

    import train_downscaling as td
    rows, sources = load_rows(spec, [inc] + ([cand] if cand else []), args.data_dir)
    near = extra_stations_near_truth(rows, stations, float(spec.get("dedupe_km", 1.0)))
    if near:
        print(f"{len(near)} extra-file station(s) within {spec.get('dedupe_km', 1.0)} km of a truth station removed "
              f"from every recipe: {sorted(near)}", flush=True)
        sources.append({"name": "extra_stations_near_truth", "removed": sorted(near),
                        "sha256": hashlib.sha256(json.dumps(sorted(near)).encode()).hexdigest()})
    excluded = truth_ids | near
    by_key = {}
    for r in rows:
        if r["station_id"] in truth_ids:
            by_key.setdefault((r["station_id"], str(r["date"])[:10]), r)
    st_by_id = {s["station_id"]: s for s in stations}
    # units use the row's climate_zone (what serving gates on); make the station's zone and group agree
    relabelled = {}
    for (sid, _), r in by_key.items():
        z = r.get("climate_zone")
        if z and z != st_by_id[sid]["zone"]:
            relabelled[sid] = (st_by_id[sid]["zone"], z)
            st_by_id[sid]["zone"], st_by_id[sid]["zone_group"] = z, sc.zone_group(z, spec)
    if relabelled:
        print(f"station zone relabelled from the row label: {relabelled}", flush=True)
    key_parts = {"scorecard_version": spec["scorecard_version"], "manifest_sha256": spec["manifest"]["sha256"],
                 "trainer_sha256": _sha256_file(td.__file__),
                 "contract_sha256": _sha256_file(os.path.join(REPO_ROOT, "src/heatready_downscaling/contract.py")),
                 "features_sha256": _sha256_file(os.path.join(REPO_ROOT, "src/heatready_downscaling/features.py")),
                 **({"grid_bias_prior_sha256": _sha256_file(os.path.join(REPO_ROOT, "src/heatready_downscaling/grid_bias_prior.py"))}
                    if "gridbias" in {inc["feature_set"], (cand or {}).get("feature_set")} else {}),
                 "truth_stations_sha256": _sha256_file(_repo_path(args.spec, spec["truth_stations"])),
                 "sources": sorted((s["name"], s["sha256"]) for s in sources)}

    frames, years_out, prov = [], [], []
    for cutoff in spec["tiers"][args.tier]["cutoffs"]:
        year = sc.score_year(cutoff)
        for target in ("tmax", "tmin"):
            keys = [(s, d) for s, d, t in manifest if t == target and int(d[:4]) == year]
            if not keys:
                prov.append({"cutoff": cutoff, "target": target, "status": "no manifest rows"})
                continue
            test_rows = [by_key[k] for k in keys]
            p_inc, prov_inc = recipe_predictions(rows, inc, inc_meta, cutoff, target, excluded, test_rows,
                                                 args.cache_dir, key_parts, args.n_jobs)
            if not p_inc:
                p_cand, prov_cand = {}, {"note": "not fit: the incumbent has nothing to compare against"}
            elif cand is not None:
                p_cand, prov_cand = recipe_predictions(rows, cand, cand_meta, cutoff, target, excluded, test_rows,
                                                       args.cache_dir, key_parts, args.n_jobs)
            else:
                p_cand = load_candidate_predictions(args.candidate_predictions, cutoff, target)
                cp = os.path.join(args.candidate_predictions, f"{cutoff}__{target}.csv.gz")
                prov_cand = {"candidate_predictions": os.path.basename(cp),
                             "sha256": _sha256_file(cp) if os.path.exists(cp) else None}
            status = "scored"
            if not p_inc or not p_cand:
                status = "unscoreable: " + ("incumbent" if not p_inc else "candidate") + " has no fit or predictions"
            elif not set(keys) <= set(p_cand) or not set(keys) <= set(p_inc):
                raise SystemExit(f"{cutoff} {target}: predictions do not cover the manifest (coverage must be "
                                 "at least the incumbent's)")
            prov.append({"cutoff": cutoff, "target": target, "status": status, "incumbent": prov_inc,
                         "candidate": prov_cand, "manifest_rows": len(keys)})
            print(f"[{cutoff} {target}] {status} ({len(keys)} rows)", flush=True)
            if status != "scored":
                continue
            # per target: an incumbent with target-specific extra rows trains each target on different stations
            tr = recipe_corpus(rows, inc, cutoff, excluded, target)
            coords = {r["station_id"]: (r["lat"], r["lon"]) for r in tr}
            lats = np.array([c[0] for c in coords.values()], float)
            lons = np.array([c[1] for c in coords.values()], float)
            inc_train_stations = {sid: _haversine_min_km(float(s["lat"]), float(s["lon"]), lats, lons)
                                  for sid, s in st_by_id.items()}
            frames.append(pd.DataFrame({
                "station_id": [k[0] for k in keys], "date": [k[1] for k in keys], "year": year, "target": target,
                "zone": [by_key[k]["climate_zone"] for k in keys],
                "delta_obs": [by_key[k][f"delta_{target}_c"] for k in keys],
                "grid_c": [by_key[k][f"grid_{target}_c"] for k in keys],
                "inc_delta": [p_inc[k][0] for k in keys], "inc_ci95": [p_inc[k][1] for k in keys],
                "cand_delta": [p_cand[k][0] for k in keys], "cand_ci95": [p_cand[k][1] for k in keys],
                "dist_km": [inc_train_stations[k[0]] for k in keys]}))
        years_out.append(year)
    if not frames:
        raise SystemExit("nothing scoreable on this tier")
    df = pd.concat(frames, ignore_index=True)
    if args.dump_rows:  # per station-day served deltas, for recomputing any number outside the scorer
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(os.path.join(args.out_dir, f"rows_{args.tier}.csv.gz"), index=False)
    st = pd.DataFrame(stations)
    st["lat"], st["lon"] = st["lat"].astype(float), st["lon"].astype(float)
    clusters = sc.city_clusters(st, float(spec.get("city_radius_km", 30)))
    df["city"] = df["station_id"].map(clusters)
    result = sc.score(df, st, spec)
    aim = sc.score_aim(df, st, spec, decl)
    decision = sc.ship_decision(result, aim, spec, kind=decl["kind"], tier=args.tier)
    private = result.pop("private")
    out = {"tier": args.tier, "declaration": decl, "declaration_sha256": _sha256_file(args.declaration),
           "incumbent": inc["model_version"], "candidate": cand["model_version"] if cand else decl["candidate"],
           "metadata_sha256": {"incumbent": _sha256_file(args.incumbent_metadata),
                               "candidate": _sha256_file(args.candidate_metadata) if cand else None},
           "key_parts": key_parts, "provenance": prov, "zone_relabelled": relabelled, "aim": aim, "decision": decision, **result}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"result_{args.tier}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    title = f"{out['candidate']} vs {out['incumbent']}, {args.tier} tier"
    with open(os.path.join(args.out_dir, f"zone_table_{args.tier}.md"), "w") as f:
        f.write(sc.render_zone_table(result, decision, title))
    if private:
        with open(os.path.join(args.out_dir, f"private_{args.tier}.json"), "w") as f:
            json.dump(private, f, indent=2)
    print(json.dumps({"decision": decision["pass"], "reasons": decision["reasons"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest")
    b.add_argument("--spec", required=True)
    b.add_argument("--data-dir", required=True, help="directory holding the spec's truth_extra_rows files")
    b.add_argument("--out-dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--tier", choices=("fast", "full"), required=True)
    r.add_argument("--declaration", required=True, help="the candidate's declared aim, committed before scoring")
    r.add_argument("--incumbent-recipe", required=True)
    r.add_argument("--incumbent-metadata", required=True, help="the served version's metadata.json (CV gate)")
    r.add_argument("--candidate-recipe")
    r.add_argument("--candidate-metadata")
    r.add_argument("--candidate-predictions", help="DIR of {cutoff}__{target}.csv.gz served deltas (layer candidates)")
    r.add_argument("--data-dir", required=True, help="directory holding every extra rows file named by the recipes")
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n-jobs", type=int, default=-1)
    r.add_argument("--dump-rows", action="store_true", help="also write the scored station-day rows to rows_<tier>.csv.gz")
    args = p.parse_args()
    if args.cmd == "run" and args.candidate_recipe and not args.candidate_metadata:
        raise SystemExit("--candidate-recipe needs --candidate-metadata (its CV gate)")
    {"build-manifest": cmd_build_manifest, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
