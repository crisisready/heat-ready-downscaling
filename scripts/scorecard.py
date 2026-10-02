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
    if recipe["feature_set"] != "base":
        raise SystemExit("scorecard v1 refits base-feature recipes only; a regime recipe needs its covariate "
                         "CSVs plumbed through first")
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
    return rows, sources


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


def recipe_corpus(rows: list[dict], recipe: dict, cutoff: str, truth_ids: set[str]) -> list[dict]:
    """The recipe's training rows at a cutoff: its own sources, dated on or before the cutoff, with
    every truth station removed at every date."""
    srcs = {"ghcn_training", *(x["name"] for x in recipe["extra_rows"])}
    return [r for r in rows if r["_src"] in srcs and str(r["date"])[:10] <= cutoff
            and r["station_id"] not in truth_ids]


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


def served_predictions(train_rows: list[dict], test_rows: list[dict], target: str, metadata: dict,
                       n_jobs: int) -> list[dict]:
    """Fit the recipe's QRF on train_rows and return the SERVED result for each test row: the DS
    serving contract with this version's own CV gate (from its metadata.json)."""
    import train_downscaling as td
    from quantile_forest import RandomForestQuantileRegressor

    from heatready_downscaling.contract import QRFModelAdapter, derive_zones_passing_cv_gate
    from heatready_downscaling.features import FEATURE_ORDER

    X, y, *_ = td.build_training_feature_matrix(train_rows, target, feature_order=FEATURE_ORDER)
    if len(y) < td._MIN_FOLD_TRAIN_ROWS:
        return []
    model = RandomForestQuantileRegressor(**{**td._QRF_PARAMS, "n_jobs": n_jobs}).fit(X, y)
    bundle = {
        f"model_{target}": model,
        "metadata": {"model_version": metadata["model_version"], "feature_order": list(FEATURE_ORDER),
                     "conformal_q95_by_zone": metadata.get("conformal_q95_by_zone", {}),
                     "conformal_q95_by_zone_tmin": metadata.get("conformal_q95_by_zone_tmin", {}),
                     "ood_aoa_threshold": metadata.get("ood_aoa_threshold")},
        "zones_passing_cv_gate": derive_zones_passing_cv_gate(metadata),
        **td._build_aoa_index(model, X, target, td._QRF_PARAMS["random_state"]),
    }
    return QRFModelAdapter(bundle).predict(test_rows, target)


def recipe_predictions(rows, recipe, metadata, cutoff, target, truth_ids, test_rows, cache_dir, key_parts,
                       n_jobs) -> tuple[dict, dict]:
    """{(station_id, date): (served_delta, ci95)} for one recipe/cutoff/target, from the cache when the
    content key matches. Returns (predictions, provenance)."""
    train = recipe_corpus(rows, recipe, cutoff, truth_ids)
    served_meta = {"gate": metadata.get("cv", {}).get("leave_region_out", {}).get(target, {}).get("by_zone"),
                   "q95": metadata.get("conformal_q95_by_zone" if target == "tmax" else "conformal_q95_by_zone_tmin"),
                   "ood": metadata.get("ood_aoa_threshold")}
    key = _cache_key({**key_parts, "recipe": recipe, "cutoff": cutoff, "target": target, "served_meta": served_meta,
                      "train_rows_sha256": rows_fingerprint(train), "test_rows_sha256": rows_fingerprint(test_rows)})
    path = os.path.join(cache_dir, f"{recipe['model_version']}__{cutoff}__{target}__{key}.csv.gz")
    if os.path.exists(path):
        preds = {}
        with gzip.open(path, "rt", newline="") as f:
            for rec in csv.DictReader(f):
                preds[(rec["station_id"], rec["date"])] = (float(rec["served_delta"]),
                                                          float(rec["ci95"]) if rec["ci95"] else float("nan"))
        return preds, {"cache": os.path.basename(path), "cache_hit": True, "sha256": _sha256_file(path)}
    results = served_predictions(train, test_rows, target, metadata, n_jobs)
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
                 "truth_stations_sha256": _sha256_file(_repo_path(args.spec, spec["truth_stations"])),
                 "sources": sorted((s["name"], s["sha256"]) for s in sources)}

    frames, years_out, prov = [], [], []
    for cutoff in spec["tiers"][args.tier]["cutoffs"]:
        year = sc.score_year(cutoff)
        inc_train_stations = None
        for target in ("tmax", "tmin"):
            keys = [(s, d) for s, d, t in manifest if t == target and int(d[:4]) == year]
            if not keys:
                prov.append({"cutoff": cutoff, "target": target, "status": "no manifest rows"})
                continue
            test_rows = [by_key[k] for k in keys]
            p_inc, prov_inc = recipe_predictions(rows, inc, inc_meta, cutoff, target, truth_ids, test_rows,
                                                 args.cache_dir, key_parts, args.n_jobs)
            if not p_inc:
                p_cand, prov_cand = {}, {"note": "not fit: the incumbent has nothing to compare against"}
            elif cand is not None:
                p_cand, prov_cand = recipe_predictions(rows, cand, cand_meta, cutoff, target, truth_ids, test_rows,
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
            if inc_train_stations is None:
                tr = recipe_corpus(rows, inc, cutoff, truth_ids)
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
    args = p.parse_args()
    if args.cmd == "run" and args.candidate_recipe and not args.candidate_metadata:
        raise SystemExit("--candidate-recipe needs --candidate-metadata (its CV gate)")
    {"build-manifest": cmd_build_manifest, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
