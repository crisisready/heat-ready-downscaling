"""
GHCN-Daily rows for the scorecard's thin-zone unseen stations (C1 thin-zone pull, E54): ghcn_training-shaped
rows for a fixed list of GHCN-D stations, 2023-2025, so the out-of-time scorecard can score zones where no
unseen station exists today (Aw, Am, Af, Cwb, Cwa, Dfa, Dfb, Dwa).

The station list is the selection CSV from heat-risk-data-api research/thin-zone-pull/select_candidates.py
(id, lat, lon, elev, cc, ...). Observations come from NOAA's Access Data Service through
ghcn.fetch_ghcn_daily_bulk_concurrent (same QC-flag path as the main corpus). Grid values, humidity, wind and
covariates come from the same helpers build_gsod_bsh_rows.py and build_training_set.py use, and rows are
assembled by build_gsod_bsh_rows.assemble_rows, so a row here is built exactly as every other ghcn_training row.

Output is a JSON file of rows (not a database write). The rows go to the scorecard's unseen set only.

Environment and PYTHONPATH as build_gsod_bsh_rows.py (heat-risk-data-api/src on PYTHONPATH).

Usage:
    python3 scripts/build_thin_zone_rows.py --stations selected_stations.csv --start-date 2023-01-01 \\
        --end-date 2025-12-31 --out-dir /tmp/thin_zone_rows [--station-ids ID ...]
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import multiprocessing

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

logger = logging.getLogger("build_thin_zone_rows")


def read_stations(path, only=None):
    """Selection CSV -> station dicts the shared row assembler takes (station_id, lat, lon,
    elevation_m, name, fips)."""
    stations = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if only and r["id"] not in only:
                continue
            try:
                elev = float(r["elev"])
            except (KeyError, TypeError, ValueError):
                elev = None
            stations.append({"station_id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                             "elevation_m": None if elev is None or elev <= -999 else elev,
                             "name": r.get("name", ""), "fips": r["id"][:2], "zone": r.get("zone", "")})
    ids = [s["station_id"] for s in stations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate station ids in the selection CSV")
    return stations


def coverage_by_year(series):
    counts = defaultdict(int)
    for obs in series:
        counts[obs["date"][:4]] += 1
    return dict(sorted(counts.items()))


def _cluster_covariates_worker(label, cluster):
    """Runs in a spawned child process: one cluster's covariate snapshot (canopy, WorldCover, GHSL, DEM,
    LandScan, LST). The batch is exactly the cluster, so lst_warm_season_anomaly_c (a 75 km radius baseline
    inside this cluster's own scratch table) is the same value the serial loop produces."""
    from build_training_set import (_bucket_from_credentials, _landscan_from_credentials,
                                    snapshot_covariates_for_stations)
    landscan_bucket, landscan_key = _landscan_from_credentials()
    return label, snapshot_covariates_for_stations(
        cluster, _bucket_from_credentials(), batch_label=label,
        landscan_bucket=landscan_bucket, landscan_key=landscan_key)


def _s3_cp(local, s3_uri):
    subprocess.run(["aws", "s3", "cp", local, s3_uri, "--only-show-errors"], check=True)


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def plan_clusters(stations, fetch_stations, cluster_fn):
    """Deterministic (label, cluster) list: per country, then clustered by bbox cells, same labels the serial
    loop used (thinzone_<fips> or thinzone_<fips>_c<i>)."""
    plan = []
    for fips in sorted({s["fips"] for s in stations}):
        country = [f for f, s in zip(fetch_stations, stations) if s["fips"] == fips]
        clusters = cluster_fn(country)
        for i, cluster in enumerate(clusters):
            plan.append((f"thinzone_{fips}" if len(clusters) == 1 else f"thinzone_{fips}_c{i}", cluster))
    return plan


def run_cluster_covariates(plan, out_dir, workers, s3_prefix=None, max_clusters=None):
    """Covariates for every cluster in `plan`, `workers` clusters at a time (spawned processes, so each has its
    own DB/S3 clients). Each finished cluster is written to out_dir/clusters/<label>.json and copied to
    s3_prefix/clusters/ right away; clusters already on disk are skipped, so a relaunch resumes. Returns
    (covariates_by_station, failed_labels)."""
    cdir = os.path.join(out_dir, "clusters")
    os.makedirs(cdir, exist_ok=True)
    saved = set()
    for l, c in plan:  # a saved file counts only if it covers exactly this cluster's stations
        path = os.path.join(cdir, l + ".json")
        if os.path.exists(path):
            with open(path) as f:
                if set(json.load(f)) == {s["station_id"] for s in c}:
                    saved.add(l)
                else:
                    logger.warning("saved cluster %s has a different station set, re-running it", l)
    todo = [(l, c) for l, c in plan if l not in saved]
    if max_clusters:
        todo = todo[:max_clusters]
    logger.info("covariates: %d cluster(s) planned, %d already saved, %d to run, %d worker(s)",
                len(plan), len(saved), len(todo), workers)
    failed = []
    if todo:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_cluster_covariates_worker, l, c): (l, len(c)) for l, c in todo}
            done = 0
            for fut in as_completed(futs):
                label, n = futs[fut]
                try:
                    _, cov = fut.result()
                except Exception:
                    logger.exception("cluster %s (%d station(s)) FAILED", label, n)
                    failed.append(label)
                    continue
                path = os.path.join(cdir, label + ".json")
                try:
                    _write_json_atomic(path, cov)
                    if s3_prefix:
                        _s3_cp(path, f"{s3_prefix}/clusters/{label}.json")
                except Exception:
                    logger.exception("cluster %s computed but save/copy FAILED", label)
                    failed.append(label)
                    continue
                done += 1
                logger.info("cluster %s done (%d station(s)); %d/%d, saved%s", label, n, done, len(todo),
                            " + copied to S3" if s3_prefix else "")
    covariates = {}
    for label, _ in plan:
        path = os.path.join(cdir, label + ".json")
        if os.path.exists(path) and label not in failed:
            with open(path) as f:
                covariates.update(json.load(f))
    return covariates, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stations", required=True)
    ap.add_argument("--start-date", type=date.fromisoformat, required=True)
    ap.add_argument("--end-date", type=date.fromisoformat, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--station-ids", nargs="+", help="restrict to these station ids (smoke tests)")
    ap.add_argument("--cluster-workers", type=int, default=1,
                    help="covariate clusters processed in parallel (separate processes); default 1")
    ap.add_argument("--s3-prefix", help="s3://bucket/prefix: restore earlier output from here at start, and copy "
                    "checkpoints and each finished cluster there as they complete (makes a relaunch resumable)")
    ap.add_argument("--max-clusters", type=int, help="only run this many not-yet-saved clusters (timing samples)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import ghcn
    from build_gsod_bsh_rows import assemble_rows
    from build_training_set import (
        _cluster_stations_by_bbox_cells,
        fetch_era5_land_for_stations_via_timeseries,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.s3_prefix:
        subprocess.run(["aws", "s3", "sync", args.s3_prefix, args.out_dir, "--only-show-errors"], check=True)
    stale = os.path.join(args.out_dir, "thin_zone_rows.json")
    if os.path.exists(stale):
        os.remove(stale)  # never let a restored earlier result stand in for this run's
    stations = read_stations(args.stations, set(args.station_ids) if args.station_ids else None)
    logger.info("%d station(s)", len(stations))

    series_by_sid = ghcn.fetch_ghcn_daily_bulk_concurrent(
        [s["station_id"] for s in stations], args.start_date, args.end_date,
        checkpoint_path=os.path.join(args.out_dir, "ghcnd_checkpoint.jsonl"))
    no_series = sorted(s["station_id"] for s in stations if not series_by_sid.get(s["station_id"]))
    if no_series:  # a failed fetch and a genuinely empty station look the same here, so say so and mark incomplete
        logger.warning("no GHCN-D series for %d station(s), left out: %s", len(no_series), no_series)
    stations = [s for s in stations if series_by_sid.get(s["station_id"])]
    coverage = {s["station_id"]: coverage_by_year(series_by_sid[s["station_id"]]) for s in stations}

    fetch_stations = [{k: s[k] for k in ("station_id", "lat", "lon", "elevation_m", "name")} for s in stations]
    grid_by_station, humidity_by_station, wind_by_station = fetch_era5_land_for_stations_via_timeseries(
        fetch_stations, args.start_date, args.end_date,
        checkpoint_path=os.path.join(args.out_dir, "era5_timeseries_checkpoint.jsonl"))
    missing = sorted(set(series_by_sid) - set(grid_by_station))
    if missing:
        logger.warning("no ERA5-Land series for %d station(s), left out: %s", len(missing), missing)

    if args.s3_prefix:
        for n in ("ghcnd_checkpoint.jsonl", "era5_timeseries_checkpoint.jsonl"):
            if os.path.exists(os.path.join(args.out_dir, n)):
                _s3_cp(os.path.join(args.out_dir, n), f"{args.s3_prefix}/{n}")
    # Same batching as build_gsod_bsh_rows.main and build_training_set.main: per country, then clustered
    # by bbox cells. lst_warm_season_anomaly_c is relative to the other stations in the batch, so a lone
    # station in its own batch would get an anomaly of about 0 by construction. The batch here is these
    # stations (a known, accepted difference from the main corpus, as for the GSOD rows). Clusters run in
    # parallel but each cluster is still its own batch, so values do not depend on --cluster-workers.
    plan = plan_clusters(stations, fetch_stations, _cluster_stations_by_bbox_cells)
    covariates_by_station, failed = run_cluster_covariates(
        plan, args.out_dir, args.cluster_workers, args.s3_prefix, args.max_clusters)
    if failed or args.max_clusters:
        logger.warning("covariates incomplete (failed clusters: %s; max_clusters=%s), no rows file written",
                       failed, args.max_clusters)
        return 1 if failed else 0

    rows, shifts = assemble_rows(stations, series_by_sid, grid_by_station, humidity_by_station,
                                 wind_by_station, covariates_by_station)
    by_station = defaultdict(int)
    for r in rows:
        by_station[r["station_id"]] += 1
    logger.info("assembled %d row(s) across %d station(s)", len(rows), len(by_station))
    out = os.path.join(args.out_dir, "thin_zone_rows.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "row_count": len(rows), "rows_by_station": dict(by_station),
                   "obs_window_shift_days": shifts, "missing_era5": missing, "no_ghcnd_series": no_series,
                   "ghcnd_coverage_by_year": coverage,
                   "start_date": args.start_date.isoformat(), "end_date": args.end_date.isoformat(),
                   "complete": not missing and not no_series}, f)
    if args.s3_prefix:
        _s3_cp(out, f"{args.s3_prefix}/thin_zone_rows.json")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    sys.exit(main() or 0)
