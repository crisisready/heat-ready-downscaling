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
import sys
from collections import defaultdict
from datetime import date

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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stations", required=True)
    ap.add_argument("--start-date", type=date.fromisoformat, required=True)
    ap.add_argument("--end-date", type=date.fromisoformat, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--station-ids", nargs="+", help="restrict to these station ids (smoke tests)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import ghcn
    from build_gsod_bsh_rows import assemble_rows
    from build_training_set import (
        _bucket_from_credentials,
        _cluster_stations_by_bbox_cells,
        _landscan_from_credentials,
        fetch_era5_land_for_stations_via_timeseries,
        snapshot_covariates_for_stations,
    )

    os.makedirs(args.out_dir, exist_ok=True)
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

    landscan_bucket, landscan_key = _landscan_from_credentials()
    vuln_bucket = _bucket_from_credentials()
    # Same batching as build_gsod_bsh_rows.main and build_training_set.main: per country, then clustered
    # by bbox cells. lst_warm_season_anomaly_c is relative to the other stations in the batch, so a lone
    # station in its own batch would get an anomaly of about 0 by construction. The batch here is these
    # stations (a known, accepted difference from the main corpus, as for the GSOD rows).
    covariates_by_station = {}
    for fips in sorted({s["fips"] for s in stations}):
        country = [f for f, s in zip(fetch_stations, stations) if s["fips"] == fips]
        clusters = _cluster_stations_by_bbox_cells(country)
        for i, cluster in enumerate(clusters):
            label = f"thinzone_{fips}" if len(clusters) == 1 else f"thinzone_{fips}_c{i}"
            logger.info("covariates: %s (%d station(s))", label, len(cluster))
            covariates_by_station.update(snapshot_covariates_for_stations(
                cluster, vuln_bucket, batch_label=label,
                landscan_bucket=landscan_bucket, landscan_key=landscan_key))

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
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
