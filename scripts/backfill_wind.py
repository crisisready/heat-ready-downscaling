"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

NOT RUNNABLE STANDALONE IN THIS REPO (2026-07-27 note, added during Phase 1
extraction): this script imports db/era5/ghcn -- private-repo-only modules
that write directly to the live Aurora ghcn_training table over a VPC-only
connection. Unlike train_downscaling.py/sweep_*.py (fixed to import from
heatready_downscaling instead), this script's job is a live DATABASE WRITE
against infrastructure this public repo has no path to and, by design
(P0.6/the VPC-isolation architecture), never will. It is landed here
verbatim for provenance and historical record -- the backfill it describes
already ran (see below, "8 countries... backfilled") -- not as part of this
repo's runnable toolkit. Running it again requires checking it out inside
crisisready/heat-risk-data-api's own environment, with its db.py/era5.py/
ghcn.py/build_training_set.py and Aurora access, not from here.

One-shot backfill of nighttime_wind_ms for the existing 274,249 ghcn_training
rows (8 countries, the 2023 training window) -- populates the wind covariate
PR #214/#215 plumbed through ingestion but never backfilled, so it can be
added to downscaling.FEATURE_ORDER and evaluated as a model feature (see
docs/pipeline-fable-consult-2026-07-19-downscaling-diagnosis.md's tmin-zone
findings and docs/plan-2026-07-19-era5-wind-radiation-integration.md).

Real ERA5-Land fetch requesting era5._ERA5_VARIABLES + [u10, v10] -- NOT the
full 6-variable _TRAINING_ERA5_VARIABLES list, since t2m/d2m/sp/humidity are
already correctly populated in the DB and don't need re-fetching, and this
backfill has no use for solar radiation. era5.extract_era5_means requires
t2m/d2m/sp to be present in the NetCDF (it looks them up via the non-optional
_find_var), so the base 3 still have to be requested even though only
wind_ms is used from the result -- 5 variables instead of 6 is the real,
correctly-scoped savings available here.

Reuses build_training_set.py's _era5_download_lock (free-slot-first across
however many CDS accounts are configured -- 3, as of 2026-07-19's dual/
triple-account split going live) and _daily_mean_nighttime_wind (the exact
nighttime-hours aggregation the training pipeline itself uses), so this
mirrors the shipped methodology exactly -- no new wind-computation logic,
just a narrower fetch and an UPDATE instead of an INSERT.

Run one country as its own OS process (--countries <FIPS>), matching
build_training_set.py's own documented concurrent-country pattern: CDS's
multi-account lock already caps real concurrent CDS load at however many
accounts are configured, so launching all 8 countries concurrently gets a
real (not just apparent) ~3x speedup on the CDS-bound step, for free, using
AWS resources already paid for and already sitting mostly idle.

Usage:
    # dry run: smallest country first (NI, 8 stations) to validate
    # correctness and get a real per-country timing baseline.
    python scripts/backfill_wind.py --countries NI

    # full backfill, run once per country as separate concurrent processes:
    for C in US FR GM UK AS IN MX NI; do
        nohup python scripts/backfill_wind.py --countries $C > backfill-wind-$C.log 2>&1 &
    done
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import os
import sys
import threading
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import build_training_set as bts
import db
import era5
import ghcn

logger = logging.getLogger(__name__)

_WIND_BACKFILL_VARIABLES = era5._ERA5_VARIABLES + [era5._WIND_U_VARIABLE, era5._WIND_V_VARIABLE]

# era5.extract_era5_means opens the downloaded NetCDF via xarray, backed by
# the HDF5 C library -- most HDF5 builds (this venv's included) are NOT
# thread-safe for concurrent access from multiple threads in one process.
# Confirmed live 2026-07-19: running 3 segments concurrently on worker
# threads produced a real segfault in libhdf5 (dmesg: "segfault ... in
# libhdf5-....so") the instant multiple threads called extract_era5_means
# at nearly the same moment. Downloads (the actual CDS-bound bottleneck)
# stay fully concurrent -- only the fast (seconds, not CDS's tens-to-
# hundreds of seconds) NetCDF-parsing step is serialized behind this lock,
# so this costs negligible wall-clock while eliminating the crash. Every
# existing production caller of extract_era5_means runs one country per OS
# PROCESS (separate address space, no HDF5 thread-safety exposure) --
# this script is the only place in the codebase that calls it from
# multiple threads within one process, so the lock lives here rather than
# inside era5.py itself.
_NETCDF_PARSE_LOCK = threading.Lock()


def _distinct_stations(region: str) -> list[dict]:
    rows = db.execute(
        "SELECT DISTINCT station_id, lon, lat FROM ghcn_training WHERE region = %s",
        (region,),
    )
    return [{"station_id": r["station_id"], "lon": r["lon"], "lat": r["lat"]} for r in rows]


def _region_date_range(region: str) -> tuple[str, str]:
    """Queries the real min/max date for this region rather than assuming
    the whole training set shares one hardcoded window -- a country added
    later, or re-backfilled independently, could legitimately span a
    different range."""
    row = db.execute(
        "SELECT min(date) mn, max(date) mx FROM ghcn_training WHERE region = %s", (region,),
    )[0]
    return row["mn"], row["mx"]


def _configured_account_count() -> int:
    return sum(1 for i in era5.all_account_indices() if era5.account_configured(i))


def _checkpoint_path(region: str) -> str:
    return f"/tmp/backfill-wind-{region}.done"


def _load_completed_segment_starts(region: str) -> set:
    """Segment-start ISO dates already known to have succeeded and been
    written to the DB in a previous run of this country -- an explicit
    done-file, not a NULL-count check. NULL-counting is unreliable here:
    _daily_mean_nighttime_wind legitimately drops station-days that fail
    the nighttime-hours completeness requirement, so a successfully
    processed month can still leave real NULLs behind."""
    path = _checkpoint_path(region)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _append_completed_segment_starts(region: str, seg_starts) -> None:
    if not seg_starts:
        return
    with open(_checkpoint_path(region), "a") as f:
        for seg_start in seg_starts:
            f.write(seg_start.isoformat() + "\n")


def _fetch_segment(region: str, bbox: str, geojson_obj: dict, seg_start, seg_end) -> list[dict]:
    """Acquire one CDS lock slot, download+extract a single calendar-month
    segment, release the slot. Runs on a worker thread -- both the blocking
    fcntl.flock wait and cdsapi's HTTP calls release the GIL, so N threads
    genuinely overlap on I/O rather than serializing behind Python's GIL."""
    t0 = time.monotonic()
    with bts._era5_download_lock() as account_index:
        logger.info(
            "[%s] acquired CDS account_index=%d for segment %s..%s",
            region, account_index, seg_start, seg_end,
        )
        nc_path = era5.download_era5(
            bbox, seg_start, seg_end,
            dataset="reanalysis-era5-land", variables=_WIND_BACKFILL_VARIABLES,
            account_index=account_index,
        )
    logger.info(
        "[%s] segment %s..%s downloaded in %.1fs", region, seg_start, seg_end, time.monotonic() - t0,
    )
    try:
        with _NETCDF_PARSE_LOCK:
            return era5.extract_era5_means(nc_path, geojson_obj)
    finally:
        os.unlink(nc_path)


def backfill_country(region: str) -> None:
    t_country = time.monotonic()
    stations = _distinct_stations(region)
    if not stations:
        logger.info("[%s] no stations found, skipping", region)
        return

    start_date, end_date = _region_date_range(region)
    bbox = bts.stations_bbox(stations)
    geojson_obj = bts.stations_to_geojson(stations)
    lons = [s["lon"] for s in stations]
    lats = [s["lat"] for s in stations]
    logger.info(
        "[%s] %d station(s), bbox span %.1f x %.1f deg, window %s..%s",
        region, len(stations), max(lons) - min(lons), max(lats) - min(lats), start_date, end_date,
    )

    t0 = time.monotonic()
    tz_map = bts._timezones_for_stations(stations)
    logger.info("[%s] timezones resolved in %.1fs", region, time.monotonic() - t0)

    # Fetch each calendar-month segment on its own worker thread, up to one
    # per configured CDS account, instead of one sequential loop. CDS's own
    # per-request server-side queue wait is highly variable (39s-507s
    # observed live for same-size requests) and independent per request --
    # a single slow month must not block the rest of a country's year
    # behind it while other slots sit idle. This also compounds with
    # _era5_download_lock's existing free-slot-first sharing: that lock
    # already lets ANY process/thread grab whichever of the (up to 3)
    # account slots is free, so up to `_configured_account_count()` segments
    # -- whether from this country alone (single-country dry run) or spread
    # across several countries running concurrently -- are always
    # genuinely in flight together, never serialized behind one slow
    # request. Both the blocking fcntl.flock wait and cdsapi's HTTP calls
    # release the GIL, so this is real overlap, not just apparent
    # concurrency.
    # Each segment's failure is isolated -- one bad month (transient CDS
    # error, a network blip, a misconfigured account) must not discard every
    # OTHER segment's already-successful download. Confirmed live 2026-07-19:
    # an IAM permissions gap on account_index=1 raised inside one segment's
    # future, and an earlier version of this loop let that exception
    # propagate straight out of `as_completed`, killing the whole country's
    # backfill and losing all already-completed months' data (nothing is
    # written to the DB until every segment's rows are collected). Catching
    # per-future here means a partial-coverage month gets logged and
    # skipped, not the entire country.
    # Skip segments a previous run of this country already completed and
    # wrote to the DB -- CDS's rate-limited fetch is the expensive part, and
    # the DB write itself is naturally idempotent, so re-fetching an
    # already-successful month on re-run is pure waste. Uses an explicit
    # done-file (not a NULL-count check -- see _load_completed_segment_starts).
    completed_starts = _load_completed_segment_starts(region)
    all_segments = era5._split_by_calendar_month(start_date, end_date)
    segments = [(s, e) for s, e in all_segments if s.isoformat() not in completed_starts]
    n_skipped = len(all_segments) - len(segments)
    if n_skipped:
        logger.info("[%s] skipping %d already-completed segment(s) per checkpoint", region, n_skipped)
    if not segments:
        logger.info("[%s] every segment already completed per checkpoint, nothing to do", region)
        return

    hourly: list[dict] = []
    failed_segments: list[tuple] = []
    max_workers = max(1, min(len(segments), _configured_account_count()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_segment = {
            pool.submit(_fetch_segment, region, bbox, geojson_obj, seg_start, seg_end): (seg_start, seg_end)
            for seg_start, seg_end in segments
        }
        for future in concurrent.futures.as_completed(future_to_segment):
            seg_start, seg_end = future_to_segment[future]
            try:
                hourly.extend(future.result())
            except Exception:
                failed_segments.append((seg_start, seg_end))
                logger.exception(
                    "[%s] segment %s..%s failed -- skipping it, continuing with the rest", region, seg_start, seg_end,
                )
    logger.info(
        "[%s] extracted %d hourly row(s) across %d/%d segment(s) (%d concurrent worker(s))",
        region, len(hourly), len(segments) - len(failed_segments), len(segments), max_workers,
    )
    if failed_segments:
        logger.warning(
            "[%s] %d segment(s) failed and were skipped -- re-run this country later to fill the gap: %s",
            region, len(failed_segments), failed_segments,
        )

    t0 = time.monotonic()
    nighttime_wind_by_station = bts._daily_mean_nighttime_wind(hourly, tz_map)
    n_station_days = sum(len(v) for v in nighttime_wind_by_station.values())
    logger.info(
        "[%s] nighttime wind computed for %d station-day(s) across %d station(s) in %.1fs",
        region, n_station_days, len(nighttime_wind_by_station), time.monotonic() - t0,
    )

    t0 = time.monotonic()
    rows = [
        (wind_ms, station_id, date_iso)
        for station_id, by_date in nighttime_wind_by_station.items()
        for date_iso, wind_ms in by_date.items()
    ]
    if rows:
        db.execute_many(
            "UPDATE ghcn_training SET nighttime_wind_ms = %s WHERE station_id = %s AND date = %s",
            rows,
        )
    # Only checkpoint segments that both fetched successfully AND made it
    # into this DB write -- a checkpoint entry must always imply the data
    # really is in the DB, never just "the fetch didn't raise."
    succeeded_starts = [seg_start for seg_start, seg_end in segments if (seg_start, seg_end) not in failed_segments]
    _append_completed_segment_starts(region, succeeded_starts)
    logger.info(
        "[%s] updated %d station-day row(s) -- country done in %.1fs total",
        region, len(rows), time.monotonic() - t_country,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", required=True, help="Comma-separated FIPS country codes.")
    args = parser.parse_args()
    countries = [c.strip().upper() for c in args.countries.split(",")]

    # Confirmed live 2026-07-19: this database's ghcn_training table
    # predates the nighttime_wind_ms column migration (ghcn.py:180) --
    # build_training_set.py's own main() already calls this at startup
    # (build_training_set.py:963), but this script never did, so the whole
    # NI backfill ran its full CDS fetch (12/12 segments, 0 failures) only
    # to fail at the final UPDATE with "column nighttime_wind_ms does not
    # exist". Idempotent (ADD COLUMN IF NOT EXISTS), safe to call from every
    # concurrently-launched country process, matching the established
    # pattern exactly.
    ghcn.create_ghcn_training_table()

    for region in countries:
        try:
            backfill_country(region)
        except Exception:
            logger.exception("[%s] country-level backfill failed entirely -- continuing with the rest", region)
    logger.info("Wind backfill complete for %s.", countries)


if __name__ == "__main__":
    main()
