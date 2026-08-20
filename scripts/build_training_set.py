"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

FORWARD-PORTED 2026-07-27 (Phase 1 section 5.3) onto main's current src/
API: dem.extract_elevation dropped its `bbox` parameter and now returns
{"status","resume_index","results"} (was a plain dict); vulnerability.
extract_worldcover's return shape gained the same {"status","resume_index",
"results"} wrapper. Both call sites below are fixed to the current
signature/shape -- see each fix's own inline comment for exactly what
changed and why. lst.compute_warm_season_composite's name_coords/
reference_radius_km parameters (also branch-only relative to main at
extraction time) needed no fix here -- they were forward-ported onto main
separately (Phase 0 P0.2), so this branch's call site already matches.

NOT RUNNABLE STANDALONE IN THIS REPO: like scripts/backfill_wind.py, this
script imports dem/era5/ghcn/heat_calcs/landscan/lst/vulnerability --
private-repo-only modules for live CDS/S3/Aurora access this public repo
has no path to and, by design, never will get (the whole point of the
export/snapshot split is that contributors never need direct DB/CDS
access). It is landed here, forward-ported and syntax-verified, because
the CODE is the reproducible methodology record this program depends on
being able to audit and re-run -- but actually RUNNING it (the monthly
regeneration job, or the one-country/one-month smoke-test gate that
verifies this forward-port) happens from crisisready/heat-risk-data-api's
own bastion, with its db.py/era5.py/dem.py/ghcn.py/heat_calcs.py/
landscan.py/lst.py/vulnerability.py and live Aurora/CDS/S3 access, not
from a checkout of this repo alone. tests/test_build_training_set.py is
excluded from this repo's own test collection for the same reason (see
conftest.py).

Offline training-set builder for the downscaling model (downscaling build
Phase 3 -- see docs/plan-2026-07-18-neighborhood-resolution-downscaling.md
sections 2.4, 3.3, 4.1, 7).

For each GHCN-Daily station in the given dense-station countries: fetches
the station's daily Tmax/Tmin, fetches ERA5-Land at the station's location
for the same window, aligns the two time series' day-boundary convention
(GHCN-Daily's station-local observation-time windows don't always match
ERA5-Land's UTC-hourly-aggregated local day), snapshots the same static
covariates the downscaling model uses at inference (canopy, WorldCover,
GHSL-SMOD, elevation), and writes one row per station-day into the global
`ghcn_training` table.

Not a Lambda dependency -- this is an offline, one-time (or periodic,
re-run to add stations/refresh data) build script. Intended to run on a
temporary EC2 instance, not gu-dev, for any real multi-country/multi-year
build: an ERA5-Land pull plus four raster covariate extractors per country
is real, sustained CDS/network/CPU load, and this repo has hit the shared-
cgroup OOM documented in feedback_use_aws_for_heavy_jobs.md before for
similarly-sized jobs. A small-scale run (one country, a short window, a
handful of stations) is fine locally for testing.

Batches by country, sub-batched by Koppen climate zone (2026-08-03): one
ERA5-Land pull and one covariate-extraction pass per zone within a country
(bbox covering that zone's stations only), not one pull per station -- the
same amortization era5.py/vulnerability.py already rely on for per-project
extraction. Sub-batching by zone (not just one bbox per country) matters
most for --station-ids-file: a curated station set can span multiple zones
across a country, and a single country-wide bbox for a geographically
dispersed set can exceed CDS's own per-request cost limit (confirmed live
2026-08-03) -- zones are naturally compact by construction, so this keeps
every request's bbox small regardless of how spread out the country's full
station set is.

Usage:
    # Small test run, one country, short window, tight geographic cluster
    # (keeps the ERA5-Land bbox tiny for a fast local check):
    python scripts/build_training_set.py --countries US --start-date 2016-06-01 \
        --end-date 2016-06-30 --max-stations-per-country 5 --profile nish-climateverse

    # A real multi-country training build (run on EC2, not gu-dev). Active-
    # station filtering (ghcn.active_station_ids) is always applied; the
    # stratified selection mode spreads the cap for climate/land-cover
    # diversity, and --max-bbox-extent-deg bounds a large country's own
    # candidate set so ERA5-Land download volume stays reasonable (volume
    # scales with geographic EXTENT, not station count -- confirmed live
    # 2026-07-18 that a full-CONUS, one-month pull is ~3.5 GB regardless of
    # how many of the 78,566 US stations are actually selected):
    python scripts/build_training_set.py --countries US,GM,FR,AS,UK \
        --start-date 2023-01-01 --end-date 2023-12-31 \
        --selection-mode stratified --max-stations-per-country 300 \
        --max-bbox-extent-deg 15

    # Running multiple countries as CONCURRENT processes (one per country,
    # each with its own --countries value) is safe and recommended for a
    # real multi-country build: LST/covariate extraction is CPU/IO-bound on
    # this instance with no cross-country resource conflict, so it
    # genuinely parallelizes (confirmed 2026-07-18: cost is instance-hours,
    # not per-core, so using otherwise-idle vCPUs this way is close to
    # free). Only the ERA5-Land CDS download itself is serialized ACROSS
    # those concurrent processes via a local file lock
    # (_era5_download_lock) -- CDS's own account-level concurrency limit
    # means simultaneous submissions risk queuing or rejection, not a real
    # speedup, so parallelizing that specific step buys nothing anyway.

    # An explicit, pre-curated station set (2026-08-03) -- e.g. a
    # downscaling-confidence gap-fill analysis that already picked exactly
    # which stations to add, across one or more countries in a single pooled
    # write. Bypasses --countries/--max-stations-per-country/--selection-mode/
    # --max-bbox-extent-deg entirely; countries are derived from the IDs'
    # own FIPS prefixes and still batch ERA5/covariate fetches per country.
    # --dry-run runs the identical pipeline (same real CDS/S3 cost) but
    # skips the final ghcn_training write, printing a per-country row count
    # and covariate-completeness summary instead -- always run this first.
    python scripts/build_training_set.py --station-ids-file /tmp/pooled_station_ids.json \
        --start-date 2016-06-01 --end-date 2023-12-31 --profile nish-climateverse --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import dem
import era5
import ghcn
import heat_calcs
import landscan
import lst
import vulnerability

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(REPO_ROOT, "credentials.yaml")

_STATION_BUFFER_DEG = 0.01  # ~1km square buffer "polygon" per station, for reuse with polygon-based extractors
_BBOX_PAD_DEG = 0.5  # padding around a country's station cluster, so the ERA5-Land/covariate bbox has margin
_ERA5_CHUNK_DAYS = 400  # bounds per-request netCDF size/memory -- mirrors manager.backfill_era5's
                        # existing chunked-window precedent. Deliberately > 365 (2026-07-18): a
                        # single training year now fits in ONE outer chunk, so era5.download_era5's
                        # OWN calendar-month splitting (added to fix a real CDS cartesian-date-field
                        # bug, see era5._split_by_calendar_month) is the only thing that decomposes
                        # the request -- confirmed live this produces exactly the natural ~14
                        # calendar-month segments a year needs, instead of the 20 a smaller outer
                        # chunk size produces by ALSO creating extra tiny segments at every internal
                        # chunk boundary's 2-day padding (a real, measured ~30% cut in the total
                        # serialized CDS requests, the confirmed dominant bottleneck for a multi-
                        # country build). Still bounded (not removed entirely): a future multi-year
                        # retrain would chunk at this boundary, keeping era5.download_era5's own
                        # in-memory segment-merge (_merge_era5_segments) from growing unboundedly.

# Cross-process mutex serializing the CDS ERA5-Land download call -- guards
# against CDS's own per-account 1-concurrent-request limit (discovered
# 2026-07-06; see manager.py's _CDS_LOCK_TABLE docstring) whenever more than
# one build_training_set.py process downloads ERA5 at the same time (e.g.
# two --countries invocations launched concurrently by hand). One lock file
# per CDS account slot -- index 0 always exists; index 1+ exist only when
# that CDS account is configured (era5.account_configured), matching
# production's manager.py fallback story (docs/plan-2026-07-19-cds-dual-
# account-split.md section 5, Q2). Derived from era5.all_account_indices()
# -- NOT a separately hardcoded list -- so a new account added only to
# era5.py's _ERA5_SECRET_ENV_VARS is picked up here automatically (found via
# code review 2026-07-19: a hardcoded parallel list here meant a 3rd
# account would silently never get its own lock file).
#
# A plain local file lock, not production's DB-backed manager.cds_request_lock
# -- that lock's 16-minute lease is sized for Lambda's 900s ceiling and would
# expire mid-download for this script's much longer, unbounded pulls, letting
# a second holder in while the first is still actively downloading. Every
# concurrent process here shares one filesystem on one machine, so a local
# flock has no such staleness concern: it only needs to outlive the processes
# holding it.
_ERA5_DOWNLOAD_LOCK_PATHS = [
    f"/tmp/build_training_set_era5_download_{chr(ord('a') + i)}.lock"
    for i in era5.all_account_indices()
]


@contextlib.contextmanager
def _era5_download_lock():
    """
    Acquire whichever configured CDS-account lock slot is free right now --
    free-slot-first, not a deterministic per-country/per-process split
    (docs/plan-2026-07-19-cds-dual-account-split.md section 5, Q1): country
    processes are launched independently with no persistent per-country
    account association, so "always the same account" idempotency buys
    nothing, while a deterministic split would load-imbalance badly at
    small country counts (or whenever one country's bbox/volume dwarfs
    another's).

    Mechanism: try each configured lock file non-blocking, in order; the
    first one successfully locked is the acquired slot. If every configured
    slot is already held, block on the first configured one (guaranteed
    eventual progress -- exactly what a single lock would always have
    given). Yields the account_index of whichever slot was acquired, to
    pass into era5.download_era5.

    When ERA5_SECRET_ARN_2 is unset, only lock A / account_index 0 is
    configured -- every acquisition blocks on that one file exactly as a
    script that had never been generalized past a single lock would.
    """
    configured = [
        (path, account_index)
        for account_index, path in enumerate(_ERA5_DOWNLOAD_LOCK_PATHS)
        if era5.account_configured(account_index)
    ]
    # account_index 0 is always configured (era5.account_configured(0) is
    # unconditionally True), so `configured` is never empty here.

    fh = None
    acquired_account_index = None
    try:
        while fh is None:
            for path, account_index in configured:
                candidate = open(path, "w")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    candidate.close()
                    continue
                fh, acquired_account_index = candidate, account_index
                break
            if fh is None:
                # Real bug found live 2026-08-20, first genuine 3-way
                # concurrent run: the original fallback here committed to a
                # single blocking flock() on configured[0] the moment every
                # slot was seen busy in one scan -- so a caller could sit
                # blocked on THAT one specific slot for many minutes even
                # after a DIFFERENT slot freed up seconds later (confirmed
                # live: two processes both saw all 3 slots busy; one grabbed
                # slot 'a' once it freed, the other stayed blocked on 'a'
                # specifically for 8+ minutes while slot 'b' sat completely
                # idle the whole time -- a real, direct throughput loss, not
                # a theoretical one). Re-scanning ALL configured slots every
                # second instead of ever committing to one preserves the
                # same "guaranteed eventual progress" property (the loop
                # never gives up) while actually grabbing whichever slot
                # frees first, not whichever happened to be first in the
                # list. Re-opening 2-3 tiny local lock files once/sec while
                # genuinely waiting on a multi-minute CDS queue anyway is
                # negligible overhead, not worth optimizing further.
                time.sleep(1.0)

        yield acquired_account_index
    finally:
        if fh is not None:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()

# The expanded ERA5-Land variable list this training builder requests --
# era5._ERA5_VARIABLES (t2m/d2m/sp) plus wind/radiation, so this module can
# compute the nighttime_wind_ms covariate (docs/plan-2026-07-19-era5-wind-
# radiation-integration.md). Built from era5._ERA5_VARIABLES rather than a
# second hardcoded copy of the base three, so it can never silently drift
# from the production default it's extending. era5._ERA5_VARIABLES itself is
# never mutated -- only this training-time list is expanded (plan section 5,
# Q1); production's daily/backfill calls keep requesting exactly 3.
# References era5.py's own _WIND_U_VARIABLE/_WIND_V_VARIABLE/
# _SOLAR_RADIATION_VARIABLE constants rather than re-typing the CDS
# variable-name strings here -- a second, independently-typed copy of these
# strings is exactly the drift risk _find_var_optional's silent-None-on-miss
# design can't catch on its own (code review 2026-07-19; see era5.py's own
# comment on these constants).
_TRAINING_ERA5_VARIABLES = era5._ERA5_VARIABLES + [
    era5._WIND_U_VARIABLE,
    era5._WIND_V_VARIABLE,
    era5._SOLAR_RADIATION_VARIABLE,
]


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


def _landscan_from_credentials() -> tuple[str, str]:
    """(bucket, key) for the single global LandScan raster -- same
    credentials.yaml-first, env-var-fallback convention as
    _bucket_from_credentials, mirroring manager._insert_population's
    LANDSCAN_BUCKET/LANDSCAN_KEY env-var read for the Lambda path."""
    if not os.path.exists(CREDENTIALS_FILE):
        bucket, key = os.environ.get("LANDSCAN_BUCKET"), os.environ.get("LANDSCAN_KEY")
        if bucket and key:
            return bucket, key
        raise FileNotFoundError(
            f"{CREDENTIALS_FILE} not found and LANDSCAN_BUCKET/LANDSCAN_KEY are unset",
        )
    import yaml
    with open(CREDENTIALS_FILE) as f:
        creds = yaml.safe_load(f)
    return creds["landscan"]["bucket"], creds["landscan"]["key"]


def stations_to_geojson(stations: list[dict]) -> dict:
    """Build one tiny square buffer "polygon" per station, named by
    station_id, so this codebase's existing per-polygon extractors
    (vulnerability.py, dem.py) can be reused as-is against station points
    without any new raster-reading code."""
    features = []
    for s in stations:
        lon, lat = s["lon"], s["lat"]
        d = _STATION_BUFFER_DEG
        features.append({
            "type": "Feature",
            "properties": {"name": s["station_id"]},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - d, lat - d], [lon + d, lat - d],
                    [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d],
                ]],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def stations_bbox(stations: list[dict], pad_deg: float = _BBOX_PAD_DEG) -> str:
    """Bounding box (west,south,east,north) covering all given stations,
    padded so ERA5-Land's nearest-centroid lookup and the raster extractors
    have real margin around edge stations."""
    lons = [s["lon"] for s in stations]
    lats = [s["lat"] for s in stations]
    return f"{min(lons) - pad_deg},{min(lats) - pad_deg},{max(lons) + pad_deg},{max(lats) + pad_deg}"


def _timezones_for_stations(stations: list[dict]) -> dict[str, str]:
    """Each station's own local timezone -- GHCN's Tmax/Tmin and
    heat_calcs.aggregate_hourly_to_daily's 06:00-local day boundary both
    need this in the station's actual local time, not UTC (verified live
    2026-07-18: hardcoding UTC silently produced zero complete local days
    for a real Phoenix, UTC-7, station). Builds one shared TimezoneFinder
    instance for the whole station batch rather than calling
    heat_calcs.get_timezone per station -- that helper constructs a fresh
    TimezoneFinder() (which loads real boundary data) on every call, fine
    for manager._lookup_timezones' per-project handful of polygons but
    wasteful for a dense country's full GHCN station list here."""
    from timezonefinder import TimezoneFinder

    tf = TimezoneFinder()
    return {
        s["station_id"]: tf.timezone_at(lng=s["lon"], lat=s["lat"]) or "UTC"
        for s in stations
    }


def _date_chunks(start_date: date, end_date: date, chunk_days: int = _ERA5_CHUNK_DAYS) -> list[tuple[date, date]]:
    """Split [start_date, end_date] into contiguous, non-overlapping chunks of
    at most chunk_days -- each chunk is independently padded (+/-2 days) when
    downloaded, so a chunk boundary never leaves a requested day incomplete
    (each chunk pulls its own margin, regardless of its neighbors)."""
    chunks = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def fetch_era5_land_for_stations(
    stations: list[dict], start_date: date, end_date: date,
    bbox: str | None = None, geojson_obj: dict | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Download ERA5-Land for the bbox covering all given stations, chunked into
    _ERA5_CHUNK_DAYS-sized windows (mirrors manager.backfill_era5's existing
    chunked-window precedent) -- a country-wide bbox pulled in one shot across
    a multi-year training window risks a multi-GB single netCDF and an
    outsized single CDS request; chunking bounds both. Extracts per-station
    daily Tmax/Tmin (via the existing heat_calcs.aggregate_hourly_to_daily --
    the same function ERA5/CAMS/METAR all use, so "a day" is defined
    identically everywhere), daily-mean specific humidity, and nighttime-mean
    wind speed (all computed directly from the hourly rows, since
    aggregate_hourly_to_daily's return shape doesn't carry d2m/sp/wind_ms
    through) for each chunk, then merges across chunks.

    Requests _TRAINING_ERA5_VARIABLES (the expanded 6-variable wind/
    radiation set), not era5's 3-variable production default -- this is the
    one call site in this codebase that needs u10/v10/ssrd at all (docs/
    plan-2026-07-19-era5-wind-radiation-integration.md).

    bbox/geojson_obj may be passed in precomputed (build_rows_for_country
    shares one bbox/geojson_obj between this and snapshot_covariates_for_stations
    rather than each recomputing the identical structure from the same
    station list); computed from `stations` if omitted, for standalone use.

    Returns (daily_tmax_tmin_by_station, humidity_by_station, nighttime_wind_by_station):
      - daily_tmax_tmin_by_station: {station_id: {date_iso: {"tmax": .., "tmin": ..}}}
      - humidity_by_station: {station_id: {date_iso: specific_humidity_kgkg}}
      - nighttime_wind_by_station: {station_id: {date_iso: nighttime_wind_ms}}
    """
    bbox = bbox if bbox is not None else stations_bbox(stations)
    geojson_obj = geojson_obj if geojson_obj is not None else stations_to_geojson(stations)
    tz_map = _timezones_for_stations(stations)

    daily_by_station: dict[str, dict[str, dict]] = {}
    humidity_by_station: dict[str, dict[str, float]] = {}
    nighttime_wind_by_station: dict[str, dict[str, float]] = {}

    # chunk_days passed explicitly (not relying on _date_chunks' own default)
    # so a test (or a future caller) can override _ERA5_CHUNK_DAYS at call
    # time -- a bare default argument is bound once at function-definition
    # time in Python, so patching the module constant afterward would
    # silently have no effect on an already-defined default.
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk_days=_ERA5_CHUNK_DAYS):
        # aggregate_hourly_to_daily only returns days with all 24 local hours present.
        # A local day can span into the adjacent UTC calendar day in either direction
        # depending on the station's UTC offset sign (confirmed live: a single-day
        # pull for a UTC-7 station produced zero complete days). Padding 2 days on
        # each side (not 1) also covers align_obs_window's +/-1-day shifted lookups
        # at the edges of the requested window -- a shift=+1 lookup for obs_day=
        # chunk_end needs local day chunk_end+1 to be complete, which itself needs
        # ERA5-Land UTC data through chunk_end+2.
        #
        # _era5_download_lock serializes ONLY this call across concurrent
        # build_training_set.py processes (CDS's own per-account concurrency
        # cap -- see the lock's own docstring); the acquired slot's
        # account_index is threaded into download_era5 so it actually
        # downloads under that slot's CDS credential.
        with _era5_download_lock() as account_index:
            nc_path = era5.download_era5(
                bbox, chunk_start - timedelta(days=2), chunk_end + timedelta(days=2),
                dataset="reanalysis-era5-land", variables=_TRAINING_ERA5_VARIABLES,
                account_index=account_index,
            )
        try:
            hourly = era5.extract_era5_means(nc_path, geojson_obj)
        finally:
            os.unlink(nc_path)

        daily = heat_calcs.aggregate_hourly_to_daily(hourly, tz_map)
        for row in daily:
            daily_by_station.setdefault(row["name"], {})[row["date"]] = {
                "tmax": row["day_t2m_max"], "tmin": row["day_t2m_min"],
            }

        for name, by_date in _daily_mean_specific_humidity(hourly, tz_map).items():
            humidity_by_station.setdefault(name, {}).update(by_date)

        for name, by_date in _daily_mean_nighttime_wind(hourly, tz_map).items():
            nighttime_wind_by_station.setdefault(name, {}).update(by_date)

    return daily_by_station, humidity_by_station, nighttime_wind_by_station


def _daily_mean_specific_humidity(hourly_rows: list[dict], tz_map: dict[str, str]) -> dict[str, dict[str, float]]:
    """Daily-mean specific humidity per station, computed from hourly d2m/sp
    via the existing heat_calcs.calc_sh -- the plan's feature 13
    (grid_specific_humidity_kgkg), a climate-regime descriptor for the
    model's transfer argument (section 4.1). Buckets by the same 06:00-
    local-shift day boundary heat_calcs.aggregate_hourly_to_daily uses, but
    computed as a separate pass here since that function's return shape
    doesn't carry d2m/sp through."""
    from zoneinfo import ZoneInfo

    buckets: dict[tuple[str, str], list[float]] = {}
    for row in hourly_rows:
        name = row["name"]
        tz = ZoneInfo(tz_map.get(name, "UTC"))
        dt_local = heat_calcs._parse_utc_dt(row["datetime"]).astimezone(tz)
        local_day = (dt_local - timedelta(hours=6)).date().isoformat()
        d2m, sp = row.get("d2m"), row.get("sp")
        if d2m is None or sp is None:
            continue
        try:
            q = heat_calcs.calc_sh(d2m, sp)
        except ZeroDivisionError:
            continue
        buckets.setdefault((name, local_day), []).append(q)

    result: dict[str, dict[str, float]] = {}
    for (name, local_day), vals in buckets.items():
        result.setdefault(name, {})[local_day] = sum(vals) / len(vals)
    return result


# 18:00-05:59 local -- matches heat_calcs.aggregate_hourly_to_daily's own
# nighttime window exactly (6 evening hours + 6 early-morning hours).
_NIGHTTIME_HOURS_EXPECTED = 12


def _daily_mean_nighttime_wind(hourly_rows: list[dict], tz_map: dict[str, str]) -> dict[str, dict[str, float]]:
    """Nighttime-mean (18:00-05:59 local) wind speed per station-day -- the
    ghcn_training.nighttime_wind_ms covariate (docs/plan-2026-07-19-era5-
    wind-radiation-integration.md sections 2 and 4). Wind mixing is the
    best-established physical modulator of urban-heat-island intensity
    (Oke 1982: UHI maximizes on clear, calm NIGHTS specifically) -- a
    literature check during this plan's drafting also confirmed the
    existing grid_diurnal_range_c covariate is NOT a wind proxy (DTR tracks
    clear-sky radiative cooling, a distinct mechanism), so this is
    deliberately its own nighttime-only mean, not a 24h daily mean and not a
    derivation from grid_diurnal_range_c.

    Computed as a separate pass for the same reason
    _daily_mean_specific_humidity is (immediately above): aggregate_hourly_
    to_daily uses wind_ms only internally for its own UTCI/WBGT calculation
    and does not expose a nighttime-wind aggregate in its return shape.
    Buckets by the same 06:00-local-shift day boundary and 18:00-05:59
    nighttime window heat_calcs.aggregate_hourly_to_daily uses, so a
    station-day's nighttime_wind_ms lines up with the same local day as its
    grid_tmax_c/grid_tmin_c/grid_specific_humidity_kgkg.

    A row missing wind_ms, or carrying a non-finite one (production-shaped
    hourly input, any hour CDS happened not to return u10/v10 for, or a
    NaN from era5.extract_era5_means's masked-cell rescue not finding a
    valid substitute cell within range -- see _nearest_valid_cell's own
    docstring), is excluded rather than imputed or allowed to poison the
    mean -- upsert_ghcn_training_rows leaves the column NULL rather than
    defaulting it, and a NULL here must propagate as "no data" all the way
    through, never silently become a 0.0 OR a NaN written to the DB as a
    literal float value (found via code review 2026-07-19: a bare
    `is None` check does not catch NaN, and `sum()` over a NaN-containing
    list silently produces NaN, which is not None and so was not caught by
    upsert_ghcn_training_rows' None-only default substitution either).

    Also requires the FULL 18:00-05:59 nighttime window (12 distinct local
    hours) to be present with a valid reading, matching aggregate_hourly_
    to_daily's own all-or-nothing completeness rule (there: all 24 local
    hours; here: all 12 nighttime hours) -- a station-day with only a
    handful of nighttime hours available would otherwise produce a real but
    much noisier mean than its column name implies, silently weaker than
    the completeness guarantee its sibling covariates
    (grid_tmax_c/grid_tmin_c/grid_specific_humidity_kgkg) already have on
    the same row.

    KEEP IN SYNC with crisisready/heat-risk-data-api's src/heat_calcs.py
    daily_mean_nighttime_wind (this function's own canonical twin, promoted
    out of an earlier version of this file into heat_calcs.py so training
    and serving share one implementation -- see that repo's devlog,
    "Sub-phase 5a"). This copy is necessarily re-inlined here rather than
    imported, since this package cannot depend on the private repo's src/ --
    but that means it does NOT automatically pick up future fixes to the
    canonical version. The `wind_ms_is_fallback` exclusion below was added
    to heat_calcs.py after this promotion (PR #240, private repo) to skip
    hours open_meteo._hourly_to_rows fabricated rather than genuinely
    observed -- currently inert here (this script only ever passes ERA5-
    sourced hourly rows, which never set that flag), but load-bearing the
    moment this file (or a descendant) ever aggregates Open-Meteo-sourced
    hourly rows the way validate_lagfill_downscaling.py/
    validate_forecast_downscaling.py already do.
    """
    from zoneinfo import ZoneInfo

    buckets: dict[tuple[str, str], dict] = {}
    for row in hourly_rows:
        name = row["name"]
        tz = ZoneInfo(tz_map.get(name, "UTC"))
        dt_local = heat_calcs._parse_utc_dt(row["datetime"]).astimezone(tz)
        local_hour = dt_local.hour
        if 6 <= local_hour <= 17:
            continue  # daytime hour -- this covariate is nighttime-only
        local_day = (dt_local - timedelta(hours=6)).date().isoformat()
        bucket = buckets.setdefault((name, local_day), {"hours": set(), "vals": []})
        bucket["hours"].add(local_hour)
        wind_ms = row.get("wind_ms")
        if wind_ms is not None and math.isfinite(wind_ms) and not row.get("wind_ms_is_fallback", False):
            bucket["vals"].append(wind_ms)

    result: dict[str, dict[str, float]] = {}
    for (name, local_day), bucket in buckets.items():
        if len(bucket["hours"]) < _NIGHTTIME_HOURS_EXPECTED or len(bucket["vals"]) < _NIGHTTIME_HOURS_EXPECTED:
            continue
        result.setdefault(name, {})[local_day] = sum(bucket["vals"]) / len(bucket["vals"])
    return result


def _canopy_resume_project_id(batch_label: str, stations: list[dict]) -> str:
    """A project_id for vulnerability.extract_canopy's resume-state
    mechanism that changes whenever the actual station set changes --
    keying purely by country/batch_label (as an earlier version of this
    script did) would let a small test run's persisted resume_tile_index
    (e.g. --max-stations-per-country 5) get silently reused, and silently
    misapplied, by a later full run for the same country: the two runs'
    quadkey lists differ in size/order, so tile indices from one don't
    correspond to the same physical tiles in the other, and stations could
    end up with a skipped-as-'already visited' canopy tile that was never
    actually read. Folding in the station count and a stable hash of the
    sorted station IDs means a different station set gets its own resume
    state instead of colliding with an unrelated one."""
    ids_key = ",".join(sorted(s["station_id"] for s in stations))
    ids_hash = hashlib.sha1(ids_key.encode()).hexdigest()[:10]
    return f"ghcn_training_{batch_label}_{len(stations)}_{ids_hash}"


def _station_buffer_area_km2(feature: dict) -> float:
    """Real area (km^2) of one station's tiny buffer square
    (stations_to_geojson), via the same EPSG:4326->EPSG:3857-around-centroid
    reprojection every other area computation in this codebase uses
    (landscan._polygon_and_pixel_area_km2, manager._polygon_geometry_stats) --
    needed because a fixed degree-sized square covers different real area
    depending on latitude (longitude compression), so this cannot be one
    precomputed constant shared across stations."""
    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape

    poly = shapely_shape(feature["geometry"])
    return float(gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("EPSG:3857").area.iloc[0] / 1e6)


def _population_density_by_station(geojson_obj: dict, landscan_bucket: str, landscan_key: str) -> dict[str, float]:
    """LandScan population density (people/km^2) at each station's buffer
    square -- a single global-raster point read, the same cheap extractor
    shape as canopy/WorldCover/GHSL/DEM (unlike LST, below)."""
    pop_rows = landscan.extract_population(geojson_obj, landscan_bucket, landscan_key)
    areas_by_name = {f["properties"]["name"]: _station_buffer_area_km2(f) for f in geojson_obj["features"]}
    result = {}
    for r in pop_rows:
        name = r["name"]
        area_km2 = areas_by_name.get(name)
        result[name] = (r["population"] / area_km2) if area_km2 else None
    return result


# Bounds the LST anomaly baseline population to roughly a real project's own
# geographic extent (a single metro area), not this batch's full country-wide
# station spread -- fixes a real train/serve mismatch found via a
# 2026-07-18 red-team review: production computes a polygon's LST anomaly
# relative to only that project's own polygons (typically one city/metro
# area), but a training batch spans an entire country's stratified station
# sample. Without this, each station's anomaly baseline would be a national
# average the model never sees at serving time. 75km comfortably covers any
# single real project's extent (every crisisready-1 pilot project is a
# single metro area) while remaining a tiny fraction of a country's own
# extent, so the existing cross-country/cross-region CV diversity goal is
# unaffected -- see lst.compute_warm_season_composite's name_coords/
# reference_radius_km docstring for the full mechanism.
_LST_REFERENCE_RADIUS_KM = 75.0


def _lst_warm_season_anomaly_by_station(
    geojson_obj: dict, bbox: str, batch_label: str, stations: list[dict],
) -> dict[str, float]:
    """
    Real warm-season LST anomaly per station, via the SAME Landsat
    scene-search + composite pipeline a real project uses
    (manager.backfill_lst's composite_prep path + lst.compute_warm_season_
    composite) -- not a cheap point read like canopy/WorldCover/GHSL/DEM.
    This is a genuinely heavy, real-cost operation (Landsat scene search
    across three ~3-month warm-season windows, requester-pays S3 reads,
    potentially hundreds of scene dates for a large country -- see
    lst.extract_lst_watermarked's own docstring, "Arizona measured at
    245-312" scene dates for ONE state); accepted deliberately (2026-07-18)
    because lst_warm_season_anomaly_c is the plan's single most predictive
    covariate (plan section 4.1) and training without it would understate
    the model's real potential.

    Uses a dedicated per-batch LST table (not a project's own table -- this
    is an offline build, not a project), but the anomaly baseline itself is
    scoped to _LST_REFERENCE_RADIUS_KM around each station (not the whole
    batch) -- see _LST_REFERENCE_RADIUS_KM's own comment and
    lst.compute_warm_season_composite's docstring for why a country-wide
    baseline would not match what a real project's anomaly measures.

    No DB-persisted resume state (unlike backfill_lst) -- this is a single
    script invocation, not a Lambda; extract_lst_watermarked's own upserts
    are idempotent, so a re-run after a crash just re-pays some already-
    processed requester-pays reads rather than risking any duplication.
    """
    country_key = re.sub(r"[^a-z0-9_]", "", batch_label.lower()) or "batch"
    table_name = f"ghcn_training_lst_{country_key}"
    lst.create_lst_table(table_name)

    seasons = lst.derive_warm_season_window(bbox)["seasons"]
    for window_i, (w_start, w_end) in enumerate(seasons, start=1):
        resume_after = None
        completed = False
        window_t0 = time.monotonic()
        batch_i = 0
        while not completed:
            batch_i += 1
            batch_t0 = time.monotonic()
            date_rows, completed = lst.extract_lst_watermarked(
                geojson_obj, bbox, w_start, w_end, resume_after=resume_after,
            )
            for scene_date, rows in date_rows:
                if rows:
                    lst.upsert_lst_rows(table_name, rows)
                resume_after = scene_date
            # Real elapsed time per batch -- the ONLY way to see LST's actual
            # rate (scene-dates/second) as it runs, rather than inferring
            # progress from the "time budget exceeded" WARNING alone, which
            # never appears on the batch that finally completes a window.
            logger.info(
                "[%s] LST window %d/%d batch %d: %d scene date(s) in %.1fs (completed=%s)",
                batch_label, window_i, len(seasons), batch_i, len(date_rows),
                time.monotonic() - batch_t0, completed,
            )
        logger.info(
            "[%s] LST window %d/%d finished in %.1fs total",
            batch_label, window_i, len(seasons), time.monotonic() - window_t0,
        )

    name_coords = {s["station_id"]: (s["lat"], s["lon"]) for s in stations}
    composite = lst.compute_warm_season_composite(
        table_name, bbox, name_coords=name_coords, reference_radius_km=_LST_REFERENCE_RADIUS_KM,
    )
    return {name: c["lst_warm_season_anomaly_c"] for name, c in composite.items()}


def snapshot_covariates_for_stations(
    stations: list[dict], vuln_bucket: str, batch_label: str, landscan_bucket: str, landscan_key: str,
    bbox: str | None = None, geojson_obj: dict | None = None,
) -> dict[str, dict]:
    """
    Snapshot the same static covariates the downscaling model uses at
    inference (plan section 4.1, features 1-9) at each station's location,
    reusing the existing per-polygon extractors unmodified -- a station's
    tiny buffer "polygon" (stations_to_geojson) is indistinguishable from a
    real project polygon to these functions.

    bbox/geojson_obj may be passed in precomputed (see
    fetch_era5_land_for_stations' docstring); computed from `stations` if
    omitted, for standalone use.
    """
    bbox = bbox if bbox is not None else stations_bbox(stations)
    geojson_obj = geojson_obj if geojson_obj is not None else stations_to_geojson(stations)

    canopy = vulnerability.extract_canopy(
        geojson_obj, bbox, vuln_bucket, project_id=_canopy_resume_project_id(batch_label, stations),
    )
    # FORWARD-PORT FIX (2026-07-27, Phase 1 section 5.3): extract_worldcover
    # gained chunked/resumable + resolution-scaled behavior (2026-07-22
    # audit, main only) after this branch diverged -- its return shape is
    # now {"status", "resume_index", "results"}, not a plain
    # {polygon: {...}} dict. Unwrap ["results"] -- this call always runs
    # to completion in one shot here (no vuln_table/resume_index/deadline
    # passed), so "results" always covers every polygon.
    worldcover = vulnerability.extract_worldcover(geojson_obj, bbox, vuln_bucket)["results"]
    ghsl = vulnerability.extract_ghsl_smod(geojson_obj, bbox, vuln_bucket)
    # Elevation (dem.extract_elevation) needs the ERA5-Land grid resolution for
    # elevation_rel_to_gridcell_m -- read from era5's own dataset->resolution
    # map (the single source of truth fetch_era5_land_for_stations' download_era5
    # call is keyed against) rather than a second hardcoded literal that could
    # silently drift from it.
    # FORWARD-PORT FIX (2026-07-27, Phase 1 section 5.3): extract_elevation
    # gained the identical chunked/resumable treatment (main only) and
    # DROPPED its `bbox` parameter entirely (tiles are now discovered per
    # polygon, same reasoning as extract_worldcover's now-unused bbox) --
    # this branch's call passed bbox positionally where grid_resolution_deg
    # now lives; fixed to the current 2-positional-arg signature, and
    # ["results"] unwrapped for the same reason as worldcover above.
    elevation = dem.extract_elevation(
        geojson_obj, grid_resolution_deg=era5._DATASET_RESOLUTION["reanalysis-era5-land"],
    )["results"]
    pop_density = _population_density_by_station(geojson_obj, landscan_bucket, landscan_key)
    lst_anomaly = _lst_warm_season_anomaly_by_station(geojson_obj, bbox, batch_label, stations)

    result = {}
    for s in stations:
        sid = s["station_id"]
        c, wc, g, e = canopy.get(sid, {}), worldcover.get(sid, {}), ghsl.get(sid, {}), elevation.get(sid, {})
        result[sid] = {
            "canopy_height_mean_m": c.get("canopy_height_mean_m"),
            "canopy_frac_over_3m": c.get("canopy_frac_over_3m"),
            "wc_built_frac": wc.get("wc_built_frac"),
            "wc_tree_frac": wc.get("wc_tree_frac"),
            "wc_water_frac": wc.get("wc_water_frac"),
            "ghsl_urban_fraction": g.get("ghsl_urban_fraction"),
            "elevation_rel_to_gridcell_m": e.get("elevation_rel_to_gridcell_m"),
            "elevation_mean_m": e.get("elevation_mean_m"),
            "slope_deg": e.get("slope_deg"),
            "aspect_deg": e.get("aspect_deg"),
            "lst_warm_season_anomaly_c": lst_anomaly.get(sid),
            "pop_density_per_km2": pop_density.get(sid),
        }
    return result


def build_rows_for_country(
    country: str, stations: list[dict], start_date: date, end_date: date, vuln_bucket: str,
    landscan_bucket: str, landscan_key: str, *,
    ghcn_max_workers: int = 8, ghcn_checkpoint_path: str | None = None,
) -> list[dict]:
    """Assemble ghcn_training rows for every station in one country/batch:
    fetch GHCN + ERA5-Land + covariates once per batch, align each
    station's obs window empirically, compute the regression target
    (delta_tmax_c/delta_tmin_c), and label region/climate_zone.

    ghcn_max_workers/ghcn_checkpoint_path pass straight through to
    ghcn.fetch_ghcn_daily_bulk_concurrent (adaptive-concurrency GHCN fetch,
    2026-08-03 -- see that function's own docstring)."""
    # Computed once here and shared with both calls below -- they'd otherwise
    # each independently rebuild the identical bbox/GeoJSON from this same
    # station list.
    bbox = stations_bbox(stations)
    geojson_obj = stations_to_geojson(stations)

    # Run all three concurrently, not sequentially -- ERA5, covariates/LST,
    # and the per-station GHCN fetch are completely independent of each
    # other (none needs another's output), but running them one after
    # another means a country spends its ENTIRE ERA5 wait (which,
    # serialized with every other country through _era5_download_lock, is a
    # real, large chunk of wall-clock time -- confirmed live 2026-07-18 via
    # py-spy: 4 of 5 concurrent country processes sat 100% idle blocked on
    # that lock, doing no covariate/LST work they could have started
    # immediately) unable to make ANY progress on the CPU-bound covariate/
    # LST work or the (equally independent) GHCN fetch. A 3-worker thread
    # pool lets one thread sit in ERA5's blocking network wait (which
    # releases the GIL) while another does real covariate/LST work and a
    # third runs the GHCN fetch's OWN internal thread pool (fetch_all,
    # bounded by ghcn_max_workers) -- cutting a country's own wall-clock
    # time toward max(era5_wait, covariate_wait, ghcn_wait) instead of their
    # sum. Added 2026-08-03 alongside fetch_ghcn_daily_bulk_concurrent --
    # the GHCN fetch used to be a separate, fully sequential (one station at
    # a time) loop after this block, which for a large pooled station set
    # (679 US+Mexicali stations) meant real wall-clock time nothing else in
    # this function could overlap with.
    def _timed_era5():
        t0 = time.monotonic()
        logger.info("[%s] ERA5 fetch starting", country)
        result = fetch_era5_land_for_stations(stations, start_date, end_date, bbox=bbox, geojson_obj=geojson_obj)
        logger.info("[%s] ERA5 fetch finished in %.1fs", country, time.monotonic() - t0)
        return result

    def _timed_covariates():
        t0 = time.monotonic()
        logger.info("[%s] covariate/LST snapshot starting", country)
        result = snapshot_covariates_for_stations(
            stations, vuln_bucket, batch_label=country,
            landscan_bucket=landscan_bucket, landscan_key=landscan_key, bbox=bbox, geojson_obj=geojson_obj,
        )
        logger.info("[%s] covariate/LST snapshot finished in %.1fs", country, time.monotonic() - t0)
        return result

    def _timed_ghcn():
        t0 = time.monotonic()
        logger.info("[%s] GHCN daily fetch starting (%d station(s), max_workers=%d)",
                    country, len(stations), ghcn_max_workers)
        result = ghcn.fetch_ghcn_daily_bulk_concurrent(
            [s["station_id"] for s in stations], start_date, end_date,
            max_workers=ghcn_max_workers, checkpoint_path=ghcn_checkpoint_path,
        )
        logger.info("[%s] GHCN daily fetch finished in %.1fs (%d/%d station(s) returned rows)",
                    country, time.monotonic() - t0, len(result), len(stations))
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        era5_future = executor.submit(_timed_era5)
        covariates_future = executor.submit(_timed_covariates)
        ghcn_future = executor.submit(_timed_ghcn)
        grid_by_station, humidity_by_station, nighttime_wind_by_station = era5_future.result()
        covariates_by_station = covariates_future.result()
        ghcn_by_station = ghcn_future.result()

    rows = []
    for s in stations:
        sid = s["station_id"]
        station_series = ghcn_by_station.get(sid, [])
        if not station_series:
            continue

        grid_series = grid_by_station.get(sid, {})
        grid_tmax_by_date = {d: v["tmax"] for d, v in grid_series.items()}
        shift = ghcn.align_obs_window(station_series, grid_tmax_by_date)

        covariates = covariates_by_station.get(sid, {})
        region = ghcn.region_from_station_id(sid)
        climate_zone = ghcn.koppen_climate_zone(s["lat"], s["lon"])
        humidity_series = humidity_by_station.get(sid, {})
        nighttime_wind_series = nighttime_wind_by_station.get(sid, {})

        for obs in station_series:
            try:
                obs_day = date.fromisoformat(obs["date"])
            except ValueError:
                # NOAA's Access Data Service is expected to always return a
                # bare YYYY-MM-DD DATE string; guard anyway (mirroring
                # ghcn.align_obs_window's own try/except around the same
                # parse) so one malformed record drops just that row
                # instead of aborting the whole country's build.
                continue
            shifted_day = (obs_day + timedelta(days=shift)).isoformat()
            grid_vals = grid_series.get(shifted_day)
            if grid_vals is None:
                continue
            grid_tmax, grid_tmin = grid_vals["tmax"], grid_vals["tmin"]
            rows.append({
                "station_id": sid,
                "date": obs["date"],
                "lon": s["lon"], "lat": s["lat"], "elevation_m": s["elevation_m"],
                "region": region, "climate_zone": climate_zone,
                "station_tmax_c": obs["station_tmax_c"], "station_tmin_c": obs["station_tmin_c"],
                "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin,
                "delta_tmax_c": obs["station_tmax_c"] - grid_tmax,
                "delta_tmin_c": obs["station_tmin_c"] - grid_tmin,
                "grid_specific_humidity_kgkg": humidity_series.get(shifted_day),
                "nighttime_wind_ms": nighttime_wind_series.get(shifted_day),
                "obs_window_shift_days": shift,
                **covariates,
            })

    return rows


def _select_geographically_compact(stations: list[dict], max_count: int) -> list[dict]:
    """Select `max_count` stations clustered around this country's own
    station centroid, instead of a naive list-order truncation.

    NOAA's inventory order is not geographically meaningful -- a plain
    `stations[:max_count]` can (and, confirmed live 2026-07-18, does) select
    stations spread coast-to-coast for a large country, which does not
    shrink stations_bbox() at all: the ERA5-Land request stays nationwide
    regardless of the cap, and CDS itself rejected exactly this shape of
    request ("cost limits exceeded... reduce your selection") for a
    single-month, 3-station US pull. A caller reaching for
    --max-stations-per-country wants a genuinely SMALL bbox (a fast test
    run, or a deliberately bounded training sample); this makes the cap
    actually deliver that instead of silently failing to bound anything."""
    if max_count >= len(stations):
        return stations
    lons = [s["lon"] for s in stations]
    lats = [s["lat"] for s in stations]
    centroid_lon, centroid_lat = sum(lons) / len(lons), sum(lats) / len(lats)
    by_distance = sorted(
        stations,
        key=lambda s: (s["lon"] - centroid_lon) ** 2 + (s["lat"] - centroid_lat) ** 2,
    )
    return by_distance[:max_count]


def _select_geographically_stratified(
    stations: list[dict], max_count: int, max_extent_deg: float | None = None,
) -> list[dict]:
    """
    Select up to `max_count` stations SPREAD across the country's full
    extent (grid-binned round robin), the opposite goal of
    _select_geographically_compact: a real production training build wants
    climate/land-cover DIVERSITY, not a tight cluster.

    Capping station COUNT alone does not bound ERA5-Land download volume --
    that scales with the station list's geographic EXTENT (bbox), not how
    many stations sit inside it (confirmed live 2026-07-18: a full-CONUS,
    one-month ERA5-Land pull was ~3.5 GB regardless of how many of the
    78,566 US stations were actually used). For a country whose stations
    span more than max_extent_deg in either dimension, this also bounds the
    CANDIDATE set to a max_extent_deg square around the country's own
    station centroid before stratifying within it -- trading full national
    coverage for a training build whose ERA5-Land volume stays reasonable.
    A smaller country (France, Germany, UK) is typically already under
    max_extent_deg and is left untouched.
    """
    candidates = stations
    if max_extent_deg is not None and len(stations) > 1:
        lons = [s["lon"] for s in stations]
        lats = [s["lat"] for s in stations]
        centroid_lon, centroid_lat = sum(lons) / len(lons), sum(lats) / len(lats)
        if (max(lons) - min(lons)) > max_extent_deg or (max(lats) - min(lats)) > max_extent_deg:
            half = max_extent_deg / 2
            candidates = [
                s for s in stations
                if abs(s["lon"] - centroid_lon) <= half and abs(s["lat"] - centroid_lat) <= half
            ]

    if max_count >= len(candidates):
        return candidates

    lons = [s["lon"] for s in candidates]
    lats = [s["lat"] for s in candidates]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_span = (max_lon - min_lon) or 1.0
    lat_span = (max_lat - min_lat) or 1.0

    # +1, not a bare int(sqrt) truncation -- int(2**0.5) == 1 would collapse
    # small caps (e.g. max_count=2 or 4) to a single 1x1 "grid" with no real
    # spatial stratification at all (confirmed by a failing test: max_count=2
    # returned two stations 0.1 degrees apart instead of spanning the extent).
    grid_size = max(2, int(max_count ** 0.5) + 1)
    cells: dict[tuple[int, int], list[dict]] = {}
    for s in candidates:
        cx = min(grid_size - 1, int((s["lon"] - min_lon) / lon_span * grid_size))
        cy = min(grid_size - 1, int((s["lat"] - min_lat) / lat_span * grid_size))
        cells.setdefault((cx, cy), []).append(s)

    cell_lists = [c for c in cells.values() if c]
    selected: list[dict] = []
    i = 0
    while len(selected) < max_count and cell_lists:
        cell = cell_lists[i % len(cell_lists)]
        selected.append(cell.pop())
        if not cell:
            cell_lists.pop(i % len(cell_lists))
        else:
            i += 1
    return selected


def _group_stations_by_country(stations: list[dict]) -> dict[str, list[dict]]:
    """Group an already-fetched station list by FIPS country prefix, so
    main() can call list_ghcn_stations once for every requested country
    instead of once per country (that function downloads/parses NOAA's
    entire global station inventory regardless of filter size, so calling
    it per-country in an N-country build redundantly repeats that same
    multi-MB fetch+parse N times)."""
    result: dict[str, list[dict]] = {}
    for s in stations:
        result.setdefault(ghcn.region_from_station_id(s["station_id"]), []).append(s)
    return result


def _group_stations_by_zone(stations: list[dict]) -> dict[str, list[dict]]:
    """Group stations by Koppen climate zone (ghcn.koppen_climate_zone) --
    used to sub-batch build_rows_for_country's ERA5-Land fetch so each
    request's bounding box stays geographically tight, instead of one
    country-wide bbox spanning every zone a curated station set happens to
    touch. Confirmed live 2026-08-03 smoke-testing the pooled US+Mexicali
    write: a country-wide bbox for a geographically dispersed station set
    (the --station-ids-file path has no --max-bbox-extent-deg bounding at
    all, by design -- it uses the curated list as-is) exceeded CDS's own
    per-request cost limit ("cost limits exceeded... reduce your
    selection"). Climate zones are naturally geographically compact by
    construction and are already the unit the curated station lists
    themselves were assembled by, so batching by zone reduces every
    request's bbox versus one country-wide batch -- but zone alone turned
    out NOT to be sufficient on its own (a single Koppen zone like BWh can
    still span a wide real area): _chunk_stations_by_extent runs on top of
    this, per zone, for the actual bbox-size guarantee. koppen_climate_zone
    is a local raster lookup (no network call), so recomputing it here even
    though build_rows_for_country's own per-station loop computes it again
    per row is cheap and not worth threading through as a precomputed
    value."""
    result: dict[str, list[dict]] = {}
    for s in stations:
        zone = ghcn.koppen_climate_zone(s["lat"], s["lon"])
        result.setdefault(zone, []).append(s)
    return result


# Default max raw station spread (in degrees, before stations_bbox's own
# _BBOX_PAD_DEG padding) per ERA5-Land request chunk. Deliberately smaller
# than _BBOX_PAD_DEG*2 -- confirmed live 2026-08-03 that _group_stations_by_
# zone alone is NOT sufficient: 2 same-zone, same-50km-cluster Mexicali
# stations (raw spread ~1.4deg lat x 0.33deg lon) still tripped CDS's
# "cost limits exceeded" rejection once padded to a 2.37deg x 1.33deg bbox
# -- padding, not station spread, was doing most of that damage (0.5deg pad
# on each side alone contributes 1.0deg of the final extent in any
# dimension). This constant bounds the RAW station spread going into
# stations_bbox, not the final padded bbox -- see _chunk_stations_by_extent.
_ERA5_MAX_CHUNK_EXTENT_DEG = 0.5


def _chunk_stations_by_extent(stations: list[dict], max_extent_deg: float) -> list[list[dict]]:
    """Partition stations into geographically-bounded groups, each spanning
    at most ~max_extent_deg of raw lon/lat spread -- unlike
    _select_geographically_stratified (which CAPS a station count, dropping
    the rest), every station is kept here; this only controls how many
    separate ERA5-Land requests they get split across. Needed alongside
    _group_stations_by_zone: a climate zone alone is not reliably small
    enough (confirmed live 2026-08-03 -- see _ERA5_MAX_CHUNK_EXTENT_DEG's
    own comment), so this bounds the actual station spread directly,
    independent of zone.

    Grid-based: stations are bucketed into max_extent_deg x max_extent_deg
    cells -- cheap, deterministic, and every group's raw spread is bounded
    by construction (a cell's own width), not by iteratively measuring and
    re-splitting an oversized group."""
    if not stations:
        return []
    cells: dict[tuple[int, int], list[dict]] = {}
    for s in stations:
        cx = int(s["lon"] // max_extent_deg)
        cy = int(s["lat"] // max_extent_deg)
        cells.setdefault((cx, cy), []).append(s)
    return list(cells.values())


# Covariates most worth eyeballing before committing to a real write: the 4
# columns ghcn.upsert_ghcn_training_rows was fixed (2026-08-03) to stop
# silently dropping from every upsert, plus the other real-extraction-cost
# covariates most likely to be silently sparse for a new station set.
# koppen_main_group_code is deliberately included even though this script
# never computes it (build_rows_for_country only computes climate_zone) --
# it will always show 0/N here, a standing reminder of that known,
# separate gap rather than a silent omission from the summary.
_COVARIATE_COMPLETENESS_COLUMNS = (
    "elevation_mean_m", "slope_deg", "aspect_deg", "koppen_main_group_code",
    "lst_warm_season_anomaly_c", "pop_density_per_km2", "nighttime_wind_ms",
    "grid_specific_humidity_kgkg",
)


def _covariate_completeness_summary(rows: list[dict]) -> str:
    """One-line %-non-null summary per covariate, for --dry-run's preview --
    lets a human eyeball data quality before committing to the actual
    write, the same role the printed gate/zone summary plays for
    publish_band_gate.py's own --dry-run."""
    if not rows:
        return "no rows"
    n = len(rows)
    parts = [
        f"{col}={sum(1 for r in rows if r.get(col) is not None)}/{n}"
        for col in _COVARIATE_COMPLETENESS_COLUMNS
    ]
    return "covariate completeness: " + ", ".join(parts)


def main() -> None:
    # This module and every extractor it calls (era5.py, lst.py, vulnerability.py,
    # ghcn.py) use the standard logging module, but nothing in the plain-script
    # execution path ever configures a handler -- Python's root logger has NONE
    # by default, so every logger.info() call (real per-request ERA5 timing, LST
    # batch progress, etc.) was silently discarded; only logger.warning()+ ever
    # appeared, via Python's "handler of last resort" (confirmed live 2026-07-18
    # investigating a stalled build: the only visible log lines were the LST
    # "time budget exceeded" WARNINGs -- every INFO-level status line, including
    # ERA5's own per-request timing, was invisibly dropped). %(asctime)s is not
    # cosmetic here -- it's the only way to get real wall-clock timestamps on
    # any of this, since print() calls throughout this script carry none.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default=None,
                         help="Comma-separated FIPS country codes (e.g. US,GM,FR,AS) -- this script "
                              "selects its own stations per country. Exactly one of --countries or "
                              "--station-ids-file is required.")
    parser.add_argument("--station-ids-file", default=None,
                         help="JSON file with a top-level \"station_ids\" list of bare GHCN station-id "
                              "strings -- an explicit, pre-curated station set to build/write (e.g. one "
                              "assembled by a downscaling-confidence gap-fill analysis), bypassing this "
                              "script's own per-country selection entirely. Countries are derived from "
                              "the IDs' own FIPS prefixes (ghcn.region_from_station_id) -- multiple "
                              "countries in one file is fine and still batches ERA5/covariate fetches "
                              "per country exactly as --countries does. Station metadata (lon/lat/"
                              "elevation_m/name) is looked up fresh from ghcn.list_ghcn_stations for the "
                              "derived countries, not read from this file -- the file is only "
                              "the *which stations*, never a second source of truth for their location "
                              "data. Exactly one of --countries or --station-ids-file is required. "
                              "Not compatible with --max-stations-per-country/--max-bbox-extent-deg "
                              "(the file's station list is used in full, not re-selected/re-capped).")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--max-stations-per-country", type=int, default=None,
                         help="Cap stations per country -- for small test runs; omit for a real build.")
    parser.add_argument("--selection-mode", choices=("compact", "stratified"), default="compact",
                         help="'compact' (default): cluster the cap tightly -- small, fast test runs. "
                              "'stratified': spread the cap across the country for climate/land-cover "
                              "diversity -- the real training-build mode.")
    parser.add_argument("--max-bbox-extent-deg", type=float, default=None,
                         help="Only used with --selection-mode stratified. Bounds a large country's "
                              "candidate stations to this many degrees around its own centroid before "
                              "stratifying -- ERA5-Land download volume scales with geographic EXTENT, "
                              "not station count, so a big country (US, Australia) needs this to keep a "
                              "real build's CDS request size reasonable (confirmed live 2026-07-18: a "
                              "full-CONUS, one-month pull was ~3.5 GB). Omit for a country small enough "
                              "that its natural extent is already fine (France, Germany, UK).")
    parser.add_argument("--min-last-year", type=int, default=None,
                         help="Skip stations whose GHCN inventory record stopped before this year "
                              "(ghcn.active_station_ids) -- defaults to --end-date's year, so a station "
                              "that stopped reporting before the requested window can't contribute "
                              "anyway. Confirmed live 2026-07-18: 3 stations with no recent data wasted "
                              "a full ERA5-Land pull before returning 0 rows.")
    parser.add_argument("--profile", default=None, help="Named AWS profile (omit on EC2 with an attached IAM role).")
    parser.add_argument("--ghcn-max-workers", type=int, default=8,
                         help="Concurrent NOAA Access Data Service requests per country batch "
                              "(ghcn.fetch_ghcn_daily_bulk_concurrent's adaptive-concurrency pool -- see "
                              "that function's own docstring). Default 8. Raise this for a large pooled "
                              "run on a big enough instance; the throttle backs off on its own if NOAA "
                              "starts rejecting/slowing requests, so a too-high value degrades rather "
                              "than fails outright.")
    parser.add_argument("--ghcn-checkpoint-dir", default=None,
                         help="Directory for a per-country GHCN-fetch checkpoint file "
                              "(ghcn_daily_{country}.jsonl) -- makes a large pooled run crash-resumable: "
                              "a killed process restarts only the stations not yet fetched, rather than "
                              "re-paying every NOAA round-trip. Omit for no cross-run resume (fine for a "
                              "small/test run).")
    parser.add_argument("--era5-max-chunk-extent-deg", type=float, default=_ERA5_MAX_CHUNK_EXTENT_DEG,
                         help="Max raw lon/lat spread (degrees, before stations_bbox's own padding) of "
                              "stations in one ERA5-Land request -- each zone-batch is further split into "
                              "chunks bounded by this (see _chunk_stations_by_extent). Lower this if CDS "
                              "still rejects a request as too large; a single climate zone is not always "
                              "small enough on its own (confirmed live 2026-08-03).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run the full fetch/covariate/row-build pipeline exactly as a real build "
                              "would, but skip the final ghcn.upsert_ghcn_training_rows call -- print a "
                              "per-country row count and covariate-completeness summary instead of "
                              "writing anything. Costs the same real CDS/S3/network time as a real run "
                              "(there is no cheap way to preview rows without actually fetching/"
                              "computing them) -- this only gates the DB write itself, matching "
                              "publish_band_gate.py's own dry-run/human-decision idiom for the one step "
                              "that actually commits something.")
    parser.add_argument("--rows-out", default=None,
                         help="Write every real computed row (across all countries/zones/chunks) to this "
                              "JSON path -- {\"rows\": [...], \"row_count\": N, \"countries\": [...], "
                              "\"start_date\", \"end_date\"}. Independent of --dry-run/--dry-run's own DB "
                              "write gate: this only controls whether the real fetch/covariate compute this "
                              "run already paid for gets persisted anywhere besides the DB, so a --dry-run "
                              "(no DB write) run's real CDS/S3/NOAA cost isn't thrown away with nothing to "
                              "show for it -- the same gap validate_lagfill_downscaling.py's own "
                              "--phase dump-rows/--rows-in split already solved for reading FROM "
                              "ghcn_training; this is the equivalent for rows computed but not yet written "
                              "TO it. Omit for the previous behavior (rows exist only in-process memory).")
    args = parser.parse_args()

    if bool(args.countries) == bool(args.station_ids_file):
        parser.error("exactly one of --countries or --station-ids-file is required")
    if args.station_ids_file and (args.max_stations_per_country is not None or args.max_bbox_extent_deg is not None):
        parser.error("--max-stations-per-country/--max-bbox-extent-deg are not compatible with "
                      "--station-ids-file (the file's station list is used in full)")

    if args.profile:
        # Unconditional override, not setdefault -- matches scripts/deploy.py's
        # own convention (env = {**os.environ, "AWS_PROFILE": profile}) for the
        # same reason CLAUDE.md flags there: a stale AWS_PROFILE already
        # exported in the shell would otherwise silently win over an explicit
        # --profile flag, pointing every AWS call at the wrong account.
        os.environ["AWS_PROFILE"] = args.profile

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    requested_ids = None
    if args.station_ids_file:
        with open(args.station_ids_file) as f:
            requested_ids = set(json.load(f)["station_ids"])
        countries = sorted({ghcn.region_from_station_id(sid) for sid in requested_ids})
        print(f"--station-ids-file: {len(requested_ids)} requested station(s) across {len(countries)} "
              f"countr(y/ies) ({','.join(countries)})")
    else:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    vuln_bucket = _bucket_from_credentials()
    landscan_bucket, landscan_key = _landscan_from_credentials()

    if not args.dry_run:
        # Real gap, found live 2026-08-20 staging a --dry-run --rows-out run with no
        # DB write credentials available yet: this was called unconditionally, before
        # dry_run was even checked, so --dry-run could never actually run without a
        # write-capable DB connection despite its own docstring's explicit promise
        # ("skip the final ghcn.upsert_ghcn_training_rows call... this only gates the
        # DB write itself") -- the CREATE TABLE/ALTER TABLE DDL here (idempotent, but
        # still a real write-capable connection and a real write statement) is exactly
        # the kind of DB interaction a dry run should never require. Gated the same way
        # the actual upsert already is.
        ghcn.create_ghcn_training_table()

    # One inventory fetch for every requested country, not one per country --
    # list_ghcn_stations downloads and parses NOAA's entire global station
    # file regardless of how many country codes are passed, so calling it
    # once per country in this loop would redundantly re-download/re-parse
    # the same multi-MB file N times for an N-country build.
    all_stations = ghcn.list_ghcn_stations(countries)

    if requested_ids is not None:
        found_ids = {s["station_id"] for s in all_stations}
        missing = requested_ids - found_ids
        if missing:
            print(f"WARNING: {len(missing)} requested station_id(s) not found in NOAA's inventory for "
                  f"{countries}, skipping: {sorted(missing)}")
        all_stations = [s for s in all_stations if s["station_id"] in requested_ids]

    stations_by_country = _group_stations_by_country(all_stations)

    min_last_year = args.min_last_year if args.min_last_year is not None else end_date.year
    active_ids = ghcn.active_station_ids(countries, min_last_year)

    total_rows = 0
    all_rows: list[dict] = [] if args.rows_out else None  # None (not []) when unused, so a caller
    # that forgets --rows-out can't mistake an always-empty list for "genuinely zero rows computed."
    for country in countries:
        stations = stations_by_country.get(country, [])
        before_active_filter = len(stations)
        stations = [s for s in stations if s["station_id"] in active_ids]
        if before_active_filter and before_active_filter != len(stations):
            print(f"[{country}] {before_active_filter} station(s) -> {len(stations)} active (last year >= {min_last_year})")

        if args.max_stations_per_country is not None:
            if args.selection_mode == "stratified":
                stations = _select_geographically_stratified(
                    stations, args.max_stations_per_country, max_extent_deg=args.max_bbox_extent_deg,
                )
            else:
                stations = _select_geographically_compact(stations, args.max_stations_per_country)
        if not stations:
            print(f"[{country}] no active stations found, skipping")
            continue

        # Sub-batch by climate zone, then further by raw geographic spread --
        # see _group_stations_by_zone's and _chunk_stations_by_extent's own
        # docstrings for why both are needed (a country-wide, or even a
        # single-zone, bbox can exceed CDS's per-request cost limit,
        # confirmed live 2026-08-03).
        zone_batches = _group_stations_by_zone(stations)
        for zone in sorted(zone_batches):
            extent_chunks = _chunk_stations_by_extent(zone_batches[zone], args.era5_max_chunk_extent_deg)
            for chunk_i, chunk_stations in enumerate(extent_chunks):
                batch_label = (
                    f"{country}_{zone}" if len(extent_chunks) == 1 else f"{country}_{zone}_c{chunk_i}"
                )
                ghcn_checkpoint_path = (
                    os.path.join(args.ghcn_checkpoint_dir, f"ghcn_daily_{batch_label}.jsonl")
                    if args.ghcn_checkpoint_dir else None
                )
                print(f"[{batch_label}] building training rows for {len(chunk_stations)} station(s)...")
                rows = build_rows_for_country(
                    batch_label, chunk_stations, start_date, end_date, vuln_bucket, landscan_bucket, landscan_key,
                    ghcn_max_workers=args.ghcn_max_workers, ghcn_checkpoint_path=ghcn_checkpoint_path,
                )
                if rows and not args.dry_run:
                    ghcn.upsert_ghcn_training_rows(rows)
                if all_rows is not None:
                    all_rows.extend(rows)
                    if args.rows_out:
                        # Real gap found live 2026-08-20: a multi-hour run (many
                        # zone/extent chunks, each with real CDS queue latency)
                        # only ever wrote --rows-out ONCE, at the very end of
                        # EVERY chunk across the whole country/station-ids-file
                        # list -- so a crash, kill, or bastion interruption after
                        # hours of real compute lost all of it, with nothing on
                        # disk to show for it. Flush after every chunk instead
                        # (same real cost class as the GHCN checkpoint dir's own
                        # per-station resumability) -- a partial file is always
                        # readable, and the final write still happens the same
                        # way at the end for the common no-interruption case.
                        with open(args.rows_out, "w") as f:
                            json.dump({
                                "rows": all_rows, "row_count": len(all_rows), "countries": countries,
                                "start_date": args.start_date, "end_date": args.end_date,
                                "dry_run": args.dry_run, "complete": False,
                            }, f)
                total_rows += len(rows)
                if args.dry_run:
                    print(f"[{batch_label}] --dry-run: would write {len(rows)} row(s), not uploading. "
                          f"{_covariate_completeness_summary(rows)}")
                else:
                    print(f"[{batch_label}] wrote {len(rows)} row(s)")

    if args.dry_run:
        print(f"--dry-run: not uploading. {total_rows} total ghcn_training row(s) would be written "
              f"across {len(countries)} countr(y/ies).")
    else:
        print(f"Done. {total_rows} total ghcn_training row(s) across {len(countries)} countr(y/ies).")

    if args.rows_out:
        with open(args.rows_out, "w") as f:
            json.dump({
                "rows": all_rows, "row_count": len(all_rows), "countries": countries,
                "start_date": args.start_date, "end_date": args.end_date,
                "dry_run": args.dry_run,  # honest provenance -- whether this same run ALSO wrote to
                # ghcn_training, or these rows exist only here, matters to anyone consuming the file later.
                "complete": True,  # distinguishes a genuine final write from one of the per-chunk
                # checkpoint writes above (same file, same shape, complete=False) -- a reader that
                # only wants the finished result should check this rather than assume the file's
                # mere existence means the run finished.
            }, f)
        print(f"--rows-out: wrote {len(all_rows)} row(s) to {args.rows_out}")


if __name__ == "__main__":
    main()
