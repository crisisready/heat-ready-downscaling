"""
NOAA GSOD (Global Summary of the Day) BSh training rows for countries where GHCN-Daily's TMAX
record has thinned out. Built for the Ahmedabad BSh refinement (heat-risk-data-api
research/ahmedabad-bsh-refinement/DIAGNOSIS.md): GHCN's Ahmedabad record (IN005010600) drops
from ~300 QC-clean days a year before 2011 to under 30 a year from 2016, so the model has seen
almost nothing from Indian BSh stations in the years the local offset appears to have moved.

Why GSOD and not a third-party aggregator: for the Indian stations checked (Ahmedabad
2008-2013, Rajkot 2010-2012), every GHCN TMAX value carries source flag 'S' (sourced from
GSOD), and GHCN-minus-GSOD is +0.001 C mean, 0.015 C sd -- rounding noise. GHCN simply stopped
propagating fresh GSOD rows for these stations after ~2015 while GSOD itself kept updating.
So these rows extend the same observation stream the corpus already trusts; they are not a
new source with its own bias to validate.

GSOD specifics handled here (checked against real downloads):
  - MAX/MIN are degrees Fahrenheit; missing is the literal 9999.9.
  - MAX_ATTRIBUTES '*' means the day's max was derived from the hourly/synoptic reports rather
    than an explicit max report. Kept (it is what GHCN's own S-flagged TMAX already is for
    these stations) and counted per station in the coverage report.
  - GSOD has no GHCN-style QFLAG, so no per-value QC flag is available. A value outside
    [-40, 60] C, or a day with tmax < tmin, is dropped as physically implausible.
  - The per-year "access" CSV can 404 for a station isd-history.csv lists as active; a 404 is
    a missing year, not an error.

Station IDs: a GSOD station whose WMO number maps to a GHCN-Daily station (ghcnd-stations.txt's
WMO column) reuses that GHCN ID, so its rows extend the existing station's history under one
(station_id, date) key instead of duplicating it. An unmapped station gets
f"{fips}G{usaf}" -- the real FIPS prefix, so ghcn.region_from_station_id() still puts it in its
own country's leave-region-out CV fold, and a 9-character ID that can't collide with an
11-character GHCN ID.

Grid values come from the CDS reanalysis-era5-land-timeseries dataset
(build_training_set.fetch_era5_land_for_stations_via_timeseries), the same ERA5-Land values the
gridded path produces (max |diff| 0.0001 C, measured 2026-09-17), one request per station for
the whole date range. Covariates come from build_training_set.snapshot_covariates_for_stations,
unchanged.

Output: a JSON file of ghcn_training-shaped rows (not a DB write), for train_downscaling.py's
--extra-rows-json or a maintainer upsert via ghcn.upsert_ghcn_training_rows.

Environment (read by the reused build_training_set helpers when credentials.yaml is absent):
VULNERABILITY_DATA_BUCKET, LANDSCAN_BUCKET, LANDSCAN_KEY, and ERA5_SECRET_ARN[_2,_3] for CDS.
PYTHONPATH must include heat-risk-data-api/src (ghcn, era5, and the covariate modules).

Usage:
    python3 scripts/build_gsod_bsh_rows.py --countries IN PK --start-date 2016-01-01 \\
        --end-date 2026-09-20 --out-dir /tmp/gsod_bsh_pull [--stations-only]
"""
import argparse
import csv
import io
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
# ghcn (and build_training_set, imported in main) are heat-risk-data-api/src modules, imported
# where they're used so the pure parsing/selection functions below stay testable in this repo.

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GHCND_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{usaf}{wban}.csv"
GSOD_MISSING = 9999.9
PLAUSIBLE_C = (-40.0, 60.0)
TARGET_ZONE = "BSh"
# ISD often lists one physical site under two USAF IDs (e.g. Bhavnagar as 420801 and 428380,
# ~1 km apart). Stations closer than this are treated as the same site.
COLOCATED_KM = 3.0

logger = logging.getLogger("build_gsod_bsh_rows")


def _get(url, retries=4):
    """GET with retry on transient errors. Returns None on HTTP 404 (a missing GSOD year)."""
    last_exc = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
        time.sleep(5 * (attempt + 1))
    raise last_exc


def _cached(out_dir, name, url):
    path = os.path.join(out_dir, "cache", name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    body = _get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(body if body is not None else b"")
    return body


def f_to_c(value):
    return (value - 32.0) * 5.0 / 9.0


def parse_gsod_csv(text):
    """GSOD per-station-year CSV -> list of {date, station_tmax_c, station_tmin_c, max_derived}.
    Drops a day with either value missing (9999.9), outside PLAUSIBLE_C, or with tmax < tmin."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            mx, mn = float(row["MAX"]), float(row["MIN"])
        except (KeyError, TypeError, ValueError):
            continue
        if mx == GSOD_MISSING or mn == GSOD_MISSING:
            continue
        tmax, tmin = f_to_c(mx), f_to_c(mn)
        lo, hi = PLAUSIBLE_C
        if not (lo <= tmin <= tmax <= hi):
            continue
        out.append({
            "date": row["DATE"], "station_tmax_c": round(tmax, 2), "station_tmin_c": round(tmin, 2),
            "max_derived": (row.get("MAX_ATTRIBUTES") or "").strip() == "*",
        })
    return out


def parse_isd_history(text, countries, start_year):
    """isd-history.csv rows for `countries` (FIPS) still reporting on or after start_year,
    with real coordinates."""
    stations = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["CTRY"] not in countries or not row["LAT"] or not row["LON"]:
            continue
        try:
            lat, lon = float(row["LAT"]), float(row["LON"])
            end_year = int(row["END"][:4])
        except ValueError:
            continue
        if end_year < start_year or (lat == 0.0 and lon == 0.0) or row["USAF"] == "999999":
            continue
        elev = row.get("ELEV(M)")
        try:
            elev = float(elev) if elev not in (None, "", "-999.0", "-999.9") else None
        except ValueError:
            elev = None
        stations.append({"usaf": row["USAF"], "wban": row["WBAN"], "fips": row["CTRY"],
                         "name": row["STATION NAME"], "lat": lat, "lon": lon, "elevation_m": elev})
    return stations


def parse_ghcnd_wmo_map(text, countries):
    """WMO station number -> GHCN-Daily ID, from ghcnd-stations.txt's fixed-width WMO column
    (cols 81-85), restricted to `countries`."""
    wmo_map = {}
    for line in text.splitlines():
        if len(line) < 85 or line[:2] not in countries:
            continue
        wmo = line[80:85].strip()
        if wmo:
            wmo_map.setdefault(wmo, line[:11].strip())
    return wmo_map


def station_id_for(station, wmo_map):
    """Reuse the GHCN ID for a WMO-numbered station (USAF = WMO number + a trailing 0), else a
    FIPS-prefixed synthetic ID (see the module docstring)."""
    usaf = station["usaf"]
    if usaf.endswith("0") and usaf[:5] in wmo_map:
        return wmo_map[usaf[:5]]
    return f"{station['fips']}G{usaf}"


def coverage_by_year(series):
    counts = defaultdict(int)
    for obs in series:
        counts[obs["date"][:4]] += 1
    return dict(sorted(counts.items()))


def select_stations(candidates, series_by_usaf, min_days_per_year, min_good_years):
    """Stations with at least min_good_years years of >= min_days_per_year valid days (and at
    least one valid day, whatever min_good_years is)."""
    kept = []
    for s in candidates:
        cov = coverage_by_year(series_by_usaf.get(s["usaf"], []))
        good = sum(1 for n in cov.values() if n >= min_days_per_year)
        s["coverage_by_year"] = cov
        s["good_years"] = good
        if cov and good >= min_good_years:
            kept.append(s)
    return kept


def _km(a, b):
    """Great-circle distance between two {lat, lon} dicts, km."""
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def drop_colocated(stations, series_by_usaf, km=COLOCATED_KM):
    """Keep one station per physical site: of any stations within `km` of each other, the one
    with the most valid days. Returns (kept, dropped), dropped as (usaf, kept-instead usaf)."""
    ranked = sorted(stations, key=lambda s: -len(series_by_usaf.get(s["usaf"], [])))
    kept, dropped = [], []
    for s in ranked:
        twin = next((k for k in kept if _km(s, k) < km), None)
        if twin is None:
            kept.append(s)
        else:
            dropped.append((s["usaf"], twin["usaf"]))
    return [s for s in stations if s in kept], dropped


def assemble_rows(stations, series_by_sid, grid_by_station, humidity_by_station,
                  nighttime_wind_by_station, covariates_by_station):
    """ghcn_training-shaped rows, built the same way build_aemet_valencia_bsh_rows.py builds
    them, except that region comes from the station ID (so it lands in its country's CV fold)."""
    import ghcn

    rows = []
    shifts = {}
    for s in stations:
        sid = s["station_id"]
        station_series = series_by_sid.get(sid, [])
        grid_series = grid_by_station.get(sid, {})
        if not station_series or not grid_series:
            continue
        grid_tmax_by_date = {d: v["tmax"] for d, v in grid_series.items()}
        shift = ghcn.align_obs_window(station_series, grid_tmax_by_date)
        shifts[sid] = shift
        covariates = covariates_by_station.get(sid, {})
        climate_zone = ghcn.koppen_climate_zone(s["lat"], s["lon"])
        humidity_series = humidity_by_station.get(sid, {})
        wind_series = nighttime_wind_by_station.get(sid, {})
        for obs in station_series:
            shifted_day = (date.fromisoformat(obs["date"]) + timedelta(days=shift)).isoformat()
            grid_vals = grid_series.get(shifted_day)
            if grid_vals is None:
                continue
            rows.append({
                "station_id": sid, "date": obs["date"],
                "lon": s["lon"], "lat": s["lat"], "elevation_m": s["elevation_m"],
                "region": ghcn.region_from_station_id(sid), "climate_zone": climate_zone,
                "station_tmax_c": obs["station_tmax_c"], "station_tmin_c": obs["station_tmin_c"],
                "grid_tmax_c": grid_vals["tmax"], "grid_tmin_c": grid_vals["tmin"],
                "delta_tmax_c": obs["station_tmax_c"] - grid_vals["tmax"],
                "delta_tmin_c": obs["station_tmin_c"] - grid_vals["tmin"],
                "grid_specific_humidity_kgkg": humidity_series.get(shifted_day),
                "nighttime_wind_ms": wind_series.get(shifted_day),
                "obs_window_shift_days": shift,
                "koppen_main_group_code": ghcn.koppen_main_group_code_from_zone(climate_zone),
                **covariates,
            })
    return rows, shifts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--countries", nargs="+", required=True, help="FIPS codes, e.g. IN PK")
    ap.add_argument("--start-date", type=date.fromisoformat, required=True)
    ap.add_argument("--end-date", type=date.fromisoformat, required=True)
    ap.add_argument("--min-days-per-year", type=int, default=300)
    ap.add_argument("--min-good-years", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--station-ids", nargs="+", help="restrict to these output station IDs (smoke tests)")
    ap.add_argument("--stations-only", action="store_true",
                    help="stop after station selection + GSOD coverage (no CDS, no covariates)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import ghcn

    os.makedirs(args.out_dir, exist_ok=True)
    countries = set(args.countries)

    isd = _cached(args.out_dir, "isd-history.csv", ISD_HISTORY_URL).decode("latin-1")
    candidates = parse_isd_history(isd, countries, args.start_date.year)
    logger.info("isd-history: %d station(s) in %s reporting since %d",
                len(candidates), sorted(countries), args.start_date.year)
    candidates = [s for s in candidates if ghcn.koppen_climate_zone(s["lat"], s["lon"]) == TARGET_ZONE]
    logger.info("%d of them classify as %s", len(candidates), TARGET_ZONE)

    series_by_usaf = {}
    for s in candidates:
        series = []
        for year in range(args.start_date.year, args.end_date.year + 1):
            body = _cached(args.out_dir, f"gsod/{year}/{s['usaf']}{s['wban']}.csv",
                           GSOD_URL.format(year=year, usaf=s["usaf"], wban=s["wban"]))
            if body:
                series.extend(parse_gsod_csv(body.decode("latin-1")))
        series_by_usaf[s["usaf"]] = [o for o in series
                                     if args.start_date.isoformat() <= o["date"] <= args.end_date.isoformat()]

    stations = select_stations(candidates, series_by_usaf, args.min_days_per_year, args.min_good_years)
    stations, colocated = drop_colocated(stations, series_by_usaf)
    for usaf, twin in colocated:
        logger.info("dropped %s: same site as %s (within %.0f km)", usaf, twin, COLOCATED_KM)
    wmo_map = parse_ghcnd_wmo_map(_cached(args.out_dir, "ghcnd-stations.txt", GHCND_STATIONS_URL)
                                  .decode("latin-1"), countries)
    for s in stations:
        s["station_id"] = station_id_for(s, wmo_map)
        series = series_by_usaf[s["usaf"]]
        s["share_max_derived"] = round(sum(o["max_derived"] for o in series) / len(series), 3)
    if args.station_ids:
        stations = [s for s in stations if s["station_id"] in set(args.station_ids)]
    ids = [s["station_id"] for s in stations]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"station_id collision among selected stations: {sorted(ids)}")
    logger.info("%d %s station(s) with >= %d years of >= %d valid days", len(stations), TARGET_ZONE,
                args.min_good_years, args.min_days_per_year)
    with open(os.path.join(args.out_dir, "gsod_bsh_stations.json"), "w") as f:
        json.dump({"stations": stations, "dropped_colocated": colocated,
                   "rejected": [{k: s[k] for k in ("usaf", "wban", "name", "good_years", "coverage_by_year")}
                                for s in candidates if s not in stations]}, f, indent=1)
    if args.stations_only or not stations:
        return

    from build_training_set import (
        _bucket_from_credentials,
        _cluster_stations_by_bbox_cells,
        _landscan_from_credentials,
        fetch_era5_land_for_stations_via_timeseries,
        snapshot_covariates_for_stations,
    )

    series_by_sid = {s["station_id"]: series_by_usaf[s["usaf"]] for s in stations}
    fetch_stations = [{"station_id": s["station_id"], "lat": s["lat"], "lon": s["lon"],
                       "elevation_m": s["elevation_m"], "name": s["name"]} for s in stations]
    grid_by_station, humidity_by_station, wind_by_station = fetch_era5_land_for_stations_via_timeseries(
        fetch_stations, args.start_date, args.end_date,
        checkpoint_path=os.path.join(args.out_dir, "era5_timeseries_checkpoint.jsonl"),
    )
    missing = sorted(set(series_by_sid) - set(grid_by_station))
    if missing:
        logger.warning("no ERA5-Land series for %d station(s), left out: %s", len(missing), missing)

    landscan_bucket, landscan_key = _landscan_from_credentials()
    vuln_bucket = _bucket_from_credentials()
    # Batch the covariate snapshot exactly as build_training_set.main does: per country, then
    # _cluster_stations_by_bbox_cells at its default cell budget. This is not just a memory
    # choice. lst_warm_season_anomaly_c is each station's LST minus a baseline drawn from the
    # SAME batch table (stations within _LST_REFERENCE_RADIUS_KM, else the whole table), so a
    # different batching gives the new rows a differently-defined anomaly than the corpus
    # they're joining. A lone station in its own batch would get an anomaly of ~0 by construction.
    covariates_by_station = {}
    fips_of = {s["station_id"]: s["fips"] for s in stations}
    for fips in sorted({fips_of[s["station_id"]] for s in fetch_stations}):
        country_stations = [s for s in fetch_stations if fips_of[s["station_id"]] == fips]
        clusters = _cluster_stations_by_bbox_cells(country_stations)
        for i, cluster in enumerate(clusters):
            label = f"gsod_{fips}_{TARGET_ZONE}" if len(clusters) == 1 else f"gsod_{fips}_{TARGET_ZONE}_c{i}"
            logger.info("covariates: %s (%d station(s))", label, len(cluster))
            covariates_by_station.update(snapshot_covariates_for_stations(
                cluster, vuln_bucket, batch_label=label,
                landscan_bucket=landscan_bucket, landscan_key=landscan_key,
            ))

    rows, shifts = assemble_rows(stations, series_by_sid, grid_by_station, humidity_by_station,
                                 wind_by_station, covariates_by_station)
    by_station = defaultdict(int)
    for r in rows:
        by_station[r["station_id"]] += 1
    logger.info("assembled %d row(s) across %d station(s)", len(rows), len(by_station))
    with open(os.path.join(args.out_dir, "gsod_bsh_rows.json"), "w") as f:
        json.dump({"rows": rows, "row_count": len(rows), "rows_by_station": dict(by_station),
                   "obs_window_shift_days": shifts, "missing_era5": missing,
                   "start_date": args.start_date.isoformat(), "end_date": args.end_date.isoformat(),
                   "complete": True}, f)
    logger.info("wrote %s", os.path.join(args.out_dir, "gsod_bsh_rows.json"))


if __name__ == "__main__":
    main()
