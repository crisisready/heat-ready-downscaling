"""
PROVENANCE: moved from crisisready/heat-risk-data-api's scripts/validate_lagfill_downscaling.py at commit a31ec14863273904ae6a9de7a6c34cb77f84f4f4 (2026-07-27, Phase 1.4, plan section 5.4). See this repository's own PROVENANCE.md.

REFACTORED during the move: `score_band`/`fidelity_report` (previously defined
in this file) are now `heatready_downscaling.score.score_band`/`fidelity_report`
-- the byte-identical scoring code every validation script and the future
scoring harness share (see score.py's own docstring). `_build_report` now
delegates to `heatready_downscaling.report.build_report`, which canonicalizes
this script's old ad-hoc `rows_nrt_paired` field to the shared envelope's
`rows_paired`. Model loading/inference goes through
`heatready_downscaling.contract.QRFModelAdapter` (`adapter.predict(...)`)
instead of the private repo's `downscaling.load_model`/`predict_downscaled`
free functions -- same underlying logic, now behind the `ModelAdapter`
protocol a future Rung-C contributor model must also satisfy.

`fold_salt` (required by score_band, see its own docstring): this script
scores directly against the live `ghcn_training` table, not a Phase-2
band-paired snapshot with its own `snapshot_version` -- there is no
submission this could be gamed against, so `args.model_version` is used as
the salt (deterministic per model version being validated, which is all
that matters for a maintainer-run retrospective check). Real snapshot-based
callers (a future `run_submission.py`/`score_forward_eval.py`) must pass
the actual `snapshot_version` instead, per score_band's own docstring.

NOT RUNNABLE STANDALONE IN THIS REPO: like scripts/build_training_set.py,
this script imports db/heat_calcs/open_meteo/api_call_manager -- private-
repo-only modules for live Aurora/Open-Meteo access this public repo has no
path to. It is landed here because the CODE is the reproducible methodology
record this program depends on being able to audit -- but actually RUNNING
it happens from crisisready/heat-risk-data-api's own bastion (with its
db.py/heat_calcs.py/open_meteo.py/api_call_manager.py and live Aurora/
Open-Meteo access), with this package (`heatready_downscaling`, pip-
installed from this repo per scripts/requirements-promote.txt) providing
score_band/fidelity_report/build_report/QRFModelAdapter. tests/
test_validate_lagfill_downscaling.py's own module-level `import db` (via
load_validation_rows) is why db.py must be importable to collect that test
file at all -- see conftest.py's collect_ignore for the same reason
build_training_set.py's test is excluded.

Retrospective leave-region-out-style validation of the shipped downscaling
model against the Open-Meteo lag-fill band's near-real-time (NRT) base
distribution.

See docs/plan-2026-07-21-forecast-lagfill-downscaling-feasibility.md section
4a (crisisready/heat-risk-data-api) for the design this implements: the
lag-fill band never has "true" ERA5 -- production serves a blended NRT
product (ERA5 + other NWP models) for the 1-5 most recent days. The
Historical Forecast API archives forecasts "as they were made" (no
look-ahead), so it reconstructs, for any past date, the same kind of NRT
value the live lag-fill band would have served. This script pairs that
reconstruction against `ghcn_training`'s real GHCN station ground truth,
applies the SHIPPED model as-is (no retrain -- matching production's
actual serving path, per the design doc's explicit choice), and reports the
same qrf_beats_grid gate shape scripts/train_downscaling.py's original CV
used, per Koppen zone, so a human can decide which (zone, target) pairs are
safe to enable for the lag-fill band.

Faithfulness to production, two deliberate choices:
  1. NRT hourly data is run through the EXACT SAME code path a live
     Open-Meteo-band day would use: open_meteo._hourly_to_rows (same
     wind_ms_is_fallback / surface-pressure-to-Pa handling) and
     heat_calcs.aggregate_hourly_to_daily (same 6am-local-day boundary,
     nighttime window). This also functions as a live smoke test of that
     plumbing against real NRT data, not just archive-band data.
  2. The "date" a reconstructed value is filed under is GHCN's own reported
     obs date, in the station's local timezone -- NOT the ERA5-shift-
     adjusted date ghcn_training stores alongside grid_tmax_c/grid_tmin_c.
     `obs_window_shift_days` corrects a GHCN-vs-ERA5-Land-UTC-grid
     convention mismatch specifically; it says nothing about how the
     Open-Meteo band's own 6am-local-day bucketing lines up with GHCN's
     date, which is a different pipeline with its own (already-shipped,
     already-relied-upon) day convention. Matching GHCN's raw date against
     the OM-band-style local-day aggregate is the faithful analogue of what
     production actually does for a live polygon. A fidelity self-check
     (--fidelity-check) reports how closely the reconstructed value tracks
     the already-stored ERA5 grid_tmax_c/grid_tmin_c, as a sanity signal on
     this choice -- large systematic drift would indicate a day-boundary
     bug, not just ordinary NRT-vs-ERA5 disagreement.

Usage:
    DB_SECRET_ARN=... DB_HOST=... VULNERABILITY_DATA_BUCKET=... \\
    OPEN_METEO_API_KEY=... AWS_REGION=us-east-1 \\
        python scripts/validate_lagfill_downscaling.py \\
            --model-version ds-2026.07-rf5 --sample 3000 \\
            --out /tmp/lagfill_validation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import heat_calcs
import open_meteo
from api_call_manager import (
    NO_RESULT, AdaptiveThrottle, HttpSession, JsonlCheckpointStore, ServiceConfig, fetch_all,
)

from heatready_downscaling.contract import QRFModelAdapter
from heatready_downscaling.report import build_report
from heatready_downscaling.score import AUTO_ENABLE_MARGIN, MIN_ZONE_N, fidelity_report, score_band

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_lagfill")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(REPO_ROOT, "credentials.yaml")

# Two endpoints for the same data -- anonymous (no key needed, historically
# used to avoid the keyed customer-redirect's degradation) vs keyed (paid
# tier, needs tuned settings). See _service_config below for which one gets
# used and why -- bounded sustained-load testing (via this same
# api_call_manager module) found the keyed customer-redirect endpoint is
# genuinely reliable at max_workers=4/timeout_s=90 (0 failures, every
# station on the first attempt, actually FASTER overall than the
# max_workers=8/timeout_s=30 anonymous-tier defaults this script used
# before) -- the earlier "customer-redirect degrades under load" finding
# was real but fixed by calling it correctly, not a reason to avoid it.
_ANON_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_KEYED_URL = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
_BAND_KEY = "lag_fill"  # this script only ever validates the lag-fill band; stamped into
                        # build_report's envelope so publish_band_gate.py can catch a
                        # report generated here being published under the wrong --band-key
_FETCH_PAD_DAYS = 1  # each side, so a local day's 24h window never runs off the fetched range
# Sentinel, not a real Phase-2 snapshot version: this script scores directly
# against the live ghcn_training table, which has no snapshot_version of its
# own. See score_band's fold_salt usage below for the same reasoning.
_SNAPSHOT_VERSION = "ghcn_training-live"


def _open_meteo_api_key() -> str | None:
    key = os.environ.get("OPEN_METEO_API_KEY")
    if key:
        return key
    if not os.path.exists(CREDENTIALS_FILE):
        return None
    import yaml
    with open(CREDENTIALS_FILE) as f:
        creds = yaml.safe_load(f) or {}
    return (creds.get("open_meteo") or {}).get("api_key")


def load_validation_rows(sample: int | None, seed: int, zones: list[str] | None = None) -> list[dict]:
    """A random sample of ghcn_training rows with everything
    heatready_downscaling.features.build_feature_matrix needs (static
    covariates), plus the raw station_id/lat/lon/date needed to fetch and
    re-bucket an NRT reconstruction the same way production's Open-Meteo
    band would.

    zones (2026-08-03, gate-variant scoping): restrict to specific Koppen
    climate_zone(s) -- lets a variant rerun (e.g. elevation-nan, fitting a
    gate for a SINGLE project's actual serving zone) validate only the
    zone(s) that project needs, rather than refitting the entire global
    fleet under a base distribution most projects don't use. None (default)
    is the original, unscoped, every-zone behavior."""
    import db

    query = """
        SELECT station_id, date, lon, lat, region, climate_zone,
               station_tmax_c, station_tmin_c, grid_tmax_c, grid_tmin_c,
               lst_warm_season_anomaly_c, canopy_height_mean_m, canopy_frac_over_3m,
               wc_built_frac, wc_tree_frac, wc_water_frac, ghsl_urban_fraction,
               pop_density_per_km2, elevation_rel_to_gridcell_m, elevation_mean_m,
               slope_deg, aspect_deg, grid_specific_humidity_kgkg, koppen_main_group_code,
               nighttime_wind_ms
        FROM ghcn_training
        WHERE region IS NOT NULL AND climate_zone IS NOT NULL
          AND station_tmax_c IS NOT NULL AND station_tmin_c IS NOT NULL
    """
    params: list = []
    if zones:
        # IN (...) with individual placeholders, not = ANY(%s) -- this
        # project's db.py uses pg8000, whose array-parameter binding has
        # been driver-version-fragile historically; a plain IN clause has
        # no such ambiguity and needs no special-casing in db.execute.
        placeholders = ",".join(["%s"] * len(zones))
        query += f" AND climate_zone IN ({placeholders})"
        params.extend(zones)
    if sample:
        query += " ORDER BY random() LIMIT %s"
        params.append(int(sample))
    rows = db.execute(query, tuple(params))
    logger.info("Loaded %d ghcn_training row(s) for validation%s", len(rows),
                f" (zones={sorted(zones)})" if zones else "")
    return rows


def _station_timezones(rows: list[dict]) -> dict[str, str]:
    """One timezone lookup per distinct station -- mirrors
    build_training_set.py's _timezones_for_stations, needed so
    aggregate_hourly_to_daily buckets NRT hours into the same local days
    GHCN's own dates represent."""
    from timezonefinder import TimezoneFinder

    tf = TimezoneFinder()
    by_station: dict[str, tuple[float, float]] = {}
    for r in rows:
        by_station.setdefault(r["station_id"], (r["lat"], r["lon"]))
    return {sid: (tf.timezone_at(lng=lon, lat=lat) or "UTC") for sid, (lat, lon) in by_station.items()}


def _service_config(api_key: str | None) -> ServiceConfig:
    """Keyed (paid tier, tuned) when a key is available, anonymous
    (untuned, historically throttle-prone) otherwise. See the module
    docstring's endpoint comment for the sustained-load test that
    validated these specific numbers."""
    if api_key:
        return ServiceConfig(
            name="lagfill_hfa_keyed", api_key=api_key, api_key_param="apikey",
            timeout_s=90.0, retry_max=4, backoff_base_s=2.0,
        )
    return ServiceConfig(name="lagfill_hfa_anon", timeout_s=30.0, retry_max=4, backoff_base_s=2.0)


def _endpoint_url(api_key: str | None) -> str:
    return _KEYED_URL if api_key else _ANON_URL


def _fetch_nrt_daily_for_station_tz(
    station_id: str, lat: float, lon: float, dates: list[date], session: HttpSession, tz: str, url: str,
    disable_elevation_correction: bool = False,
) -> dict[str, dict]:
    """One HTTP call (one calendar year of hourly data) via the shared,
    throttled HttpSession -- threads the station's real timezone through to
    aggregate_hourly_to_daily instead of the module-level UTC placeholder.

    disable_elevation_correction (2026-08-03): a single-station request, so
    (unlike production's batched multi-point requests -- see
    heat-risk-data-api's own open_meteo.py fix for why a batch needs one
    "nan" per point) a bare "nan" scalar is correct here without any
    per-point expansion."""
    start = min(dates) - timedelta(days=_FETCH_PAD_DAYS)
    end = max(dates) + timedelta(days=_FETCH_PAD_DAYS)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(open_meteo._HOURLY_VARS),
        "timezone": "UTC",
        # Open-Meteo's windspeed default is km/h; production's own
        # open_meteo.py always overrides this (see its "wind_speed_unit":
        # "ms" comment, "default is km/h!") -- omitting it here would feed
        # km/h values straight into nighttime_wind_ms as if they were
        # already m/s, ~3.6x too high.
        "wind_speed_unit": "ms",
    }
    if disable_elevation_correction:
        params["elevation"] = "nan"

    hourly_data = session.get_json(url, params)
    if hourly_data is NO_RESULT or "hourly" not in hourly_data:
        return {}

    tz_map = {station_id: tz}
    rows = open_meteo._hourly_to_rows(hourly_data["hourly"], [station_id])
    daily = heat_calcs.aggregate_hourly_to_daily(rows, tz_map)
    wind_by_date = heat_calcs.daily_mean_nighttime_wind(rows, tz_map).get(station_id, {})
    sh_by_date = heat_calcs.daily_mean_specific_humidity(rows, tz_map).get(station_id, {})

    out: dict[str, dict] = {}
    for d in daily:
        out[d["date"]] = {
            "tmax": d["day_t2m_max"],
            "tmin": d["day_t2m_min"],
            "nighttime_wind_ms": wind_by_date.get(d["date"]),
            "grid_specific_humidity_kgkg": sh_by_date.get(d["date"]),
        }
    return out


def _process_one_station(station: dict, session: HttpSession) -> dict:
    """fetch_all's fetch_fn -- one station's worth of build_paired_rows'
    work (may issue >1 HTTP call internally, one per distinct year in its
    date range, all routed through the same shared/throttled session).
    Returns a single combined payload ({nrt_rows, fidelity_rows}) since
    fetch_all/CheckpointStore only support one payload per item -- the two
    output streams this station produces are bundled together and split
    back apart by the caller after collect()."""
    station_id = station["station_id"]
    station_rows = station["rows"]
    tz = station["tz"]
    url = station["url"]
    disable_elevation_correction = station.get("disable_elevation_correction", False)
    lat, lon = station_rows[0]["lat"], station_rows[0]["lon"]
    dates = [r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"]) for r in station_rows]

    # Historical Forecast API caps how much a single call should span --
    # chunk by year so one long-lived station doesn't produce a multi-year
    # hourly payload in one request.
    by_year: dict[int, list[date]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    daily_by_date: dict[str, dict] = {}
    for year, year_dates in by_year.items():
        chunk = _fetch_nrt_daily_for_station_tz(
            station_id, lat, lon, year_dates, session, tz, url,
            disable_elevation_correction=disable_elevation_correction,
        )
        daily_by_date.update(chunk)

    nrt_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for r in station_rows:
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
        nrt = daily_by_date.get(d.isoformat())
        if nrt is None or nrt["tmax"] is None or nrt["tmin"] is None:
            continue
        new_row = dict(r)
        new_row["grid_tmax_c"] = nrt["tmax"]
        new_row["grid_tmin_c"] = nrt["tmin"]
        # Fall back to the row's existing ERA5-derived covariate when the
        # NRT reconstruction itself couldn't produce one (e.g. a
        # fabricated-wind hour excluded it) -- these two covariates are
        # meant to describe the grid cell's typical humidity/wind regime,
        # not a quantity unique to one data source, so this is a
        # reasonable proxy rather than dropping the row outright.
        if nrt["nighttime_wind_ms"] is not None:
            new_row["nighttime_wind_ms"] = nrt["nighttime_wind_ms"]
        if nrt["grid_specific_humidity_kgkg"] is not None:
            new_row["grid_specific_humidity_kgkg"] = nrt["grid_specific_humidity_kgkg"]
        nrt_rows.append(new_row)

        # math.isfinite, not just `is not None`: ghcn_training's stored
        # grid_tmax_c/grid_tmin_c can hold a literal float('nan') for a
        # handful of rows (confirmed live -- 275/272,021 in a real
        # forecast-lead run, a genuine upstream gap, not a rounding
        # artifact), which `is not None` does not catch. A single NaN
        # entering fidelity_rows poisons np.mean() for the ENTIRE
        # fidelity_check, disabling this exact sanity-check net for the
        # whole report instead of just skipping the one bad row.
        if (
            r.get("grid_tmax_c") is not None and r.get("grid_tmin_c") is not None
            and math.isfinite(r["grid_tmax_c"]) and math.isfinite(r["grid_tmin_c"])
        ):
            fidelity_rows.append({
                "station_id": station_id, "date": d.isoformat(),
                "era5_tmax": r["grid_tmax_c"], "era5_tmin": r["grid_tmin_c"],
                "nrt_tmax": nrt["tmax"], "nrt_tmin": nrt["tmin"],
            })

    if not nrt_rows:
        return NO_RESULT
    return {"nrt_rows": nrt_rows, "fidelity_rows": fidelity_rows}


def build_paired_rows(
    rows: list[dict], tz_by_station: dict[str, str], api_key: str | None,
    max_workers: int | None = None, checkpoint_path: str | None = None,
    disable_elevation_correction: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (nrt_rows, fidelity_rows) -- see _process_one_station's
    docstring for the per-station payload shape this flattens.

    Stations are processed CONCURRENTLY via api_call_manager.fetch_all --
    one station = one fetch_all item (may itself issue >1 HTTP call, one
    per year in its date range), all routed through one shared
    AdaptiveThrottle + HttpSession so real concurrent pressure on Open-Meteo
    stays bounded and adaptive regardless of how many years any one station
    spans. Checkpointed to a JsonlCheckpointStore -- a kill/crash mid-run
    loses at most the stations still in flight, and a re-run with the same
    checkpoint_path resumes instead of re-fetching everything.

    max_workers: defaults to 4 for the keyed endpoint (tuned, validated
    sustained-load test: 0 failures, faster overall than 8 workers on the
    untuned defaults) or 8 for anonymous (matches the prior default; the
    anonymous endpoint was never observed to need the same conservative
    tuning the keyed customer-redirect does).
    """
    if max_workers is None:
        max_workers = 4 if api_key else 8

    by_station: dict[str, list[dict]] = {}
    for r in rows:
        by_station.setdefault(r["station_id"], []).append(r)

    url = _endpoint_url(api_key)
    stations = [
        {
            "station_id": sid, "rows": station_rows, "tz": tz_by_station.get(sid, "UTC"), "url": url,
            "disable_elevation_correction": disable_elevation_correction,
        }
        for sid, station_rows in by_station.items()
    ]

    cfg = _service_config(api_key)
    throttle = AdaptiveThrottle(max_workers=max_workers)
    session = HttpSession(cfg, throttle)
    store = JsonlCheckpointStore(checkpoint_path or "/tmp/lagfill_fetch_checkpoint.jsonl")

    summary = fetch_all(
        stations, fetch_fn=_process_one_station, key_fn=lambda s: s["station_id"],
        store=store, session=session, max_workers=max_workers, progress_every=25,
    )
    logger.info(
        "fetch_all done: %d/%d stations fetched, %d failed, %d already checkpointed. Throttle stats: %s",
        summary.n_fetched, summary.n_total, summary.n_failed, summary.n_done_preexisting,
        json.dumps(summary.throttle_stats),
    )

    nrt_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for _key, payload in store.collect():
        nrt_rows.extend(payload["nrt_rows"])
        fidelity_rows.extend(payload["fidelity_rows"])
    return nrt_rows, fidelity_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOT required=True at the argparse level: phase=dump-rows and
    # phase=fetch never call QRFModelAdapter.load, so forcing a value here
    # made those two phases unusable from a machine with no knowledge of/
    # access to the model registry, contradicting the phase split's own
    # point. Validated conditionally below instead.
    parser.add_argument("--model-version", default=None, help="required for --phase all or --phase score")
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--sample", type=int, default=3000, help="random ghcn_training row sample size; 0 = full table")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-workers", type=int, default=None,
                         help="concurrent stations in flight during --phase fetch/all; defaults to 4 (keyed) "
                              "or 8 (anonymous) -- see build_paired_rows' own docstring for the sustained-load "
                              "test these tuned defaults came from")
    parser.add_argument("--checkpoint", default=None,
                         help="phase=fetch/all: JsonlCheckpointStore path -- a kill/rerun with the same path "
                              "resumes instead of re-fetching. Defaults to /tmp/lagfill_fetch_checkpoint_"
                              "s{sample}_n{rows}.jsonl (config-specific, to avoid silently reusing another "
                              "run's data) -- pass explicitly to intentionally resume across runs.")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    # --phase: the full run needs BOTH DB access (only available from an EC2
    # box inside the VPC) AND a source IP Open-Meteo hasn't throttled -- a
    # several-hour sustained run from ONE EC2 box can eventually trip an
    # IP-level 429 wall that blocks even KEYED requests (a paid subscription
    # doesn't help if the IP itself is blocked), while a different machine's
    # IP (no DB access, but not throttled) serves the same key fine.
    # Splitting into phases lets the DB-bound step run where the DB is
    # reachable and the Open-Meteo-bound step run wherever the IP is
    # currently clean, with the row data handed off as plain JSON (dates
    # round-trip as ISO strings; every downstream function already accepts
    # either a date object or an ISO string, see build_paired_rows). "all"
    # (default) is the original single-machine behavior, unchanged.
    parser.add_argument(
        "--phase", choices=["all", "dump-rows", "fetch", "score"], default="all",
        help="all (default, needs DB+clean IP+model access all at once), dump-rows (DB only), "
             "fetch (Open-Meteo only, reads --rows-in), score (model only, reads --paired-in)",
    )
    parser.add_argument("--rows-in", default=None, help="phase=fetch: JSON from a prior --phase dump-rows run")
    parser.add_argument("--paired-in", default=None, help="phase=score: JSON from a prior --phase fetch run")
    # 2026-08-03, gate-variant scoping (see downscaling.load_band_gate's own
    # docstring in heat-risk-data-api): a project whose Open-Meteo requests
    # disable elevation correction is served under a DIFFERENT base
    # distribution than the one this harness's published gate was fitted
    # against. --elevation-nan refits under that base; --zones scopes the
    # (expensive, global-by-default) refit to only the zone(s) that project
    # actually needs, rather than the whole fleet. Passing --elevation-nan
    # stamps "base_variant": "native_noelev" into the report (see
    # _build_report) so publish_band_gate.py can refuse a report published
    # under the wrong --variant.
    parser.add_argument("--elevation-nan", action="store_true",
                         help="disable Open-Meteo's elevation lapse-rate correction on every NRT request "
                              "(single-point requests, so a bare elevation=nan is correct -- see "
                              "_fetch_nrt_daily_for_station_tz's own docstring). Stamps base_variant="
                              "'native_noelev' into the report.")
    parser.add_argument("--zones", default=None,
                         help="comma-separated Koppen climate_zone(s) to restrict validation to (e.g. Cfb) -- "
                              "default is unscoped, every zone with data. Use with --elevation-nan to refit "
                              "only the zone(s) a specific project's variant override actually needs.")
    args = parser.parse_args()
    zones = [z.strip() for z in args.zones.split(",") if z.strip()] if args.zones else None
    # 2026-08-03, adversarial review finding: --zones "," or --zones ""
    # parses to an empty list, which the `if zones:` guards elsewhere in
    # this file treat as "no filter" -- silently turning an intended
    # single-zone run into a full global refit. Raise rather than guess.
    if args.zones and not zones:
        raise SystemExit(f"--zones {args.zones!r} parsed to an empty zone list -- refusing to "
                          "silently fall back to an unscoped (global) run")
    base_variant = "native_noelev" if args.elevation_nan else None

    if args.phase == "dump-rows":
        rows = load_validation_rows(args.sample or None, args.seed, zones=zones)
        if not rows:
            logger.error("No ghcn_training rows returned -- nothing to validate")
            raise SystemExit(1)
        out_path = args.out or "/tmp/lagfill_rows.json"
        # sample_requested travels WITH the data (not just args.sample at
        # whatever phase happens to read it later) -- score's own --sample
        # CLI value has no relation to what dump-rows actually used, and
        # could silently mismatch the real rows_sampled count in the final
        # report. base_variant/zones travel the same way (2026-08-03) --
        # score's report stamp must reflect what THIS run's data actually
        # is, not whatever --elevation-nan/--zones happen to be set to on a
        # later, possibly different, score invocation.
        with open(out_path, "w") as f:
            json.dump(
                {"sample_requested": args.sample, "rows": rows, "base_variant": base_variant, "zones": zones},
                f, default=str,
            )
        logger.info("Dumped %d ghcn_training row(s) to %s%s", len(rows), out_path,
                    f" (zones={zones})" if zones else "")
        return

    if args.phase == "fetch":
        if not args.rows_in:
            raise SystemExit("--phase fetch requires --rows-in")
        api_key = _open_meteo_api_key()
        if not api_key:
            logger.warning("No Open-Meteo API key found -- proceeding anonymous, expect tighter rate limits")
        with open(args.rows_in) as f:
            dumped = json.load(f)
        rows = dumped["rows"]
        if not rows:
            logger.error("Input file %s has no rows -- nothing to fetch", args.rows_in)
            raise SystemExit(1)
        # base_variant/zones travel with the dumped rows (see dump-rows'
        # own comment) -- NOT re-read from this invocation's own
        # --elevation-nan/--zones, which could silently disagree with what
        # dump-rows actually queried/is about to be fetched under.
        dumped_variant = dumped.get("base_variant")
        dumped_zones = dumped.get("zones")
        tz_by_station = _station_timezones(rows)
        logger.info("Resolved timezones for %d distinct station(s)", len(tz_by_station))
        # Default path includes the input config (sample_requested + actual
        # row count + base_variant) -- a FIXED default path across
        # different --sample/--rows-in/--elevation-nan configs would
        # silently reuse a prior run's per-station payloads as if they were
        # current for a totally different base distribution, with no
        # error (2026-08-03: this is exactly the class of bug this comment
        # already warned about for sample/rows_in -- elevation mode is the
        # same risk, not a new one). Explicit --checkpoint still overrides
        # this whenever real resumption across configs is actually intended.
        variant_tag = dumped_variant or "default"
        checkpoint_path = (
            args.checkpoint
            or f"/tmp/lagfill_fetch_checkpoint_s{dumped['sample_requested']}_n{len(rows)}_{variant_tag}.jsonl"
        )
        nrt_rows, fidelity_rows = build_paired_rows(
            rows, tz_by_station, api_key, args.max_workers, checkpoint_path,
            disable_elevation_correction=bool(dumped_variant),
        )
        logger.info("Built %d NRT-paired row(s) from %d sampled row(s) (%.1f%% coverage)",
                    len(nrt_rows), len(rows), 100.0 * len(nrt_rows) / len(rows) if rows else 0.0)
        out_path = args.out or "/tmp/lagfill_paired.json"
        with open(out_path, "w") as f:
            json.dump({
                "sample_requested": dumped["sample_requested"], "base_variant": dumped_variant,
                "zones": dumped_zones,
                "rows_sampled": len(rows), "nrt_rows": nrt_rows, "fidelity_rows": fidelity_rows,
            }, f)
        logger.info("Wrote %d paired row(s) to %s", len(nrt_rows), out_path)
        return

    if args.phase == "score":
        if not args.paired_in:
            raise SystemExit("--phase score requires --paired-in")
        if not args.model_version:
            raise SystemExit("--phase score requires --model-version")
        bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
        adapter = QRFModelAdapter.load(args.model_version, bucket=bucket)
        with open(args.paired_in) as f:
            paired = json.load(f)
        nrt_rows, fidelity_rows = paired["nrt_rows"], paired["fidelity_rows"]
        report = _build_report(
            args.model_version, paired["sample_requested"], paired["rows_sampled"],
            nrt_rows, fidelity_rows, adapter, base_variant=paired.get("base_variant"),
            zones=paired.get("zones"),
        )
        out_path = args.out or os.path.join("/tmp", f"lagfill_validation_{args.model_version}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote report to %s", out_path)
        return

    if not args.model_version:
        raise SystemExit("--model-version is required for --phase all")

    # --phase all: original single-machine behavior, unchanged.
    api_key = _open_meteo_api_key()
    if not api_key:
        logger.warning("No Open-Meteo API key found (OPEN_METEO_API_KEY env or credentials.yaml) -- proceeding anonymous, expect tighter rate limits")

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    adapter = QRFModelAdapter.load(args.model_version, bucket=bucket)

    rows = load_validation_rows(args.sample or None, args.seed, zones=zones)
    if not rows:
        logger.error("No ghcn_training rows returned -- nothing to validate")
        return

    tz_by_station = _station_timezones(rows)
    logger.info("Resolved timezones for %d distinct station(s)", len(tz_by_station))

    # variant_tag in the default path (2026-08-03): a FIXED default across
    # different --elevation-nan configs would silently reuse a prior run's
    # per-station payloads as if they were current for a different base
    # distribution, no error -- same risk this comment already covers for
    # sample/seed.
    variant_tag = base_variant or "default"
    checkpoint_path = (
        args.checkpoint or f"/tmp/lagfill_fetch_checkpoint_s{args.sample}_seed{args.seed}_{variant_tag}.jsonl"
    )
    nrt_rows, fidelity_rows = build_paired_rows(
        rows, tz_by_station, api_key, args.max_workers, checkpoint_path,
        disable_elevation_correction=args.elevation_nan,
    )
    logger.info("Built %d NRT-paired row(s) from %d sampled row(s) (%.1f%% coverage)",
                len(nrt_rows), len(rows), 100.0 * len(nrt_rows) / len(rows) if rows else 0.0)

    report = _build_report(
        args.model_version, args.sample, len(rows), nrt_rows, fidelity_rows, adapter,
        base_variant=base_variant, zones=zones,
    )
    out_path = args.out or os.path.join("/tmp", f"lagfill_validation_{args.model_version}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report to %s", out_path)


def _build_report(
    model_version: str, sample_requested: int, rows_sampled: int,
    nrt_rows: list[dict], fidelity_rows: list[dict], adapter: QRFModelAdapter,
    base_variant: str | None = None, zones: list[str] | None = None,
) -> dict:
    """Shared by --phase all and --phase score so the two paths can never
    silently diverge in what a 'report' contains. sample_requested/
    rows_sampled are passed explicitly (not read off args) so --phase score
    reports the value the ORIGINAL dump-rows invocation actually used, not
    whatever --sample happens to be set to on the score invocation itself
    (these can be different machines/runs). base_variant (2026-08-03) is
    the same idea, for the SAME reason: identifies which alternate base
    distribution (if any) this run's NRT rows were actually fetched under
    (see load_band_gate's own docstring in heat-risk-data-api) -- stamped
    into the report so publish_band_gate.py can refuse a report published
    under a --variant it doesn't actually match.

    zones (2026-08-03, adversarial review finding): stamped for the SAME
    reason as base_variant, but the risk it closes is different and more
    dangerous by default -- a --zones-scoped run with NO --elevation-nan
    produces a report with base_variant=None, which passes build_gate's
    variant check cleanly (None matches the default key) and would
    silently REPLACE the entire fleet-wide default gate with only the
    scoped zone(s)' data (build_gate constructs the gate fresh from the
    report alone, no merge with what's currently published). Stamping
    zones here lets build_gate refuse that specific case even when variant
    itself was never in play."""
    by_target: dict[str, dict] = {}
    for target in ("tmax", "tmin"):
        by_zone = score_band(adapter, nrt_rows, target, fold_salt=model_version)
        by_target[target] = by_zone
        passing = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is True)
        failing = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is False)
        insufficient = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is None)
        auto_enable = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid_with_margin"] is True)
        logger.info("[%s] zones PASSING plain gate: %s", target, passing)
        logger.info("[%s] zones FAILING plain gate: %s", target, failing)
        logger.info("[%s] zones with insufficient n (< %d): %s", target, MIN_ZONE_N, insufficient)
        logger.info("[%s] zones clearing the %.0f%%-margin AUTO-ENABLE bar: %s", target, AUTO_ENABLE_MARGIN * 100, auto_enable)

    return build_report(
        model_version=model_version,
        band_key=_BAND_KEY,
        snapshot_version=_SNAPSHOT_VERSION,
        sample_requested=sample_requested,
        rows_sampled=rows_sampled,
        rows_paired=len(nrt_rows),
        fidelity_check=fidelity_report(fidelity_rows),
        by_target=by_target,
        extra=({"base_variant": base_variant} if base_variant else {})
              | ({"zones": sorted(zones)} if zones else {})
              or None,
    )


if __name__ == "__main__":
    main()
