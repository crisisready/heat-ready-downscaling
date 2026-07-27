"""
PROVENANCE: moved from crisisready/heat-risk-data-api's scripts/validate_forecast_downscaling.py at commit a31ec14863273904ae6a9de7a6c34cb77f84f4f4 (2026-07-27, Phase 1.4, plan section 5.4). See this repository's own PROVENANCE.md.

REFACTORED during the move: previously imported `score_band`/`fidelity_report`/
`_MIN_ZONE_N` from validate_lagfill_downscaling.py -- the "two scripts
reaching into each other's module" anti-pattern score.py's own docstring
flags as fixed by this extraction. Now imports directly from
`heatready_downscaling.score`, same as validate_lagfill_downscaling.py does.
Model loading/inference goes through `heatready_downscaling.contract.
QRFModelAdapter` instead of the private repo's `downscaling.load_model`/
`predict_downscaled`. Report envelope now built via `heatready_downscaling.
report.build_report`, which canonicalizes this script's old `rows_lead_paired`
field to the shared envelope's `rows_paired` (with `lead_days` carried
through via `build_report`'s `extra` kwarg).

`fold_salt`: same `args.model_version`-as-salt reasoning as
validate_lagfill_downscaling.py -- see that script's own docstring.

NOT RUNNABLE STANDALONE IN THIS REPO: like scripts/build_training_set.py,
this script imports heat_calcs/open_meteo/api_call_manager -- private-repo-
only modules for live Open-Meteo access this public repo has no path to.
Actually RUNNING it happens from crisisready/heat-risk-data-api's own
bastion -- see validate_lagfill_downscaling.py's own docstring for the full
explanation, identical here.

Retrospective validation of the shipped downscaling model against the
forecast band, using Open-Meteo's Previous Runs API (`temperature_2m_previous_dayN`
etc.) to reconstruct, for a past date, exactly what an N-day-ahead forecast
said at the time it was issued -- a look-ahead-safe reconstruction of what
production's forecast band actually serves.

See docs/plan-2026-07-21-forecast-lagfill-downscaling-feasibility.md
(crisisready/heat-risk-data-api) section 3d/4b: the anonymous-tier null
result that originally blocked this was a wrong-hostname research bug
(customer-api.open-meteo.com was guessed instead of the real
previous-runs-api.open-meteo.com); re-tested, both anonymous and keyed
access work. Production serves forecast lead 1-7 days ahead (manager.py's
FORECAST_DAYS, default 7) -- lead-time-dependent bias drift is a known risk
for NWP output (feasibility doc section 3c), so this script validates ONE
lead time per run (--lead-days), not a single pooled gate across every
lead. Run once per lead of interest; start with lead=1 (the nearest-term,
most heavily-used forecast day, and the one with the most retrospective
data available at a given zone's sample size).

Shares its production-code-path philosophy and NRT-fetch/day-bucketing
design with scripts/validate_lagfill_downscaling.py (see that module's
docstring for the two faithfulness choices -- both apply here unchanged: run
NRT/forecast hourly data through open_meteo._hourly_to_rows +
heat_calcs.aggregate_hourly_to_daily exactly as production would, and bucket
by GHCN's own reported local date, not the ERA5-shift-adjusted one). Radiation
variables (shortwave/direct) are NOT requested here -- the Previous Runs API
was only confirmed to carry temperature/dewpoint/wind/pressure lead-time
variants, and this validation doesn't need UTCI/WBGT, only day_t2m_max/min +
nighttime_wind_ms + specific humidity.

Usage:
    DB_SECRET_ARN=... DB_HOST=... VULNERABILITY_DATA_BUCKET=... \\
    OPEN_METEO_API_KEY=... AWS_REGION=us-east-1 \\
        python scripts/validate_forecast_downscaling.py \\
            --model-version ds-2026.07-rf5 --lead-days 1 --sample 3000 \\
            --out /tmp/forecast_lead1_validation_report.json
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
from heatready_downscaling.score import MIN_ZONE_N, fidelity_report, score_band

from validate_lagfill_downscaling import (  # noqa: E402 -- shared plumbing, see module docstring
    _open_meteo_api_key,
    _station_timezones,
    load_validation_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_forecast")

_ANON_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_KEYED_PREVIOUS_RUNS_URL = "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
# CORRECTED (live incident): an earlier version of this module's docstring
# claimed "apikey is just a query param on the plain hostname, no redirect
# needed" -- this was WRONG and never actually verified end-to-end. Direct
# test: previous-runs-api.open-meteo.com with NO key at all returns the
# exact same 429 "Daily API request limit exceeded" as a request WITH
# apikey attached -- the plain hostname silently ignores the key and always
# serves the anonymous tier's shared daily quota, identical to lag-fill's
# historical-forecast-api needing its own "customer-" prefixed hostname to
# actually engage the paid account. Every forecast-band validation call
# before this fix was hitting the anonymous quota regardless of the key,
# which is why it exhausted "barely any real pulls" -- it was never our
# account's usage being counted. customer-previous-runs-api.open-meteo.com
# + apikey verified live: 200 OK, real data, no quota hit.
# NOT sustained-load tested the way lag-fill's keyed endpoint was --
# timeout/concurrency below are carried over from the pre-migration
# defaults, not independently validated; treat them as a starting point,
# not a proven-safe tuning, until this endpoint gets its own bounded
# concurrency test.
_FETCH_PAD_DAYS = 1
# Subset of open_meteo._HOURLY_VARS this endpoint is confirmed to carry a
# _previous_dayN variant for (live-probed) -- shortwave/direct radiation
# were never checked and aren't needed for this validation's tmax/tmin/
# wind/humidity targets (module docstring).
_LEAD_VARS = ["temperature_2m", "dewpoint_2m", "windspeed_10m", "surface_pressure"]
# Live-probed: temperature_2m_previous_dayN has deep coverage (confirmed
# working for 2023 dates -- ghcn_training's ENTIRE date range, see below)
# via the "gfs_seamless" model explicitly; Open-Meteo's default "best_match"
# model silently returns null for an entire region for an older date
# (confirmed: London/2023 returns 0/24 non-null on best_match, 24/24 on
# gfs_seamless) -- best_match's regional-model routing isn't reliably
# backed by a deep archive everywhere, so this script always pins
# models=gfs_seamless rather than relying on the default. The companion
# variables (dewpoint/windspeed/surface_pressure previous_dayN) only have
# data from ~2024-01 onward regardless of model (confirmed: all-null for
# 2023, populated for 2024) -- for a 2023-dated row this naturally falls
# through to build_paired_rows' existing "keep the row's original
# ERA5-based covariate when the reconstruction doesn't have one" fallback
# (same logic validate_lagfill_downscaling.py uses), NOT a reason to drop
# the row. 2021-03 is Open-Meteo's own documented floor for GFS 2m
# temperature.
_COVERAGE_START = date(2021, 3, 1)
_FORECAST_MODEL = "gfs_seamless"
# Sentinel, not a real Phase-2 snapshot version -- see
# validate_lagfill_downscaling.py's identical constant/comment.
_SNAPSHOT_VERSION = "ghcn_training-live"


def fetch_lead_daily_for_station(
    station_id: str, lat: float, lon: float, dates: list[date], lead_days: int, session: HttpSession, tz: str,
    url: str,
) -> dict[str, dict]:
    """Same shape as validate_lagfill_downscaling._fetch_nrt_daily_for_station_tz,
    but reconstructs the `lead_days`-ahead forecast value instead of the NRT
    analysis: fetches `{var}_previous_day{lead_days}` for each of _LEAD_VARS,
    remaps to the unsuffixed names open_meteo._hourly_to_rows expects, then
    runs the SAME production aggregation path. api_key (if any) is attached
    by the shared HttpSession/ServiceConfig, not built into params here."""
    dates = [d for d in dates if d >= _COVERAGE_START]
    if not dates:
        return {}
    start = min(dates) - timedelta(days=_FETCH_PAD_DAYS)
    end = max(dates) + timedelta(days=_FETCH_PAD_DAYS)

    hourly_params = ",".join(f"{v}_previous_day{lead_days}" for v in _LEAD_VARS)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": hourly_params,
        "models": _FORECAST_MODEL,
        "timezone": "UTC",
        "wind_speed_unit": "ms",  # see validate_lagfill_downscaling's identical comment -- default is km/h
    }

    resp = session.get_json(url, params)
    if resp is NO_RESULT or "hourly" not in resp:
        return {}

    suffix = f"_previous_day{lead_days}"
    hourly = resp["hourly"]
    remapped = {"time": hourly.get("time", [])}
    for v in _LEAD_VARS:
        remapped[v] = hourly.get(f"{v}{suffix}", [])
    # open_meteo._hourly_to_rows reads these under their archive-API names --
    # dewpoint_2m/windspeed_10m/surface_pressure already match; only
    # temperature_2m needs no rename either. shortwave_radiation is simply
    # absent (not requested), which _hourly_to_rows already handles (defaults
    # solar_wm2 to 0.0, see its own docstring/code).

    # open_meteo._hourly_to_rows requires BOTH t2m and d2m non-null to keep an
    # hour at all -- fine for production (both always come from the same
    # request), but dewpoint_2m_previous_dayN has no data before ~2024 (this
    # module's docstring) while temperature_2m_previous_dayN does, so running
    # 2023-dated rows through that gate drops every hour (confirmed live:
    # 24/24 non-null temperatures, 0 rows survived _hourly_to_rows) even
    # though the regression TARGET (tmax/tmin) is fully computable. tmax/tmin
    # are bucketed directly here so temperature-only dates still produce a
    # row; wind/humidity go through the normal production path and are simply
    # None when d2m isn't available (build_paired_rows' existing fallback to
    # the row's original ERA5-based covariate handles that, same as
    # validate_lagfill_downscaling.py).
    daily_temp = _bucket_temperature_only(remapped["time"], remapped["temperature_2m"], tz)

    tz_map = {station_id: tz}
    rows = open_meteo._hourly_to_rows(remapped, [station_id])
    wind_by_date = heat_calcs.daily_mean_nighttime_wind(rows, tz_map).get(station_id, {}) if rows else {}
    sh_by_date = heat_calcs.daily_mean_specific_humidity(rows, tz_map).get(station_id, {}) if rows else {}

    out: dict[str, dict] = {}
    for d, extrema in daily_temp.items():
        out[d] = {
            "tmax": extrema["tmax"],
            "tmin": extrema["tmin"],
            "nighttime_wind_ms": wind_by_date.get(d),
            "grid_specific_humidity_kgkg": sh_by_date.get(d),
        }
    return out


def _bucket_temperature_only(times: list[str], t2m_vals: list, tz: str) -> dict[str, dict]:
    """Day-bucket a bare temperature_2m series into local-day tmax/tmin using
    heat_calcs' own 06:00-local-day boundary (local_day_iso) -- the same
    convention aggregate_hourly_to_daily uses -- WITHOUT requiring dewpoint
    per hour (see the caller's comment for why that gate would otherwise drop
    every 2023-dated hour). Only days with all 24 local hours present are
    returned, matching aggregate_hourly_to_daily's own completeness rule."""
    from zoneinfo import ZoneInfo

    tzinfo = ZoneInfo(tz)
    buckets: dict[str, dict] = {}
    for t_str, t2m in zip(times, t2m_vals):
        if t2m is None:
            continue
        dt_local = heat_calcs._parse_utc_dt(t_str).astimezone(tzinfo)
        day = heat_calcs.local_day_iso(dt_local)
        b = buckets.setdefault(day, {"hours": set(), "vals": []})
        b["hours"].add(dt_local.hour)
        b["vals"].append(t2m)

    return {
        day: {"tmax": max(b["vals"]), "tmin": min(b["vals"])}
        for day, b in buckets.items() if len(b["hours"]) >= 24
    }


def _process_one_station_lead(station: dict, session: HttpSession) -> dict:
    """fetch_all's fetch_fn for one station at one lead -- see
    validate_lagfill_downscaling._process_one_station for the identical
    pattern (this is its forecast-band sibling)."""
    station_id = station["station_id"]
    station_rows = station["rows"]
    tz = station["tz"]
    lead_days = station["lead_days"]
    url = station["url"]
    lat, lon = station_rows[0]["lat"], station_rows[0]["lon"]
    dates = [r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"]) for r in station_rows]

    by_year: dict[int, list[date]] = {}
    dropped_pre_coverage = 0
    for d in dates:
        if d < _COVERAGE_START:
            dropped_pre_coverage += 1
            continue
        by_year.setdefault(d.year, []).append(d)
    if dropped_pre_coverage:
        # Logged immediately, not aggregated through the return payload --
        # a station whose EVERY date predates _COVERAGE_START ends up with
        # zero lead_rows and returns NO_RESULT below, which fetch_all never
        # persists -- any count folded into that payload would be silently
        # lost for exactly the stations most likely to have a high drop
        # count.
        logger.debug(
            "station=%s: dropped %d date(s) before Previous Runs API coverage start (%s)",
            station_id, dropped_pre_coverage, _COVERAGE_START,
        )

    daily_by_date: dict[str, dict] = {}
    for year, year_dates in by_year.items():
        chunk = fetch_lead_daily_for_station(station_id, lat, lon, year_dates, lead_days, session, tz, url)
        daily_by_date.update(chunk)

    lead_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for r in station_rows:
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
        fc = daily_by_date.get(d.isoformat())
        if fc is None or fc["tmax"] is None or fc["tmin"] is None:
            continue
        new_row = dict(r)
        new_row["grid_tmax_c"] = fc["tmax"]
        new_row["grid_tmin_c"] = fc["tmin"]
        if fc["nighttime_wind_ms"] is not None:
            new_row["nighttime_wind_ms"] = fc["nighttime_wind_ms"]
        if fc["grid_specific_humidity_kgkg"] is not None:
            new_row["grid_specific_humidity_kgkg"] = fc["grid_specific_humidity_kgkg"]
        lead_rows.append(new_row)

        # math.isfinite, not just `is not None` -- same fix as
        # validate_lagfill_downscaling.py's own fidelity_rows builder (see
        # that one's comment for the full "275/272,021 real NaN rows in
        # ghcn_training" incident detail): a single NaN entering
        # fidelity_rows poisons np.mean() for the ENTIRE fidelity_check,
        # silently disabling this sanity-check net for the whole report.
        if (
            r.get("grid_tmax_c") is not None and r.get("grid_tmin_c") is not None
            and math.isfinite(r["grid_tmax_c"]) and math.isfinite(r["grid_tmin_c"])
        ):
            fidelity_rows.append({
                "station_id": station_id, "date": d.isoformat(),
                "era5_tmax": r["grid_tmax_c"], "era5_tmin": r["grid_tmin_c"],
                "nrt_tmax": fc["tmax"], "nrt_tmin": fc["tmin"],
            })

    if not lead_rows:
        return NO_RESULT
    return {"lead_rows": lead_rows, "fidelity_rows": fidelity_rows}


def build_paired_rows(
    rows: list[dict], tz_by_station: dict[str, str], lead_days: int, api_key: str | None,
    max_workers: int = 4, checkpoint_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Same contract/concurrency model as
    validate_lagfill_downscaling.build_paired_rows -- see that function's
    docstring. max_workers defaults conservatively (4, not lag-fill's
    tuned-and-validated value) since this endpoint hasn't had its own
    sustained-load test yet -- see the module docstring."""
    if not api_key:
        logger.warning(
            "No Open-Meteo API key -- forecast validation will hit the ANONYMOUS "
            "previous-runs-api.open-meteo.com quota, which is shared/limited and NOT "
            "what this validation needs real coverage from (see module docstring's "
            "incident note)."
        )
    url = _KEYED_PREVIOUS_RUNS_URL if api_key else _ANON_PREVIOUS_RUNS_URL

    by_station: dict[str, list[dict]] = {}
    for r in rows:
        by_station.setdefault(r["station_id"], []).append(r)

    stations = [
        {"station_id": sid, "rows": station_rows, "tz": tz_by_station.get(sid, "UTC"),
         "lead_days": lead_days, "url": url}
        for sid, station_rows in by_station.items()
    ]

    cfg = ServiceConfig(
        name=f"forecast_lead{lead_days}", api_key=api_key, api_key_param="apikey",
        timeout_s=60.0, retry_max=4, backoff_base_s=2.0,
    )
    throttle = AdaptiveThrottle(max_workers=max_workers)
    session = HttpSession(cfg, throttle)
    store = JsonlCheckpointStore(checkpoint_path or f"/tmp/forecast_lead{lead_days}_fetch_checkpoint.jsonl")

    summary = fetch_all(
        stations, fetch_fn=_process_one_station_lead, key_fn=lambda s: s["station_id"],
        store=store, session=session, max_workers=max_workers, progress_every=25,
    )
    logger.info(
        "Lead-%d fetch_all done: %d/%d stations fetched, %d failed, %d already checkpointed. Throttle stats: %s",
        lead_days, summary.n_fetched, summary.n_total, summary.n_failed, summary.n_done_preexisting,
        json.dumps(summary.throttle_stats),
    )

    lead_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for _key, payload in store.collect():
        lead_rows.extend(payload["lead_rows"])
        fidelity_rows.extend(payload["fidelity_rows"])

    return lead_rows, fidelity_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--lead-days", type=int, required=True, help="forecast lead time to validate, 1-7")
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--sample", type=int, default=3000, help="random ghcn_training row sample size; 0 = full table")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-workers", type=int, default=4,
                         help="concurrent stations in flight -- conservative default, this endpoint hasn't "
                              "had its own sustained-load test the way lag-fill's keyed endpoint has")
    parser.add_argument("--checkpoint", default=None,
                         help="JsonlCheckpointStore path -- a kill/rerun with the same path resumes instead "
                              "of re-fetching. Defaults to /tmp/forecast_lead{N}_fetch_checkpoint_s{sample}_"
                              "seed{seed}.jsonl (config-specific, to avoid silently reusing another run's "
                              "data) -- pass explicitly to intentionally resume across runs.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not (1 <= args.lead_days <= 7):
        raise SystemExit("--lead-days must be 1-7 (production's FORECAST_DAYS horizon)")

    api_key = _open_meteo_api_key()
    if not api_key:
        logger.warning("No Open-Meteo API key found -- proceeding anonymous")

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    adapter = QRFModelAdapter.load(args.model_version, bucket=bucket)

    rows = load_validation_rows(args.sample or None, args.seed)
    if not rows:
        logger.error("No ghcn_training rows returned -- nothing to validate")
        return

    tz_by_station = _station_timezones(rows)
    logger.info("Resolved timezones for %d distinct station(s)", len(tz_by_station))

    checkpoint_path = args.checkpoint or f"/tmp/forecast_lead{args.lead_days}_fetch_checkpoint_s{args.sample}_seed{args.seed}.jsonl"
    lead_rows, fidelity_rows = build_paired_rows(
        rows, tz_by_station, args.lead_days, api_key, args.max_workers, checkpoint_path,
    )
    logger.info("Built %d lead-%d-paired row(s) from %d sampled row(s) (%.1f%% coverage)",
                len(lead_rows), args.lead_days, len(rows),
                100.0 * len(lead_rows) / len(rows) if rows else 0.0)

    band_key = f"forecast_lead{args.lead_days}"
    by_target: dict[str, dict] = {}
    for target in ("tmax", "tmin"):
        by_zone = score_band(adapter, lead_rows, target, fold_salt=args.model_version)
        by_target[target] = by_zone
        passing = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is True)
        failing = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is False)
        insufficient = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid"] is None)
        auto_enable = sorted(z for z, m in by_zone.items() if m["qrf_beats_grid_with_margin"] is True)
        logger.info("[lead=%d, %s] zones PASSING plain gate: %s", args.lead_days, target, passing)
        logger.info("[lead=%d, %s] zones FAILING plain gate: %s", args.lead_days, target, failing)
        logger.info("[lead=%d, %s] zones with insufficient n (< %d): %s", args.lead_days, target, MIN_ZONE_N, insufficient)
        logger.info("[lead=%d, %s] zones clearing the auto-enable margin bar: %s", args.lead_days, target, auto_enable)
        # Symmetry with validate_lagfill_downscaling.py's own logging --
        # bias_correction_c is only set for zones validated via the CV path
        # (score_band's own docstring); surfacing it here means a human
        # deciding whether to run publish_band_gate.py can see which
        # auto-enabled zones are relying on a bias correction without
        # opening the raw JSON report.
        corrected = {z: round(m["bias_correction_c"], 3) for z, m in by_zone.items() if m.get("bias_correction_c") is not None}
        if corrected:
            logger.info("[lead=%d, %s] bias corrections that will be published: %s", args.lead_days, target, corrected)

    report = build_report(
        model_version=args.model_version,
        band_key=band_key,
        snapshot_version=_SNAPSHOT_VERSION,
        sample_requested=args.sample,
        rows_sampled=len(rows),
        rows_paired=len(lead_rows),
        fidelity_check=fidelity_report(fidelity_rows),
        by_target=by_target,
        extra={"lead_days": args.lead_days},
    )

    out_path = args.out or os.path.join("/tmp", f"forecast_lead{args.lead_days}_validation_{args.model_version}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report to %s", out_path)


if __name__ == "__main__":
    main()
