"""
Build the band-paired training/validation snapshot (plan section 6,
crisisready/heat-risk-data-api's docs/plan-2026-07-27-heatready-
crowdsourced-model-improvement.md). One row per station_id x date x band,
long format, hive-partitioned by band and month -- see
heatready_downscaling.snapshot's own docstring for the on-disk layout and
PROVENANCE.md for how this script relates to it.

Three phases, mirroring the DB-access-vs-clean-IP split validate_lagfill_
downscaling.py/validate_forecast_downscaling.py already established (same
underlying problem: `--phase export` needs S3 read access to the
export-ghcn-training output, `--phase fetch` needs a clean, keyed Open-Meteo
IP, `--phase pack` needs neither):

- `--phase export`: download and parse the export-ghcn-training S3 output
  (crisisready/heat-risk-data-api's docs/api.md `export-ghcn-training`
  action) -- gzipped JSONL parts, `SELECT * FROM ghcn_training`. No DB, no
  Open-Meteo. Requires only S3 read on VULNERABILITY_DATA_BUCKET.
- `--phase fetch --band <band>`: one band's worth of Open-Meteo NRT/lead
  reconstruction for every station-day in the export. Reuses (does not
  reimplement) validate_lagfill_downscaling.py's `_fetch_nrt_daily_for_station_tz`
  for `lag_fill` and validate_forecast_downscaling.py's
  `fetch_lead_daily_for_station` for `forecast_lead1..7` -- same production
  code path, same day-bucketing/unit-handling correctness those two scripts
  already earned the hard way. `era5` never needs this phase: its
  grid_tmax_c/grid_tmin_c/etc. are already sitt ing in the export as-is.
- `--phase pack`: shape every band's rows into the snapshot schema
  (heatready_downscaling.snapshot's _pa_schema), partition by band/month,
  build stations.parquet + sample.csv + MANIFEST.json, compute the
  provisional-scoring holdout (snapshot.compute_holdout) and write a
  PUBLIC copy with holdout-station rows and holdout.json excluded
  entirely (plan section 3.4: "Holdout rows are excluded from the
  published snapshot entirely") alongside the FULL private working copy.

NOT RUNNABLE STANDALONE IN THIS REPO for --phase fetch: like validate_
lagfill_downscaling.py/validate_forecast_downscaling.py, it imports heat_calcs/
open_meteo/api_call_manager -- private-repo-only modules. --phase export and
--phase pack need only boto3/pyarrow (already this package's own
dependencies) and run standalone here.

Per-band provenance (plan section 6.1, fixes F7 -- forecast bands are
temperature-only for 2023, humidity/wind silently fall back to ERA5):
every row carries base_source/base_model/humidity_source/wind_source, so a
contributor scoring a snapshot partition can see exactly which values are
genuine band reconstructions versus ERA5 fallbacks, never inferred or
assumed.

Usage:
    # Phase 1: export (S3 only)
    AWS_PROFILE=nish-climateverse VULNERABILITY_DATA_BUCKET=... \\
        python scripts/build_band_paired_snapshot.py --phase export \\
            --run-id <run_id from export-ghcn-training> --out /tmp/ghcn_export_rows.json

    # Phase 2: fetch, once per band (8 invocations for lag_fill + forecast_lead1..7;
    # era5 never needs this -- run these concurrently as separate processes,
    # there is no cross-band rate limit to coordinate on the paid Open-Meteo tier)
    OPEN_METEO_API_KEY=... python scripts/build_band_paired_snapshot.py --phase fetch \\
            --band lag_fill --rows-in /tmp/ghcn_export_rows.json --out /tmp/snapshot_lag_fill.json

    # Phase 3: pack (no DB, no Open-Meteo)
    python scripts/build_band_paired_snapshot.py --phase pack \\
            --snapshot-version v2026.07 --rows-in /tmp/ghcn_export_rows.json \\
            --band-input lag_fill=/tmp/snapshot_lag_fill.json \\
            --band-input forecast_lead1=/tmp/snapshot_forecast_lead1.json \\
            ... (one --band-input per fetched band) \\
            --out-dir /tmp/snapshot_v2026.07_full --public-out-dir /tmp/snapshot_v2026.07_public
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from heatready_downscaling import snapshot as snap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_band_paired_snapshot")

_GHCN_EXPORT_S3_PREFIX = "downscaling/exports/ghcn_training"  # must match private repo's handler.py constant

# Open-Meteo endpoint URLs, duplicated as plain string literals rather than
# imported from validate_lagfill_downscaling.py/validate_forecast_downscaling.py
# -- these two scripts (and heat_calcs/open_meteo/api_call_manager, which
# they in turn import) are private-repo-only and NOT available when running
# --phase export/pack in this repo alone (see module docstring). Keeping
# these 5 literal strings here, not imported, is what lets _FETCH_BAND_CONFIG
# stay a module-level dict usable by argparse's --band choices regardless of
# which phase is being run -- matching this codebase's own established idiom
# of documented local redefinition over an artificial cross-module dependency
# for a handful of constants (see validate_lagfill_downscaling.py's own
# comment on _PRACTICAL_BIAS_FLOOR_C for the precedent). The actual fetch
# logic that USES these URLs is still imported lazily, inside _phase_fetch/
# _dispatch_fetch_one_station, never reimplemented.
_ANON_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_KEYED_URL = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
_ANON_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_KEYED_PREVIOUS_RUNS_URL = "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
_FORECAST_MODEL = "gfs_seamless"

# Constants carried verbatim into every row's covariate-provenance columns
# (plan section 6.1) -- these describe HOW the static covariates were
# computed, not values that vary per row, so they're module constants here
# rather than read from anywhere per-row.
_POP_DENSITY_SOURCE = "landscan_global"
_POP_DENSITY_BUFFER_DEG = 0.01
_LST_REFERENCE_RADIUS_KM = 75.0

# Columns copied verbatim from an export row into every band row for that
# station-day -- station identity, fold keys, and static covariates, all
# constant regardless of which band's base value is being paired. Excludes
# delta_tmax_c/delta_tmin_c (training-time-derived, score_band recomputes
# these itself from station_*/grid_* and has no use for a stored delta) and
# grid_tmax_c/grid_tmin_c/grid_specific_humidity_kgkg/nighttime_wind_ms
# (those ARE the base-value columns that differ per band -- handled
# separately by _era5_band_rows/_fetched_band_rows below).
_PASSTHROUGH_COLUMNS = (
    "lon", "lat", "elevation_m", "region", "climate_zone", "koppen_main_group_code",
    "obs_window_shift_days", "lst_warm_season_anomaly_c", "canopy_height_mean_m",
    "canopy_frac_over_3m", "wc_built_frac", "wc_tree_frac", "wc_water_frac",
    "ghsl_urban_fraction", "pop_density_per_km2", "elevation_rel_to_gridcell_m",
    "elevation_mean_m", "slope_deg", "aspect_deg",
)

# Per-band fetch configuration -- endpoints, provenance strings, and
# whether the band needs a lead_days argument. "era5" is deliberately
# absent: it's the one band that never goes through --phase fetch at all.
_FETCH_BAND_CONFIG = {
    "lag_fill": {
        "anon_url": _ANON_URL, "keyed_url": _KEYED_URL,
        "base_source": "open_meteo_hfa", "base_model": None, "lead_days": None,
    },
    **{
        f"forecast_lead{n}": {
            "anon_url": _ANON_PREVIOUS_RUNS_URL, "keyed_url": _KEYED_PREVIOUS_RUNS_URL,
            "base_source": "open_meteo_previous_runs", "base_model": _FORECAST_MODEL, "lead_days": n,
        }
        for n in range(1, 8)
    },
}


def _phase_export(args: argparse.Namespace) -> None:
    """Download and parse every part-N.jsonl.gz under
    s3://{bucket}/downscaling/exports/ghcn_training/{run_id}/ -- the
    private repo's `export-ghcn-training` root action output (SELECT *
    FROM ghcn_training). No DB access -- S3 read only."""
    import boto3

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    prefix = f"{_GHCN_EXPORT_S3_PREFIX}/{args.run_id}/"
    client = boto3.client("s3")

    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith(".jsonl.gz"))
    if not keys:
        raise SystemExit(f"No export parts found at s3://{bucket}/{prefix} -- check --run-id")
    keys.sort()  # part-0, part-1, ... -- not load-bearing for correctness, just deterministic logs

    rows: list[dict] = []
    for key in keys:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        text = gzip.decompress(body).decode("utf-8")
        for line in text.split("\n"):
            if line.strip():
                rows.append(json.loads(line))
        logger.info("Parsed %s (%d part(s) so far, %d row(s) total)", key, keys.index(key) + 1, len(rows))

    logger.info("Export parse complete: %d row(s) across %d part(s)", len(rows), len(keys))
    with open(args.out, "w") as f:
        json.dump({"run_id": args.run_id, "rows": rows}, f)
    logger.info("Wrote %d row(s) to %s", len(rows), args.out)


def _dispatch_fetch_one_station(station: dict, session: HttpSession) -> dict:
    """fetch_all's fetch_fn for --phase fetch -- one station's worth of
    reconstruction for the ONE band this invocation was started with
    (band/lead_days/url are baked into every station dict by _phase_fetch,
    not read from a module global, so this function has no per-invocation
    state of its own). Delegates the actual HTTP/day-bucketing work to
    _fetch_nrt_daily_for_station_tz (lag_fill) or fetch_lead_daily_for_station
    (forecast_lead*) -- see this module's own docstring for why those are
    reused rather than reimplemented.

    Returns {date_iso: {tmax, tmin, nighttime_wind_ms, grid_specific_humidity_kgkg}},
    the same shape both reused functions already produce -- _phase_fetch
    itself does the source-flag/schema shaping once every station's result
    is collected, not per station."""
    from api_call_manager import NO_RESULT
    from validate_forecast_downscaling import _COVERAGE_START, fetch_lead_daily_for_station
    from validate_lagfill_downscaling import _fetch_nrt_daily_for_station_tz

    station_id = station["station_id"]
    dates = station["dates"]
    tz = station["tz"]
    lat, lon = station["lat"], station["lon"]
    lead_days = station["lead_days"]
    url = station["url"]

    by_year: dict[int, list[date]] = {}
    for d in dates:
        if lead_days is not None and d < _COVERAGE_START:
            continue  # forecast bands: Previous Runs API's own documented coverage floor
        by_year.setdefault(d.year, []).append(d)
    if not by_year:
        return NO_RESULT

    daily: dict[str, dict] = {}
    for year, year_dates in by_year.items():
        if lead_days is None:
            chunk = _fetch_nrt_daily_for_station_tz(station_id, lat, lon, year_dates, session, tz, url)
        else:
            chunk = fetch_lead_daily_for_station(station_id, lat, lon, year_dates, lead_days, session, tz, url)
        daily.update(chunk)

    if not daily:
        return NO_RESULT
    return daily


def _phase_fetch(args: argparse.Namespace) -> None:
    from api_call_manager import AdaptiveThrottle, HttpSession, JsonlCheckpointStore, ServiceConfig, fetch_all
    from validate_lagfill_downscaling import _open_meteo_api_key, _service_config, _station_timezones

    if args.band not in _FETCH_BAND_CONFIG:
        raise SystemExit(f"--band must be one of {sorted(_FETCH_BAND_CONFIG)} (era5 never needs --phase fetch)")
    cfg = _FETCH_BAND_CONFIG[args.band]

    with open(args.rows_in) as f:
        export = json.load(f)
    rows = export["rows"]
    logger.info("Loaded %d export row(s) for band=%s", len(rows), args.band)

    api_key = _open_meteo_api_key()
    if not api_key:
        raise SystemExit(
            "OPEN_METEO_API_KEY not set (or credentials.yaml has no open_meteo.api_key) -- the snapshot "
            "fetch must run against the paid, keyed tier, never anonymous (see this module's docstring)."
        )
    url = cfg["keyed_url"]

    tz_by_station = _station_timezones(rows)
    logger.info("Resolved timezones for %d distinct station(s)", len(tz_by_station))

    by_station: dict[str, dict] = {}
    for r in rows:
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
        st = by_station.setdefault(r["station_id"], {
            "station_id": r["station_id"], "lat": r["lat"], "lon": r["lon"],
            "tz": tz_by_station.get(r["station_id"], "UTC"), "lead_days": cfg["lead_days"], "url": url,
            "dates": [], "rows_by_date": {},
        })
        st["dates"].append(d)
        st["rows_by_date"][d.isoformat()] = r

    stations = list(by_station.values())
    # timeout_s matches each band's own already-validated ServiceConfig:
    # 90s for lag_fill (_service_config, historical-forecast-api), 60s for
    # forecast leads (validate_forecast_downscaling.build_paired_rows'
    # own ServiceConfig, previous-runs-api).
    service_cfg = _service_config(api_key) if cfg["lead_days"] is None else ServiceConfig(
        name=f"snapshot_{args.band}", api_key=api_key, api_key_param="apikey",
        timeout_s=60.0, retry_max=4, backoff_base_s=2.0,
    )
    throttle = AdaptiveThrottle(max_workers=args.max_workers)
    session = HttpSession(service_cfg, throttle)
    checkpoint_path = args.checkpoint or f"/tmp/snapshot_{args.band}_{args.snapshot_version}_checkpoint.jsonl"
    store = JsonlCheckpointStore(checkpoint_path)

    summary = fetch_all(
        stations, fetch_fn=_dispatch_fetch_one_station, key_fn=lambda s: s["station_id"],
        store=store, session=session, max_workers=args.max_workers, progress_every=25,
    )
    logger.info(
        "fetch_all done for band=%s: %d/%d stations fetched, %d failed, %d already checkpointed. Throttle: %s",
        args.band, summary.n_fetched, summary.n_total, summary.n_failed, summary.n_done_preexisting,
        json.dumps(summary.throttle_stats),
    )

    payload_by_station = dict(store.collect())
    out_rows: list[dict] = []
    for station in stations:
        payload = payload_by_station.get(station["station_id"])
        if not payload:
            continue
        for d in station["dates"]:
            recon = payload.get(d.isoformat())
            if recon is None or recon["tmax"] is None or recon["tmin"] is None:
                continue
            export_row = station["rows_by_date"][d.isoformat()]
            out_rows.append(_shape_fetched_row(export_row, d, args.band, cfg, recon, args.snapshot_version))

    logger.info("Built %d %s-band row(s) from %d export row(s) (%.1f%% coverage)",
                len(out_rows), args.band, len(rows), 100.0 * len(out_rows) / len(rows) if rows else 0.0)
    with open(args.out, "w") as f:
        json.dump({"band": args.band, "rows": out_rows}, f, default=str)
    logger.info("Wrote %d row(s) to %s", len(out_rows), args.out)


def _shape_fetched_row(
    export_row: dict, d: date, band: str, cfg: dict, recon: dict, snapshot_version: str,
) -> dict:
    """One export row + one day's Open-Meteo reconstruction -> one full
    snapshot-schema row. humidity_source/wind_source are set per-FIELD
    (not per-row) since a forecast-band row can genuinely have a real
    temperature reconstruction alongside a fallback-to-ERA5 humidity/wind
    value (F7: dewpoint/wind/pressure previous_dayN has no 2023 coverage
    even though temperature does) -- collapsing this to one row-level flag
    would silently misrepresent exactly the cases F7 exists to disclose."""
    humidity = recon.get("grid_specific_humidity_kgkg")
    wind = recon.get("nighttime_wind_ms")
    row = {
        "station_id": export_row["station_id"], "date": d, "band": band,
        "station_tmax_c": export_row["station_tmax_c"], "station_tmin_c": export_row["station_tmin_c"],
        "grid_tmax_c": recon["tmax"], "grid_tmin_c": recon["tmin"],
        "grid_specific_humidity_kgkg": humidity if humidity is not None else export_row.get("grid_specific_humidity_kgkg"),
        "nighttime_wind_ms": wind if wind is not None else export_row.get("nighttime_wind_ms"),
        "base_source": cfg["base_source"], "base_model": cfg["base_model"],
        "humidity_source": "band" if humidity is not None else "era5_fallback",
        "wind_source": "band" if wind is not None else "era5_fallback",
        "pop_density_source": _POP_DENSITY_SOURCE, "pop_density_buffer_deg": _POP_DENSITY_BUFFER_DEG,
        "lst_reference_radius_km": _LST_REFERENCE_RADIUS_KM, "snapshot_version": snapshot_version,
    }
    row.update({col: export_row.get(col) for col in _PASSTHROUGH_COLUMNS})
    return row


def _era5_band_rows(export_rows: list[dict], snapshot_version: str) -> list[dict]:
    """era5 band rows built directly from the export -- no fetch needed,
    grid_tmax_c/grid_tmin_c/grid_specific_humidity_kgkg/nighttime_wind_ms
    are already genuine ERA5-Land values in ghcn_training itself."""
    out: list[dict] = []
    for r in export_rows:
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
        row = {
            "station_id": r["station_id"], "date": d, "band": "era5",
            "station_tmax_c": r["station_tmax_c"], "station_tmin_c": r["station_tmin_c"],
            "grid_tmax_c": r["grid_tmax_c"], "grid_tmin_c": r["grid_tmin_c"],
            "grid_specific_humidity_kgkg": r.get("grid_specific_humidity_kgkg"),
            "nighttime_wind_ms": r.get("nighttime_wind_ms"),
            "base_source": "era5_land_cds", "base_model": None,
            "humidity_source": "band", "wind_source": "band",
            "pop_density_source": _POP_DENSITY_SOURCE, "pop_density_buffer_deg": _POP_DENSITY_BUFFER_DEG,
            "lst_reference_radius_km": _LST_REFERENCE_RADIUS_KM, "snapshot_version": snapshot_version,
        }
        row.update({col: r.get(col) for col in _PASSTHROUGH_COLUMNS})
        out.append(row)
    return out


def _stations_from_export(export_rows: list[dict]) -> list[dict]:
    """One row per distinct station_id -- identity/fold-key + static-
    covariate columns only (no band, no date), deduped from the export's
    own per-station-day rows (these columns are constant per station)."""
    seen: dict[str, dict] = {}
    for r in export_rows:
        if r["station_id"] in seen:
            continue
        seen[r["station_id"]] = {"station_id": r["station_id"], **{col: r.get(col) for col in _PASSTHROUGH_COLUMNS}}
    return list(seen.values())


def _write_sample_csv(path: str, station_id: str, band_rows_by_band: dict[str, list[dict]]) -> None:
    """1 station x every fetched band x however many days that station has
    -- a small, git-committable illustration of the schema (plan section
    6: "sample.csv, committed") without needing pyarrow to inspect it."""
    all_rows = [r for rows in band_rows_by_band.values() for r in rows if r["station_id"] == station_id]
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_rows, key=lambda r: (r["band"], r["date"])):
            writer.writerow(r)


def _phase_pack(args: argparse.Namespace) -> None:
    with open(args.rows_in) as f:
        export = json.load(f)
    export_rows = export["rows"]

    band_rows: dict[str, list[dict]] = {"era5": _era5_band_rows(export_rows, args.snapshot_version)}
    for spec in args.band_input:
        band, path = spec.split("=", 1)
        with open(path) as f:
            fetched = json.load(f)
        rows = fetched["rows"]
        for r in rows:
            r["date"] = r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
        band_rows[band] = rows
        logger.info("Loaded %d row(s) for band=%s from %s", len(rows), band, path)

    stations = _stations_from_export(export_rows)
    holdout = snap.compute_holdout(stations)
    logger.info("Computed provisional holdout: %d station(s) across %d zone(s)",
                len(holdout), len({s["climate_zone"] for s in stations}))

    def _write_snapshot(out_dir: str, *, exclude_holdout: bool) -> None:
        partitions: list[dict] = []
        for band, rows in band_rows.items():
            filtered = [r for r in rows if not (exclude_holdout and r["station_id"] in holdout)] if exclude_holdout else rows
            by_month: dict[str, list[dict]] = {}
            for r in filtered:
                by_month.setdefault(r["date"].strftime("%Y-%m"), []).append(r)
            for month, month_rows in by_month.items():
                rel_path, sha256, row_count = snap.write_partition(out_dir, band, month, month_rows)
                partitions.append({"path": rel_path, "sha256": sha256, "row_count": row_count})

        station_rows = [s for s in stations if not (exclude_holdout and s["station_id"] in holdout)]
        snap.write_stations(out_dir, station_rows)

        # sample.csv illustrates the schema for anyone browsing the public
        # snapshot -- always a NON-holdout station (safe to include either
        # copy) so this one file doesn't have to vary between the two.
        # holdout.json itself stays private-copy-only (plan section 3.4:
        # "published only AFTER the cycle closes").
        sample_station = next(iter(sorted(s["station_id"] for s in stations if s["station_id"] not in holdout)), None)
        if sample_station:
            _write_sample_csv(os.path.join(out_dir, "sample.csv"), sample_station, band_rows)
        if not exclude_holdout:
            snap.write_holdout(out_dir, holdout)

        snap.write_manifest(
            out_dir, args.snapshot_version, partitions,
            generating_commit=args.generating_commit or "unknown",
            package_version=args.package_version or "unknown",
            license_info={"code": "Apache-2.0", "data": "CC-BY-4.0"},
            f7_caveat=(
                "Forecast bands (forecast_lead1..7) are temperature-only for 2023 -- "
                "dewpoint/wind/pressure previous_dayN has no data before ~2024-01, so "
                "humidity_source/wind_source='era5_fallback' for most 2023-dated forecast-band rows. "
                "Check humidity_source/wind_source per row rather than assuming band coverage."
            ),
            f3_disclosure=(
                "pop_density_per_km2 is LandScan Global over a ~1km station buffer "
                f"(pop_density_buffer_deg={_POP_DENSITY_BUFFER_DEG}), attributed to ORNL. Serving uses "
                "worldpop_pop_total / polygon_area_km2 instead -- a real train/serve mismatch in both "
                "numerator and denominator, disclosed here rather than reconciled."
            ),
        )
        logger.info("Wrote snapshot to %s (%d partition(s), exclude_holdout=%s)", out_dir, len(partitions), exclude_holdout)
        # manifest_sha256 (sha256 of the literal MANIFEST.json bytes,
        # defined in scripts/run_submission.py's own verify_snapshot
        # docstring) is what a contributor's manifest.yaml must pin -- print
        # it explicitly rather than making them compute it themselves.
        with open(os.path.join(out_dir, "MANIFEST.json"), "rb") as f:
            manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
        logger.info("manifest_sha256 for %s: %s", out_dir, manifest_sha256)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_snapshot(args.out_dir, exclude_holdout=False)

    if args.public_out_dir:
        os.makedirs(args.public_out_dir, exist_ok=True)
        _write_snapshot(args.public_out_dir, exclude_holdout=True)


_ALL_BANDS = ("era5",) + tuple(_FETCH_BAND_CONFIG)


def _freeze_predictions_for_band(adapter, rows: list[dict], model_version: str) -> list[dict]:
    """Run the REAL model (via QRFModelAdapter, loaded with private S3
    credentials) over one band's rows for both targets, called WITHOUT
    extra_zone_gate/bias_correction -- i.e. the base model's raw result,
    gated only by its own trained-on-ERA5 zones_passing_cv_gate. This is
    the exact contract contract.FrozenPredictionAdapter's own docstring
    requires: re-gating/bias-correction happen later, at score time,
    against this frozen baseline -- never bake them in here."""
    frozen: list[dict] = []
    for target in ("tmax", "tmin"):
        preds = adapter.predict(rows, target)
        for r, p in zip(rows, preds):
            frozen.append({
                "station_id": r["station_id"], "date": r["date"], "target": target,
                "delta_c": p["delta_c"], "ci95_c": p["ci95_c"], "confidence": p["confidence"],
                "applied": p["applied"], "out_of_distribution": p["out_of_distribution"],
                "covariates_missing": json.dumps(p["covariates_missing"]),
                "cv_gate_passed": p["cv_gate_passed"], "model_version": model_version,
            })
    return frozen


def _phase_predict(args: argparse.Namespace) -> None:
    """Add a frozen predictions/ subtree to an ALREADY-PACKED snapshot
    directory (--phase pack must have run first). Needs private S3
    credentials for contract.QRFModelAdapter.load -- unlike export/fetch/
    pack, this phase is maintainer-only by necessity, not just by
    convention. Only boto3/joblib/sklearn/quantile-forest are needed (this
    package's own pyproject.toml already pins these for exactly this use,
    per its own comment) -- no db/heat_calcs/open_meteo/api_call_manager."""
    from heatready_downscaling.contract import QRFModelAdapter

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    adapter = QRFModelAdapter.load(args.model_version, bucket=bucket)

    manifest = snap.read_manifest(args.snapshot_dir)
    partitions = list(manifest["partitions"])

    for band in _ALL_BANDS:
        rows = snap.read_band_partitions(args.snapshot_dir, band)
        if not rows:
            logger.warning("No rows for band=%s in %s -- skipping", band, args.snapshot_dir)
            continue
        frozen = _freeze_predictions_for_band(adapter, rows, args.model_version)
        by_month: dict[str, list[dict]] = {}
        for r in frozen:
            by_month.setdefault(r["date"].strftime("%Y-%m"), []).append(r)
        for month, month_rows in by_month.items():
            rel_path, sha256, row_count = snap.write_predictions_partition(
                args.snapshot_dir, args.model_version, band, month, month_rows,
            )
            partitions.append({"path": rel_path, "sha256": sha256, "row_count": row_count})
        logger.info("Froze %d prediction row(s) for band=%s, model=%s", len(frozen), band, args.model_version)

    snap.write_manifest(
        args.snapshot_dir, manifest["snapshot_version"], partitions,
        generating_commit=manifest["generating_commit"], package_version=manifest["package_version"],
        license_info=manifest["license"], f7_caveat=manifest["f7_caveat"], f3_disclosure=manifest["f3_disclosure"],
        release_url=manifest.get("release_url"), doi_url=manifest.get("doi_url"),
    )
    logger.info("Updated MANIFEST.json with %d total partition(s)", len(partitions))
    # --phase predict rewrites MANIFEST.json (new prediction partitions
    # added), so ITS manifest_sha256 -- not --phase pack's -- is the one a
    # contributor's manifest.yaml should actually pin, since predict is the
    # last step before publishing.
    with open(os.path.join(args.snapshot_dir, "MANIFEST.json"), "rb") as f:
        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info("manifest_sha256 for %s (post-predict, final): %s", args.snapshot_dir, manifest_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["export", "fetch", "pack", "predict"], required=True)
    parser.add_argument("--snapshot-version", default=None, help="required for --phase fetch/pack")
    parser.add_argument("--bucket", default=None,
                         help="defaults to VULNERABILITY_DATA_BUCKET env var (--phase export/predict)")
    parser.add_argument("--model-version", default=None, help="--phase predict only")
    parser.add_argument("--snapshot-dir", default=None,
                         help="--phase predict: an already-packed snapshot directory (from --phase pack)")
    parser.add_argument("--run-id", default=None, help="export-ghcn-training run_id (--phase export)")
    parser.add_argument("--rows-in", default=None, help="--phase fetch/pack: JSON from a prior --phase export run")
    parser.add_argument("--band", default=None, choices=sorted(_FETCH_BAND_CONFIG), help="--phase fetch only")
    parser.add_argument("--max-workers", type=int, default=16,
                         help="concurrent stations in flight (--phase fetch) -- higher than the tuned validation-"
                              "script default (4) since the paid Open-Meteo tier in use has no published rate "
                              "limit (verify empirically at small scale before a full run, see this module's docstring)")
    parser.add_argument("--checkpoint", default=None, help="--phase fetch: JsonlCheckpointStore path")
    parser.add_argument("--band-input", action="append", default=[], help="--phase pack: band=path, repeatable")
    parser.add_argument("--out", default=None, help="--phase export/fetch: output JSON path")
    parser.add_argument("--out-dir", default=None, help="--phase pack: full (private) snapshot directory")
    parser.add_argument("--public-out-dir", default=None,
                         help="--phase pack: if given, also write a holdout-excluded public copy here")
    parser.add_argument("--generating-commit", default=None, help="--phase pack: this repo's commit, for MANIFEST.json")
    parser.add_argument("--package-version", default=None, help="--phase pack: heatready_downscaling version, for MANIFEST.json")
    args = parser.parse_args()

    if args.phase == "export":
        if not args.run_id or not args.out:
            raise SystemExit("--phase export requires --run-id and --out")
        _phase_export(args)
    elif args.phase == "fetch":
        if not args.band or not args.rows_in or not args.out or not args.snapshot_version:
            raise SystemExit("--phase fetch requires --band, --rows-in, --out, --snapshot-version")
        _phase_fetch(args)
    elif args.phase == "pack":
        if not args.rows_in or not args.out_dir or not args.snapshot_version:
            raise SystemExit("--phase pack requires --rows-in, --out-dir, --snapshot-version")
        _phase_pack(args)
    elif args.phase == "predict":
        if not args.model_version or not args.snapshot_dir:
            raise SystemExit("--phase predict requires --model-version, --snapshot-dir")
        _phase_predict(args)


if __name__ == "__main__":
    main()
