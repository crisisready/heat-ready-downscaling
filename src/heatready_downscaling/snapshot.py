"""
Band-paired training/validation snapshot: reader + writer. New in this
package (no private-repo equivalent) -- schema matches
docs/plan-2026-07-27-heatready-crowdsourced-model-improvement.md sections
6.1/6.2 in the private repo. See PROVENANCE.md.

Layout on disk (or after download from a GitHub Release / Zenodo DOI):

    snapshots/<version>/
    ├── MANIFEST.json
    ├── stations.parquet          one row per station
    ├── holdout.json              withheld station IDs -- published only
    │                             AFTER the scoring cycle closes
    ├── sample.csv                1 station x 9 bands x 365 days, committed
    ├── paired/band=<band>/month=<YYYY-MM>/part-0.parquet
    └── predictions/model=<model_version>/band=<band>/month=<YYYY-MM>/part-0.parquet
                                  frozen model output -- see contract.FrozenPredictionAdapter's
                                  own docstring for why this exists (Phase 3, 2026-07-27):
                                  QRFModelAdapter.load needs private S3 credentials no external
                                  contributor or CI job has, so scoring in that context reads
                                  these frozen per-row predictions instead of running live
                                  inference. Written by the maintainer only (from a machine that
                                  DOES have the private bucket's credentials), same as every other
                                  partition here -- contributors only ever read this directory.

Row-dict-in/row-dict-out on the read side (not a DataFrame-returning API)
is deliberate: it keeps score.py/features.py Parquet-unaware, matching how
they already operate on list-of-dict rows. Only this module and
scripts/build_band_paired_snapshot.py (not yet written -- Phase 2) need to
know Parquet exists at all. `pyarrow` stays an optional extra
(`heatready-downscaling[snapshot]`) for exactly that reason.
"""

import hashlib
import json
import os

SNAPSHOT_SCHEMA_VERSION = 1

_BANDS = ("era5", "lag_fill") + tuple(f"forecast_lead{n}" for n in range(1, 8))


def _pa_schema():
    """Lazy-built pyarrow schema -- matches plan section 6.1's column list
    exactly. A function, not a module-level constant, so importing this
    module never requires pyarrow unless a caller actually reads/writes a
    partition."""
    import pyarrow as pa

    return pa.schema([
        ("station_id", pa.string()),
        ("date", pa.date32()),
        ("band", pa.string()),  # partition key -- present as a column too for a self-sufficient partition
        # truth -- denormalized across bands, costs ~20MB, worth it: one
        # band partition is then self-sufficient for scoring
        ("station_tmax_c", pa.float64()),
        ("station_tmin_c", pa.float64()),
        # base value for THIS band
        ("grid_tmax_c", pa.float64()),
        ("grid_tmin_c", pa.float64()),
        ("grid_specific_humidity_kgkg", pa.float64()),
        ("nighttime_wind_ms", pa.float64()),
        # per-band provenance -- fixes F7 (forecast bands are temperature-
        # only for 2023; humidity/wind fall back to ERA5), prevents silent
        # divergence between contributors who don't realize a band's
        # humidity/wind is a fallback value.
        ("base_source", pa.string()),  # era5_land_cds | open_meteo_hfa | open_meteo_previous_runs
        ("base_model", pa.string()),  # null | "gfs_seamless"
        ("humidity_source", pa.string()),  # "band" | "era5_fallback"
        ("wind_source", pa.string()),  # "band" | "era5_fallback"
        # station identity / fold keys
        ("lon", pa.float64()),
        ("lat", pa.float64()),
        ("elevation_m", pa.float64()),
        ("region", pa.string()),  # GHCN FIPS -- the leave-region-out fold key
        ("climate_zone", pa.string()),  # free-form Koppen or latitude-band fallback
        ("koppen_main_group_code", pa.int16()),
        ("obs_window_shift_days", pa.int16()),
        # static covariates (constant per station)
        ("lst_warm_season_anomaly_c", pa.float64()),
        ("canopy_height_mean_m", pa.float64()),
        ("canopy_frac_over_3m", pa.float64()),
        ("wc_built_frac", pa.float64()),
        ("wc_tree_frac", pa.float64()),
        ("wc_water_frac", pa.float64()),
        ("ghsl_urban_fraction", pa.float64()),
        ("pop_density_per_km2", pa.float64()),
        ("elevation_rel_to_gridcell_m", pa.float64()),
        ("elevation_mean_m", pa.float64()),
        ("slope_deg", pa.float64()),
        ("aspect_deg", pa.float64()),
        # covariate provenance
        ("pop_density_source", pa.string()),  # "landscan_global"
        ("pop_density_buffer_deg", pa.float64()),  # 0.01
        ("lst_reference_radius_km", pa.float64()),  # 75.0
        ("snapshot_version", pa.string()),
    ])


def write_partition(snapshot_dir: str, band: str, month: str, rows: list[dict]) -> tuple[str, str, int]:
    """Write paired/band={band}/month={month}/part-0.parquet, ZSTD, sorted
    by (station_id, date) -- long format (not wide), since the program is
    per-band and monthly-append: a new lead or month is a new partition,
    never a rewrite. Sort ensures per-station constants dictionary/RLE-
    compress to nearly nothing.

    Returns (relative_path, sha256, row_count) for the caller to fold into
    MANIFEST.json via write_manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if band not in _BANDS:
        raise ValueError(f"unrecognized band {band!r} -- must be one of {_BANDS}")

    rel_dir = os.path.join("paired", f"band={band}", f"month={month}")
    abs_dir = os.path.join(snapshot_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    rel_path = os.path.join(rel_dir, "part-0.parquet")
    abs_path = os.path.join(snapshot_dir, rel_path)

    sorted_rows = sorted(rows, key=lambda r: (r["station_id"], r["date"]))
    table = pa.Table.from_pylist(sorted_rows, schema=_pa_schema())
    pq.write_table(table, abs_path, compression="zstd")

    sha256 = _file_sha256(abs_path)
    return rel_path, sha256, len(sorted_rows)


def read_band_partitions(snapshot_dir: str, band: str, months: list[str] | None = None) -> list[dict]:
    """Read and concatenate every paired/band={band}/month=*/part-0.parquet
    (or just the given months if provided), returning list[dict] -- the
    same row shape score.score_band/features.build_feature_matrix already
    expect, so callers never have to know Parquet exists."""
    import glob

    import pyarrow.parquet as pq

    band_dir = os.path.join(snapshot_dir, "paired", f"band={band}")
    if months is not None:
        paths = [os.path.join(band_dir, f"month={m}", "part-0.parquet") for m in months]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted(glob.glob(os.path.join(band_dir, "month=*", "part-0.parquet")))

    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


# Frozen model-prediction partitions (Phase 3, 2026-07-27) -- one row per
# (station_id, date, target), for a specific (model_version, band). Exists
# so scoring can happen with zero AWS access and zero sklearn/joblib pickle
# compatibility: contract.QRFModelAdapter.load needs private S3 credentials
# no external contributor or CI job has -- see contract.FrozenPredictionAdapter's
# own docstring for the full reasoning. `covariates_missing` is stored as a
# JSON-encoded string, not a native pyarrow list column -- keeps this
# partition's schema as uniformly scalar as the paired one above, and the
# list is only ever a handful of feature names, never large enough for that
# to matter for size or query performance.
def _predictions_pa_schema():
    import pyarrow as pa

    return pa.schema([
        ("station_id", pa.string()),
        ("date", pa.date32()),
        ("target", pa.string()),  # "tmax" | "tmin"
        ("delta_c", pa.float64()),  # null when not applied
        ("ci95_c", pa.float64()),  # null when not applied
        ("confidence", pa.string()),  # "high" | "medium" | "low" -- frozen AS COMPUTED at
                                        # snapshot-build time, not re-derived later, so a future
                                        # change to contract._confidence_class's thresholds can
                                        # never silently reinterpret an already-published snapshot
        ("applied", pa.bool_()),
        ("out_of_distribution", pa.bool_()),
        ("covariates_missing", pa.string()),  # JSON-encoded list[str], e.g. "[]" or '["slope_deg"]'
        ("cv_gate_passed", pa.bool_()),
        ("model_version", pa.string()),
    ])


def write_predictions_partition(
    snapshot_dir: str, model_version: str, band: str, month: str, rows: list[dict],
) -> tuple[str, str, int]:
    """Write predictions/model={model_version}/band={band}/month={month}/part-0.parquet.
    `rows` are the FROZEN per-row prediction dicts -- see contract.
    FrozenPredictionAdapter's own docstring for the exact contract this
    freezes (must be QRFModelAdapter.predict's own output, called with
    extra_zone_gate=None/bias_correction=None -- i.e. the base model's raw,
    uncorrected result -- never a result that already had a band-specific
    gate or bias correction baked in)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if band not in _BANDS:
        raise ValueError(f"unrecognized band {band!r} -- must be one of {_BANDS}")

    rel_dir = os.path.join("predictions", f"model={model_version}", f"band={band}", f"month={month}")
    abs_dir = os.path.join(snapshot_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    rel_path = os.path.join(rel_dir, "part-0.parquet")
    abs_path = os.path.join(snapshot_dir, rel_path)

    sorted_rows = sorted(rows, key=lambda r: (r["station_id"], r["date"], r["target"]))
    table = pa.Table.from_pylist(sorted_rows, schema=_predictions_pa_schema())
    pq.write_table(table, abs_path, compression="zstd")

    sha256 = _file_sha256(abs_path)
    return rel_path, sha256, len(sorted_rows)


def read_predictions_partitions(
    snapshot_dir: str, model_version: str, band: str, months: list[str] | None = None,
) -> list[dict]:
    """Read and concatenate every predictions/model={model_version}/band={band}/month=*/part-0.parquet
    (or just the given months). `covariates_missing` is JSON-decoded back
    into a real list[str] on the way out, mirroring QRFModelAdapter.predict's
    own return shape exactly."""
    import glob

    import pyarrow.parquet as pq

    band_dir = os.path.join(snapshot_dir, "predictions", f"model={model_version}", f"band={band}")
    if months is not None:
        paths = [os.path.join(band_dir, f"month={m}", "part-0.parquet") for m in months]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted(glob.glob(os.path.join(band_dir, "month=*", "part-0.parquet")))

    rows: list[dict] = []
    for path in paths:
        for r in pq.read_table(path).to_pylist():
            r["covariates_missing"] = json.loads(r["covariates_missing"])
            rows.append(r)
    return rows


def read_stations(snapshot_dir: str) -> list[dict]:
    """Read stations.parquet -- one row per station (lon/lat/elevation_m/
    region/climate_zone/static covariates), independent of any band."""
    import pyarrow.parquet as pq

    path = os.path.join(snapshot_dir, "stations.parquet")
    return pq.read_table(path).to_pylist()


def write_stations(snapshot_dir: str, rows: list[dict]) -> None:
    """Write stations.parquet -- one row per station_id. No fixed schema
    enforced here (unlike write_partition's per-band _pa_schema) -- the
    caller (build_band_paired_snapshot.py's pack phase) already dedupes a
    band partition's own per-station-constant columns (identity/fold keys
    + static covariates from plan section 6.1), so this just persists
    whatever dict shape it's given via pyarrow's own type inference, the
    same way read_stations' own test never assumes a fixed column set
    either. Sorted by station_id for the same reason write_partition sorts
    its own rows -- deterministic byte-for-byte output given the same
    input, so re-running pack on unchanged data reproduces an identical
    sha256."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sorted_rows = sorted(rows, key=lambda r: r["station_id"])
    table = pa.Table.from_pylist(sorted_rows)
    pq.write_table(table, os.path.join(snapshot_dir, "stations.parquet"), compression="zstd")


def compute_holdout(stations: list[dict], held_out_fraction: float = 0.15) -> set[str]:
    """Zone-stratified provisional-scoring holdout (plan section 3.4): per
    climate_zone, hold out max(1, round(held_out_fraction * n_zone_stations))
    stations, chosen deterministically by LOWEST md5(station_id) within that
    zone -- not a flat md5(station_id) % 100 < 15 over the whole corpus,
    which would let a large zone's stations dominate the cut and could zero
    out a thin zone entirely. Deliberately NOT salted by snapshot_version
    (unlike score.score_band's bias-CV fold assignment) -- this is a stable,
    release-over-release provisional holdout, not a per-submission anti-
    gaming fold; salting it would let the same station rotate in and out of
    "public training data" release over release, which is a data-leakage
    risk of its own kind (a station privately held out in v1 could appear
    in v2's public partitions with no one having decided that was safe).

    `stations` needs only "station_id" and "climate_zone" per row (e.g. the
    same rows read_stations returns, or the export's raw per-station rows
    before dedup). Callers exclude the returned IDs' rows from every public
    band partition entirely (plan section 3.4: "Holdout rows are excluded
    from the published snapshot entirely") -- this function only decides
    WHICH stations, not what to do with their rows."""
    import hashlib

    by_zone: dict[str, list[str]] = {}
    for s in stations:
        by_zone.setdefault(s["climate_zone"], []).append(s["station_id"])

    holdout: set[str] = set()
    for zone, station_ids in by_zone.items():
        n_holdout = max(1, round(held_out_fraction * len(station_ids)))
        ranked = sorted(station_ids, key=lambda sid: hashlib.md5(sid.encode()).hexdigest())
        holdout.update(ranked[:n_holdout])
    return holdout


def write_holdout(snapshot_dir: str, station_ids: set[str]) -> None:
    """Write holdout.json to `snapshot_dir` -- the caller is responsible for
    keeping this file OUT of whatever gets published to the public GitHub
    Release/Zenodo asset (plan section 6: "published only AFTER the cycle
    closes"). Writing it into the private S3 working copy alongside the
    full (including held-out-station) partitions is the intended use --
    the scoring harness (Phase 3) needs it to know which rows a candidate
    was never allowed to train against; the public release doesn't get
    this file (or those stations' rows at all) until a cycle actually closes."""
    with open(os.path.join(snapshot_dir, "holdout.json"), "w") as f:
        json.dump(sorted(station_ids), f, indent=2)


def read_holdout(snapshot_dir: str) -> set[str] | None:
    """Read holdout.json (withheld station IDs) if it exists -- returns
    None (not an empty set) when the file is absent, since this file is
    deliberately published only AFTER the scoring cycle closes; an empty
    set would falsely imply "no stations are held out" rather than "this
    information isn't public yet"."""
    path = os.path.join(snapshot_dir, "holdout.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return set(json.load(f))


def write_manifest(
    snapshot_dir: str,
    snapshot_version: str,
    partitions: list[dict],
    *,
    generating_commit: str,
    package_version: str,
    license_info: dict,
    f7_caveat: str,
    f3_disclosure: str,
    release_url: str | None = None,
    doi_url: str | None = None,
) -> None:
    """Write MANIFEST.json. `partitions` is a list of
    {"path": ..., "sha256": ..., "row_count": ...} dicts, one per
    write_partition call this snapshot build made -- the manifest's job is
    to pin exactly what a `manifest_sha256` reference in a submission
    manifest.yaml is checking against (verify_manifest below), so every
    partition's own hash must be recorded, not just an aggregate."""
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_version": snapshot_version,
        "generating_commit": generating_commit,
        "package_version": package_version,
        "partitions": partitions,
        "license": license_info,
        "f7_caveat": f7_caveat,
        "f3_disclosure": f3_disclosure,
        "release_url": release_url,
        "doi_url": doi_url,
    }
    with open(os.path.join(snapshot_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(snapshot_dir: str) -> dict:
    with open(os.path.join(snapshot_dir, "MANIFEST.json")) as f:
        return json.load(f)


def verify_manifest(snapshot_dir: str) -> None:
    """Re-hash every partition file listed in MANIFEST.json and compare
    against its recorded sha256 -- raises ValueError on any mismatch. This
    is the actual check behind `manifest_sha256`-pinning in a submission's
    manifest.yaml and promote_from_public.py's 'verify manifest_sha256'
    step -- a submission (or a scoring run) must never trust that a
    snapshot directory is what it claims to be without this."""
    manifest = read_manifest(snapshot_dir)
    mismatches = []
    for partition in manifest["partitions"]:
        abs_path = os.path.join(snapshot_dir, partition["path"])
        if not os.path.exists(abs_path):
            mismatches.append({"path": partition["path"], "error": "missing"})
            continue
        actual = _file_sha256(abs_path)
        if actual != partition["sha256"]:
            mismatches.append({"path": partition["path"], "expected": partition["sha256"], "actual": actual})
    if mismatches:
        raise ValueError(f"snapshot at {snapshot_dir!r} failed manifest verification: {mismatches}")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
