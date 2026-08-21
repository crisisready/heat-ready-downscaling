"""
Real ECA&D-sourced BSh training rows for the Valencia coastal Mediterranean
corridor (Almeria -> Alicante -> Valencia), 41 real stations confirmed BSh
via live kgcpy lookup (NOT the Csa classification PHASE4_ECAD_REAL_SPAIN_
EVIDENCE.md assumed -- that doc predates this session's direct verification).

Reuses build_training_set_openmeteo.py's own already-tested
fetch_era5_land_for_stations_via_openmeteo/snapshot_covariates_for_stations
(same Open-Meteo grid fetch, same real covariate snapshot pipeline used for
every other station written to ghcn_training tonight) and ghcn.py's own
align_obs_window/koppen_climate_zone/upsert_ghcn_training_rows -- only the
station-observation SOURCE changes (ECA&D's keyless bulk blend files instead
of GHCN-Daily), matching build_rows_for_country's exact row-assembly shape
so downstream code (build_feature_matrix, eval_spatial_ranking_clusters.py)
needs zero changes to consume these rows.

Station IDs prefixed "EC" + 6-digit ECA&D STAID -- deliberately NOT a real
GHCN 2-letter country prefix, so ghcn.region_from_station_id/dedup logic
never confuses these with a real GHCN station.
"""
import json
import re
import sys
import zipfile
from datetime import date, timedelta

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import ghcn  # noqa: E402
from build_training_set_openmeteo import (  # noqa: E402
    fetch_era5_land_for_stations_via_openmeteo,
    snapshot_covariates_for_stations,
    _bucket_from_credentials,
    _landscan_from_credentials,
)

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)
ZIP_DIR = "/tmp/ecad_pull"
BSH_STATIONS_JSON = "/tmp/ecad_pull/ecad_bsh_stations.json"


def _dms_to_dd(s):
    sign = -1 if s.strip().startswith("-") else 1
    d, m, sec = (int(x) for x in s.strip().lstrip("+-").split(":"))
    return sign * (d + m / 60 + sec / 3600)


def load_ecad_valencia_bsh_stations():
    """The 41 real BSh stations already identified this session (Cyprus +
    Canary Islands + the real coastal Mediterranean corridor Almeria ->
    Alicante -> Valencia). Scoped here to just the MAINLAND coastal corridor
    (lat 36-40, lon -3 to 0) -- the geographically/climatically coherent,
    Valencia-relevant subset; Canary Islands stations are real BSh too but a
    different (volcanic-island, trade-wind) sub-regime, out of scope for this
    specific pull (a real, disclosed choice, not an oversight)."""
    all_bsh = json.load(open(BSH_STATIONS_JSON))
    corridor = [s for s in all_bsh if s["cn"] == "ES" and 36.0 <= s["lat"] <= 40.0 and -3.0 <= s["lon"] <= 0.0]
    return corridor


def parse_ecad_series(zip_path, member_prefix, staid, element_col):
    """Parse one real ECA&D blended-series file directly from the zip
    (no full extraction needed) into {date_iso: value_c}, quality-filtered
    to Q=0 (valid) only, matching ecad_common.py's own convention."""
    fname = f"{member_prefix}{staid:06d}.txt"
    with zipfile.ZipFile(zip_path) as zf:
        try:
            raw = zf.read(fname).decode("latin-1")
        except KeyError:
            return {}
    out = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not re.match(r"^\d+$", parts[0]):
            continue
        _staid, _souid, date_s, val_s, q_s = parts
        if q_s != "0":
            continue
        try:
            d = date.fromisoformat(f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}")
        except ValueError:
            continue
        if not (START_DATE <= d <= END_DATE):
            continue
        val = int(val_s)
        if val == -9999:
            continue
        out[d.isoformat()] = val / 10.0
    return out


def main():
    corridor = load_ecad_valencia_bsh_stations()
    print(f"real corridor BSh station(s): {len(corridor)}")

    stations = [
        {"station_id": f"EC{s['staid']:06d}", "lat": s["lat"], "lon": s["lon"], "elevation_m": None, "name": s["name"]}
        for s in corridor
    ]

    ghcn_by_station = {}
    for s in corridor:
        sid = f"EC{s['staid']:06d}"
        tx = parse_ecad_series(f"{ZIP_DIR}/tx.zip", "TX_STAID", s["staid"], "TX")
        tn = parse_ecad_series(f"{ZIP_DIR}/tn.zip", "TN_STAID", s["staid"], "TN")
        common_dates = sorted(set(tx) & set(tn))
        series = [{"date": d, "station_tmax_c": tx[d], "station_tmin_c": tn[d]} for d in common_dates]
        if series:
            ghcn_by_station[sid] = series
        print(f"  {sid} {s['name']}: {len(series)} real quality-valid day(s)")

    vuln_bucket = _bucket_from_credentials()
    landscan_bucket, landscan_key = _landscan_from_credentials()

    print("fetching real Open-Meteo ERA5-Land grid values...")
    grid_by_station, humidity_by_station, nighttime_wind_by_station = fetch_era5_land_for_stations_via_openmeteo(
        stations, START_DATE, END_DATE, checkpoint_path="/tmp/ecad_pull/era5_checkpoint.jsonl",
    )

    print("snapshotting real covariates...")
    covariates_by_station = snapshot_covariates_for_stations(
        stations, vuln_bucket, batch_label="ES_valencia_corridor_ecad_bsh",
        landscan_bucket=landscan_bucket, landscan_key=landscan_key,
    )

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
        climate_zone = ghcn.koppen_climate_zone(s["lat"], s["lon"])
        humidity_series = humidity_by_station.get(sid, {})
        nighttime_wind_series = nighttime_wind_by_station.get(sid, {})

        for obs in station_series:
            obs_day = date.fromisoformat(obs["date"])
            shifted_day = (obs_day + timedelta(days=shift)).isoformat()
            grid_vals = grid_series.get(shifted_day)
            if grid_vals is None:
                continue
            grid_tmax, grid_tmin = grid_vals["tmax"], grid_vals["tmin"]
            rows.append({
                "station_id": sid,
                "date": obs["date"],
                "lon": s["lon"], "lat": s["lat"], "elevation_m": s["elevation_m"],
                "region": "ECAD_ES", "climate_zone": climate_zone,
                "station_tmax_c": obs["station_tmax_c"], "station_tmin_c": obs["station_tmin_c"],
                "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin,
                "delta_tmax_c": obs["station_tmax_c"] - grid_tmax,
                "delta_tmin_c": obs["station_tmin_c"] - grid_tmin,
                "grid_specific_humidity_kgkg": humidity_series.get(shifted_day),
                "nighttime_wind_ms": nighttime_wind_series.get(shifted_day),
                "obs_window_shift_days": shift,
                **covariates,
            })

    print(f"real assembled row(s): {len(rows)}")
    zones = sorted(set(r["climate_zone"] for r in rows))
    print(f"real climate_zone(s) present: {zones}")
    json.dump({"rows": rows, "row_count": len(rows), "complete": True}, open("/tmp/ecad_pull/ecad_bsh_rows.json", "w"))
    print("wrote /tmp/ecad_pull/ecad_bsh_rows.json")


if __name__ == "__main__":
    main()
