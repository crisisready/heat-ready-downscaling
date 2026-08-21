"""
Real AEMET OpenData (opendata.aemet.es) BSh training rows for the Valencia
region -- second real Spain source tonight, alongside ECA&D. SIAR (the
originally preferred source, denser national coverage) was confirmed
genuinely blocked: its registration requires Cl@ve, Spain's citizen
digital-identity system, unavailable to a non-Spanish-citizen (confirmed
live by the user directly, not assumed).

AEMET's real API is a two-step indirection: the first call (with an
api_key header) returns a JSON envelope {"estado", "datos": <url>,
"metadatos": <url>}; the actual payload is a SEPARATE fetch of the `datos`
URL, which is flaky/short-lived in practice (live-verified tonight: the
first attempt timed out with zero bytes, a same-URL retry a few seconds
later succeeded in under a second) -- retried with backoff here rather
than treated as a hard failure on the first miss.

Station coordinates are given in a real, unusual DMS-packed string format
(e.g. "384648N" = 38 deg 46' 48" N, "004745W" = 0 deg 47' 45" W -- lon
carries a 3-digit degree field, lat a 2-digit one) -- parsed explicitly,
not assumed to match any other source's format.

tmax/tmin values use Spanish-locale comma decimals ("37,2" not "37.2").

Reuses build_training_set.py's own already-tested Open-Meteo ERA5-Land
fetch/covariate-snapshot code, exactly like build_ecad_valencia_bsh_rows.py
-- only the station-observation source changes.

Station IDs prefixed "AE" + AEMET's own `indicativo` code, matching the
"EC"-for-ECA&D convention so ghcn.py's region/dedup logic never confuses
these with a real GHCN station.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import ghcn  # noqa: E402
from build_training_set import (  # noqa: E402
    fetch_era5_land_for_stations_via_openmeteo,
    snapshot_covariates_for_stations,
    _bucket_from_credentials,
    _landscan_from_credentials,
)

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)
BASE = "https://opendata.aemet.es/opendata"
CREDENTIALS_FILE = os.path.join(_REPO_ROOT, "credentials.yaml")
OUT_DIR = "/tmp/aemet_pull"


def _api_key():
    import yaml
    with open(CREDENTIALS_FILE) as f:
        creds = yaml.safe_load(f)
    return creds["aemet"]["api_key"]


def _fetch_json(url, api_key=None, retries=4):
    """AEMET's `datos` URLs are real but flaky -- a fresh URL can time out
    on the first attempt and succeed on an immediate retry (live-verified
    tonight). Retried with a short backoff, not treated as a hard failure
    on one miss."""
    headers = {"api_key": api_key, "Accept": "application/json"} if api_key else {}
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("latin-1"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(3)
    raise last_exc


def _dms_to_dd(dms, is_lon):
    """AEMET's packed DMS format: 2-digit-degree + 2-digit-minute +
    2-digit-second + hemisphere, e.g. "384648N" = 38 deg 46' 48" N,
    "004745W" = 0 deg 47' 45" W -- deg_len=2 for BOTH lat and lon
    (live-verified against 3 real known station locations before this
    script was run for real: an initial 3-digit-degree assumption for
    longitude, based on the string simply being one character longer,
    produced nonsense coordinates -- e.g. Jalance at -10.9 degrees W,
    mid-Atlantic -- while deg_len=2 for both matched all three real
    stations' actual known locations exactly). `is_lon` is kept as a
    parameter (unused in the formula itself) only because a caller
    reading this function's call sites should still see which axis each
    value is, not because the parse differs by axis."""
    hemi = dms[-1]
    digits = dms[:-1]
    deg_len = 2
    deg = int(digits[:deg_len])
    minute = int(digits[deg_len:deg_len + 2])
    sec = int(digits[deg_len + 2:deg_len + 4])
    dd = deg + minute / 60 + sec / 3600
    return -dd if hemi in ("S", "W") else dd


def _parse_es_float(s):
    if s is None or s == "" or s == "Ip":  # "Ip" = trace precipitation, not relevant here
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def fetch_all_stations(api_key):
    envelope = _fetch_json(f"{BASE}/api/valores/climatologicos/inventarioestaciones/todasestaciones", api_key)
    stations = _fetch_json(envelope["datos"])
    return stations


def fetch_station_daily(indicativo, api_key, start, end):
    """AEMET's date-range endpoint real, live-verified cap: "El rango de
    fechas no puede ser superior a 6 meses" (estado 404) on a genuine
    full-year request -- an initial assumption of a multi-year cap (based
    on general familiarity with similar gov APIs, never actually checked
    against this one) was wrong. Chunked into <=180-day windows and
    concatenated; each chunk is its own real two-step fetch."""
    results = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=179), end)
        fecha_ini = f"{chunk_start.isoformat()}T00:00:00UTC"
        fecha_fin = f"{chunk_end.isoformat()}T23:59:59UTC"
        url = f"{BASE}/api/valores/climatologicos/diarios/datos/fechaini/{fecha_ini}/fechafin/{fecha_fin}/estacion/{indicativo}"
        envelope = _fetch_json(url, api_key)
        if envelope.get("estado") == 200 and envelope.get("datos"):
            results.extend(_fetch_json(envelope["datos"]))
        chunk_start = chunk_end + timedelta(days=1)
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    api_key = _api_key()

    print("fetching real AEMET station inventory...")
    all_stations = fetch_all_stations(api_key)
    print(f"real station(s) fleet-wide: {len(all_stations)}")

    import kgcpy
    bsh_stations = []
    for s in all_stations:
        try:
            lat = _dms_to_dd(s["latitud"], is_lon=False)
            lon = _dms_to_dd(s["longitud"], is_lon=True)
        except Exception:
            continue
        # Scoped to the coastal Mediterranean corridor (same real, disclosed
        # bbox as the ECA&D pull) -- avoids classifying AEMET's many inland
        # mountain stations (Ademuz, Utiel etc., real but a different,
        # non-BSh climate) one by one against kgcpy for no benefit.
        if not (36.0 <= lat <= 40.0 and -1.0 <= lon <= 1.0):
            continue
        zone = ghcn.koppen_climate_zone(lat, lon)
        if zone == "BSh":
            bsh_stations.append({
                "station_id": f"AE{s['indicativo']}", "aemet_id": s["indicativo"],
                "lat": lat, "lon": lon, "elevation_m": _parse_es_float(s.get("altitud")),
                "name": s.get("nombre"),
            })
    print(f"real BSh station(s) in the coastal corridor: {len(bsh_stations)}")
    for s in bsh_stations:
        print(f"  {s['station_id']} {s['name']} lat={s['lat']:.4f} lon={s['lon']:.4f}")

    with open(f"{OUT_DIR}/aemet_bsh_stations.json", "w") as f:
        json.dump(bsh_stations, f, indent=2)

    print("fetching real daily climatological data per station...")
    ghcn_by_station = {}
    for s in bsh_stations:
        sid = s["station_id"]
        try:
            daily = fetch_station_daily(s["aemet_id"], api_key, START_DATE, END_DATE)
        except Exception as exc:
            print(f"  {sid}: FAILED {exc!r}")
            continue
        series = []
        for row in daily:
            tmax = _parse_es_float(row.get("tmax"))
            tmin = _parse_es_float(row.get("tmin"))
            if tmax is None or tmin is None:
                continue
            series.append({"date": row["fecha"], "station_tmax_c": tmax, "station_tmin_c": tmin})
        if series:
            ghcn_by_station[sid] = series
        print(f"  {sid} {s['name']}: {len(series)} real quality day(s)")

    stations = [
        {"station_id": s["station_id"], "lat": s["lat"], "lon": s["lon"],
         "elevation_m": s["elevation_m"], "name": s["name"]}
        for s in bsh_stations if s["station_id"] in ghcn_by_station
    ]
    print(f"real station(s) with usable data (fetch scoped to these): {len(stations)}")
    if not stations:
        print("no usable stations -- stopping before the covariate/ERA5 fetch (stations_bbox "
              "crashes on an empty list, a confusing failure mode for what's really just zero data)")
        return

    vuln_bucket = _bucket_from_credentials()
    landscan_bucket, landscan_key = _landscan_from_credentials()

    print("fetching real Open-Meteo ERA5-Land grid values...")
    grid_by_station, humidity_by_station, nighttime_wind_by_station = fetch_era5_land_for_stations_via_openmeteo(
        stations, START_DATE, END_DATE, checkpoint_path=f"{OUT_DIR}/era5_checkpoint.jsonl",
    )

    print("snapshotting real covariates...")
    covariates_by_station = snapshot_covariates_for_stations(
        stations, vuln_bucket, batch_label="ES_valencia_corridor_aemet_bsh",
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
                "region": "AEMET_ES", "climate_zone": climate_zone,
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
    with open(f"{OUT_DIR}/aemet_bsh_rows.json", "w") as f:
        json.dump({"rows": rows, "row_count": len(rows), "complete": True}, f)
    print(f"wrote {OUT_DIR}/aemet_bsh_rows.json")


if __name__ == "__main__":
    main()
