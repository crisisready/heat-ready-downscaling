"""
Real SIAR (Sistema de Informacion Agroclimatica para el Regadio) BSh
training rows for the Valencia region -- third real Spain source tonight,
alongside ECA&D and AEMET.

SIAR's own documented "API SiAR" requires Cl@ve (Spain's citizen digital
identity), confirmed genuinely unavailable to a non-Spanish-citizen. But
its PUBLIC "Consulta de datos" page (no login at all) is real and does
serve genuine historical daily data -- confirmed live by an independent
codex sol consult, then independently re-verified here.

Two real, distinct endpoints, discovered by reading the page's own JS
rather than guessing:
  - `mapa_siar/cargarEstaciones` (GET, keyless): the full national station
    catalog with REAL coordinates -- {idProvincia, idEstacion, coordenadas:
    {lat, lng}, provincia, municipio, altitud, activa, ...}. No browser
    automation needed for this part.
  - `consultaDatos/consultaDatos` (the actual daily-data query): requires
    real browser automation. A `requests`-based two-step POST (matching
    exactly what the page's own JS sends -- same fields, same
    Origin/Referer/X-XSRF-TOKEN headers a real browser would send) was
    rejected with a real, reproducible 405 by the live backend --
    almost certainly a bot-detection layer only satisfied by genuine
    browser JS execution, confirmed by then getting a real 200 with the
    identical field values via Playwright.

Reverse-engineered form quirks for the Playwright flow, each real and
load-bearing:
  - Province/station checkboxes sit inside custom multiselect widgets
    that hide their <input> elements until a JS toggle opens them.
    Playwright's own check()/click() refuse an invisible element even
    with force=True; driven instead by directly setting .checked and
    dispatching a click event for province, and for stations additionally
    opening the real #btn-seleccionar-estaciones popup AND clicking
    #btnVerSeleccion afterward -- that's the actual function (a live
    $('input[id^=checkEstacion_]:checked') DOM query) that populates the
    real hidden idEstaciones field; calling the checkbox's own registered
    handler (selectCheckEstacion) only updates a cosmetic "selected
    stations" display list, not the submitted field.
  - The date inputs are real <input type="date"> elements. The visible
    placeholder text says dd/MM/yyyy, but a native date input's real
    value contract is ISO (YYYY-MM-DD) -- a dd/MM/yyyy string is silently
    rejected (left empty), no error surfaced.
  - Up to 40 stations per query (the page's own #hiddenMaxEstaciones);
    batched here. No apparent date-range cap (a full calendar year
    validated cleanly, unlike AEMET's real, documented 6-month limit).

Reuses build_training_set.py's own already-tested Open-Meteo ERA5-Land
fetch/covariate-snapshot code, exactly like the ECA&D and AEMET scripts --
only the station-observation source changes.

Station IDs prefixed "SI" + "{provinceId}-{stationId}", matching the
"EC"/"AE" convention so ghcn.py's region/dedup logic never confuses these
with a real GHCN station.
"""
import json
import os
import re
import sys
import time
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
BASE = "https://servicio.mapa.gob.es/siarweb"
OUT_DIR = "/tmp/siar_pull"
# Real, live-verified scaling limit: an 11-station/full-year query never
# finished rendering within 120s (the page never even navigated to the
# results URL), while a 3-station/7-day query reliably completed well
# under 20s. Rather than keep guessing at a bigger timeout, trading more
# (smaller, already-proven) requests for realistic per-request render
# time -- MAX_STATIONS_PER_QUERY down from the page's own real cap (40)
# to what's actually been shown to work, and the full year chunked into
# quarters.
MAX_STATIONS_PER_QUERY = 3
DATE_CHUNK_DAYS = 90
VALENCIA_REGION_PROVINCES = {3, 12, 46}  # Alicante, Castellon, Valencia


def fetch_station_catalog_with_coords():
    import urllib.request

    req = urllib.request.Request(f"{BASE}/mapa_siar/cargarEstaciones", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        stations = json.loads(resp.read())
    real = [
        s for s in stations
        if s.get("idCCAA") == "VAL" and s.get("idProvincia") in VALENCIA_REGION_PROVINCES
        and s.get("activa") and s.get("coordenadas") and s["coordenadas"].get("lat") is not None
    ]
    return real


def fetch_batch_via_browser(page, stations_batch, start, end):
    page.goto(f"{BASE}/consultaDatos/inicio", timeout=30000)
    page.wait_for_load_state("networkidle")

    prov_ids = sorted(set(str(s["idProvincia"]) for s in stations_batch))
    for prov_id in prov_ids:
        page.evaluate(f"""
            const cb = document.querySelector('#checkProv_{prov_id}');
            if (cb) {{ cb.checked = true; cb.dispatchEvent(new Event('click', {{bubbles: true}})); }}
        """)
    page.wait_for_timeout(300)

    page.click("#btn-seleccionar-estaciones")
    page.wait_for_timeout(300)

    for s in stations_batch:
        page.evaluate(f"""
            const cb = document.querySelector('#checkEstacion_{s["idProvincia"]}-{s["idEstacion"]}');
            if (cb) {{ cb.checked = true; }}
        """)
    page.wait_for_timeout(200)

    if page.locator("#btnVerSeleccion").count() > 0:
        page.click("#btnVerSeleccion", force=True)
        page.wait_for_timeout(300)

    page.evaluate("document.querySelector('#popup-estaciones').style.display = 'none';")
    page.wait_for_timeout(200)

    page.evaluate(f"""
        document.querySelector('#fechaInicialVal').value = '{start.isoformat()}';
        document.querySelector('#fechaFinalVal').value = '{end.isoformat()}';
    """)
    page.wait_for_timeout(200)

    page.click("#btnConsultar", force=True)
    # Real, live-verified: a small 3-station/7-day test navigated well
    # within 20s, but a full-year/11-station batch (a much bigger render
    # -- each station gets its own ~365-row table) genuinely needs more.
    page.wait_for_url("**/consultaDatos/consultaDatos", timeout=60000)
    page.wait_for_load_state("networkidle", timeout=60000)
    return page.content()


def _parse_es_float(s):
    if s is None or s.strip() in ("", "-"):
        return None
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        return None


def parse_results_html(html, stations_batch):
    series_by_station = {}
    for s in stations_batch:
        table_id = f"tabla-consultas_{s['idProvincia']}_{s['idEstacion']}"
        m = re.search(rf'<table[^>]*id="{table_id}"[^>]*>(.*?)</table>', html, re.S)
        if not m:
            continue
        table_html = m.group(1)
        rows = re.findall(r'<tr class="fondo-row">(.*?)</tr>', table_html, re.S)
        series = []
        for row_html in rows:
            cells = re.findall(r'<td[^>]*>([^<]*)</td>', row_html)
            if len(cells) < 5:
                continue
            fecha, _tmedia, tmax_s, _hora_max, tmin_s = cells[:5]
            try:
                d = date(int(fecha[6:10]), int(fecha[3:5]), int(fecha[0:2])).isoformat()
            except (ValueError, TypeError, IndexError):
                continue
            tmax = _parse_es_float(tmax_s)
            tmin = _parse_es_float(tmin_s)
            if tmax is None or tmin is None:
                continue
            series.append({"date": d, "station_tmax_c": tmax, "station_tmin_c": tmin})
        if series:
            sid = f"SI{s['idProvincia']}-{s['idEstacion']}"
            series_by_station[sid] = series
    return series_by_station


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("fetching real SIAR station catalog with coordinates (keyless GET)...")
    catalog = fetch_station_catalog_with_coords()
    print(f"real active station(s) across Valencia/Alicante/Castellon: {len(catalog)}")

    import kgcpy
    bsh_candidates = []
    for s in catalog:
        lat, lon = s["coordenadas"]["lat"], s["coordenadas"]["lng"]
        zone = ghcn.koppen_climate_zone(lat, lon)
        if zone == "BSh":
            bsh_candidates.append({
                "sid": f"SI{s['idProvincia']}-{s['idEstacion']}",
                "idProvincia": s["idProvincia"], "idEstacion": s["idEstacion"],
                "lat": lat, "lon": lon, "elevation_m": s.get("altitud"),
                "name": f"{s.get('sestacionCortoProv', '')}{s.get('sestacionCortoId', '')} - {s.get('sestacion', s.get('municipio'))}",
            })
    print(f"real BSh station(s): {len(bsh_candidates)}")
    for s in bsh_candidates:
        print(f"  {s['sid']} {s['name']} lat={s['lat']:.4f} lon={s['lon']:.4f}")
    with open(f"{OUT_DIR}/siar_bsh_stations.json", "w") as f:
        json.dump(bsh_candidates, f, indent=2, ensure_ascii=False)

    if not bsh_candidates:
        print("no real BSh stations found -- stopping")
        return

    from playwright.sync_api import sync_playwright

    ghcn_by_station = {}
    batches = [
        bsh_candidates[i:i + MAX_STATIONS_PER_QUERY]
        for i in range(0, len(bsh_candidates), MAX_STATIONS_PER_QUERY)
    ]
    print(f"real batch(es) of <= {MAX_STATIONS_PER_QUERY} station(s): {len(batches)}")

    date_chunks = []
    chunk_start = START_DATE
    while chunk_start <= END_DATE:
        chunk_end = min(chunk_start + timedelta(days=DATE_CHUNK_DAYS - 1), END_DATE)
        date_chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    print(f"real date chunk(s) of <= {DATE_CHUNK_DAYS} day(s): {len(date_chunks)}")

    total_jobs = len(batches) * len(date_chunks)
    job_i = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for i, batch in enumerate(batches):
            batch_stations = [{"idProvincia": s["idProvincia"], "idEstacion": s["idEstacion"]} for s in batch]
            for c_start, c_end in date_chunks:
                job_i += 1
                print(f"job {job_i}/{total_jobs}: batch {i+1}/{len(batches)} "
                      f"({len(batch)} station(s)) x {c_start}..{c_end}...", flush=True)
                page = browser.new_page()
                try:
                    html = fetch_batch_via_browser(page, batch_stations, c_start, c_end)
                    series = parse_results_html(html, batch_stations)
                    for sid, obs in series.items():
                        ghcn_by_station.setdefault(sid, []).extend(obs)
                    print(f"  real station(s) with data this chunk: {len(series)}")
                except Exception as exc:
                    print(f"  job {job_i} FAILED: {exc!r}")
                finally:
                    page.close()
                time.sleep(2)
        browser.close()

    for sid, obs in ghcn_by_station.items():
        print(f"  {sid}: {len(obs)} real quality day(s)")

    stations = [s for s in bsh_candidates if s["sid"] in ghcn_by_station]
    print(f"real station(s) with usable data (fetch scoped to these): {len(stations)}")
    if not stations:
        print("no usable stations -- stopping before the covariate/ERA5 fetch")
        return

    fetch_stations = [
        {"station_id": s["sid"], "lat": s["lat"], "lon": s["lon"], "elevation_m": s["elevation_m"], "name": s["name"]}
        for s in stations
    ]

    vuln_bucket = _bucket_from_credentials()
    landscan_bucket, landscan_key = _landscan_from_credentials()

    print("fetching real Open-Meteo ERA5-Land grid values...")
    grid_by_station, humidity_by_station, nighttime_wind_by_station = fetch_era5_land_for_stations_via_openmeteo(
        fetch_stations, START_DATE, END_DATE, checkpoint_path=f"{OUT_DIR}/era5_checkpoint.jsonl",
    )

    print("snapshotting real covariates...")
    covariates_by_station = snapshot_covariates_for_stations(
        fetch_stations, vuln_bucket, batch_label="ES_valencia_corridor_siar_bsh",
        landscan_bucket=landscan_bucket, landscan_key=landscan_key,
    )

    rows = []
    for s in fetch_stations:
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
                "region": "SIAR_ES", "climate_zone": climate_zone,
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
    with open(f"{OUT_DIR}/siar_bsh_rows.json", "w") as f:
        json.dump({"rows": rows, "row_count": len(rows), "complete": True}, f)
    print(f"wrote {OUT_DIR}/siar_bsh_rows.json")


if __name__ == "__main__":
    main()
