"""Unit tests for scripts/build_gsod_bsh_rows.py's pure parsing/selection/assembly logic. No
network, CDS, S3 or DB: ghcn (a heat-risk-data-api module) is stubbed for assemble_rows."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_gsod_bsh_rows as g

GSOD_HEADER = ('"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","TEMP","TEMP_ATTRIBUTES",'
               '"MAX","MAX_ATTRIBUTES","MIN","MIN_ATTRIBUTES"\n')


def _gsod(*days):
    return GSOD_HEADER + "".join(
        f'"42647099999","{d}","23.07","72.63","57.6","AHMEDABAD, IN","80.0","24","{mx}","{fl}","{mn}"," "\n'
        for d, mx, fl, mn in days)


def test_parse_gsod_converts_fahrenheit_and_flags_derived_max():
    rows = g.parse_gsod_csv(_gsod(("2020-01-01", "  74.1", "*", "  55.0"), ("2020-01-02", "212.0", " ", "32.0")))
    assert rows[0] == {"date": "2020-01-01", "station_tmax_c": 23.39, "station_tmin_c": 12.78, "max_derived": True}
    assert len(rows) == 1  # 212 F = 100 C is outside the plausible range and dropped


def test_parse_gsod_drops_missing_and_inverted_days():
    rows = g.parse_gsod_csv(_gsod(("2020-01-01", "9999.9", "", "55.0"),
                                  ("2020-01-02", "90.0", "", "9999.9"),
                                  ("2020-01-03", "60.0", "", "70.0"),
                                  ("2020-01-04", "100.0", "", "80.0")))
    assert [r["date"] for r in rows] == ["2020-01-04"]


def test_parse_isd_history_filters_country_year_and_null_island():
    text = ('"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
            '"426470","99999","AHMEDABAD","IN","","VAAH","+23.077","+072.635","+0055.0","19730101","20260920"\n'
            '"426540","99999","OLD","IN","","","+23.2","+072.6","+0080.0","19730101","20101231"\n'
            '"000000","99999","ZERO","IN","","","+00.000","+000.000","","19730101","20260920"\n'
            '"416750","99999","MULTAN","PK","","OPMT","+30.203","+071.419","+0122.0","19730101","20260920"\n'
            '"999999","12345","NOID","IN","","","+23.0","+072.0","","19730101","20260920"\n')
    got = g.parse_isd_history(text, {"IN"}, 2016)
    assert [s["usaf"] for s in got] == ["426470"]
    assert got[0]["elevation_m"] == 55.0 and got[0]["fips"] == "IN"


def test_wmo_map_and_station_id():
    line = "IN005010600  23.0670   72.6330   55.0    AHMADABAD                      GSN     42647"
    wmo = g.parse_ghcnd_wmo_map(line + "\nUSW00012345  0 0 0 X" + " " * 70 + "99999", {"IN"})
    assert wmo == {"42647": "IN005010600"}
    assert g.station_id_for({"usaf": "426470", "fips": "IN"}, wmo) == "IN005010600"
    assert g.station_id_for({"usaf": "427480", "fips": "IN"}, wmo) == "ING427480"
    # A non-WMO USAF (doesn't end in 0) never maps, even if its first 5 digits collide.
    assert g.station_id_for({"usaf": "426471", "fips": "IN"}, wmo) == "ING426471"


def test_select_stations_counts_good_years():
    full = [{"date": f"2020-{m:02d}-{d:02d}"} for m in range(1, 13) for d in range(1, 29)]  # 336 days
    thin = [{"date": f"2021-01-{d:02d}"} for d in range(1, 29)]
    cands = [{"usaf": "A"}, {"usaf": "B"}]
    kept = g.select_stations(cands, {"A": full + thin, "B": thin}, min_days_per_year=300, min_good_years=1)
    assert [s["usaf"] for s in kept] == ["A"]
    assert kept[0]["coverage_by_year"] == {"2020": 336, "2021": 28} and kept[0]["good_years"] == 1


def test_assemble_rows_region_from_id_and_shift(monkeypatch):
    fake = types.SimpleNamespace(
        align_obs_window=lambda series, grid: 1,
        koppen_climate_zone=lambda lat, lon: "BSh",
        region_from_station_id=lambda sid: sid[:2].upper(),
        koppen_main_group_code_from_zone=lambda zone: {"BSh": 2}[zone],
    )
    monkeypatch.setitem(sys.modules, "ghcn", fake)
    stations = [{"station_id": "ING427480", "lat": 22.3, "lon": 73.2, "elevation_m": 40.0}]
    series = {"ING427480": [{"date": "2020-05-01", "station_tmax_c": 42.0, "station_tmin_c": 28.0},
                            {"date": "2020-05-02", "station_tmax_c": 43.0, "station_tmin_c": 29.0}]}
    grid = {"ING427480": {"2020-05-02": {"tmax": 40.0, "tmin": 27.5}}}
    rows, shifts = g.assemble_rows(stations, series, grid, {}, {}, {"ING427480": {"wc_built_frac": 0.5}})
    assert shifts == {"ING427480": 1}
    assert len(rows) == 1  # 2020-05-02 + 1 day has no grid value
    r = rows[0]
    assert (r["region"], r["climate_zone"], r["date"]) == ("IN", "BSh", "2020-05-01")
    assert r["delta_tmax_c"] == pytest.approx(2.0) and r["delta_tmin_c"] == pytest.approx(0.5)
    assert r["wc_built_frac"] == 0.5 and r["obs_window_shift_days"] == 1
    assert r["koppen_main_group_code"] == 2


def test_drop_colocated_keeps_the_longer_record():
    a = {"usaf": "420801", "lat": 21.75, "lon": 72.19}
    b = {"usaf": "428380", "lat": 21.752, "lon": 72.20}  # ~1 km from a
    c = {"usaf": "427300", "lat": 22.48, "lon": 69.12}
    series = {"420801": [0] * 10, "428380": [0] * 20, "427300": [0] * 5}
    kept, dropped = g.drop_colocated([a, b, c], series)
    assert [s["usaf"] for s in kept] == ["428380", "427300"]
    assert dropped == [("420801", "428380")]


def test_select_stations_never_keeps_a_station_with_no_days():
    assert g.select_stations([{"usaf": "A"}], {"A": []}, min_days_per_year=300, min_good_years=0) == []
