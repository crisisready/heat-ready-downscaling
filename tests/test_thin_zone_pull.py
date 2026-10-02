import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from build_thin_zone_rows import coverage_by_year, read_stations  # noqa: E402
from scorecard_build_truth_set import airport_tag, hav_km  # noqa: E402
from scorecard_extend_truth_set import extend  # noqa: E402


def _v1():
    return [{"station_id": "AA1", "lat": "10.0", "lon": "20.0", "zone": "Aw", "zone_group": "tropical", "origin": "x",
             "source": "s", "airport": "no", "setting": "", "region": "", "visibility": "public", "name": ""}]


def _sel(sid, lat, lon, name="TOWN"):
    return {"id": sid, "lat": str(lat), "lon": str(lon), "name": name}


def test_extend_adds_new_and_skips_duplicates_and_unbuilt():
    selected = [_sel("N1", 11.0, 21.0, "TOWN AIRPORT"), _sel("N2", 10.0001, 20.0001), _sel("N3", 12.0, 22.0),
                _sel("AA1", 10.0, 20.0)]
    payload = {"rows_by_station": {"N1": 5, "N2": 5, "AA1": 5}}
    out, skipped = extend(_v1(), selected, payload, lambda la, lo: "Aw", lambda z: "tropical", airport_tag, hav_km)
    assert [r["station_id"] for r in out] == ["AA1", "N1"]
    assert out[1]["airport"] == "yes" and out[1]["origin"] == "thin_zone_pull_2026-10-02"
    reasons = dict(skipped)
    assert reasons["N2"].startswith("within 1 km") and reasons["N3"] == "no rows built" and reasons["AA1"] == "already in v1 truth"


def test_read_stations_filters_and_rejects_duplicates(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,zone,lat,lon,elev,name\nA,Aw,1,2,-999,X\nB,Aw,3,4,12.5,Y\n")
    got = read_stations(str(p), {"B"})
    assert [s["station_id"] for s in got] == ["B"] and got[0]["elevation_m"] == 12.5 and got[0]["fips"] == "B"
    assert read_stations(str(p))[0]["elevation_m"] is None
    p.write_text("id,zone,lat,lon,elev,name\nA,Aw,1,2,1,X\nA,Aw,3,4,1,Y\n")
    with pytest.raises(ValueError):
        read_stations(str(p))


def test_coverage_by_year():
    assert coverage_by_year([{"date": "2023-01-01"}, {"date": "2023-01-02"}, {"date": "2025-01-01"}]) == {"2023": 2, "2025": 1}


def test_plan_clusters_labels_and_order():
    from build_thin_zone_rows import plan_clusters
    stations = [{"station_id": "AAA", "fips": "US"}, {"station_id": "BBB", "fips": "US"}, {"station_id": "CCC", "fips": "CH"}]
    fetch = [{"station_id": s["station_id"]} for s in stations]
    plan = plan_clusters(stations, fetch, lambda c: [c[:1], c[1:]] if len(c) == 2 else [c])
    assert [l for l, _ in plan] == ["thinzone_CH", "thinzone_US_c0", "thinzone_US_c1"]


def test_run_cluster_covariates_resumes_from_saved_clusters(tmp_path):
    import json
    from build_thin_zone_rows import run_cluster_covariates
    cdir = tmp_path / "clusters"
    cdir.mkdir()
    (cdir / "thinzone_US.json").write_text(json.dumps({"AAA": {"lst_warm_season_anomaly_c": 1.5}}))
    cov, failed = run_cluster_covariates([("thinzone_US", [{"station_id": "AAA"}])], str(tmp_path), workers=1)
    assert cov == {"AAA": {"lst_warm_season_anomaly_c": 1.5}} and failed == []
