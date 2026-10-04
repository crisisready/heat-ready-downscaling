import json
from datetime import date
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from heatready_downscaling import grid_bias_prior as gbp  # noqa: E402
from heatready_downscaling.features import (FEATURE_ORDER, FEATURE_ORDER_GRIDBIAS,  # noqa: E402
                                            SUPPORTED_FEATURE_ORDERS, build_feature_matrix)


def _rows(sid, lat, lon, month, delta, n=12):
    return [(sid, lat, lon, f"2023-{month:02d}-{d + 1:02d}", delta) for d in range(n)]


def _table(*groups):
    rows = [r for g in groups for r in g]
    return gbp.station_month_table(*zip(*rows))


def test_station_month_table_needs_min_days_and_skips_nonfinite():
    rows = _rows("A", 40.0, 120.0, 7, 1.0) + _rows("B", 40.5, 120.0, 7, 2.0, n=gbp.MIN_DAYS - 1)
    rows += [("A", 40.0, 120.0, "2023-07-30", float("nan")), ("A", 40.0, 120.0, "2023-07-31", None)]
    t = gbp.station_month_table(*zip(*rows))
    assert list(t[7][3]) == ["A"] and t[7][2][0] == pytest.approx(1.0)


def test_prior_skips_own_site_and_shrinks_with_distance():
    t = _table(_rows("A", 40.0, 120.0, 7, 2.0), _rows("B", 41.0, 120.0, 7, -1.0))
    # at A's own site only B (111 km) counts
    p, w = gbp.prior_at(t, 40.0, 120.0, 7)
    wb = np.exp(-0.5 * (111.2 / gbp.LENGTH_KM) ** 2)
    assert w == pytest.approx(wb, rel=1e-2) and p == pytest.approx(-1.0 * wb / (wb + gbp.SHRINK_WEIGHT), rel=1e-2)
    # beyond the radius, and in a month with no table: no prior
    assert gbp.prior_at(t, 10.0, 120.0, 7) == (0.0, 0.0)
    assert gbp.prior_at(t, 40.5, 120.0, 1) == (0.0, 0.0)


def test_priors_for_rows_and_json_round_trip():
    t = _table(_rows("A", 40.0, 120.0, 7, 2.0), _rows("B", 40.3, 120.0, 7, 1.0))
    t2 = gbp.from_json(json.loads(json.dumps(gbp.to_json(t))))
    p1, w1 = gbp.priors_for_rows(t, [40.1, 40.1], [120.0, 120.0], ["2023-07-05", "2023-07-06"])
    p2, w2 = gbp.priors_for_rows(t2, [40.1], [120.0], ["2023-07-05"])
    assert p1[0] == p1[1] == pytest.approx(p2[0]) and w1[0] == pytest.approx(w2[0])
    assert 1.0 < p1[0] / (w1[0] / (w1[0] + gbp.SHRINK_WEIGHT)) < 2.0  # an average of 1 and 2, shrunk


def test_gridbias_feature_order_is_supported_and_built():
    assert SUPPORTED_FEATURE_ORDERS["gridbias"] == FEATURE_ORDER_GRIDBIAS == FEATURE_ORDER + gbp.FEATURES
    row = {c: 1.0 for c in ("grid_tmax_c", "grid_tmin_c", "lat", "lon", "lst_warm_season_anomaly_c",
                            "canopy_height_mean_m", "canopy_frac_over_3m", "wc_built_frac", "wc_tree_frac",
                            "wc_water_frac", "ghsl_urban_fraction", "pop_density_per_km2",
                            "elevation_rel_to_gridcell_m", "elevation_mean_m", "slope_deg", "aspect_deg",
                            "grid_specific_humidity_kgkg", "nighttime_wind_ms", "koppen_main_group_code")}
    row["date"] = "2023-07-01"
    X, ok, missing = build_feature_matrix([row], "tmax", feature_order=FEATURE_ORDER_GRIDBIAS)
    assert not ok[0] and missing[0] == ["grid_bias_prior_c", "grid_bias_support"]
    X, ok, _ = build_feature_matrix([{**row, "grid_bias_prior_c": 0.7, "grid_bias_support": 2.5}], "tmax",
                                    feature_order=FEATURE_ORDER_GRIDBIAS)
    assert ok[0] and X[0, -2] == 0.7 and X[0, -1] == 2.5


def test_refit_grid_bias_columns_uses_fold_training_rows_only():
    import train_downscaling as td
    # region R1 (two nearby stations, warm) is held out; region R2 is far away
    rows = (_rows("R1A", 45.0, 125.0, 7, 3.0) + _rows("R1B", 45.3, 125.0, 7, 3.0)
            + _rows("R2A", 10.0, 0.0, 7, -1.0) + _rows("R2B", 10.3, 0.0, 7, -1.0))
    sids, lats, lons, dates, y = zip(*rows)
    ctx = {"cols": (0, 1), "station_id": list(sids), "lat": list(lats), "lon": list(lons), "date": list(dates)}
    X = np.full((len(rows), 2), 99.0)
    held = np.array([s.startswith("R1") for s in sids])
    Xf = td.refit_grid_bias_columns(X, np.array(y), ~held, ctx)
    assert np.all(Xf[held] == 0.0)  # R1 never sees its own region, and R2 is out of range
    assert np.all(Xf[~held, 0] < 0) and np.all(Xf[~held, 1] > 0)  # R2 rows see the other R2 station
    assert np.all(X == 99.0)  # caller's matrix untouched


def test_attach_grid_bias_and_region_map(tmp_path):
    import train_downscaling as td
    rows = [{"station_id": s, "lat": la, "lon": lo, "date": d, "delta_tmax_c": v, "delta_tmin_c": None}
            for s, la, lo, d, v in _rows("A", 40.0, 120.0, 7, 2.0) + _rows("B", 40.3, 120.0, 7, 1.0)]
    td.attach_grid_bias(rows, "tmax")
    a = next(r for r in rows if r["station_id"] == "A")
    assert 0 < a["grid_bias_prior_c"] < 1.0 and a["grid_bias_support"] > 0  # from B alone, shrunk
    td.attach_grid_bias(rows, "tmin")  # no tmin deltas at all: no prior anywhere
    assert all(r["grid_bias_prior_c"] == 0.0 and r["grid_bias_support"] == 0.0 for r in rows)
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"CH": "NE_ASIA", "KN": "NE_ASIA"}))
    regions, mapping = td.apply_cv_region_map(["CH", "KN", "US"], str(p))
    assert regions == ["NE_ASIA", "NE_ASIA", "US"] and mapping["CH"] == "NE_ASIA"
    assert td.apply_cv_region_map(["CH"], None) == (["CH"], None)


def test_scorecard_accepts_gridbias_recipe(tmp_path):
    import scorecard
    p = tmp_path / "r.yaml"
    p.write_text("model_version: x\nfeature_set: gridbias\nextra_rows: []\n")
    assert scorecard.load_recipe(str(p))["feature_set"] == "gridbias"
    p.write_text("model_version: x\nfeature_set: regime\nextra_rows: []\n")
    with pytest.raises(SystemExit):
        scorecard.load_recipe(str(p))


def test_scorecard_grid_bias_copies_rows_and_uses_train_table():
    import scorecard
    train = [{"station_id": s, "lat": la, "lon": lo, "date": d, "delta_tmax_c": v}
             for s, la, lo, d, v in _rows("A", 40.0, 120.0, 7, 2.0) + _rows("B", 40.3, 120.0, 7, 2.0)]
    test = [{"station_id": "T", "lat": 40.1, "lon": 120.0, "date": "2025-07-03", "delta_tmax_c": -5.0}]
    tr2, te2 = scorecard.attach_grid_bias_for_fit(train, test, "tmax")
    assert "grid_bias_prior_c" not in train[0] and "grid_bias_prior_c" not in test[0]
    assert te2[0]["grid_bias_prior_c"] > 0  # from A and B only; the test station's own -5 never enters


def test_truth_overrides_replace_drop_and_null_tmin(tmp_path):
    import hashlib
    import scorecard
    p = tmp_path / "o.csv"
    p.write_text("station_id,date,tmax_c\nA,2025-01-01,30.0\nA,2025-01-02,\n")
    ov = scorecard.load_truth_overrides(str(p), hashlib.sha256(p.read_bytes()).hexdigest())
    rows = [{"station_id": "A", "date": "2025-01-01", "grid_tmax_c": 28.0, "station_tmax_c": 27.0, "delta_tmax_c": -1.0,
             "station_tmin_c": 20.0, "delta_tmin_c": 1.0},
            {"station_id": "A", "date": "2025-01-02", "grid_tmax_c": 28.0, "station_tmax_c": 27.0, "delta_tmax_c": -1.0,
             "station_tmin_c": 20.0, "delta_tmin_c": 1.0},
            {"station_id": "A", "date": "2025-01-03", "grid_tmax_c": 28.0, "station_tmax_c": 27.0, "delta_tmax_c": -1.0,
             "station_tmin_c": 20.0, "delta_tmin_c": 1.0}]
    counts = scorecard.apply_truth_overrides(rows, ov)
    assert counts == {"overrides": 2, "rows_replaced": 1, "rows_dropped": 1}
    assert rows[0]["delta_tmax_c"] == pytest.approx(2.0) and rows[0]["delta_tmin_c"] is None
    assert rows[1]["delta_tmax_c"] is None and rows[1]["delta_tmin_c"] is None
    assert rows[2]["delta_tmax_c"] == -1.0 and rows[2]["delta_tmin_c"] == 1.0
    with pytest.raises(SystemExit):
        scorecard.load_truth_overrides(str(p), "0" * 64)


def test_trainer_end_to_end_gridbias_with_region_map_and_tmax_only_extra_rows(tmp_path):
    from unittest.mock import patch

    import train_downscaling as td
    from test_train_downscaling import _make_rows
    rows = _make_rows(80, ["US", "FR", "CH"], {"US": "Cfa", "FR": "Cfb", "CH": "Dwa"})
    for r in rows:  # 4 stations per region, fixed positions, 20 days each in January
        k = int(r["station_id"].split("_")[1])
        r["station_id"] = f"{r['region']}_{k % 4}"
        r["lat"], r["lon"] = 30.0 + (k % 4) * 0.5 + {"US": 0, "FR": 2, "CH": 4}[r["region"]], -100.0
        r["date"] = date(2023, 1, 1 + k // 4)
    extra = [{**r, "station_id": "KN_x", "region": "KN", "delta_tmin_c": None, "station_tmin_c": None,
              "lat": 35.0, "date": r["date"].isoformat()} for r in rows[:20]]
    xp = tmp_path / "extra.json"
    xp.write_text(json.dumps({"complete": True, "rows": extra}))
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"CH": "NE_ASIA", "KN": "NE_ASIA"}))
    with patch.object(td, "load_training_rows", return_value=rows), \
         patch.object(td, "_bucket_from_credentials", return_value="b"), \
         patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), \
         patch.object(td, "save_model_artifacts") as save, \
         patch("sys.argv", ["t", "--model-version", "x", "--candidate-only", "--feature-set", "gridbias",
                            "--cv-region-map", str(mp), "--extra-rows-json", str(xp), "--cv-n-jobs", "1"]):
        td.main()
    _, _, bundle, meta = save.call_args[0]
    assert meta["feature_order"][-2:] == ["grid_bias_prior_c", "grid_bias_support"]
    assert meta["cv_region_map"] == {"CH": "NE_ASIA", "KN": "NE_ASIA"}
    assert meta["grid_bias_prior"]["tmax"]["station_months"] > 0
    assert "grid_bias_table_tmax" in bundle and "grid_bias_table_tmin" in bundle
    assert set(meta["cv"]["leave_region_out"]["tmax"]["by_zone"]) >= {"Cfa", "Cfb", "Dwa"}


def test_trainer_refuses_gridbias_without_candidate_only():
    from unittest.mock import patch

    import train_downscaling as td
    with patch("sys.argv", ["t", "--model-version", "x", "--feature-set", "gridbias"]):
        with pytest.raises(SystemExit):
            td.main()
