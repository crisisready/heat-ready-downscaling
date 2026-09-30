"""rf9 regime feature order: FEATURE_ORDER is untouched, FEATURE_ORDER_REGIME appends REGIME_FEATURES,
build_feature_matrix builds either, the contract accepts only supported orders, and the trainer's
regime join matches rows by (station_id, date)."""
import csv
import importlib.util
import pathlib

import pytest

from heatready_downscaling.contract import validate_feature_order
from heatready_downscaling.features import (FEATURE_ORDER, FEATURE_ORDER_REGIME, REGIME_FEATURES,
                                            build_feature_matrix)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _row(**kw):
    r = {"grid_tmax_c": 35.0, "grid_tmin_c": 25.0, "lat": 23.0, "lon": 72.6, "date": "2025-07-01",
         "pop_density_per_km2": 1000.0, "lst_warm_season_anomaly_c": 0.1, "canopy_height_mean_m": 4.0,
         "canopy_frac_over_3m": 0.1, "wc_built_frac": 0.7, "wc_tree_frac": 0.1, "wc_water_frac": 0.0,
         "ghsl_urban_fraction": 0.9, "elevation_rel_to_gridcell_m": 1.0, "elevation_mean_m": 50.0,
         "slope_deg": 0.5, "aspect_deg": 90.0, "grid_specific_humidity_kgkg": 0.018,
         "koppen_main_group_code": 1, "nighttime_wind_ms": 2.0}
    r.update(kw)
    return r


def test_base_order_unchanged_and_regime_appends():
    assert len(FEATURE_ORDER) == 21
    assert FEATURE_ORDER_REGIME[:21] == FEATURE_ORDER
    assert FEATURE_ORDER_REGIME[21:] == REGIME_FEATURES


def test_base_matrix_ignores_regime_keys():
    X, ok, _ = build_feature_matrix([_row()], "tmax")
    assert X.shape == (1, 21) and ok == [True]


def test_regime_matrix_requires_regime_values():
    reg = dict(zip(REGIME_FEATURES, (2.5, 20.1, 14.0, 70.0)))
    X, ok, _ = build_feature_matrix([_row(**reg), _row()], "tmax", feature_order=FEATURE_ORDER_REGIME)
    assert X.shape == (2, 25) and ok == [True, False]
    assert list(X[0, 21:]) == [2.5, 20.1, 14.0, 70.0]


def test_unsupported_order_refused():
    with pytest.raises(ValueError):
        build_feature_matrix([_row()], "tmax", feature_order=FEATURE_ORDER[::-1])
    with pytest.raises(ValueError):
        validate_feature_order(list(FEATURE_ORDER)[:-1])
    assert validate_feature_order(list(FEATURE_ORDER_REGIME)) == FEATURE_ORDER_REGIME
    assert validate_feature_order(list(FEATURE_ORDER)) == FEATURE_ORDER


def test_trainer_attach_regime_features(tmp_path):
    spec = importlib.util.spec_from_file_location("td", ROOT / "scripts" / "train_downscaling.py")
    td = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(td)
    except ImportError as e:  # trainer's optional heavy deps
        pytest.skip(f"trainer import needs {e}")
    p = tmp_path / "r.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "date", "precipitation_sum", "shortwave_radiation_sum",
                    "wind_speed_10m_max", "relative_humidity_2m_mean"])
        w.writerow(["S1", "2025-07-01", "3.0", "18.0", "12.0", "80.0"])
        w.writerow(["S3", "2025-07-01", "nan", "18.0", "12.0", "80.0"])
    rows = [{"station_id": "S1", "date": "2025-07-01"}, {"station_id": "S2", "date": "2025-07-01"},
            {"station_id": "S3", "date": "2025-07-01"}]
    c = td.attach_regime_features(rows, [str(p)])
    assert c["matched"] == 2
    assert rows[0]["era5_precip_sum_mm"] == 3.0 and rows[1]["era5_precip_sum_mm"] is None
    assert rows[2]["era5_precip_sum_mm"] is None  # literal nan -> missing, never a NaN feature


def test_exclude_station_since():
    spec = importlib.util.spec_from_file_location("td2", ROOT / "scripts" / "train_downscaling.py")
    td = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(td)
    except ImportError as e:
        pytest.skip(f"trainer import needs {e}")
    rows = [{"station_id": "A", "date": "2022-12-31"}, {"station_id": "A", "date": "2023-01-01"},
            {"station_id": "B", "date": "2024-01-01"}]
    kept, dropped = td.exclude_station_since(rows, ["A:2023-01-01"])
    assert [(r["station_id"], r["date"]) for r in kept] == [("A", "2022-12-31"), ("B", "2024-01-01")]
    assert dropped == {"A:2023-01-01": 1}
    with pytest.raises(SystemExit):
        td.exclude_station_since(rows, ["A-2023"])
