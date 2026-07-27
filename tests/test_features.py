"""Unit tests for heatready_downscaling.features."""

import math
from datetime import date
from unittest.mock import patch

import pytest

from heatready_downscaling import features

_COMPLETE_ROW = {
    "grid_tmax_c": 35.0, "grid_tmin_c": 22.0, "lat": 33.4, "lon": -112.0,
    "date": date(2023, 7, 15),
    "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0,
    "canopy_frac_over_3m": 0.2, "wc_built_frac": 0.6, "wc_tree_frac": 0.1,
    "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
    "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5,
    "elevation_mean_m": 340.0, "slope_deg": 3.5, "aspect_deg": 180.0,
    "grid_specific_humidity_kgkg": 0.012,
    "nighttime_wind_ms": 2.4,
    "koppen_main_group_code": 1,
}


class TestBuildFeatureMatrix:
    def test_complete_row_is_marked_complete_with_no_missing_features(self):
        X, complete_mask, missing_by_row = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        assert complete_mask == [True]
        assert missing_by_row == [[]]
        assert X.shape == (1, len(features.FEATURE_ORDER))

    def test_index_zero_is_grid_tmax_for_tmax_target_and_grid_tmin_for_tmin_target(self):
        X_tmax, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        X_tmin, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmin")
        assert X_tmax[0, 0] == 35.0
        assert X_tmin[0, 0] == 22.0

    def test_doy_sin_cos_computed_from_date(self):
        X, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        doy = date(2023, 7, 15).timetuple().tm_yday
        angle = 2 * math.pi * doy / 365.25
        idx_sin = features.FEATURE_ORDER.index("doy_sin")
        idx_cos = features.FEATURE_ORDER.index("doy_cos")
        assert X[0, idx_sin] == pytest.approx(math.sin(angle))
        assert X[0, idx_cos] == pytest.approx(math.cos(angle))

    def test_log1p_applied_to_population_density(self):
        X, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        idx = features.FEATURE_ORDER.index("log1p_pop_density")
        assert X[0, idx] == pytest.approx(math.log1p(1500.0))

    def test_missing_single_feature_marks_row_incomplete(self):
        row = {**_COMPLETE_ROW, "lst_warm_season_anomaly_c": None}
        X, complete_mask, missing_by_row = features.build_feature_matrix([row], "tmax")
        assert complete_mask == [False]
        assert missing_by_row == [["lst_warm_season_anomaly_c"]]

    def test_missing_grid_value_marks_row_incomplete(self):
        row = {**_COMPLETE_ROW, "grid_tmax_c": None}
        _, complete_mask, missing_by_row = features.build_feature_matrix([row], "tmax")
        assert complete_mask == [False]
        assert "grid_daily_value_c" in missing_by_row[0]

    def test_date_as_iso_string_is_accepted(self):
        row = {**_COMPLETE_ROW, "date": "2023-07-15"}
        X, complete_mask, _ = features.build_feature_matrix([row], "tmax")
        assert complete_mask == [True]

    def test_koppen_main_group_code_computed_live_when_absent(self):
        # _COMPLETE_ROW's (33.4, -112.0) is Phoenix -- real desert (BWh,
        # main group B, code 1).
        row = {**_COMPLETE_ROW, "koppen_main_group_code": None}
        X, complete_mask, _ = features.build_feature_matrix([row], "tmax")
        idx = features.FEATURE_ORDER.index("koppen_main_group_code")
        assert complete_mask == [True]
        assert X[0, idx] == pytest.approx(1.0)

    def test_stored_koppen_code_used_without_recomputing(self):
        with patch("heatready_downscaling.koppen.koppen_main_group_code") as mock_koppen:
            X, complete_mask, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        mock_koppen.assert_not_called()
        assert complete_mask == [True]
        assert X[0, features.FEATURE_ORDER.index("koppen_main_group_code")] == pytest.approx(1.0)

    def test_missing_lon_marks_koppen_feature_missing_when_code_absent(self):
        row = {**_COMPLETE_ROW, "koppen_main_group_code": None, "lon": None}
        _, complete_mask, missing_by_row = features.build_feature_matrix([row], "tmax")
        assert complete_mask == [False]
        assert "koppen_main_group_code" in missing_by_row[0]

    def test_elevation_mean_m_and_slope_deg_pulled_through_directly(self):
        X, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        assert X[0, features.FEATURE_ORDER.index("elevation_mean_m")] == pytest.approx(340.0)
        assert X[0, features.FEATURE_ORDER.index("slope_deg")] == pytest.approx(3.5)

    def test_aspect_deg_encoded_as_sin_cos(self):
        # aspect_deg=180 (due south) -> sin=0, cos=-1.
        X, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        assert X[0, features.FEATURE_ORDER.index("aspect_sin")] == pytest.approx(0.0, abs=1e-9)
        assert X[0, features.FEATURE_ORDER.index("aspect_cos")] == pytest.approx(-1.0)

    def test_missing_aspect_deg_marks_both_sin_and_cos_missing(self):
        row = {**_COMPLETE_ROW, "aspect_deg": None}
        _, complete_mask, missing_by_row = features.build_feature_matrix([row], "tmax")
        assert complete_mask == [False]
        assert "aspect_sin" in missing_by_row[0]
        assert "aspect_cos" in missing_by_row[0]

    def test_grid_diurnal_range_is_tmax_minus_tmin(self):
        X, _, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmax")
        idx = features.FEATURE_ORDER.index("grid_diurnal_range_c")
        assert X[0, idx] == pytest.approx(35.0 - 22.0)

    def test_grid_diurnal_range_present_even_for_tmin_target(self):
        X, complete_mask, _ = features.build_feature_matrix([_COMPLETE_ROW], "tmin")
        idx = features.FEATURE_ORDER.index("grid_diurnal_range_c")
        assert complete_mask == [True]
        assert X[0, idx] == pytest.approx(35.0 - 22.0)
