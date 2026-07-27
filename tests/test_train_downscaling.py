"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

Unit tests for scripts/train_downscaling.py — no DB, S3, or CDS calls.

Uses real quantile_forest fits against small synthetic datasets (fast enough
at this scale) rather than mocking sklearn/quantile_forest internals, so the
CV/conformal/AOA math is exercised against real model output shapes.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import train_downscaling as td


def _make_rows(n_per_region: int, regions: list[str], zone_by_region: dict[str, str], seed: int = 0) -> list[dict]:
    rng = np.random.RandomState(seed)
    rows = []
    for region in regions:
        for i in range(n_per_region):
            grid_tmax, grid_tmin = 30.0 + rng.uniform(-2, 2), 20.0 + rng.uniform(-2, 2)
            delta_tmax, delta_tmin = rng.uniform(-1, 1), rng.uniform(-1, 1)
            rows.append({
                "station_id": f"{region}_{i}", "date": date(2023, 1, 1 + (i % 27)),
                "lon": -100.0 + rng.uniform(-5, 5), "lat": 30.0 + rng.uniform(-5, 5),
                "region": region, "climate_zone": zone_by_region[region],
                "station_tmax_c": grid_tmax + delta_tmax, "station_tmin_c": grid_tmin + delta_tmin,
                "grid_tmax_c": grid_tmax, "grid_tmin_c": grid_tmin,
                "delta_tmax_c": delta_tmax, "delta_tmin_c": delta_tmin,
                "lst_warm_season_anomaly_c": rng.uniform(-1, 1), "canopy_height_mean_m": rng.uniform(0, 10),
                "canopy_frac_over_3m": rng.uniform(0, 1), "wc_built_frac": rng.uniform(0, 1),
                "wc_tree_frac": rng.uniform(0, 1), "wc_water_frac": 0.0,
                "ghsl_urban_fraction": rng.uniform(0, 1), "pop_density_per_km2": rng.uniform(0, 5000),
                "elevation_rel_to_gridcell_m": rng.uniform(-50, 50),
                "elevation_mean_m": rng.uniform(0, 500),
                "slope_deg": rng.uniform(0, 15), "aspect_deg": rng.uniform(0, 360),
                "grid_specific_humidity_kgkg": rng.uniform(0.005, 0.02),
                "nighttime_wind_ms": rng.uniform(0, 8),
            })
    return rows


class TestBuildTrainingFeatureMatrix:
    def test_row_count_and_target_selection(self):
        rows = _make_rows(5, ["US"], {"US": "Cfa"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        assert X.shape == (5, len(td.FEATURE_ORDER))
        assert len(y) == 5
        assert set(regions) == {"US"}
        assert set(zones) == {"Cfa"}
        assert y[0] == pytest.approx(rows[0]["delta_tmax_c"])

    def test_tmin_target_uses_tmin_grid_and_delta(self):
        rows = _make_rows(3, ["US"], {"US": "Cfa"})
        X, y, _, _, _, _ = td.build_training_feature_matrix(rows, "tmin")
        grid_idx = 0  # feature 0 is the target-specific grid value
        assert X[0, grid_idx] == pytest.approx(rows[0]["grid_tmin_c"])
        assert y[0] == pytest.approx(rows[0]["delta_tmin_c"])

    def test_row_missing_a_feature_is_dropped(self):
        rows = _make_rows(3, ["US"], {"US": "Cfa"})
        rows[1]["lst_warm_season_anomaly_c"] = None
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        assert len(y) == 2
        assert len(regions) == 2 and len(zones) == 2

    def test_nan_target_row_is_dropped_even_though_features_are_complete(self):
        # A real-world case: a grid cell that's entirely ocean-masked (a tiny
        # island station) can leave grid_tmax_c/delta_tmax_c as an actual
        # float NaN in the DB, not SQL NULL -- "IS NOT NULL" doesn't catch
        # it, so this must be filtered here or the QRF fit crashes deep in a
        # joblib worker with an opaque error far from the real cause.
        rows = _make_rows(3, ["US"], {"US": "Cfa"})
        rows[1]["delta_tmax_c"] = float("nan")
        rows[1]["grid_tmax_c"] = float("nan")
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        assert len(y) == 2
        assert not np.isnan(y).any()

    def test_stored_koppen_code_is_passed_through_without_recomputing(self):
        # ghcn_training rows carry koppen_main_group_code (backfilled
        # 2026-07-19) -- the covariate_rows re-projection inside this
        # function must forward it, or downscaling.build_feature_matrix sees
        # a missing key and silently falls back to a live kgcpy lookup per
        # row (the real bug this test pins: the SQL/DB fix in
        # downscaling.py alone wasn't enough, this re-projection dropped the
        # column a second time).
        rows = _make_rows(3, ["US"], {"US": "Cfa"})
        for r in rows:
            r["koppen_main_group_code"] = 4
        with patch("heatready_downscaling.koppen.koppen_main_group_code") as mock_koppen:
            td.build_training_feature_matrix(rows, "tmax")
        mock_koppen.assert_not_called()


class TestLeaveRegionOutCv:
    def test_every_row_gets_exactly_one_out_of_fold_prediction(self):
        rows = _make_rows(60, ["US", "FR", "GM"], {"US": "Cfa", "FR": "Cfb", "GM": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10):
            cv = td.leave_region_out_cv(X, y, regions, n_jobs=1)
        assert cv["valid"].all()
        assert not np.isnan(cv["oof_median"]).any()

    def test_fold_with_too_few_training_rows_is_skipped(self):
        # TINY region has plenty of ITS OWN rows, but every fold trains on
        # every OTHER region -- so a fold is skipped when the OTHER regions
        # combined don't clear the minimum, not based on the held-out
        # region's own size.
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        rows += _make_rows(5, ["TINY"], {"TINY": "Dfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 1000):
            cv = td.leave_region_out_cv(X, y, regions, n_jobs=1)
        # every fold's "other regions" training set is under the inflated
        # minimum, so nothing gets a prediction
        assert not cv["valid"].any()

    def test_aoa_dissimilarity_is_populated_for_held_out_rows(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10):
            cv = td.leave_region_out_cv(X, y, regions, n_jobs=1)
        assert not np.isnan(cv["oof_di"][cv["valid"]]).any()
        assert (cv["oof_di"][cv["valid"]] >= 0).all()


class TestCvMetricsByZone:
    def _cv(self, y, pred_median, half_width=1.0):
        n = len(y)
        return {
            "valid": np.ones(n, dtype=bool),
            "oof_median": np.array(pred_median),
            "oof_lo": np.array(pred_median) - half_width,
            "oof_hi": np.array(pred_median) + half_width,
            "oof_di": np.zeros(n),
        }

    def test_perfect_predictions_beat_grid(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        cv = self._cv(y, y)  # predicts truth exactly -> rmse_downscaled == 0
        zones = ["A", "A", "A", "A"]
        result = td.cv_metrics_by_zone(y, zones, cv)
        assert result["by_zone"]["A"]["qrf_beats_grid"] is True
        assert result["by_zone"]["A"]["rmse_qrf_c"] == pytest.approx(0.0)

    def test_predicting_zero_matches_grid_rmse_exactly(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        cv = self._cv(y, [0.0, 0.0, 0.0, 0.0])  # predicting delta=0 == raw grid
        zones = ["A"] * 4
        result = td.cv_metrics_by_zone(y, zones, cv)
        m = result["by_zone"]["A"]
        assert m["rmse_qrf_c"] == pytest.approx(m["rmse_grid_c"])
        assert m["qrf_beats_grid"] is False  # equal, not strictly better

    def test_splits_metrics_per_zone(self):
        y = np.array([1.0, 1.0, 5.0, 5.0])
        cv = self._cv(y, [1.0, 1.0, 0.0, 0.0])  # zone A perfect, zone B way off
        zones = ["A", "A", "B", "B"]
        result = td.cv_metrics_by_zone(y, zones, cv)
        assert result["by_zone"]["A"]["qrf_beats_grid"] is True
        assert result["by_zone"]["B"]["qrf_beats_grid"] is False


class TestCvMetricsByZoneKrigingComparison:
    def _cv(self, y, pred_median, half_width=1.0):
        n = len(y)
        return {
            "valid": np.ones(n, dtype=bool),
            "oof_median": np.array(pred_median),
            "oof_lo": np.array(pred_median) - half_width,
            "oof_hi": np.array(pred_median) + half_width,
            "oof_di": np.zeros(n),
        }

    def test_kriging_columns_absent_when_not_provided(self):
        y = np.array([1.0, -1.0])
        cv = self._cv(y, y)
        result = td.cv_metrics_by_zone(y, ["A", "A"], cv)
        assert "rmse_kriging_c" not in result["by_zone"]["A"]

    def test_qrf_beats_kriging_when_kriging_is_worse(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        cv = self._cv(y, y)  # QRF predicts perfectly
        kriging_pred = np.array([0.0, 0.0, 0.0, 0.0])  # kriging predicts delta=0 (== grid)
        result = td.cv_metrics_by_zone(y, ["A"] * 4, cv, kriging_oof_median=kriging_pred)
        m = result["by_zone"]["A"]
        assert m["qrf_beats_kriging"] is True
        assert m["rmse_qrf_c"] < m["rmse_kriging_c"]

    def test_kriging_can_beat_qrf(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        cv = self._cv(y, [0.0, 0.0, 0.0, 0.0])  # QRF predicts delta=0
        kriging_pred = y  # kriging predicts perfectly
        result = td.cv_metrics_by_zone(y, ["A"] * 4, cv, kriging_oof_median=kriging_pred)
        m = result["by_zone"]["A"]
        assert m["qrf_beats_kriging"] is False

    def test_nan_kriging_predictions_excluded_from_kriging_metric(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        cv = self._cv(y, y)
        kriging_pred = np.array([0.0, 0.0, np.nan, np.nan])  # only 2 folds had kriging predictions
        result = td.cv_metrics_by_zone(y, ["A"] * 4, cv, kriging_oof_median=kriging_pred)
        assert result["by_zone"]["A"]["rmse_kriging_c"] == pytest.approx(1.0)  # RMS of [1,-1] errors


class TestRegressionKrigingCv:
    def test_predictions_populated_for_valid_folds(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10):
            oof = td.regression_kriging_cv(X, y, regions, lons, lats, n_jobs=1)
        assert not np.isnan(oof).any()

    def test_fold_with_too_few_training_rows_is_skipped(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 1000):
            oof = td.regression_kriging_cv(X, y, regions, lons, lats, n_jobs=1)
        assert np.isnan(oof).all()

    def test_kriging_failure_falls_back_to_trend_only_without_crashing(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), \
             patch("pykrige.ok.OrdinaryKriging", side_effect=RuntimeError("degenerate variogram")):
            oof = td.regression_kriging_cv(X, y, regions, lons, lats, n_jobs=1)
        # falls back to the linear trend alone (kriged residual = 0) rather
        # than crashing or leaving NaNs for a fold that had enough data
        assert not np.isnan(oof).any()

    def test_large_fold_subsamples_before_fitting_the_variogram(self):
        # pykrige's OrdinaryKriging builds a full O(n^2) pairwise-distance
        # matrix over every point passed to its constructor -- a real
        # training fold (150k-250k rows) blew this up to a 100+ GiB
        # allocation attempt (2026-07-19). This pins the fix: a fold larger
        # than the sample cap must call OrdinaryKriging with an array
        # bounded at the cap, never the raw fold size.
        rows = _make_rows(1500, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})  # 1500/region -> 1500 train rows/fold
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)

        captured = {}

        class _FakeOK:
            def __init__(self, lons_arg, lats_arg, resid_arg, **kwargs):
                captured["n_points"] = len(lons_arg)

            def execute(self, *args, **kwargs):
                n_test = int((regions_arr == "US").sum())
                return np.zeros(n_test), None

        with patch.object(td, "_KRIGING_VARIOGRAM_SAMPLE_CAP", 200), \
             patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), \
             patch("pykrige.ok.OrdinaryKriging", _FakeOK):
            td._fit_one_kriging_fold("US", X, y, regions_arr, lons, lats, 30, 10, variogram_sample_cap=200)

        assert captured["n_points"] == 200  # capped, not the full ~1500-row FR training fold

    def test_small_fold_is_not_subsampled(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, _, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        n_train = int((regions_arr == "FR").sum())

        captured = {}

        class _FakeOK:
            def __init__(self, lons_arg, lats_arg, resid_arg, **kwargs):
                captured["n_points"] = len(lons_arg)

            def execute(self, *args, **kwargs):
                n_test = int((regions_arr == "US").sum())
                return np.zeros(n_test), None

        with patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), patch("pykrige.ok.OrdinaryKriging", _FakeOK):
            td._fit_one_kriging_fold("US", X, y, regions_arr, lons, lats, 30, 10)

        assert captured["n_points"] == n_train  # well under the cap -- uses every training row


class TestConformalQ95ByZone:
    def _cv(self, n, half_width=1.0):
        return {
            "valid": np.ones(n, dtype=bool),
            "oof_median": np.zeros(n),
            "oof_lo": np.full(n, -half_width),
            "oof_hi": np.full(n, half_width),
        }

    def test_small_zone_falls_back_to_default(self):
        y = np.zeros(5)
        cv = self._cv(5)
        zones = ["TINY"] * 5
        with patch.object(td, "_MIN_CONFORMAL_POINTS", 20):
            result = td.conformal_q95_by_zone(y, zones, cv)
        assert "TINY" not in result
        assert "_default" in result

    def test_large_zone_gets_its_own_q95(self):
        rng = np.random.RandomState(0)
        n = 100
        y = rng.uniform(-1, 1, n)
        cv = self._cv(n)
        zones = ["BIG"] * n
        with patch.object(td, "_MIN_CONFORMAL_POINTS", 20):
            result = td.conformal_q95_by_zone(y, zones, cv)
        assert "BIG" in result
        assert result["BIG"] > 0


class TestConformalEmpiricalCoverage:
    def test_coverage_is_one_when_interval_always_contains_truth(self):
        y = np.array([0.1, -0.1, 0.05])
        cv = {
            "valid": np.ones(3, dtype=bool), "oof_median": np.zeros(3),
            "oof_lo": np.full(3, -1.0), "oof_hi": np.full(3, 1.0),
        }
        coverage = td.conformal_empirical_coverage(y, ["A"] * 3, cv, {"A": 1.0, "_default": 1.0})
        assert coverage == pytest.approx(1.0)

    def test_coverage_is_zero_when_interval_never_contains_truth(self):
        y = np.array([5.0, -5.0, 6.0])
        cv = {
            "valid": np.ones(3, dtype=bool), "oof_median": np.zeros(3),
            "oof_lo": np.full(3, -1.0), "oof_hi": np.full(3, 1.0),
        }
        coverage = td.conformal_empirical_coverage(y, ["A"] * 3, cv, {"A": 1.0, "_default": 1.0})
        assert coverage == pytest.approx(0.0)


class TestBuildAoaIndex:
    def test_returns_full_training_set_when_under_cap(self):
        model = MagicMock(feature_importances_=np.array([1.0, 1.0]))
        X = np.random.rand(10, 2)
        with patch.object(td, "_AOA_TRAINING_SAMPLE_CAP", 100):
            index = td._build_aoa_index(model, X, "tmax", seed=1)
        assert index["aoa_train_features_tmax"].shape == (10, 2)

    def test_subsamples_when_over_cap(self):
        model = MagicMock(feature_importances_=np.array([1.0, 1.0]))
        X = np.random.rand(500, 2)
        with patch.object(td, "_AOA_TRAINING_SAMPLE_CAP", 50):
            index = td._build_aoa_index(model, X, "tmax", seed=1)
        assert index["aoa_train_features_tmax"].shape == (50, 2)

    def test_deterministic_given_same_seed(self):
        model = MagicMock(feature_importances_=np.array([1.0, 1.0]))
        X = np.random.RandomState(0).rand(500, 2)
        with patch.object(td, "_AOA_TRAINING_SAMPLE_CAP", 50):
            index1 = td._build_aoa_index(model, X, "tmax", seed=42)
            index2 = td._build_aoa_index(model, X, "tmax", seed=42)
        assert np.array_equal(index1["aoa_train_features_tmax"], index2["aoa_train_features_tmax"])

    def test_keys_are_suffixed_by_target_not_shared(self):
        model = MagicMock(feature_importances_=np.array([1.0, 1.0]))
        X = np.random.rand(10, 2)
        index_tmax = td._build_aoa_index(model, X, "tmax", seed=1)
        index_tmin = td._build_aoa_index(model, X, "tmin", seed=1)
        assert "aoa_train_features_tmax" in index_tmax and "aoa_train_features_tmin" not in index_tmax
        assert "aoa_train_features_tmin" in index_tmin and "aoa_train_features_tmax" not in index_tmin


class TestSaveModelArtifacts:
    def test_writes_model_and_metadata_to_expected_keys(self):
        mock_s3 = MagicMock()
        with patch("boto3.client", return_value=mock_s3):
            td.save_model_artifacts("test-bucket", "ds-2026.07-rf1", {"model_tmax": "X"}, {"model_version": "ds-2026.07-rf1"})
        assert mock_s3.put_object.call_count == 2
        keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
        assert "downscaling/models/ds-2026.07-rf1/model.joblib" in keys
        assert "downscaling/models/ds-2026.07-rf1/metadata.json" in keys
        for c in mock_s3.put_object.call_args_list:
            assert c.kwargs["Bucket"] == "test-bucket"


class TestMain:
    def test_no_rows_returns_without_calling_save(self):
        with patch.object(td, "load_training_rows", return_value=[]), \
             patch.object(td, "_bucket_from_credentials", return_value="test-bucket"), \
             patch.object(td, "save_model_artifacts") as mock_save, \
             patch("sys.argv", ["train_downscaling.py", "--model-version", "ds-test"]):
            td.main()
        mock_save.assert_not_called()

    def test_end_to_end_with_synthetic_data_calls_save_with_both_targets(self):
        rows = _make_rows(60, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        with patch.object(td, "load_training_rows", return_value=rows), \
             patch.object(td, "_bucket_from_credentials", return_value="test-bucket"), \
             patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), \
             patch.object(td, "save_model_artifacts") as mock_save, \
             patch("sys.argv", ["train_downscaling.py", "--model-version", "ds-test-1"]):
            td.main()

        mock_save.assert_called_once()
        bucket, model_version, artifact_bundle, metadata = mock_save.call_args[0]
        assert bucket == "test-bucket"
        assert model_version == "ds-test-1"
        assert "model_tmax" in artifact_bundle and "model_tmin" in artifact_bundle
        assert metadata["feature_order"] == list(td.FEATURE_ORDER)
        assert "conformal_q95_by_zone" in metadata
        assert "conformal_q95_by_zone_tmin" in metadata
        assert metadata["ood_aoa_threshold"] is not None
