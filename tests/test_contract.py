"""Unit tests for heatready_downscaling.contract -- real (small)
quantile_forest fits against synthetic data, not mocked internals, so the
AOA/gate/conformal math is exercised against real model output shapes."""

import numpy as np
import pytest
from quantile_forest import RandomForestQuantileRegressor

from heatready_downscaling import contract
from heatready_downscaling.features import FEATURE_ORDER


def _fit_qrf(rng, n=200):
    X = rng.normal(size=(n, len(FEATURE_ORDER)))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    return RandomForestQuantileRegressor(n_estimators=20, min_samples_leaf=5, random_state=0).fit(X, y), X


def _bundle(rng):
    model_tmax, X_train = _fit_qrf(rng)
    model_tmin, _ = _fit_qrf(rng)
    weights = contract.feature_importance_weights(model_tmax)
    return {
        "model_tmax": model_tmax, "model_tmin": model_tmin,
        "metadata": {
            "model_version": "ds-test", "feature_order": list(FEATURE_ORDER),
            "conformal_q95_by_zone": {"_default": 1.0}, "conformal_q95_by_zone_tmin": {"_default": 1.0},
            "ood_aoa_threshold": 999.0,
            "cv": {"leave_region_out": {
                "tmax": {"by_zone": {"BWh": {"qrf_beats_grid": True}}},
                "tmin": {"by_zone": {"BWh": {"qrf_beats_grid": False}}},
            }},
        },
        "aoa_train_features_tmax": X_train, "aoa_feature_weights_tmax": weights,
        "aoa_train_mean_tmax": X_train.mean(axis=0), "aoa_train_std_tmax": X_train.std(axis=0),
        "aoa_train_features_tmin": X_train, "aoa_feature_weights_tmin": weights,
        "aoa_train_mean_tmin": X_train.mean(axis=0), "aoa_train_std_tmin": X_train.std(axis=0),
        "zones_passing_cv_gate": {"tmax": {"BWh": True}, "tmin": {"BWh": False}},
    }


_ROW = {
    "climate_zone": "BWh", "grid_tmax_c": 30.0, "grid_tmin_c": 20.0,
    "lat": 33.4, "lon": -112.0, "date": "2023-07-01",
    "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
    "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
    "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5,
    "koppen_main_group_code": 1, "elevation_mean_m": 340.0, "slope_deg": 3.5, "aspect_deg": 180.0,
    "grid_specific_humidity_kgkg": 0.012, "nighttime_wind_ms": 2.4,
}


class TestValidateFeatureOrder:
    def test_matching_order_does_not_raise(self):
        contract.validate_feature_order(FEATURE_ORDER)

    def test_mismatched_order_raises(self):
        reordered = list(FEATURE_ORDER)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with pytest.raises(ValueError, match="feature_order"):
            contract.validate_feature_order(reordered)


class TestQRFModelAdapterPredict:
    def test_complete_row_gets_a_prediction(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds = adapter.predict([_ROW], "tmax")
        assert len(preds) == 1
        assert preds[0]["applied"] is True
        assert preds[0]["model_version"] == "ds-test"
        assert preds[0]["covariates_missing"] == []

    def test_incomplete_row_is_not_applied_and_lists_missing_features(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        row = {**_ROW, "lst_warm_season_anomaly_c": None}
        preds = adapter.predict([row], "tmax")
        assert preds[0]["applied"] is False
        assert preds[0]["delta_c"] is None
        assert "lst_warm_season_anomaly_c" in preds[0]["covariates_missing"]
        assert preds[0]["cv_gate_passed"] is None  # never reached the gate check

    def test_gate_failed_zone_does_not_apply(self):
        # tmin's CV gate fails for BWh (see _bundle's metadata) -- must not
        # apply even though covariates are complete.
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds = adapter.predict([_ROW], "tmin")
        assert preds[0]["applied"] is False
        assert preds[0]["cv_gate_passed"] is False
        assert preds[0]["delta_c"] is None

    def test_unrecognized_zone_fails_closed(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        row = {**_ROW, "climate_zone": "NeverSeenZone"}
        preds = adapter.predict([row], "tmax")
        assert preds[0]["applied"] is False
        assert preds[0]["cv_gate_passed"] is False

    def test_extra_zone_gate_denies_a_zone_the_own_gate_alone_would_pass(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds = adapter.predict([_ROW], "tmax", extra_zone_gate={"tmax": {}, "tmin": {}})
        assert preds[0]["applied"] is False

    def test_extra_zone_gate_allows_a_zone_present_in_both_gates(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds = adapter.predict([_ROW], "tmax", extra_zone_gate={"tmax": {"BWh": True}, "tmin": {}})
        assert preds[0]["applied"] is True

    def test_bias_correction_added_to_delta_c(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds_uncorrected = adapter.predict([_ROW], "tmax")
        preds_corrected = adapter.predict([_ROW], "tmax", bias_correction={"tmax": {"BWh": 1.5}, "tmin": {}})
        assert preds_corrected[0]["delta_c"] == pytest.approx(preds_uncorrected[0]["delta_c"] + 1.5)
        # ci95_c is unaffected by a bias correction -- it recenters delta_c
        # only, the model's own conformal width is unchanged.
        assert preds_corrected[0]["ci95_c"] == pytest.approx(preds_uncorrected[0]["ci95_c"])

    def test_bias_correction_zone_absent_from_dict_applies_zero(self):
        adapter = contract.QRFModelAdapter(_bundle(np.random.RandomState(0)))
        preds_uncorrected = adapter.predict([_ROW], "tmax")
        preds_missing_zone = adapter.predict([_ROW], "tmax", bias_correction={"tmax": {"OtherZone": 5.0}, "tmin": {}})
        assert preds_missing_zone[0]["delta_c"] == pytest.approx(preds_uncorrected[0]["delta_c"])


class TestDeriveZonesPassingCvGate:
    def test_derives_flat_dict_from_nested_cv_report(self):
        metadata = {"cv": {"leave_region_out": {
            "tmax": {"by_zone": {"BWh": {"qrf_beats_grid": True}, "Cfb": {"qrf_beats_grid": False}}},
            "tmin": {"by_zone": {}},
        }}}
        gates = contract.derive_zones_passing_cv_gate(metadata)
        assert gates == {"tmax": {"BWh": True, "Cfb": False}, "tmin": {}}

    def test_missing_cv_section_yields_empty_gates(self):
        assert contract.derive_zones_passing_cv_gate({}) == {"tmax": {}, "tmin": {}}

    def test_falsy_qrf_beats_grid_values_coerced_to_bool(self):
        metadata = {"cv": {"leave_region_out": {
            "tmax": {"by_zone": {"BWh": {"qrf_beats_grid": None}}}, "tmin": {"by_zone": {}},
        }}}
        gates = contract.derive_zones_passing_cv_gate(metadata)
        assert gates["tmax"]["BWh"] is False


class TestConfidenceClass:
    def test_none_ci95_is_low(self):
        assert contract._confidence_class(None, False) == "low"

    def test_out_of_distribution_is_always_low(self):
        assert contract._confidence_class(0.1, True) == "low"

    def test_high_below_threshold(self):
        assert contract._confidence_class(0.5, False) == "high"

    def test_medium_between_thresholds(self):
        assert contract._confidence_class(2.0, False) == "medium"

    def test_low_above_medium_threshold(self):
        assert contract._confidence_class(3.0, False) == "low"
