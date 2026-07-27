"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

Unit tests for scripts/sweep_gbm.py — no DB/S3/CDS calls, real (small)
LightGBM fits against synthetic data (fast enough at this scale, same
convention test_train_downscaling.py already uses for real quantile_forest
fits)."""

import os
import sys
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import train_downscaling as td
import sweep_gbm as sg

# Small, fast params for tests -- the real _GBM_GRID's configs (up to 2500
# trees) would make the suite slow for no correctness benefit.
_TEST_GBM_PARAMS = {**sg._GBM_COMMON, "learning_rate": 0.2, "n_estimators": 20, "min_child_samples": 5}


def _make_rows(n_per_region: int, regions: list[str], zone_by_region: dict[str, str], seed: int = 0) -> list[dict]:
    rng = np.random.RandomState(seed)
    rows = []
    for region in regions:
        for i in range(n_per_region):
            grid_tmax = 30.0 + rng.uniform(-2, 2)
            delta_tmax = rng.uniform(-1, 1)
            rows.append({
                "station_id": f"{region}_{i}", "date": date(2023, 1, 1 + (i % 27)),
                "lon": -100.0 + rng.uniform(-5, 5), "lat": 30.0 + rng.uniform(-5, 5),
                "region": region, "climate_zone": zone_by_region[region],
                "station_tmax_c": grid_tmax + delta_tmax, "grid_tmax_c": grid_tmax,
                "station_tmin_c": 20.0, "grid_tmin_c": 20.0,
                "delta_tmax_c": delta_tmax, "delta_tmin_c": 0.0,
                "lst_warm_season_anomaly_c": rng.uniform(-1, 1), "canopy_height_mean_m": rng.uniform(0, 10),
                "canopy_frac_over_3m": rng.uniform(0, 1), "wc_built_frac": rng.uniform(0, 1),
                "wc_tree_frac": rng.uniform(0, 1), "wc_water_frac": 0.0,
                "ghsl_urban_fraction": rng.uniform(0, 1), "pop_density_per_km2": rng.uniform(0, 5000),
                "elevation_rel_to_gridcell_m": rng.uniform(-50, 50),
                "elevation_mean_m": rng.uniform(0, 500),
                "slope_deg": rng.uniform(0, 15), "aspect_deg": rng.uniform(0, 360),
                "grid_specific_humidity_kgkg": rng.uniform(0.005, 0.02),
                "koppen_main_group_code": 1,
                "nighttime_wind_ms": rng.uniform(0, 8),
            })
    return rows


class TestGbmGrid:
    def test_six_configs_three_budgets_times_two_leaf_sizes(self):
        assert len(sg._GBM_GRID) == 6
        assert {c["min_child_samples"] for c in sg._GBM_GRID} == {50, 200}
        budgets = {(c["learning_rate"], c["n_estimators"]) for c in sg._GBM_GRID}
        assert budgets == {(0.10, 500), (0.05, 1000), (0.02, 2500)}

    def test_config_ids_are_unique(self):
        ids = {c["config_id"] for c in sg._GBM_GRID}
        assert len(ids) == len(sg._GBM_GRID)

    def test_common_params_use_stochastic_feature_and_row_subsampling(self):
        # The design review's non-negotiable: feature_fraction is the GBM
        # analog of the max_features="sqrt" fix that resolved QRF's own
        # CV-gate failure -- must not silently regress back to using all
        # features/rows every split.
        assert sg._GBM_COMMON["feature_fraction"] < 1.0
        assert sg._GBM_COMMON["bagging_fraction"] < 1.0


class TestFitOneGbmQuantile:
    def test_returns_none_when_fold_too_small(self):
        rows = _make_rows(3, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        result = sg._fit_one_gbm_quantile("US", 0.5, X, y, regions_arr, _TEST_GBM_PARAMS, min_fold_train_rows=1000)
        assert result is None

    def test_fits_and_predicts_for_a_valid_fold(self):
        rows = _make_rows(30, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        result = sg._fit_one_gbm_quantile("US", 0.5, X, y, regions_arr, _TEST_GBM_PARAMS, min_fold_train_rows=10)
        assert result is not None
        assert result["region"] == "US"
        assert result["quantile"] == 0.5
        assert result["test_mask"].sum() == 30
        assert len(result["preds"]) == 30


class TestLeaveRegionOutCvGbm:
    def test_every_row_gets_exactly_one_out_of_fold_prediction(self):
        rows = _make_rows(40, ["US", "FR", "GM"], {"US": "Cfa", "FR": "Cfb", "GM": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 10)
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)
        assert cv["valid"].all()
        assert not np.isnan(cv["oof_median"]).any()
        # post-rearrangement, monotonicity must hold for EVERY row, not just on average
        assert np.all(cv["oof_lo"] <= cv["oof_median"])
        assert np.all(cv["oof_median"] <= cv["oof_hi"])
        assert "crossing_rate" in cv

    def test_scores_cleanly_through_cv_metrics_by_zone(self):
        rows = _make_rows(40, ["US", "FR", "GM"], {"US": "Cfa", "FR": "Cfb", "GM": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 10)
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)
        metrics = td.cv_metrics_by_zone(y, zones, cv)
        assert metrics["overall"]["n"] == len(y)
        assert set(metrics["by_zone"].keys()) == {"Cfa", "Cfb"}

    def test_fold_with_too_few_training_rows_is_skipped(self):
        # Every fold trains on every OTHER region's rows, so the skip
        # threshold is about the combined OTHER-region total, not the
        # held-out region's own size (matches
        # test_train_downscaling.py's TestLeaveRegionOutCv test of the
        # same name/semantics) -- set the threshold above the total
        # available data so every fold is skipped.
        rows = _make_rows(40, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        rows += _make_rows(3, ["TINY"], {"TINY": "Dfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 1000)
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)
        assert np.isnan(cv["oof_median"]).all()


class TestMonotoneRearrangement:
    """Chernozhukov, Fernandez-Val & Galichon (2010) monotone rearrangement:
    per-row sort of the 3 independently fitted quantiles. Pins the actual
    bug the design review found -- an unsorted crossed pair produces a
    negative interval half-width in conformal_q95_by_zone, silently
    dropping that row from calibration."""

    def _fake_fit(self, canned: dict[tuple[str, float], np.ndarray]):
        def _fit(held_region, quantile, X, y, regions_arr, gbm_params, min_fold_train_rows):
            test_mask = regions_arr == held_region
            if (~test_mask).sum() < min_fold_train_rows or test_mask.sum() == 0:
                return None
            return {
                "region": held_region, "quantile": quantile, "test_mask": test_mask,
                "preds": canned[(held_region, quantile)],
            }
        return _fit

    def test_crossed_predictions_are_sorted_into_monotone_order(self):
        rows = _make_rows(10, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        n_us = int((regions_arr == "US").sum())

        # Deliberately crossed for US: median ABOVE the "hi" prediction.
        canned = {
            ("US", 0.025): np.full(n_us, 0.0),
            ("US", 0.5): np.full(n_us, 5.0),   # crossed: exceeds the 0.975 prediction below
            ("US", 0.975): np.full(n_us, 2.0),
            ("FR", 0.025): np.full(int((regions_arr == "FR").sum()), -1.0),
            ("FR", 0.5): np.full(int((regions_arr == "FR").sum()), 0.0),
            ("FR", 0.975): np.full(int((regions_arr == "FR").sum()), 1.0),
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 5)
            mp.setattr(sg, "_fit_one_gbm_quantile", self._fake_fit(canned))
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)

        # Monotone everywhere after rearrangement, including the US rows
        # that were deliberately crossed.
        assert np.all(cv["oof_lo"] <= cv["oof_median"])
        assert np.all(cv["oof_median"] <= cv["oof_hi"])
        # Rearrangement = sort: {0, 5, 2} -> {0, 2, 5}
        us_mask = regions_arr == "US"
        assert np.allclose(cv["oof_lo"][us_mask], 0.0)
        assert np.allclose(cv["oof_median"][us_mask], 2.0)
        assert np.allclose(cv["oof_hi"][us_mask], 5.0)
        # FR was never crossed -- rearrangement must be a no-op there.
        fr_mask = regions_arr == "FR"
        assert np.allclose(cv["oof_lo"][fr_mask], -1.0)
        assert np.allclose(cv["oof_median"][fr_mask], 0.0)
        assert np.allclose(cv["oof_hi"][fr_mask], 1.0)
        # All of US's rows were crossed, none of FR's -- crossing_rate is
        # exactly US's share of the valid rows.
        assert cv["crossing_rate"] == pytest.approx(n_us / len(y))

    def test_no_crossing_gives_zero_crossing_rate(self):
        rows = _make_rows(10, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        canned = {
            (r, q): np.full(int((regions_arr == r).sum()), v)
            for r in ("US", "FR") for q, v in [(0.025, -1.0), (0.5, 0.0), (0.975, 1.0)]
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 5)
            mp.setattr(sg, "_fit_one_gbm_quantile", self._fake_fit(canned))
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)
        assert cv["crossing_rate"] == 0.0

    def test_crossing_would_otherwise_poison_conformal_calibration(self):
        """Direct regression test for the design review's actual finding:
        without rearrangement, a crossed pair's negative half-width becomes
        NaN nonconformity in conformal_q95_by_zone, silently dropping that
        row from calibration. With rearrangement (the code path actually
        shipped), every valid row must contribute a real, non-NaN q95."""
        rows = _make_rows(10, ["US", "FR"], {"US": "Cfa", "FR": "Cfb"})
        X, y, regions, zones, lons, lats = td.build_training_feature_matrix(rows, "tmax")
        regions_arr = np.array(regions)
        n_us = int((regions_arr == "US").sum())
        canned = {
            ("US", 0.025): np.full(n_us, 0.0),
            ("US", 0.5): np.full(n_us, 5.0),
            ("US", 0.975): np.full(n_us, 2.0),
            ("FR", 0.025): np.full(int((regions_arr == "FR").sum()), -1.0),
            ("FR", 0.5): np.full(int((regions_arr == "FR").sum()), 0.0),
            ("FR", 0.975): np.full(int((regions_arr == "FR").sum()), 1.0),
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(td, "_MIN_FOLD_TRAIN_ROWS", 5)
            mp.setattr(sg, "_fit_one_gbm_quantile", self._fake_fit(canned))
            cv = sg.leave_region_out_cv_gbm(X, y, regions, _TEST_GBM_PARAMS, n_jobs=1)

        q95 = td.conformal_q95_by_zone(y, zones, cv)
        for zone in set(zones):
            key = zone if zone in q95 else "_default"
            assert key in q95
            assert np.isfinite(q95[key])
