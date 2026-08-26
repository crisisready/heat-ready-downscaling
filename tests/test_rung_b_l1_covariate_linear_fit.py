"""Unit tests for examples/rung_b_l1_covariate_linear_fit.py's own fitting
logic (fit_covariate_linear) -- pure, no network/snapshot needed. The
network-dependent end-to-end path (download + reproduce()) is exercised by
.github/workflows/examples.yml directly, matching CONTRIBUTING.md's own
"every example runs in CI" bar rather than being mocked here."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import rung_b_l1_covariate_linear_fit as rb


def _row(station_id, covariate_value, station_tmax_c, grid_tmax_c, zone="TestZone"):
    return {
        "station_id": station_id, "climate_zone": zone,
        "station_tmax_c": station_tmax_c, "grid_tmax_c": grid_tmax_c,
        "wc_tree_frac": covariate_value,
    }


class TestFitCovariateLinear:
    def test_recovers_a_known_linear_relationship(self):
        """residual = 2.0 - 3.0*covariate exactly, over enough distinct
        points that lstsq must recover it exactly (noiseless)."""
        rows = [
            _row(f"s{i}", cov, grid_c + (2.0 - 3.0 * cov), grid_c)
            for i, (cov, grid_c) in enumerate([(0.0, 20.0), (0.2, 21.0), (0.4, 19.5), (0.6, 22.0), (0.8, 18.0)])
        ]
        entry, n_stations = rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")
        assert n_stations == 5
        assert entry["basis"] == "raw_grid"
        assert entry["intercept"] == pytest.approx(2.0, abs=1e-9)
        assert entry["terms"] == [{"covariate": "wc_tree_frac", "slope": pytest.approx(-3.0, abs=1e-9)}]

    def test_valid_range_spans_observed_covariate_values(self):
        rows = [_row(f"s{i}", cov, 20.0, 20.0) for i, cov in enumerate([0.1, 0.5, 0.9])]
        entry, _ = rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")
        assert entry["valid_range"] == [[0.1, 0.9]]

    def test_excludes_other_zones(self):
        rows = [
            _row("s1", 0.1, 21.0, 20.0, zone="TestZone"),
            _row("s2", 0.5, 21.0, 20.0, zone="TestZone"),
            _row("s3", 0.9, 99.0, 0.0, zone="OtherZone"),
        ]
        entry, n_stations = rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")
        assert n_stations == 2

    def test_excludes_rows_missing_the_covariate(self):
        rows = [
            _row("s1", 0.1, 21.0, 20.0),
            _row("s2", 0.5, 21.0, 20.0),
            {**_row("s3", 0.9, 21.0, 20.0), "wc_tree_frac": None},
        ]
        entry, n_stations = rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")
        assert n_stations == 2

    def test_excludes_rows_missing_truth_or_grid(self):
        rows = [
            _row("s1", 0.1, 21.0, 20.0),
            _row("s2", 0.5, 21.0, 20.0),
            {**_row("s3", 0.9, 21.0, 20.0), "station_tmax_c": None},
        ]
        entry, n_stations = rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")
        assert n_stations == 2

    def test_raises_with_fewer_than_two_stations(self):
        rows = [_row("s1", 0.1, 21.0, 20.0)]
        with pytest.raises(SystemExit, match="only 1 station"):
            rb.fit_covariate_linear(rows, "TestZone", "tmax", "wc_tree_frac")

    def test_raises_with_zero_stations(self):
        with pytest.raises(SystemExit, match="only 0 station"):
            rb.fit_covariate_linear([], "TestZone", "tmax", "wc_tree_frac")

    def test_works_for_tmin_target(self):
        rows = [
            {"station_id": f"s{i}", "climate_zone": "TestZone", "station_tmin_c": grid_c + 1.0,
             "grid_tmin_c": grid_c, "wc_tree_frac": cov}
            for i, (cov, grid_c) in enumerate([(0.1, 10.0), (0.5, 11.0), (0.9, 9.0)])
        ]
        entry, n_stations = rb.fit_covariate_linear(rows, "TestZone", "tmin", "wc_tree_frac")
        assert n_stations == 3
        assert entry["intercept"] == pytest.approx(1.0, abs=1e-9)
