"""Unit tests for scripts/era5_zarr_extract.py -- builds a small synthetic
Zarr tile on local disk (no S3, no CDS) and verifies station extraction,
including the masked-cell (open-water) rescue path, matches
era5.extract_era5_means's own row shape and unit conversions."""
import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import era5_zarr_extract as ez


def _build_synthetic_tile(tmp_path, with_wind=True):
    """3x3 grid, 2 hourly timesteps. Center cell (index 1,1) is NaN (open
    water) on purpose, to exercise the rescue path. All temperatures in
    Kelvin (matching real ERA5 units) so the K->C conversion is exercised."""
    lats = np.array([10.0, 10.1, 10.2])
    lons = np.array([20.0, 20.1, 20.2])
    times = np.array(["2023-06-01T00:00:00", "2023-06-01T01:00:00"], dtype="datetime64[ns]")

    t2m = np.full((2, 3, 3), 300.0)  # 300K = 26.85C
    t2m[:, 1, 1] = np.nan  # masked/open-water cell
    d2m = np.full((2, 3, 3), 290.0)
    sp = np.full((2, 3, 3), 101000.0)

    data_vars = {
        "t2m": (("time", "latitude", "longitude"), t2m),
        "d2m": (("time", "latitude", "longitude"), d2m),
        "sp": (("time", "latitude", "longitude"), sp),
    }
    if with_wind:
        data_vars["u10"] = (("time", "latitude", "longitude"), np.full((2, 3, 3), 3.0))
        data_vars["v10"] = (("time", "latitude", "longitude"), np.full((2, 3, 3), 4.0))

    ds = xr.Dataset(data_vars, coords={"time": times, "latitude": lats, "longitude": lons})
    store_path = str(tmp_path / "tile.zarr")
    ds.to_zarr(store_path, mode="w", consolidated=True)
    return store_path


class TestExtractEra5MeansFromZarr:
    def test_extracts_a_normal_station_with_unit_conversion_and_wind(self, tmp_path):
        store = _build_synthetic_tile(tmp_path)
        ds = ez.open_era5_zarr_store(store)
        stations = [{"name": "st_a", "lat": 10.0, "lon": 20.0}]  # exact grid point (0,0)

        rows = ez.extract_era5_means_from_zarr(ds, stations)

        assert len(rows) == 2
        assert rows[0]["name"] == "st_a"
        assert rows[0]["t2m"] == pytest.approx(26.85, abs=0.01)  # K -> C
        assert rows[0]["d2m"] == pytest.approx(16.85, abs=0.01)
        assert rows[0]["sp"] == pytest.approx(101000.0)
        assert rows[0]["wind_ms"] == pytest.approx(5.0)  # sqrt(3^2+4^2)
        assert rows[0]["datetime"] == "2023-06-01T00:00:00"
        assert rows[1]["datetime"] == "2023-06-01T01:00:00"

    def test_masked_cell_is_rescued_to_nearest_unmasked_neighbor(self, tmp_path):
        store = _build_synthetic_tile(tmp_path)
        ds = ez.open_era5_zarr_store(store)
        # (10.1, 20.1) is the exact masked center cell (index 1,1).
        stations = [{"name": "coastal", "lat": 10.1, "lon": 20.1}]

        rows = ez.extract_era5_means_from_zarr(ds, stations)

        # Rescued to a real neighbor -- never NaN, never the masked cell's own value.
        assert len(rows) == 2
        assert not (rows[0]["t2m"] != rows[0]["t2m"])  # not NaN
        assert rows[0]["t2m"] == pytest.approx(26.85, abs=0.01)

    def test_no_wind_columns_when_source_lacks_wind_vars(self, tmp_path):
        store = _build_synthetic_tile(tmp_path, with_wind=False)
        ds = ez.open_era5_zarr_store(store)
        stations = [{"name": "st_a", "lat": 10.0, "lon": 20.0}]

        rows = ez.extract_era5_means_from_zarr(ds, stations)

        assert "wind_ms" not in rows[0]
        assert "solar_wm2" not in rows[0]

    def test_multiple_stations_each_get_their_own_full_timeseries(self, tmp_path):
        store = _build_synthetic_tile(tmp_path)
        ds = ez.open_era5_zarr_store(store)
        stations = [
            {"name": "st_a", "lat": 10.0, "lon": 20.0},
            {"name": "st_b", "lat": 10.2, "lon": 20.2},
        ]

        rows = ez.extract_era5_means_from_zarr(ds, stations)

        assert len(rows) == 4  # 2 stations x 2 timesteps
        names = {r["name"] for r in rows}
        assert names == {"st_a", "st_b"}


class TestNearestIndex:
    def test_finds_exact_match(self):
        assert ez._nearest_index([10.0, 10.1, 10.2], 10.1) == 1

    def test_finds_closest_when_no_exact_match(self):
        assert ez._nearest_index([10.0, 10.1, 10.2], 10.14) == 1
        assert ez._nearest_index([10.0, 10.1, 10.2], 10.16) == 2
