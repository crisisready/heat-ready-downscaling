"""Unit tests for scripts/build_band_paired_snapshot.py's pure-logic
helpers -- no network, no DB, no private-repo modules needed (unlike
_phase_fetch/_dispatch_fetch_one_station, which lazily import heat_calcs/
open_meteo/api_call_manager and are thin orchestration around
validate_lagfill_downscaling.py/validate_forecast_downscaling.py's already-
tested fetch functions, similarly untested at that level -- matching this
codebase's own established idiom for thin DB/S3/HTTP orchestration code).
What's worth pinning here is the row-shaping logic: source-flag derivation,
the era5-direct-from-export path, and station dedup."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_band_paired_snapshot as bps

_EXPORT_ROW = {
    "station_id": "USW00023183", "date": "2023-06-15", "lon": -112.0, "lat": 33.4,
    "elevation_m": 339.2, "region": "US", "climate_zone": "BWh", "koppen_main_group_code": 1,
    "obs_window_shift_days": 0, "station_tmax_c": 38.3, "station_tmin_c": 23.3,
    "grid_tmax_c": 37.0, "grid_tmin_c": 22.5, "grid_specific_humidity_kgkg": 0.012,
    "nighttime_wind_ms": 2.4, "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0,
    "canopy_frac_over_3m": 0.2, "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0,
    "ghsl_urban_fraction": 0.8, "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5,
    "elevation_mean_m": 340.0, "slope_deg": 3.5, "aspect_deg": 180.0,
    "delta_tmax_c": 1.3, "delta_tmin_c": 0.8,  # training-only, must NOT leak into snapshot rows
}


class TestEra5BandRows:
    def test_builds_one_row_per_export_row(self):
        rows = bps._era5_band_rows([_EXPORT_ROW], "v2026.07")
        assert len(rows) == 1
        r = rows[0]
        assert r["band"] == "era5"
        assert r["date"] == date(2023, 6, 15)
        assert r["grid_tmax_c"] == 37.0
        assert r["snapshot_version"] == "v2026.07"

    def test_source_provenance_is_genuine_band_not_fallback(self):
        """era5 never falls back -- its humidity/wind ARE the base ERA5
        values already, not a reconstruction with a possible gap."""
        r = bps._era5_band_rows([_EXPORT_ROW], "v2026.07")[0]
        assert r["base_source"] == "era5_land_cds"
        assert r["base_model"] is None
        assert r["humidity_source"] == "band"
        assert r["wind_source"] == "band"

    def test_covariate_provenance_constants_present(self):
        r = bps._era5_band_rows([_EXPORT_ROW], "v2026.07")[0]
        assert r["pop_density_source"] == "landscan_global"
        assert r["pop_density_buffer_deg"] == 0.01
        assert r["lst_reference_radius_km"] == 75.0

    def test_delta_columns_do_not_leak_into_snapshot_row(self):
        """delta_tmax_c/delta_tmin_c are training-time-derived, not part of
        the plan's schema -- score_band recomputes deltas itself."""
        r = bps._era5_band_rows([_EXPORT_ROW], "v2026.07")[0]
        assert "delta_tmax_c" not in r
        assert "delta_tmin_c" not in r

    def test_passthrough_columns_copied_verbatim(self):
        r = bps._era5_band_rows([_EXPORT_ROW], "v2026.07")[0]
        for col in bps._PASSTHROUGH_COLUMNS:
            assert r[col] == _EXPORT_ROW[col]


class TestShapeFetchedRow:
    def _cfg(self, band="lag_fill"):
        return bps._FETCH_BAND_CONFIG[band]

    def test_genuine_reconstruction_marks_band_source(self):
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": 3.1, "grid_specific_humidity_kgkg": 0.011}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "lag_fill", self._cfg(), recon, "v2026.07")
        assert row["grid_tmax_c"] == 36.0
        assert row["humidity_source"] == "band"
        assert row["wind_source"] == "band"
        assert row["grid_specific_humidity_kgkg"] == 0.011
        assert row["nighttime_wind_ms"] == 3.1

    def test_missing_wind_falls_back_to_export_row_value_and_is_flagged(self):
        """F7: forecast bands (and occasionally lag_fill) can reconstruct
        tmax/tmin without a companion wind value -- the row must fall back
        to the export's own ERA5-derived nighttime_wind_ms AND disclose
        that fallback via wind_source, not silently look like a genuine
        band value."""
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": None, "grid_specific_humidity_kgkg": 0.011}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "lag_fill", self._cfg(), recon, "v2026.07")
        assert row["nighttime_wind_ms"] == _EXPORT_ROW["nighttime_wind_ms"]
        assert row["wind_source"] == "era5_fallback"
        assert row["humidity_source"] == "band"  # independent per-field, not row-level

    def test_missing_humidity_falls_back_and_is_flagged_independently_of_wind(self):
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": 3.1, "grid_specific_humidity_kgkg": None}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "lag_fill", self._cfg(), recon, "v2026.07")
        assert row["grid_specific_humidity_kgkg"] == _EXPORT_ROW["grid_specific_humidity_kgkg"]
        assert row["humidity_source"] == "era5_fallback"
        assert row["wind_source"] == "band"

    def test_forecast_lead_band_carries_its_own_base_source_and_model(self):
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": None, "grid_specific_humidity_kgkg": None}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "forecast_lead3", self._cfg("forecast_lead3"), recon, "v2026.07")
        assert row["base_source"] == "open_meteo_previous_runs"
        assert row["base_model"] == "gfs_seamless"

    def test_lag_fill_has_no_model(self):
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": 1.0, "grid_specific_humidity_kgkg": 0.01}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "lag_fill", self._cfg(), recon, "v2026.07")
        assert row["base_source"] == "open_meteo_hfa"
        assert row["base_model"] is None

    def test_delta_columns_do_not_leak(self):
        recon = {"tmax": 36.0, "tmin": 21.0, "nighttime_wind_ms": 1.0, "grid_specific_humidity_kgkg": 0.01}
        row = bps._shape_fetched_row(_EXPORT_ROW, date(2023, 6, 15), "lag_fill", self._cfg(), recon, "v2026.07")
        assert "delta_tmax_c" not in row and "delta_tmin_c" not in row


class TestStationsFromExport:
    def test_dedups_by_station_id(self):
        rows = [
            {**_EXPORT_ROW, "date": "2023-01-01"},
            {**_EXPORT_ROW, "date": "2023-01-02"},
        ]
        stations = bps._stations_from_export(rows)
        assert len(stations) == 1
        assert stations[0]["station_id"] == "USW00023183"

    def test_no_band_or_date_columns_in_output(self):
        stations = bps._stations_from_export([_EXPORT_ROW])
        assert "date" not in stations[0]
        assert "band" not in stations[0]
        assert "grid_tmax_c" not in stations[0]  # band-varying, not a station-identity column

    def test_multiple_distinct_stations(self):
        rows = [_EXPORT_ROW, {**_EXPORT_ROW, "station_id": "OTHER"}]
        stations = bps._stations_from_export(rows)
        assert {s["station_id"] for s in stations} == {"USW00023183", "OTHER"}


class TestFetchBandConfig:
    def test_era5_is_not_a_fetchable_band(self):
        assert "era5" not in bps._FETCH_BAND_CONFIG

    def test_every_forecast_lead_1_through_7_present(self):
        for n in range(1, 8):
            assert f"forecast_lead{n}" in bps._FETCH_BAND_CONFIG

    def test_forecast_leads_all_pin_gfs_seamless(self):
        for n in range(1, 8):
            assert bps._FETCH_BAND_CONFIG[f"forecast_lead{n}"]["base_model"] == "gfs_seamless"
