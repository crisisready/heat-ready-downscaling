"""Unit tests for heatready_downscaling.snapshot -- writes real (tiny)
Parquet files to a tmp_path rather than mocking pyarrow, so the actual
schema/compression/sha256 machinery gets genuine coverage."""

from datetime import date

import pytest

from heatready_downscaling import snapshot

_ROW = {
    "station_id": "USW00023183", "date": date(2023, 6, 15), "band": "lag_fill",
    "station_tmax_c": 38.3, "station_tmin_c": 23.3, "grid_tmax_c": 37.0, "grid_tmin_c": 22.5,
    "grid_specific_humidity_kgkg": 0.012, "nighttime_wind_ms": 2.4,
    "base_source": "open_meteo_hfa", "base_model": None, "humidity_source": "era5_fallback", "wind_source": "band",
    "lon": -112.0, "lat": 33.4, "elevation_m": 339.2, "region": "US", "climate_zone": "BWh",
    "koppen_main_group_code": 1, "obs_window_shift_days": 0,
    "lst_warm_season_anomaly_c": 2.1, "canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2,
    "wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0, "ghsl_urban_fraction": 0.8,
    "pop_density_per_km2": 1500.0, "elevation_rel_to_gridcell_m": 12.5, "elevation_mean_m": 340.0,
    "slope_deg": 3.5, "aspect_deg": 180.0,
    "pop_density_source": "landscan_global", "pop_density_buffer_deg": 0.01, "lst_reference_radius_km": 75.0,
    "snapshot_version": "v2026.08",
}


def _row(station_id="USW00023183", d=date(2023, 6, 15)):
    return {**_ROW, "station_id": station_id, "date": d}


class TestWritePartitionAndReadBandPartitions:
    def test_roundtrip_preserves_row_values(self, tmp_path):
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        rows = snapshot.read_band_partitions(str(tmp_path), "lag_fill")
        assert len(rows) == 1
        assert rows[0]["station_id"] == "USW00023183"
        assert rows[0]["grid_tmax_c"] == pytest.approx(37.0)
        assert rows[0]["climate_zone"] == "BWh"

    def test_unrecognized_band_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unrecognized band"):
            snapshot.write_partition(str(tmp_path), "not_a_real_band", "2023-06", [_row()])

    def test_multiple_months_concatenated(self, tmp_path):
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row(d=date(2023, 6, 15))])
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-07", [_row(d=date(2023, 7, 15))])
        rows = snapshot.read_band_partitions(str(tmp_path), "lag_fill")
        assert len(rows) == 2

    def test_months_filter_restricts_which_partitions_are_read(self, tmp_path):
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row(d=date(2023, 6, 15))])
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-07", [_row(d=date(2023, 7, 15))])
        rows = snapshot.read_band_partitions(str(tmp_path), "lag_fill", months=["2023-06"])
        assert len(rows) == 1

    def test_different_bands_are_isolated(self, tmp_path):
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        snapshot.write_partition(str(tmp_path), "era5", "2023-06", [_row(station_id="OTHER")])
        assert len(snapshot.read_band_partitions(str(tmp_path), "lag_fill")) == 1
        assert len(snapshot.read_band_partitions(str(tmp_path), "era5")) == 1

    def test_rows_sorted_by_station_id_then_date_on_write(self, tmp_path):
        rows_in = [
            _row(station_id="B", d=date(2023, 6, 2)),
            _row(station_id="A", d=date(2023, 6, 15)),
            _row(station_id="A", d=date(2023, 6, 1)),
        ]
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", rows_in)
        rows_out = snapshot.read_band_partitions(str(tmp_path), "lag_fill")
        assert [(r["station_id"], r["date"]) for r in rows_out] == [
            ("A", date(2023, 6, 1)), ("A", date(2023, 6, 15)), ("B", date(2023, 6, 2)),
        ]


class TestReadStations:
    def test_reads_stations_parquet(self, tmp_path, monkeypatch):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([
            {"station_id": "A", "lon": -112.0, "lat": 33.4},
            {"station_id": "B", "lon": -100.0, "lat": 40.0},
        ])
        pq.write_table(table, str(tmp_path / "stations.parquet"))
        stations = snapshot.read_stations(str(tmp_path))
        assert len(stations) == 2
        assert {s["station_id"] for s in stations} == {"A", "B"}


class TestWriteStations:
    def test_roundtrip_via_read_stations(self, tmp_path):
        rows = [
            {"station_id": "B", "lon": -100.0, "lat": 40.0, "climate_zone": "Dfb"},
            {"station_id": "A", "lon": -112.0, "lat": 33.4, "climate_zone": "BWh"},
        ]
        snapshot.write_stations(str(tmp_path), rows)
        stations = snapshot.read_stations(str(tmp_path))
        assert {s["station_id"] for s in stations} == {"A", "B"}
        assert next(s for s in stations if s["station_id"] == "A")["climate_zone"] == "BWh"

    def test_sorted_by_station_id(self, tmp_path):
        snapshot.write_stations(str(tmp_path), [
            {"station_id": "C", "lon": 0.0}, {"station_id": "A", "lon": 0.0}, {"station_id": "B", "lon": 0.0},
        ])
        stations = snapshot.read_stations(str(tmp_path))
        assert [s["station_id"] for s in stations] == ["A", "B", "C"]


class TestComputeHoldout:
    def _stations(self, zone_counts: dict[str, int]) -> list[dict]:
        stations = []
        for zone, n in zone_counts.items():
            stations.extend({"station_id": f"{zone}{i:03d}", "climate_zone": zone} for i in range(n))
        return stations

    def test_holds_out_15_percent_per_zone(self):
        stations = self._stations({"Cfb": 100, "BWh": 20})
        holdout = snapshot.compute_holdout(stations)
        cfb_held = {sid for sid in holdout if sid.startswith("Cfb")}
        bwh_held = {sid for sid in holdout if sid.startswith("BWh")}
        assert len(cfb_held) == 15  # round(0.15 * 100)
        assert len(bwh_held) == 3  # round(0.15 * 20)

    def test_thin_zone_holds_out_at_least_one(self):
        """A zone with too few stations for round(0.15 * n) to reach 1 must
        still hold out exactly 1 -- max(1, ...) in the plan's own rule, so
        a thin zone is never silently excluded from provisional scoring."""
        stations = self._stations({"EF": 3})
        holdout = snapshot.compute_holdout(stations)
        assert len(holdout) == 1

    def test_deterministic_across_calls(self):
        stations = self._stations({"Cfb": 50})
        assert snapshot.compute_holdout(stations) == snapshot.compute_holdout(stations)

    def test_not_salted_by_snapshot_version(self):
        """Unlike score.score_band's bias-CV fold assignment, the
        provisional holdout is deliberately stable release-over-release --
        compute_holdout takes no fold_salt/snapshot_version argument at all."""
        import inspect

        params = inspect.signature(snapshot.compute_holdout).parameters
        assert "fold_salt" not in params and "snapshot_version" not in params

    def test_zones_pool_independently(self):
        """A large zone's stations must not affect a small zone's own cut
        -- each zone's md5 ranking and holdout count are computed within
        that zone alone."""
        small_only = snapshot.compute_holdout(self._stations({"BWh": 20}))
        combined = snapshot.compute_holdout(self._stations({"Cfb": 100, "BWh": 20}))
        bwh_small_only = {sid for sid in small_only if sid.startswith("BWh")}
        bwh_combined = {sid for sid in combined if sid.startswith("BWh")}
        assert bwh_small_only == bwh_combined


class TestWriteHoldoutAndReadHoldout:
    def test_returns_none_when_file_absent(self, tmp_path):
        assert snapshot.read_holdout(str(tmp_path)) is None

    def test_reads_holdout_station_ids(self, tmp_path):
        import json
        (tmp_path / "holdout.json").write_text(json.dumps(["A", "B", "C"]))
        assert snapshot.read_holdout(str(tmp_path)) == {"A", "B", "C"}

    def test_write_then_read_roundtrip(self, tmp_path):
        snapshot.write_holdout(str(tmp_path), {"A", "B", "C"})
        assert snapshot.read_holdout(str(tmp_path)) == {"A", "B", "C"}


class TestManifest:
    def test_write_read_roundtrip(self, tmp_path):
        rel_path, sha256, row_count = snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        snapshot.write_manifest(
            str(tmp_path), "v2026.08", [{"path": rel_path, "sha256": sha256, "row_count": row_count}],
            generating_commit="abc123", package_version="0.1.0",
            license_info={"code": "Apache-2.0", "data": "CC-BY-4.0"},
            f7_caveat="forecast bands are temperature-only for 2023",
            f3_disclosure="pop_density_per_km2 from LandScan over a 1km station buffer",
        )
        manifest = snapshot.read_manifest(str(tmp_path))
        assert manifest["snapshot_version"] == "v2026.08"
        assert manifest["partitions"][0]["sha256"] == sha256

    def test_verify_manifest_passes_for_untampered_snapshot(self, tmp_path):
        rel_path, sha256, row_count = snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        snapshot.write_manifest(
            str(tmp_path), "v2026.08", [{"path": rel_path, "sha256": sha256, "row_count": row_count}],
            generating_commit="abc123", package_version="0.1.0",
            license_info={}, f7_caveat="", f3_disclosure="",
        )
        snapshot.verify_manifest(str(tmp_path))  # must not raise

    def test_verify_manifest_raises_on_tampered_partition(self, tmp_path):
        rel_path, sha256, row_count = snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        snapshot.write_manifest(
            str(tmp_path), "v2026.08", [{"path": rel_path, "sha256": sha256, "row_count": row_count}],
            generating_commit="abc123", package_version="0.1.0",
            license_info={}, f7_caveat="", f3_disclosure="",
        )
        # Tamper: overwrite the partition with a second, different row set.
        snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row(station_id="TAMPERED")])
        with pytest.raises(ValueError, match="manifest verification"):
            snapshot.verify_manifest(str(tmp_path))

    def test_verify_manifest_raises_on_missing_partition(self, tmp_path):
        rel_path, sha256, row_count = snapshot.write_partition(str(tmp_path), "lag_fill", "2023-06", [_row()])
        snapshot.write_manifest(
            str(tmp_path), "v2026.08", [{"path": rel_path, "sha256": sha256, "row_count": row_count}],
            generating_commit="abc123", package_version="0.1.0",
            license_info={}, f7_caveat="", f3_disclosure="",
        )
        import os
        os.remove(tmp_path / rel_path)
        with pytest.raises(ValueError, match="manifest verification"):
            snapshot.verify_manifest(str(tmp_path))
