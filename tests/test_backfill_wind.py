"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

Unit tests for scripts/backfill_wind.py — no DB/S3/CDS calls, everything mocked."""

import contextlib
import os
import sys
import threading
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import backfill_wind as bw
import era5


class TestWindBackfillVariables:
    def test_requests_base_three_plus_wind_only_not_solar(self):
        # extract_era5_means requires t2m/d2m/sp (non-optional _find_var),
        # so the base 3 must still be requested even though only wind_ms
        # is used from the result. ssrd is deliberately NOT requested --
        # this backfill has no use for solar radiation, and skipping it is
        # the real, correctly-scoped savings over the full 6-variable
        # _TRAINING_ERA5_VARIABLES list.
        assert set(era5._ERA5_VARIABLES).issubset(set(bw._WIND_BACKFILL_VARIABLES))
        assert era5._WIND_U_VARIABLE in bw._WIND_BACKFILL_VARIABLES
        assert era5._WIND_V_VARIABLE in bw._WIND_BACKFILL_VARIABLES
        assert era5._SOLAR_RADIATION_VARIABLE not in bw._WIND_BACKFILL_VARIABLES
        assert len(bw._WIND_BACKFILL_VARIABLES) == 5


class TestConfiguredAccountCount:
    def test_counts_only_configured_slots(self):
        with patch.object(bw.era5, "all_account_indices", return_value=[0, 1, 2]), patch.object(
            bw.era5, "account_configured", side_effect=lambda i: i in (0, 1),
        ):
            assert bw._configured_account_count() == 2


class TestCheckpoint:
    def test_load_returns_empty_set_when_no_checkpoint_file_exists(self, tmp_path):
        with patch.object(bw, "_checkpoint_path", return_value=str(tmp_path / "nonexistent.done")):
            assert bw._load_completed_segment_starts("ZZ") == set()

    def test_append_then_load_round_trips(self, tmp_path):
        path = str(tmp_path / "US.done")
        with patch.object(bw, "_checkpoint_path", return_value=path):
            bw._append_completed_segment_starts("US", [date(2023, 1, 1), date(2023, 2, 1)])
            assert bw._load_completed_segment_starts("US") == {"2023-01-01", "2023-02-01"}

    def test_append_with_empty_list_does_not_create_a_file(self, tmp_path):
        path = str(tmp_path / "US.done")
        with patch.object(bw, "_checkpoint_path", return_value=path):
            bw._append_completed_segment_starts("US", [])
            assert not os.path.exists(path)

    def test_append_is_additive_across_multiple_calls(self, tmp_path):
        # Simulates two separate runs: first run completes Jan, a later
        # re-run (after a partial failure) completes Feb -- both must
        # accumulate, not overwrite.
        path = str(tmp_path / "US.done")
        with patch.object(bw, "_checkpoint_path", return_value=path):
            bw._append_completed_segment_starts("US", [date(2023, 1, 1)])
            bw._append_completed_segment_starts("US", [date(2023, 2, 1)])
            assert bw._load_completed_segment_starts("US") == {"2023-01-01", "2023-02-01"}


class TestDistinctStations:
    def test_maps_db_rows_to_station_dicts(self):
        with patch.object(bw.db, "execute", return_value=[
            {"station_id": "USW00023183", "lon": -112.0, "lat": 33.4},
        ]) as mock_exec:
            stations = bw._distinct_stations("US")
        assert stations == [{"station_id": "USW00023183", "lon": -112.0, "lat": 33.4}]
        args, kwargs = mock_exec.call_args
        assert args[1] == ("US",)


class TestRegionDateRange:
    def test_returns_min_max_date_for_region(self):
        with patch.object(bw.db, "execute", return_value=[{"mn": "2023-01-01", "mx": "2023-12-31"}]):
            start, end = bw._region_date_range("US")
        assert start == "2023-01-01"
        assert end == "2023-12-31"


class TestBackfillCountry:
    # A single-month window (not a full year) keeps most tests focused on
    # one segment -- test_acquires_the_lock_fresh_per_calendar_month_segment
    # below is the one that exercises a real multi-segment span.
    def _patch_all(self, stations=None, nighttime_wind=None, account_index=0, date_range=None):
        stations = stations if stations is not None else [
            {"station_id": "S1", "lon": -112.0, "lat": 33.4},
        ]
        nighttime_wind = nighttime_wind if nighttime_wind is not None else {
            "S1": {"2023-01-01": 3.2, "2023-01-02": 2.8},
        }
        date_range = date_range if date_range is not None else (date(2023, 1, 1), date(2023, 1, 31))

        @contextlib.contextmanager
        def fake_lock():
            yield account_index

        return patch.multiple(
            bw,
            _distinct_stations=MagicMock(return_value=stations),
            _region_date_range=MagicMock(return_value=date_range),
            # Checkpointing does real file I/O at a fixed /tmp path keyed
            # only by region -- isolate tests from real filesystem state
            # (and from each other) by default. Checkpoint-specific
            # behavior is covered by its own dedicated tests below, which
            # override these individually.
            _load_completed_segment_starts=MagicMock(return_value=set()),
            _append_completed_segment_starts=MagicMock(),
        ), patch.multiple(
            bw.bts,
            stations_bbox=MagicMock(return_value="fake-bbox"),
            stations_to_geojson=MagicMock(return_value={"type": "FeatureCollection", "features": []}),
            _timezones_for_stations=MagicMock(return_value={"S1": "America/Phoenix"}),
            _era5_download_lock=fake_lock,
            _daily_mean_nighttime_wind=MagicMock(return_value=nighttime_wind),
        ), patch.multiple(
            bw.era5,
            download_era5=MagicMock(return_value="/tmp/fake.nc"),
            extract_era5_means=MagicMock(return_value=[{"name": "S1", "datetime": "2023-01-01T00:00:00"}]),
        ), patch.object(bw.db, "execute_many")

    def test_no_stations_skips_without_error(self):
        with patch.object(bw, "_distinct_stations", return_value=[]):
            bw.backfill_country("EMPTY")  # must not raise

    def test_requests_the_wind_backfill_variable_list(self):
        p1, p2, p3, p4 = self._patch_all()
        with p1, p2, p3, p4, patch("os.unlink"):
            bw.backfill_country("US")
            _, kwargs = bw.era5.download_era5.call_args
            assert kwargs["variables"] == bw._WIND_BACKFILL_VARIABLES
            assert kwargs["dataset"] == "reanalysis-era5-land"

    def test_passes_the_acquired_lock_slots_account_index(self):
        p1, p2, p3, p4 = self._patch_all(account_index=2)
        with p1, p2, p3, p4, patch("os.unlink"):
            bw.backfill_country("US")
            _, kwargs = bw.era5.download_era5.call_args
            assert kwargs["account_index"] == 2

    def test_updates_nighttime_wind_ms_via_upsert_style_update(self):
        p1, p2, p3, p4 = self._patch_all(nighttime_wind={"S1": {"2023-01-01": 3.2, "2023-01-02": 2.8}})
        with p1, p2, p3, p4, patch("os.unlink"):
            bw.backfill_country("US")
            sql, rows = bw.db.execute_many.call_args[0]
            assert "UPDATE ghcn_training SET nighttime_wind_ms" in sql
            assert "WHERE station_id = %s AND date = %s" in sql
            assert set(rows) == {(3.2, "S1", "2023-01-01"), (2.8, "S1", "2023-01-02")}

    def test_empty_nighttime_wind_result_does_not_call_execute_many(self):
        p1, p2, p3, p4 = self._patch_all(nighttime_wind={})
        with p1, p2, p3, p4, patch("os.unlink"):
            bw.backfill_country("US")
            bw.db.execute_many.assert_not_called()

    def test_temp_netcdf_file_is_always_unlinked(self):
        p1, p2, p3, p4 = self._patch_all()
        with p1, p2, p3, p4, patch("os.unlink") as mock_unlink:
            bw.backfill_country("US")
        mock_unlink.assert_called_once_with("/tmp/fake.nc")

    def test_acquires_the_lock_fresh_per_calendar_month_segment(self):
        # The whole point of the per-segment restructure: a country's lock
        # slot must be released after EACH calendar-month download, not held
        # for the entire multi-month span. A Jan-Mar window is 3 segments,
        # so download_era5 (and therefore the lock) must be invoked 3 times,
        # not once for the full range. Segments now run concurrently on a
        # thread pool, so completion order isn't guaranteed -- assert the
        # SET of requested ranges, not a fixed order.
        p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 3, 31)))
        with p1, p2, p3, p4, patch("os.unlink"):
            bw.backfill_country("US")
            assert bw.era5.download_era5.call_count == 3
            call_ranges = {(c.args[1], c.args[2]) for c in bw.era5.download_era5.call_args_list}
            assert call_ranges == {
                (date(2023, 1, 1), date(2023, 1, 31)),
                (date(2023, 2, 1), date(2023, 2, 28)),
                (date(2023, 3, 1), date(2023, 3, 31)),
            }

    def test_concurrency_is_bounded_by_configured_account_count(self):
        with patch.object(bw, "_configured_account_count", return_value=2):
            p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 4, 30)))
            with p1, p2, p3, p4, patch("os.unlink"), patch.object(
                bw.concurrent.futures, "ThreadPoolExecutor", wraps=bw.concurrent.futures.ThreadPoolExecutor,
            ) as mock_pool:
                bw.backfill_country("US")
                assert mock_pool.call_args.kwargs["max_workers"] == 2

    def test_concurrency_never_exceeds_segment_count(self):
        # A single-month window shouldn't spin up 3 idle worker threads for 1 task.
        with patch.object(bw, "_configured_account_count", return_value=3):
            p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 1, 31)))
            with p1, p2, p3, p4, patch("os.unlink"), patch.object(
                bw.concurrent.futures, "ThreadPoolExecutor", wraps=bw.concurrent.futures.ThreadPoolExecutor,
            ) as mock_pool:
                bw.backfill_country("US")
                assert mock_pool.call_args.kwargs["max_workers"] == 1

    def test_hourly_rows_accumulate_across_segments(self):
        p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 2, 28)))
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw.bts, "_daily_mean_nighttime_wind", MagicMock(return_value={"S1": {"2023-01-01": 3.2}}),
        ) as mock_nightly:
            bw.backfill_country("US")
        # extract_era5_means returns 1 row per call; 2 segments -> 2 rows passed through.
        hourly_arg = mock_nightly.call_args[0][0]
        assert len(hourly_arg) == 2

    def test_one_failed_segment_does_not_lose_the_others(self):
        # Confirmed live 2026-07-19: an IAM AccessDeniedException on ONE
        # segment (account_index=1's secret) propagated out of
        # `future.result()` and killed the whole country's backfill,
        # discarding every OTHER already-succeeded month. A single flaky
        # segment must be isolated -- logged and skipped -- not fatal to
        # the rest.
        def flaky_download(bbox, seg_start, seg_end, **kwargs):
            if seg_start == date(2023, 2, 1):
                raise RuntimeError("simulated transient CDS failure")
            return "/tmp/fake.nc"

        p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 3, 31)))
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw.era5, "download_era5", side_effect=flaky_download,
        ), patch.object(
            bw.bts, "_daily_mean_nighttime_wind", MagicMock(return_value={"S1": {"2023-01-01": 3.2}}),
        ) as mock_nightly:
            bw.backfill_country("US")  # must not raise
        # Jan + Mar succeed (1 row each via extract_era5_means); Feb's
        # segment raised and contributed nothing, but didn't abort the rest.
        hourly_arg = mock_nightly.call_args[0][0]
        assert len(hourly_arg) == 2

    def test_all_segments_failing_still_completes_without_raising(self):
        # With every segment failing, hourly stays empty -- realistically
        # _daily_mean_nighttime_wind([], tz_map) returns {}, so pin the
        # fixture's mock to that instead of its default non-empty canned
        # value (which would falsely imply a DB write happened).
        p1, p2, p3, p4 = self._patch_all(nighttime_wind={})
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw.era5, "download_era5", side_effect=RuntimeError("simulated"),
        ):
            bw.backfill_country("US")  # must not raise
            bw.db.execute_many.assert_not_called()


    def test_skips_segments_already_completed_per_checkpoint(self):
        p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 3, 31)))
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw, "_load_completed_segment_starts", return_value={"2023-02-01"},
        ):
            bw.backfill_country("US")
            assert bw.era5.download_era5.call_count == 2
            call_starts = {c.args[1] for c in bw.era5.download_era5.call_args_list}
            assert call_starts == {date(2023, 1, 1), date(2023, 3, 1)}

    def test_all_segments_already_completed_skips_entirely_without_fetching(self):
        p1, p2, p3, p4 = self._patch_all()
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw, "_load_completed_segment_starts", return_value={"2023-01-01"},
        ):
            bw.backfill_country("US")
            bw.era5.download_era5.assert_not_called()
            bw.db.execute_many.assert_not_called()

    def test_appends_only_successfully_fetched_segments_to_checkpoint(self):
        def flaky_download(bbox, seg_start, seg_end, **kwargs):
            if seg_start == date(2023, 2, 1):
                raise RuntimeError("simulated")
            return "/tmp/fake.nc"

        p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 3, 31)))
        with p1, p2, p3, p4, patch("os.unlink"), patch.object(
            bw.era5, "download_era5", side_effect=flaky_download,
        ), patch.object(
            bw, "_append_completed_segment_starts",
        ) as mock_append:
            bw.backfill_country("US")
        args, _ = mock_append.call_args
        region_arg, starts_arg = args
        assert region_arg == "US"
        # Jan + Mar succeeded; Feb raised and must NOT be checkpointed.
        assert set(starts_arg) == {date(2023, 1, 1), date(2023, 3, 1)}

    def test_netcdf_parsing_is_serialized_across_concurrent_threads(self):
        # era5.extract_era5_means opens the NetCDF via xarray/HDF5, which
        # is not safe for concurrent access from multiple threads in one
        # process -- confirmed live 2026-07-19 via a real segfault in
        # libhdf5 when 3 segments finished at nearly the same moment.
        # _NETCDF_PARSE_LOCK must serialize every call to it even though
        # downloads run concurrently.
        state = {"active": 0, "overlap_detected": False}
        state_lock = threading.Lock()

        def slow_extract(nc_path, geojson_obj):
            with state_lock:
                state["active"] += 1
                if state["active"] > 1:
                    state["overlap_detected"] = True
            time.sleep(0.05)
            with state_lock:
                state["active"] -= 1
            return [{"name": "S1", "datetime": "2023-01-01T00:00:00"}]

        with patch.object(bw, "_configured_account_count", return_value=3):
            p1, p2, p3, p4 = self._patch_all(date_range=(date(2023, 1, 1), date(2023, 3, 31)))
            with p1, p2, p3, p4, patch("os.unlink"), patch.object(
                bw.era5, "extract_era5_means", side_effect=slow_extract,
            ):
                bw.backfill_country("US")
        assert not state["overlap_detected"]


class TestMain:
    def test_runs_the_ghcn_training_table_migration_before_any_country(self):
        # Confirmed live 2026-07-19: this script ran a full country's CDS
        # fetch (12/12 segments) only to fail at the final UPDATE with
        # "column nighttime_wind_ms does not exist" -- build_training_set.py
        # already calls this at startup; this script must too.
        with patch.object(bw.ghcn, "create_ghcn_training_table") as mock_migrate, patch.object(
            bw, "backfill_country",
        ), patch.object(sys, "argv", ["backfill_wind.py", "--countries", "US"]):
            bw.main()
        mock_migrate.assert_called_once()

    def test_one_countrys_total_failure_does_not_stop_the_rest(self):
        calls = []

        def fake_backfill(region):
            calls.append(region)
            if region == "BAD":
                raise RuntimeError("simulated total failure")

        with patch.object(bw.ghcn, "create_ghcn_training_table"), patch.object(
            bw, "backfill_country", side_effect=fake_backfill,
        ), patch.object(
            sys, "argv", ["backfill_wind.py", "--countries", "BAD,GOOD"],
        ):
            bw.main()  # must not raise
        assert calls == ["BAD", "GOOD"]
