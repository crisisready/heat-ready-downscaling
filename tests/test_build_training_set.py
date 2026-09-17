"""
PROVENANCE: extracted verbatim (not merged) from crisisready/heat-risk-data-api's origin/feature/downscaling-phase4-model-training at tip commit 9d8a678c594fbe2878033373b750cc8465a9d80e on 2026-07-27. See this repository's own PROVENANCE.md for why this branch was extracted rather than merged.

NOT IN THE ORIGINAL PLAN'S SECTION 5.1 FILE LIST -- found and extracted
2026-07-27 during the build_training_set.py forward-port (section 5.3):
the plan's extraction list omitted this file, but it exists on the branch
with real coverage for the single riskiest file in this whole port.
Extracted for the same reason as the other four tests/test_*.py files.

Unit tests for scripts/build_training_set.py — no network, DB, or AWS calls."""

import contextlib
import fcntl
import json
import os
import sys
import threading
import time
from datetime import date
from unittest.mock import MagicMock, patch

import cads_api_client.processing
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_training_set as bts

_STATIONS = [
    {"station_id": "USW00023183", "lon": -112.0036, "lat": 33.4278, "elevation_m": 339.2, "name": "PHOENIX AP"},
    {"station_id": "USW00003812", "lon": -111.9800, "lat": 33.4500, "elevation_m": 350.0, "name": "TEMPE"},
]


# ---------------------------------------------------------------------------
# stations_to_geojson / stations_bbox
# ---------------------------------------------------------------------------


class TestStationsToGeojson:
    def test_one_feature_per_station_named_by_station_id(self):
        fc = bts.stations_to_geojson(_STATIONS)
        assert fc["type"] == "FeatureCollection"
        assert [f["properties"]["name"] for f in fc["features"]] == ["USW00023183", "USW00003812"]

    def test_buffer_polygon_centered_on_station(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        coords = fc["features"][0]["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        d = bts._STATION_BUFFER_DEG
        assert min(lons) == -112.0036 - d
        assert max(lons) == -112.0036 + d
        assert min(lats) == 33.4278 - d
        assert max(lats) == 33.4278 + d


class TestStationsBbox:
    def test_padded_bbox_covers_all_stations(self):
        bbox = bts.stations_bbox(_STATIONS, pad_deg=0.5)
        west, south, east, north = (float(v) for v in bbox.split(","))
        assert west == -112.0036 - 0.5
        assert east == -111.98 + 0.5
        assert south == 33.4278 - 0.5
        assert north == 33.45 + 0.5


# ---------------------------------------------------------------------------
# fetch_era5_land_for_stations
# ---------------------------------------------------------------------------


_HOURLY_ROWS = [
    {
        "name": "USW00023183", "datetime": f"2016-06-15T{h:02d}:00:00",
        "t2m": 30.0 + h * 0.1, "d2m": 15.0, "sp": 95000.0, "wind_ms": 2.0 + h * 0.05,
    }
    for h in range(24)
] + [
    {
        "name": "USW00023183", "datetime": f"2016-06-16T{h:02d}:00:00",
        "t2m": 28.0, "d2m": 15.0, "sp": 95000.0, "wind_ms": 3.0,
    }
    for h in range(24)
]


class TestFetchEra5LandForStations:
    def test_downloads_era5_land_dataset_and_extracts_daily_values(self):
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=_HOURLY_ROWS) as mock_extract, \
             patch("os.unlink") as mock_unlink:
            daily, humidity, nighttime_wind = bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
            )

        mock_dl.assert_called_once()
        args, kwargs = mock_dl.call_args
        assert kwargs.get("dataset") == "reanalysis-era5-land" or "reanalysis-era5-land" in args
        mock_extract.assert_called_once()
        mock_unlink.assert_called_once_with("/tmp/fake.nc")

        assert "USW00023183" in daily
        assert set(daily["USW00023183"].keys())  # at least one full local day aggregated
        for day_vals in daily["USW00023183"].values():
            assert "tmax" in day_vals and "tmin" in day_vals

        assert "USW00023183" in nighttime_wind
        for wind_val in nighttime_wind["USW00023183"].values():
            assert 2.0 <= wind_val <= 4.0  # sane wind range for this fixture

    def test_requests_the_expanded_wind_radiation_variable_list(self):
        """This is the one call site that must request wind/radiation, not
        era5's 3-variable production default -- it's what makes the
        nighttime_wind_ms covariate possible at all."""
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.unlink"):
            bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))

        _, kwargs = mock_dl.call_args
        assert kwargs.get("variables") == bts._TRAINING_ERA5_VARIABLES
        assert "10m_u_component_of_wind" in kwargs["variables"]
        assert "10m_v_component_of_wind" in kwargs["variables"]
        assert "surface_solar_radiation_downwards" in kwargs["variables"]
        # base 3 production variables still requested too, not replaced
        assert set(bts.era5._ERA5_VARIABLES).issubset(set(kwargs["variables"]))

    def test_download_call_is_made_while_holding_the_era5_lock(self):
        """Regression coverage for the cross-process ERA5-download lock
        (safe concurrent-country runs, 2026-07-18): the CDS call itself
        must happen strictly between acquire and release, not just
        somewhere in the same function."""
        events = []
        real_lock = bts.fcntl.flock

        def _tracking_flock(f, op):
            # The free-slot-first lock probes with LOCK_EX | LOCK_NB (not
            # bare LOCK_EX), so classify on the UNLOCK bit rather than an
            # exact match -- anything that isn't LOCK_UN is an acquire,
            # blocking or not.
            events.append("UNLOCK" if op == bts.fcntl.LOCK_UN else "LOCK")
            return real_lock(f, op)

        def _tracking_download(*args, **kwargs):
            events.append("DOWNLOAD")
            return "/tmp/fake.nc"

        with patch.object(bts.fcntl, "flock", side_effect=_tracking_flock), \
             patch.object(bts.era5, "download_era5", side_effect=_tracking_download), \
             patch.object(bts.era5, "extract_era5_means", return_value=_HOURLY_ROWS), \
             patch("os.unlink"):
            bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))

        assert events == ["LOCK", "DOWNLOAD", "UNLOCK"]

    def test_requested_window_padded_two_days_each_side(self):
        """aggregate_hourly_to_daily only returns local days with all 24
        hours present, and a local day can span into the adjacent UTC
        calendar day in either direction depending on the station's UTC
        offset sign -- confirmed live 2026-07-18: an unpadded single-day
        pull for a real UTC-7 station produced zero complete local days.
        The pad is 2 days (not 1) so that align_obs_window's +/-1-day
        shifted lookups at the edges of the requested window also land
        inside the guaranteed-complete range."""
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.unlink"):
            bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))

        called_args, kwargs = mock_dl.call_args
        start_date_arg = kwargs.get("start_date", called_args[1] if len(called_args) > 1 else None)
        end_date_arg = kwargs.get("end_date", called_args[2] if len(called_args) > 2 else None)
        assert start_date_arg == date(2016, 6, 13)
        assert end_date_arg == date(2016, 6, 18)

    def test_temp_file_unlinked_even_if_extract_fails(self):
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc"), \
             patch.object(bts.era5, "extract_era5_means", side_effect=RuntimeError("boom")), \
             patch("os.unlink") as mock_unlink:
            try:
                bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))
            except RuntimeError:
                pass
        mock_unlink.assert_called_once_with("/tmp/fake.nc")

    def test_multi_chunk_window_makes_one_download_per_chunk_and_merges_results(self):
        """A window longer than _ERA5_CHUNK_DAYS must split into multiple
        download_era5 calls (bounds per-request netCDF size for a country-wide,
        multi-year training pull) rather than one giant request -- and the
        per-chunk daily/humidity results must merge, not clobber each other."""
        import itertools
        from datetime import timedelta as _td

        def _full_bucket_day(day_iso):
            # aggregate_hourly_to_daily buckets on (dt_utc - 6h).date() -- a
            # complete bucket for `day_iso` needs UTC hours 06:00 that day
            # through 05:00 the next day, so two consecutive full UTC days
            # of hourly rows are needed to produce exactly one complete,
            # predictably-keyed local day under UTC.
            d0 = date.fromisoformat(day_iso)
            d1 = d0 + _td(days=1)
            return [
                {"name": "USW00023183", "datetime": f"{d.isoformat()}T{h:02d}:00:00",
                 "t2m": 20.0, "d2m": 10.0, "sp": 95000.0}
                for d in (d0, d1) for h in range(24)
            ]

        rows_by_call = itertools.cycle([_full_bucket_day("2016-01-15"), _full_bucket_day("2016-06-15")])
        # Pin a small chunk size explicitly -- this test exercises the OUTER
        # date-chunking mechanism itself, independent of whatever the real
        # production _ERA5_CHUNK_DAYS default is (2026-07-18: deliberately
        # raised past 365 so a single training year relies on era5.py's own
        # calendar-month splitting instead, not this outer mechanism). Also
        # passthroughs era5._split_by_calendar_month (2026-09-17: now called
        # directly by _resolve_era5_range, one calendar-month sub-range at a
        # time, instead of being hidden inside the mocked download_era5 call)
        # so this test still isolates the outer mechanism -- without it, a
        # 90-day outer chunk spanning a real month boundary would split
        # again here and try to merge the same fake "/tmp/fake.nc" path
        # with itself, which is a different behavior from what this test
        # means to cover.
        with patch.object(bts, "_ERA5_CHUNK_DAYS", 90), \
             patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", side_effect=rows_by_call), \
             patch.object(bts, "_timezones_for_stations", return_value={"USW00023183": "UTC"}), \
             patch.object(bts.era5, "_split_by_calendar_month", side_effect=lambda s, e: [(s, e)]), \
             patch("os.unlink"):
            daily, humidity, nighttime_wind = bts.fetch_era5_land_for_stations(
                _STATIONS[:1], date(2016, 1, 1), date(2016, 12, 31),
            )

        assert mock_dl.call_count == len(bts._date_chunks(date(2016, 1, 1), date(2016, 12, 31), chunk_days=90))
        assert mock_dl.call_count > 1
        # both chunks' aggregated days must be present, not just the last chunk's
        assert "2016-01-15" in daily["USW00023183"]
        assert "2016-06-15" in daily["USW00023183"]
        assert "2016-01-15" in humidity["USW00023183"]
        assert "2016-06-15" in humidity["USW00023183"]


class TestCdsRejectedRetry:
    """issue #648: a CDS job that resolves to 'rejected' raised a generic
    cads_api_client.processing.ProcessingFailedError("Unknown API state 'rejected'") that
    propagated all the way out of build_rows_for_country and killed the whole process -- lost
    the still-queued US/VM/VQ countries in one lane to a single rejected request live
    2026-09-16. _download_era5_with_rejected_retry must retry that specific outcome with
    backoff instead of letting it propagate on the first attempt."""

    def _rejected_error(self):
        return cads_api_client.processing.ProcessingFailedError("Unknown API state 'rejected'")

    def test_retries_on_rejected_and_eventually_succeeds(self, monkeypatch):
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        with patch.object(bts.era5, "download_era5",
                           side_effect=[self._rejected_error(), self._rejected_error(), "/tmp/fake.nc"]) as mock_dl, \
             patch.object(bts.time, "sleep") as mock_sleep:
            nc_path, owns = bts._download_era5_with_rejected_retry(
                None, "-115,30,-110,35", date(2016, 6, 15), date(2016, 6, 16), "US_Af",
            )
        assert (nc_path, owns) == ("/tmp/fake.nc", True)
        assert mock_dl.call_count == 3
        # backoff, not a flat delay -- each retry waits longer than the last
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == sorted(delays) and len(set(delays)) == len(delays)

    def test_gives_up_after_max_retries_and_raises(self, monkeypatch):
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        with patch.object(bts.era5, "download_era5",
                           side_effect=self._rejected_error()) as mock_dl, \
             patch.object(bts.time, "sleep"), \
             pytest.raises(cads_api_client.processing.ProcessingFailedError):
            bts._download_era5_with_rejected_retry(
                None, "-115,30,-110,35", date(2016, 6, 15), date(2016, 6, 16), "US_Af",
            )
        # the first attempt plus exactly _CDS_REJECTED_MAX_RETRIES retries, not one more/fewer
        assert mock_dl.call_count == bts._CDS_REJECTED_MAX_RETRIES + 1

    def test_non_rejected_processing_failure_is_not_retried(self, monkeypatch):
        """'dismissed'/'deleted' terminal states raise the exact same exception class as
        'rejected' -- only the 'rejected' outcome specifically is worth retrying."""
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        dismissed_error = cads_api_client.processing.ProcessingFailedError("API state 'dismissed'")
        with patch.object(bts.era5, "download_era5", side_effect=dismissed_error) as mock_dl, \
             patch.object(bts.time, "sleep") as mock_sleep, \
             pytest.raises(cads_api_client.processing.ProcessingFailedError):
            bts._download_era5_with_rejected_retry(
                None, "-115,30,-110,35", date(2016, 6, 15), date(2016, 6, 16), "US_Af",
            )
        mock_dl.assert_called_once()
        mock_sleep.assert_not_called()

    def test_unrelated_exception_is_not_retried(self, monkeypatch):
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        with patch.object(bts.era5, "download_era5", side_effect=RuntimeError("connection reset")) as mock_dl, \
             patch.object(bts.time, "sleep") as mock_sleep, \
             pytest.raises(RuntimeError, match="connection reset"):
            bts._download_era5_with_rejected_retry(
                None, "-115,30,-110,35", date(2016, 6, 15), date(2016, 6, 16), "US_Af",
            )
        mock_dl.assert_called_once()
        mock_sleep.assert_not_called()

    def test_backoff_sleep_happens_with_the_lock_released(self, monkeypatch):
        """A rejection means CDS is contended -- holding this process's CDS concurrency slot
        idle for the whole backoff delay would only make that worse for every other lane/
        process waiting on the same slot, so each retry must re-acquire the lock fresh rather
        than sleep while still holding it."""
        lock_held = []

        @contextlib.contextmanager
        def tracking_lock(deprioritize_account_index=None):
            lock_held.append(True)
            try:
                yield 0
            finally:
                lock_held.append(False)

        monkeypatch.setattr(bts, "_era5_download_lock", tracking_lock)

        def fake_sleep(_delay):
            # the lock must already be released by the time we're sleeping
            assert lock_held[-1] is False

        with patch.object(bts.era5, "download_era5",
                           side_effect=[self._rejected_error(), "/tmp/fake.nc"]), \
             patch.object(bts.time, "sleep", side_effect=fake_sleep):
            bts._download_era5_with_rejected_retry(
                None, "-115,30,-110,35", date(2016, 6, 15), date(2016, 6, 16), "US_Af",
            )


class TestResolveEra5Range:
    """Found live, 2026-09-17, diagnosing a Nishant-flagged "our CDS request pattern must be
    wrong" report: _resolve_era5_segment used to be called ONCE for a whole multi-month
    padded range, so era5.download_era5's own internal calendar-month split happened inside
    ONE _era5_download_lock acquisition and ONE _download_era5_with_rejected_retry attempt --
    a rejection on month 12 of 14 threw away months 1-11's already-succeeded downloads and
    re-submitted all 14 from scratch, and the lock was held (confirmed live via flock -n/
    fuser on the bastion: 11+ hours, unreleased) for the WHOLE sequence, starving every other
    batch waiting on the same CDS account. _resolve_era5_range fixes this by splitting the
    range into calendar-month sub-segments itself (era5._split_by_calendar_month -- same
    boundaries era5.download_era5 would have produced internally, so this doesn't change
    request count/boundaries for the single-account, single-batch case) and resolving each
    one independently."""

    def test_multi_month_range_resolves_one_segment_at_a_time(self, monkeypatch):
        """A range spanning 4 calendar months must call era5.download_era5 once per month,
        each with that month's own exact start/end, not once for the whole range."""
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        with patch.object(bts.era5, "download_era5", return_value="/tmp/seg.nc") as mock_dl, \
             patch.object(bts.era5, "_merge_era5_segments", return_value="/tmp/merged.nc"):
            nc_path, owns, used_cache = bts._resolve_era5_range(
                "-10,40,10,50", date(2016, 1, 30), date(2016, 4, 2), "batch", cache_dir=None)
        assert (nc_path, owns, used_cache) == ("/tmp/merged.nc", True, False)
        assert mock_dl.call_count == 4
        called_ranges = [(c.args[1], c.args[2]) for c in mock_dl.call_args_list]
        assert called_ranges == [
            (date(2016, 1, 30), date(2016, 1, 31)),
            (date(2016, 2, 1), date(2016, 2, 29)),  # 2016 is a leap year
            (date(2016, 3, 1), date(2016, 3, 31)),
            (date(2016, 4, 1), date(2016, 4, 2)),
        ]

    def test_rejection_on_a_later_segment_does_not_redownload_earlier_ones(self, monkeypatch):
        """The core bug: a rejection used to blow up the WHOLE multi-month call, discarding
        already-succeeded months and re-submitting them on every retry. Per-segment
        resolution means only the failed segment retries."""
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        rejected = cads_api_client.processing.ProcessingFailedError("Unknown API state 'rejected'")
        with patch.object(bts.era5, "download_era5",
                           side_effect=["/tmp/jan.nc", rejected, "/tmp/feb.nc"]) as mock_dl, \
             patch.object(bts.era5, "_merge_era5_segments", return_value="/tmp/merged.nc"), \
             patch.object(bts.time, "sleep"):
            nc_path, owns, used_cache = bts._resolve_era5_range(
                "-10,40,10,50", date(2016, 1, 30), date(2016, 2, 5), "batch", cache_dir=None)
        assert (nc_path, owns, used_cache) == ("/tmp/merged.nc", True, False)
        # January (already succeeded) called exactly once; February retried exactly once
        # after its rejection -- never re-submitted January to get there.
        assert mock_dl.call_count == 3
        jan_calls = [c for c in mock_dl.call_args_list if c.args[1] == date(2016, 1, 30)]
        assert len(jan_calls) == 1

    def test_lock_is_acquired_independently_per_segment_not_once_for_the_whole_range(self):
        """The other half of the same bug: one process holding the account lock for an
        entire multi-month sequence starves every other batch waiting on that CDS account
        the whole time. The lock must be released between segments, not held across all of
        them, so a waiting batch can interleave."""
        held_during_each_call = []

        @contextlib.contextmanager
        def tracking_lock(deprioritize_account_index=None):
            held_during_each_call.append(True)
            try:
                yield 0
            finally:
                held_during_each_call[-1] = False

        def fake_download(bbox, start, end, **kwargs):
            # lock must be held DURING this call...
            assert held_during_each_call[-1] is True
            return f"/tmp/{start.isoformat()}.nc"

        with patch.object(bts, "_era5_download_lock", tracking_lock), \
             patch.object(bts.era5, "download_era5", side_effect=fake_download) as mock_dl, \
             patch.object(bts.era5, "_merge_era5_segments", return_value="/tmp/merged.nc"):
            bts._resolve_era5_range("-10,40,10,50", date(2016, 1, 30), date(2016, 3, 2),
                                     "batch", cache_dir=None)
        assert mock_dl.call_count == 3
        # ...but released again after every single call, not just after the last one --
        # a snapshot mid-loop would find it free, unlike the old whole-range hold.
        assert held_during_each_call == [False, False, False]

    def test_used_cache_is_true_when_any_segment_is_a_cache_hit(self, tmp_path):
        """Round-1 /code-review finding, real: a multi-segment merge always returns a fresh
        temp file (owns_nc_path=True), even when every constituent segment was a pure cache
        hit -- fetch_era5_land_for_stations used to gate its corrupted-cache retry on
        owns_nc_path alone, which this made permanently dead for any multi-month range (the
        exact case this whole module exists for). used_cache must reflect whether ANY segment
        actually came from cache, independent of owns_nc_path."""
        cache_dir = str(tmp_path)
        # Pre-seed a real cache hit for the first (January) segment only.
        jan_cache_path = bts._era5_segment_cache_path(
            cache_dir, "batch", date(2016, 1, 30), date(2016, 1, 31), "-10,40,10,50",
            "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES)
        with open(jan_cache_path, "w") as f:
            f.write("fake cached netcdf bytes")

        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 0

        with patch.object(bts, "_era5_download_lock", fake_lock), \
             patch.object(bts.era5, "download_era5", return_value="/tmp/feb.nc") as mock_dl, \
             patch.object(bts.era5, "_merge_era5_segments", return_value="/tmp/merged.nc"):
            nc_path, owns, used_cache = bts._resolve_era5_range(
                "-10,40,10,50", date(2016, 1, 30), date(2016, 2, 5), "batch", cache_dir=cache_dir)
        assert (nc_path, owns) == ("/tmp/merged.nc", True)
        assert used_cache is True
        # February (no cache entry) still had to be downloaded; January didn't.
        mock_dl.assert_called_once()
        assert mock_dl.call_args.args[1] == date(2016, 2, 1)


class TestFetchEra5LandForStationsCache:
    def test_cache_miss_downloads_and_persists_a_copy(self, tmp_path):
        """A fresh cache_dir has no file yet -- must download normally
        (through the lock) and leave a persisted copy behind for a future
        relaunch to reuse."""
        cache_dir = str(tmp_path / "era5_cache")
        real_file = tmp_path / "fake.nc"
        real_file.write_bytes(b"fake netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(real_file)) as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]):
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        mock_dl.assert_called_once()
        cached_files = list(os.scandir(cache_dir))
        assert len(cached_files) == 1
        assert cached_files[0].name.startswith("US_test_2016-06-13_2016-06-18_")
        assert open(cached_files[0].path, "rb").read() == b"fake netcdf bytes"
        # the original downloaded tmp file is still cleaned up as before
        assert not real_file.exists()

    def test_cache_hit_skips_lock_and_download_entirely(self, tmp_path):
        """A relaunched run must not re-acquire the lock or re-download a
        segment it already has cached -- this is the entire point of
        cache_dir (resuming a killed multi-hour pooled write without
        re-paying every CDS round-trip)."""
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        bbox = bts.stations_bbox(_STATIONS)
        cache_path = bts._era5_segment_cache_path(
            cache_dir, "US_test", date(2016, 6, 13), date(2016, 6, 18),
            bbox, "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )
        with open(cache_path, "wb") as f:
            f.write(b"cached netcdf bytes")

        with patch.object(bts, "_era5_download_lock") as mock_lock, \
             patch.object(bts.era5, "download_era5") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]) as mock_extract, \
             patch("os.unlink") as mock_unlink:
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        mock_lock.assert_not_called()
        mock_dl.assert_not_called()
        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][0] == cache_path
        # a cache hit's file must survive for the next call/relaunch -- never unlinked
        mock_unlink.assert_not_called()

    def test_cache_key_differs_across_zones_sharing_the_same_padded_window(self, tmp_path):
        """Two different zones/batches downloading the same calendar window
        must never collide on one cache file -- bbox is part of the key,
        batch_label alone (cosmetic) is not relied on for uniqueness."""
        cache_dir = str(tmp_path / "era5_cache")
        path_a = bts._era5_segment_cache_path(
            cache_dir, "US_BSk", date(2023, 1, 1), date(2023, 2, 1),
            "-115,30,-110,35", "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )
        path_b = bts._era5_segment_cache_path(
            cache_dir, "US_BWh", date(2023, 1, 1), date(2023, 2, 1),
            "-100,20,-95,25", "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )
        assert path_a != path_b

    def test_no_cache_dir_means_no_persistence_and_no_lookup(self):
        """Default (cache_dir=None) behavior must be unchanged from before
        this fix -- always downloads, never touches a cache path."""
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.unlink") as mock_unlink:
            bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))

        mock_dl.assert_called_once()
        mock_unlink.assert_called_once_with("/tmp/fake.nc")

    def test_corrupted_cache_entry_is_deleted_and_retried_once(self, tmp_path):
        """Round-2 review finding, real: a cache hit was previously trusted purely on
        os.path.exists with no integrity check -- a corrupted/truncated cache entry
        permanently poisoned that segment (every relaunch hit the same exception at the same
        cache_path forever). Must now delete the bad entry and retry as a genuine fresh
        download, succeeding on the retry."""
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        bbox = bts.stations_bbox(_STATIONS)
        cache_path = bts._era5_segment_cache_path(
            cache_dir, "US_test", date(2016, 6, 13), date(2016, 6, 18),
            bbox, "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )
        with open(cache_path, "wb") as f:
            f.write(b"corrupted netcdf bytes")

        fresh_file = tmp_path / "fresh.nc"
        fresh_file.write_bytes(b"real netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(fresh_file)) as mock_dl, \
             patch.object(bts.era5, "extract_era5_means",
                           side_effect=[ValueError("bad netcdf"), []]) as mock_extract:
            daily, _, _ = bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        assert mock_extract.call_count == 2
        mock_dl.assert_called_once()  # exactly one real fresh download for the retry
        assert not os.path.exists(cache_path.replace(".nc", "-does-not-exist"))  # sanity
        # the original corrupted entry is gone, replaced by the freshly re-downloaded one
        assert os.path.exists(cache_path)
        assert open(cache_path, "rb").read() == b"real netcdf bytes"
        assert not fresh_file.exists()  # the fresh download's own tmp file was still cleaned up

    def test_freshly_downloaded_segment_failing_to_parse_is_not_retried(self, tmp_path):
        """A FRESH download (not a cache hit) failing to parse is a real error -- retrying it
        as if it were a caching artifact would mask a genuine bug/bad CDS response."""
        cache_dir = str(tmp_path / "era5_cache")
        real_file = tmp_path / "fake.nc"
        real_file.write_bytes(b"fake netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(real_file)) as mock_dl, \
             patch.object(bts.era5, "extract_era5_means",
                           side_effect=ValueError("bad netcdf")) as mock_extract, \
             pytest.raises(ValueError):
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        mock_dl.assert_called_once()
        mock_extract.assert_called_once()
        assert not real_file.exists()  # still cleaned up despite the raised exception

    def test_cache_reappears_after_lock_acquired_skips_redundant_download(self, tmp_path):
        """Round-2 review finding, real: the cache-existence check used to sit entirely
        outside _era5_download_lock, so two processes racing on the same missing segment could
        both see a miss before either's write landed and both redundantly download. Simulates
        a concurrent process finishing its own download+cache-write during the time this call
        spent waiting for the lock -- the segment must be reused, not re-downloaded."""
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        bbox = bts.stations_bbox(_STATIONS)
        cache_path = bts._era5_segment_cache_path(
            cache_dir, "US_test", date(2016, 6, 13), date(2016, 6, 18),
            bbox, "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )

        class _FakeLock:
            def __enter__(self):
                # simulates a concurrent process finishing its own cache write while this
                # call was waiting to acquire the lock
                with open(cache_path, "wb") as f:
                    f.write(b"written by a concurrent process")
                return 0

            def __exit__(self, *a):
                return False

        with patch.object(bts, "_era5_download_lock", return_value=_FakeLock()), \
             patch.object(bts.era5, "download_era5") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]) as mock_extract:
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        mock_dl.assert_not_called()  # the re-check inside the lock found it -- no download
        mock_extract.assert_called_once_with(cache_path, mock_extract.call_args[0][1])

    def test_cache_path_sanitizes_batch_label(self, tmp_path):
        """Round-2 review finding, real: batch_label was used raw in the cache filename,
        inconsistent with this same file's own established sanitization precedent for the
        identical value (build_rows_for_country's own country_key)."""
        cache_dir = str(tmp_path / "era5_cache")
        path = bts._era5_segment_cache_path(
            cache_dir, "US/../evil name!", date(2023, 1, 1), date(2023, 2, 1),
            "-115,30,-110,35", "reanalysis-era5-land", bts._TRAINING_ERA5_VARIABLES,
        )
        fname = os.path.basename(path)
        assert "/" not in fname and ".." not in fname and " " not in fname and "!" not in fname
        assert fname.startswith("USevilname_")

    def test_cache_persist_prefers_hard_link_over_full_copy(self, tmp_path):
        """Round-2 review finding, real: shutil.copy2 always did a full duplicate read+write of
        a potentially multi-GB file even when cache_dir shares a filesystem with the download's
        own tmp dir, where a hard link is a same-filesystem, near-zero-cost alternative."""
        cache_dir = str(tmp_path / "era5_cache")
        real_file = tmp_path / "fake.nc"
        real_file.write_bytes(b"fake netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(real_file)), \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.link", wraps=os.link) as mock_link, \
             patch("shutil.copy2") as mock_copy:
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        mock_link.assert_called_once()
        mock_copy.assert_not_called()

    def test_cache_persist_falls_back_to_copy_when_hard_link_fails(self, tmp_path):
        """Cross-device (or hard-link-unsupported) cache_dir must still work via a real copy."""
        cache_dir = str(tmp_path / "era5_cache")
        real_file = tmp_path / "fake.nc"
        real_file.write_bytes(b"fake netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(real_file)), \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.link", side_effect=OSError("cross-device link")):
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        cached_files = list(os.scandir(cache_dir))
        assert len(cached_files) == 1
        assert open(cached_files[0].path, "rb").read() == b"fake netcdf bytes"

    def test_cache_persist_exception_class_widened_beyond_oserror(self, tmp_path):
        """Round-2 review finding, real: the persist block only caught OSError -- a non-OSError
        failure during the copy/rename used to escape it entirely, skipping the caller's own
        try/finally cleanup of the downloaded temp file for that exception class."""
        cache_dir = str(tmp_path / "era5_cache")
        real_file = tmp_path / "fake.nc"
        real_file.write_bytes(b"fake netcdf bytes")

        with patch.object(bts.era5, "download_era5", return_value=str(real_file)), \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.link", side_effect=OSError), \
             patch("shutil.copy2", side_effect=MemoryError("simulated")):
            # must not raise -- caching is best-effort, and the temp file must still be cleaned up
            bts.fetch_era5_land_for_stations(
                _STATIONS, date(2016, 6, 15), date(2016, 6, 16),
                cache_dir=cache_dir, batch_label="US_test",
            )

        assert not real_file.exists()
        assert list(os.scandir(cache_dir)) == []


class TestSweepStaleEra5CacheTmpFiles:
    def setup_method(self):
        bts._ERA5_CACHE_TMP_SWEPT_DIRS.clear()

    def test_removes_stale_tmp_files_older_than_threshold(self, tmp_path):
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        stale = os.path.join(cache_dir, "seg_2023-01-01_2023-02-01_abc.nc.tmp.123.deadbeef")
        with open(stale, "wb") as f:
            f.write(b"orphaned")
        old_time = time.time() - bts._ERA5_CACHE_TMP_STALE_AGE_SECONDS - 3600
        os.utime(stale, (old_time, old_time))

        bts._sweep_stale_era5_cache_tmp_files(cache_dir)

        assert not os.path.exists(stale)

    def test_leaves_recent_tmp_files_alone(self, tmp_path):
        """A tmp file from a genuinely-in-progress concurrent write must survive."""
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        recent = os.path.join(cache_dir, "seg_2023-01-01_2023-02-01_abc.nc.tmp.123.deadbeef")
        with open(recent, "wb") as f:
            f.write(b"in progress")

        bts._sweep_stale_era5_cache_tmp_files(cache_dir)

        assert os.path.exists(recent)

    def test_only_sweeps_once_per_cache_dir_per_process(self, tmp_path):
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        with patch("os.listdir", wraps=os.listdir) as mock_listdir:
            bts._sweep_stale_era5_cache_tmp_files(cache_dir)
            bts._sweep_stale_era5_cache_tmp_files(cache_dir)
        mock_listdir.assert_called_once()

    def test_leaves_non_tmp_cache_files_alone(self, tmp_path):
        cache_dir = str(tmp_path / "era5_cache")
        os.makedirs(cache_dir)
        real_cache_entry = os.path.join(cache_dir, "seg_2023-01-01_2023-02-01_abc.nc")
        with open(real_cache_entry, "wb") as f:
            f.write(b"a real, complete cache entry")
        old_time = time.time() - bts._ERA5_CACHE_TMP_STALE_AGE_SECONDS - 3600
        os.utime(real_cache_entry, (old_time, old_time))

        bts._sweep_stale_era5_cache_tmp_files(cache_dir)

        assert os.path.exists(real_cache_entry)


class TestDateChunks:
    def test_splits_long_window_into_bounded_chunks(self):
        chunks = bts._date_chunks(date(2016, 1, 1), date(2016, 12, 31), chunk_days=90)
        assert chunks[0] == (date(2016, 1, 1), date(2016, 3, 30))
        # contiguous: each chunk starts the day after the previous one ends
        for (s1, e1), (s2, _) in zip(chunks, chunks[1:]):
            assert s2 == e1 + bts.timedelta(days=1)
        assert chunks[-1][1] == date(2016, 12, 31)

    def test_window_shorter_than_chunk_size_is_a_single_chunk(self):
        chunks = bts._date_chunks(date(2016, 6, 15), date(2016, 6, 16), chunk_days=90)
        assert chunks == [(date(2016, 6, 15), date(2016, 6, 16))]

    def test_default_chunk_size_fits_a_full_training_year_in_one_outer_chunk(self):
        """_ERA5_CHUNK_DAYS is deliberately > 365 (2026-07-18): a single
        training year (the real scripts/build_training_set.py use case)
        must produce exactly ONE outer chunk, so era5.download_era5's own
        calendar-month splitting is the only thing decomposing the request
        -- confirmed live this cuts total CDS requests by ~30% versus a
        smaller outer chunk size that also creates extra tiny segments at
        every internal chunk boundary's padding."""
        chunks = bts._date_chunks(date(2023, 1, 1), date(2023, 12, 31))
        assert len(chunks) == 1
        assert chunks == [(date(2023, 1, 1), date(2023, 12, 31))]


class TestFetchEra5LandForStationsAccountIndex:
    def test_download_era5_receives_the_acquired_lock_slots_account_index(self, monkeypatch):
        """The 2-slot _era5_download_lock's yielded account_index must
        reach era5.download_era5 -- not just always 0 regardless of which
        slot was actually acquired (docs/plan-2026-07-19-cds-dual-account-
        split.md, Phase 3, test item 18)."""
        @contextlib.contextmanager
        def fake_lock(deprioritize_account_index=None):
            yield 1

        monkeypatch.setattr(bts, "_era5_download_lock", fake_lock)
        with patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", return_value=[]), \
             patch("os.unlink"):
            bts.fetch_era5_land_for_stations(_STATIONS, date(2016, 6, 15), date(2016, 6, 16))
        assert mock_dl.call_args.kwargs["account_index"] == 1


# ---------------------------------------------------------------------------
# _era5_download_lock -- 2-slot cross-process file lock (docs/plan-2026-07-
# 19-cds-dual-account-split.md, Phase 3). A real fcntl.flock exercised
# against tmp_path files, not a mock and not threading.Lock -- the lock's
# whole reason to exist is that it serializes SEPARATE OS PROCESSES sharing
# one filesystem (see the function's own docstring), and flock is scoped to
# the open file description, not the process, so two independent open()
# calls on the same path genuinely contend even within one test process --
# a faithful stand-in for "two separate build_training_set.py invocations."
# ---------------------------------------------------------------------------

class TestEra5DownloadLockFreeSlotFirst:
    def _patch_lock_paths(self, tmp_path, monkeypatch):
        lock_a = str(tmp_path / "a.lock")
        lock_b = str(tmp_path / "b.lock")
        monkeypatch.setattr(bts, "_ERA5_DOWNLOAD_LOCK_PATHS", [lock_a, lock_b])
        return lock_a, lock_b

    def test_yields_account_index_zero_when_account_one_unconfigured(self, tmp_path, monkeypatch):
        """Q2/test-item-17 single-lock fallback: with ERA5_SECRET_ARN_2
        unset, only lock A / account_index 0 is ever considered."""
        monkeypatch.delenv("ERA5_SECRET_ARN_2", raising=False)
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        self._patch_lock_paths(tmp_path, monkeypatch)
        with bts._era5_download_lock() as account_index:
            assert account_index == 0

    def test_yields_some_configured_slot_when_both_are_free(self, tmp_path, monkeypatch):
        """Was test_yields_account_index_zero_when_both_slots_free, asserting
        "first free configured slot wins". That assertion encoded the defect:
        scanning in index order meant account 0 won EVERY uncontended
        acquisition, so accounts 1 and 2 went unused (measured 322/111/82 jobs
        in one bastion session, with all 106 rejections on account 0). The
        contract is now "some free configured slot", with the choice randomised
        -- see TestEra5DownloadLockAccountRotation for the load-spreading
        assertions."""
        monkeypatch.setenv("ERA5_SECRET_ARN_2", "arn:aws:secretsmanager:us-east-1:123:secret:era5-2")
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        self._patch_lock_paths(tmp_path, monkeypatch)
        with bts._era5_download_lock() as account_index:
            assert account_index in (0, 1)

    def test_falls_through_to_second_slot_when_first_is_already_held(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ERA5_SECRET_ARN_2", "arn:aws:secretsmanager:us-east-1:123:secret:era5-2")
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        holder = open(lock_a, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with bts._era5_download_lock() as account_index:
                assert account_index == 1
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    def test_second_slot_release_does_not_free_the_first(self, tmp_path, monkeypatch):
        """Releasing the acquired (second) slot must never touch the first
        slot's independent lock state -- the two files are unrelated locks,
        not one logical 2-unit semaphore."""
        monkeypatch.setenv("ERA5_SECRET_ARN_2", "arn:aws:secretsmanager:us-east-1:123:secret:era5-2")
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        holder = open(lock_a, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with bts._era5_download_lock() as account_index:
                assert account_index == 1
            # slot A is still held by `holder` after the `with` block exits --
            # a fresh non-blocking attempt on it must still fail.
            probe = open(lock_a, "w")
            with pytest.raises(OSError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            probe.close()
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    def test_blocks_on_first_slot_until_it_frees_when_every_slot_is_held(self, tmp_path, monkeypatch):
        """When every configured slot is busy, acquisition must eventually
        succeed (blocking on the first configured slot) rather than raising
        or spinning without making progress -- this is what guarantees the
        training build as a whole still completes under contention, just
        serialized, exactly as a single (pre-dual-account) lock always
        behaved."""
        monkeypatch.delenv("ERA5_SECRET_ARN_2", raising=False)  # only slot A configured
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        holder = open(lock_a, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []

        def release_soon():
            time.sleep(0.2)
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

        releaser = threading.Thread(target=release_soon)
        releaser.start()
        try:
            with bts._era5_download_lock() as account_index:
                results.append(account_index)
        finally:
            releaser.join(timeout=5)
        assert results == [0]

    def test_grabs_a_different_slot_freeing_first_instead_of_waiting_on_the_one_scanned_first(
        self, tmp_path, monkeypatch,
    ):
        """Real bug found live 2026-08-20, first genuine 3-way concurrent
        build_training_set.py run: when every configured slot was busy at
        scan time, the original implementation committed to a single
        blocking flock() on configured[0] specifically -- so a caller could
        stay blocked on THAT slot for many minutes even after a DIFFERENT
        slot freed up seconds later (confirmed live: two real processes both
        saw all 3 slots busy; one correctly grabbed slot 'a' once it freed,
        the other stayed blocked on 'a' specifically for 8+ minutes while
        slot 'b' sat completely idle the whole time -- a real, measured
        throughput loss). This test holds BOTH slots, releases slot B
        (the second one) first while slot A stays held, and asserts the
        waiting caller grabs B -- not that it stays stuck waiting on A."""
        monkeypatch.setenv("ERA5_SECRET_ARN_2", "arn:aws:secretsmanager:us-east-1:123:secret:era5-2")
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        holder_a = open(lock_a, "w")
        fcntl.flock(holder_a, fcntl.LOCK_EX | fcntl.LOCK_NB)
        holder_b = open(lock_b, "w")
        fcntl.flock(holder_b, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release_b_only_soon():
            time.sleep(0.3)
            fcntl.flock(holder_b, fcntl.LOCK_UN)
            holder_b.close()

        releaser = threading.Thread(target=release_b_only_soon)
        releaser.start()
        try:
            with bts._era5_download_lock() as account_index:
                # slot A (index 0) is still held by holder_a throughout --
                # only slot B (index 1) ever freed, so this MUST be 1.
                assert account_index == 1
        finally:
            releaser.join(timeout=5)
            fcntl.flock(holder_a, fcntl.LOCK_UN)
            holder_a.close()

    def test_lock_released_on_normal_exit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ERA5_SECRET_ARN_2", raising=False)
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        with bts._era5_download_lock():
            pass

        # A fresh acquisition attempt must succeed immediately -- the first
        # `with` block's release must have actually happened, not merely
        # returned without unlocking.
        probe = open(lock_a, "w")
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        fcntl.flock(probe, fcntl.LOCK_UN)
        probe.close()

    def test_lock_released_even_if_body_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ERA5_SECRET_ARN_2", raising=False)
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        lock_a, lock_b = self._patch_lock_paths(tmp_path, monkeypatch)

        with pytest.raises(RuntimeError):
            with bts._era5_download_lock():
                raise RuntimeError("simulated download failure")

        probe = open(lock_a, "w")
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        fcntl.flock(probe, fcntl.LOCK_UN)
        probe.close()


class TestDailyMeanSpecificHumidity:
    def test_computes_mean_specific_humidity_per_station_day(self):
        result = bts._daily_mean_specific_humidity(_HOURLY_ROWS, {"USW00023183": "UTC"})
        assert "USW00023183" in result
        for q in result["USW00023183"].values():
            assert 0.0 < q < 0.05  # sane specific humidity range (kg/kg)

    def test_rows_missing_d2m_or_sp_are_skipped(self):
        rows = [{"name": "S1", "datetime": "2016-06-15T12:00:00", "t2m": 30.0, "d2m": None, "sp": 95000.0}]
        result = bts._daily_mean_specific_humidity(rows, {"S1": "UTC"})
        assert result == {}


class TestDailyMeanNighttimeWind:
    def test_computes_nighttime_only_mean_wind_per_station_day(self):
        # Under "UTC", local_hour == UTC hour, so the 18:00-05:59 nighttime
        # window is hours 18-23 and 0-5 (12 nighttime hours per local day).
        result = bts._daily_mean_nighttime_wind(_HOURLY_ROWS, {"USW00023183": "UTC"})
        assert "USW00023183" in result
        for wind_val in result["USW00023183"].values():
            assert 2.0 <= wind_val <= 4.0

    def test_daytime_hours_excluded_from_the_mean(self):
        """A day built entirely from daytime (06:00-17:59) hours must yield
        no entry at all -- not a 0.0 -- since this covariate is nighttime-
        only by design (Oke 1982's "clear and calm NIGHTS" framing)."""
        rows = [
            {"name": "S1", "datetime": f"2016-06-15T{h:02d}:00:00", "wind_ms": 10.0}
            for h in range(6, 18)  # only daytime hours, no nighttime hours at all
        ]
        result = bts._daily_mean_nighttime_wind(rows, {"S1": "UTC"})
        assert result == {}

    def test_rows_missing_wind_ms_are_skipped_not_imputed(self):
        """A missing wind_ms (e.g. production-shaped input with no
        wind/radiation) must be skipped, never defaulted to some fallback
        value -- the model's explicit non-goal is imputing a missing
        covariate (docs/plan-2026-07-19-era5-wind-radiation-integration.md
        section 14)."""
        rows = [{"name": "S1", "datetime": "2016-06-15T20:00:00", "wind_ms": None}]
        result = bts._daily_mean_nighttime_wind(rows, {"S1": "UTC"})
        assert result == {}

    def test_uses_the_same_06h_local_day_shift_as_specific_humidity(self):
        """A nighttime hour just after local midnight (00:00-05:59) belongs
        to the PREVIOUS local day under the -6h shift, same as
        _daily_mean_specific_humidity and heat_calcs.aggregate_hourly_to_daily
        -- so nighttime_wind_ms lines up with the same local day as
        grid_specific_humidity_kgkg for a given station-day. Supplies the
        full 12-hour nighttime window (18:00-23:00 on the 15th, 00:00-05:00
        on the 16th) since a partial window no longer yields a result
        (see test_partial_nighttime_window_yields_no_result)."""
        rows = [
            {"name": "S1", "datetime": f"2016-06-15T{h:02d}:00:00", "wind_ms": 5.0}
            for h in range(18, 24)
        ] + [
            {"name": "S1", "datetime": f"2016-06-16T{h:02d}:00:00", "wind_ms": 5.0}
            for h in range(0, 6)
        ]
        result = bts._daily_mean_nighttime_wind(rows, {"S1": "UTC"})
        assert result == {"S1": {"2016-06-15": 5.0}}

    def test_partial_nighttime_window_yields_no_result(self):
        """Fewer than the full 12 nighttime hours (18:00-05:59) for a local
        day must yield no entry at all, matching aggregate_hourly_to_daily's
        own all-or-nothing completeness rule -- a noisier partial mean must
        never silently stand in for the real thing (code review 2026-07-19)."""
        rows = [
            {"name": "S1", "datetime": f"2016-06-15T{h:02d}:00:00", "wind_ms": 5.0}
            for h in range(18, 24)
        ]  # only 6 of the 12 expected nighttime hours
        result = bts._daily_mean_nighttime_wind(rows, {"S1": "UTC"})
        assert result == {}

    def test_nan_wind_ms_excluded_like_a_missing_value_not_averaged_in(self):
        """A NaN wind_ms (e.g. era5.extract_era5_means's masked-cell rescue
        failing to find a valid substitute within range) must be excluded
        exactly like a missing value, never averaged in -- sum() over a
        NaN-containing list silently produces NaN, which `is not None` and
        so would NOT have been caught by upsert_ghcn_training_rows' None-
        only default substitution (code review 2026-07-19)."""
        import math
        rows = [
            {"name": "S1", "datetime": f"2016-06-15T{h:02d}:00:00", "wind_ms": math.nan}
            for h in range(18, 24)
        ] + [
            {"name": "S1", "datetime": f"2016-06-16T{h:02d}:00:00", "wind_ms": 5.0}
            for h in range(0, 6)
        ]
        result = bts._daily_mean_nighttime_wind(rows, {"S1": "UTC"})
        # only 6 of the 12 nighttime hours have a finite value -- below the
        # completeness threshold, so this local day yields no result at all,
        # and critically the result is not NaN.
        assert result == {}


# ---------------------------------------------------------------------------
# snapshot_covariates_for_stations
# ---------------------------------------------------------------------------


class TestSnapshotCovariatesForStations:
    def _patch_pop_and_lst(self, pop_density=None, lst_anomaly=None):
        pop_density = pop_density if pop_density is not None else {}
        lst_anomaly = lst_anomaly if lst_anomaly is not None else {}
        return patch.object(bts, "_population_density_by_station", return_value=pop_density), \
            patch.object(bts, "_lst_warm_season_anomaly_by_station", return_value=lst_anomaly)

    def test_maps_extractor_outputs_by_station_id(self):
        canopy = {"USW00023183": {"canopy_height_mean_m": 5.0, "canopy_frac_over_3m": 0.2}}
        # extract_worldcover/extract_elevation both return {"status",
        # "resume_index", "results"} now (2026-07-27 forward-port,
        # snapshot_covariates_for_stations' own comment) -- these mocks must
        # match that shape since the production code unwraps ["results"].
        worldcover = {"status": "complete", "resume_index": None,
                      "results": {"USW00023183": {"wc_built_frac": 0.6, "wc_tree_frac": 0.1, "wc_water_frac": 0.0}}}
        ghsl = {"USW00023183": {"ghsl_urban_fraction": 0.8}}
        elevation = {"status": "complete", "resume_index": None,
                     "results": {"USW00023183": {"elevation_rel_to_gridcell_m": 12.5}}}
        p3, p4 = self._patch_pop_and_lst(
            pop_density={"USW00023183": 1500.0}, lst_anomaly={"USW00023183": 2.1},
        )

        with patch.object(bts.vulnerability, "extract_canopy", return_value=canopy) as mock_canopy, \
             patch.object(bts.vulnerability, "extract_worldcover", return_value=worldcover), \
             patch.object(bts.vulnerability, "extract_ghsl_smod", return_value=ghsl), \
             patch.object(bts.dem, "extract_elevation", return_value=elevation), \
             p3, p4:
            result = bts.snapshot_covariates_for_stations(
                _STATIONS[:1], "some-bucket", batch_label="US",
                landscan_bucket="ls-bucket", landscan_key="ls-key",
            )

        mock_canopy.assert_called_once()
        _, kwargs = mock_canopy.call_args
        # project_id includes station count + a stable hash of station IDs
        # (not just batch_label) so a differently-sized run for the same
        # country doesn't collide with -- and silently misapply -- another
        # run's persisted canopy resume state.
        assert kwargs.get("project_id") == bts._canopy_resume_project_id("US", _STATIONS[:1])
        assert kwargs["project_id"].startswith("ghcn_training_US_1_")

        row = result["USW00023183"]
        assert row["canopy_height_mean_m"] == 5.0
        assert row["wc_built_frac"] == 0.6
        assert row["ghsl_urban_fraction"] == 0.8
        assert row["elevation_rel_to_gridcell_m"] == 12.5
        assert row["pop_density_per_km2"] == 1500.0
        assert row["lst_warm_season_anomaly_c"] == 2.1

    def test_missing_station_in_extractor_output_degrades_to_none(self):
        p3, p4 = self._patch_pop_and_lst()
        with patch.object(bts.vulnerability, "extract_canopy", return_value={}), \
             patch.object(bts.vulnerability, "extract_worldcover", return_value={"status": "complete", "resume_index": None, "results": {}}), \
             patch.object(bts.vulnerability, "extract_ghsl_smod", return_value={}), \
             patch.object(bts.dem, "extract_elevation", return_value={"status": "complete", "resume_index": None, "results": {}}), \
             p3, p4:
            result = bts.snapshot_covariates_for_stations(
                _STATIONS[:1], "some-bucket", batch_label="US",
                landscan_bucket="ls-bucket", landscan_key="ls-key",
            )

        row = result["USW00023183"]
        assert row["canopy_height_mean_m"] is None
        assert row["elevation_rel_to_gridcell_m"] is None
        assert row["pop_density_per_km2"] is None
        assert row["lst_warm_season_anomaly_c"] is None


# ---------------------------------------------------------------------------
# _population_density_by_station / _lst_warm_season_anomaly_by_station
# ---------------------------------------------------------------------------


class TestPopulationDensityByStation:
    def test_divides_population_by_real_buffer_area(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        with patch.object(bts.landscan, "extract_population",
                           return_value=[{"name": "USW00023183", "population": 100.0}]), \
             patch.object(bts, "_station_buffer_area_km2", return_value=2.0):
            result = bts._population_density_by_station(fc, "ls-bucket", "ls-key")
        assert result == {"USW00023183": 50.0}

    def test_zero_area_degrades_to_none_not_zerodivisionerror(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        with patch.object(bts.landscan, "extract_population",
                           return_value=[{"name": "USW00023183", "population": 100.0}]), \
             patch.object(bts, "_station_buffer_area_km2", return_value=0.0):
            result = bts._population_density_by_station(fc, "ls-bucket", "ls-key")
        assert result == {"USW00023183": None}


class TestLstWarmSeasonAnomalyByStation:
    def test_table_name_derived_from_sanitized_batch_label(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        with patch.object(bts.lst, "create_lst_table") as mock_create, \
             patch.object(bts.lst, "derive_warm_season_window", return_value={"seasons": []}), \
             patch.object(bts.lst, "compute_warm_season_composite", return_value={}):
            bts._lst_warm_season_anomaly_by_station(fc, "bbox", "US", _STATIONS[:1])
        mock_create.assert_called_once_with("ghcn_training_lst_us")

    def test_loops_extract_lst_watermarked_until_completed(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        composite = {"USW00023183": {"lst_warm_season_anomaly_c": 3.4}}
        with patch.object(bts.lst, "create_lst_table"), \
             patch.object(bts.lst, "derive_warm_season_window",
                          return_value={"seasons": [(date(2016, 6, 1), date(2016, 8, 31))]}), \
             patch.object(bts.lst, "extract_lst_watermarked", side_effect=[
                 ([(date(2016, 6, 15), [{"name": "USW00023183", "scene_date": date(2016, 6, 15)}])], False),
                 ([(date(2016, 7, 1), [{"name": "USW00023183", "scene_date": date(2016, 7, 1)}])], True),
             ]) as mock_extract, \
             patch.object(bts.lst, "upsert_lst_rows") as mock_upsert, \
             patch.object(bts.lst, "compute_warm_season_composite", return_value=composite) as mock_composite:
            result = bts._lst_warm_season_anomaly_by_station(fc, "bbox", "US", _STATIONS[:1])

        assert mock_extract.call_count == 2
        assert mock_upsert.call_count == 2
        # second call resumes after the last scene_date the first call returned
        _, second_kwargs = mock_extract.call_args_list[1]
        assert second_kwargs.get("resume_after") == date(2016, 6, 15)
        assert result == {"USW00023183": 3.4}
        # the reference-radius scoping is always passed -- not an optional
        # opt-in some call sites forget (see lst.compute_warm_season_composite's
        # docstring for why a bare country-wide baseline is a real train/serve bug)
        _, composite_kwargs = mock_composite.call_args
        assert composite_kwargs["name_coords"] == {"USW00023183": (33.4278, -112.0036)}
        assert composite_kwargs["reference_radius_km"] == bts._LST_REFERENCE_RADIUS_KM

    def test_sanitizes_non_alnum_batch_label(self):
        fc = bts.stations_to_geojson(_STATIONS[:1])
        with patch.object(bts.lst, "create_lst_table") as mock_create, \
             patch.object(bts.lst, "derive_warm_season_window", return_value={"seasons": []}), \
             patch.object(bts.lst, "compute_warm_season_composite", return_value={}):
            bts._lst_warm_season_anomaly_by_station(fc, "bbox", "US-1; DROP TABLE x", _STATIONS[:1])
        mock_create.assert_called_once_with("ghcn_training_lst_us1droptablex")


# ---------------------------------------------------------------------------
# build_rows_for_country
# ---------------------------------------------------------------------------


class TestBuildRowsForCountry:
    def _patch_all(self, grid_by_station=None, humidity_by_station=None, nighttime_wind_by_station=None,
                    covariates=None, station_series=None, shift=0, climate_zone="BWh"):
        grid_by_station = grid_by_station if grid_by_station is not None else {}
        humidity_by_station = humidity_by_station if humidity_by_station is not None else {}
        nighttime_wind_by_station = nighttime_wind_by_station if nighttime_wind_by_station is not None else {}
        covariates = covariates if covariates is not None else {}
        # _STATIONS[:1]'s single station_id, keyed as fetch_ghcn_daily_bulk_
        # concurrent's real return shape does -- absent entirely (not an
        # empty-list value) for "no data", matching that function's own
        # documented contract.
        ghcn_by_station = {"USW00023183": station_series} if station_series else {}
        return patch.multiple(
            bts,
            fetch_era5_land_for_stations=MagicMock(
                return_value=(grid_by_station, humidity_by_station, nighttime_wind_by_station)
            ),
            snapshot_covariates_for_stations=MagicMock(return_value=covariates),
        ), patch.multiple(
            bts.ghcn,
            fetch_ghcn_daily_bulk_concurrent=MagicMock(return_value=ghcn_by_station),
            align_obs_window=MagicMock(return_value=shift),
            region_from_station_id=MagicMock(return_value="US"),
            koppen_climate_zone=MagicMock(return_value=climate_zone),
        )

    def test_assembles_row_with_correct_deltas(self):
        p1, p2 = self._patch_all(
            grid_by_station={"USW00023183": {"2016-06-15": {"tmax": 37.0, "tmin": 22.0}}},
            humidity_by_station={"USW00023183": {"2016-06-15": 0.012}},
            nighttime_wind_by_station={"USW00023183": {"2016-06-15": 2.7}},
            covariates={"USW00023183": {"canopy_height_mean_m": 5.0}},
            station_series=[{"date": "2016-06-15", "station_tmax_c": 38.3, "station_tmin_c": 23.3}],
            shift=0,
        )
        with p1, p2:
            rows = bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )

        assert len(rows) == 1
        row = rows[0]
        assert row["station_id"] == "USW00023183"
        assert row["date"] == "2016-06-15"
        assert row["grid_tmax_c"] == 37.0
        assert row["delta_tmax_c"] == 38.3 - 37.0
        assert row["delta_tmin_c"] == 23.3 - 22.0
        assert row["region"] == "US"
        assert row["climate_zone"] == "BWh"
        assert row["obs_window_shift_days"] == 0
        assert row["grid_specific_humidity_kgkg"] == 0.012
        assert row["nighttime_wind_ms"] == 2.7
        assert row["canopy_height_mean_m"] == 5.0

    def test_era5_covariate_and_ghcn_fetch_run_concurrently(self):
        """Regression coverage for the concurrency fix (2026-07-18, extended
        2026-08-03 to 3-way when the GHCN fetch joined the same pool): all
        three independent calls must actually be able to make progress at
        the same time, not just be wrapped in a ThreadPoolExecutor that
        happens to run them one after another on a single reused worker
        thread (which a naive "different thread ident" check can't rule out
        for instant mocks -- the pool is free to dispatch instant calls to
        the same idle thread). A shared Barrier(3) proves it properly: if
        any call were ever serialized onto one thread instead of getting
        its own, the first would block on the barrier forever waiting for
        parties that can never arrive (queued behind it), and the test
        would hang/timeout."""
        import threading

        barrier = threading.Barrier(3, timeout=5)

        def _sync_era5(*args, **kwargs):
            barrier.wait()
            return {}, {}, {}

        def _sync_covariates(*args, **kwargs):
            barrier.wait()
            return {}

        def _sync_ghcn(*args, **kwargs):
            barrier.wait()
            return {}

        with patch.multiple(
            bts,
            fetch_era5_land_for_stations=MagicMock(side_effect=_sync_era5),
            snapshot_covariates_for_stations=MagicMock(side_effect=_sync_covariates),
        ), patch.multiple(
            bts.ghcn,
            fetch_ghcn_daily_bulk_concurrent=MagicMock(side_effect=_sync_ghcn),
        ):
            # Raises threading.BrokenBarrierError (via the executor future)
            # if any of the three calls were ever actually serialized onto
            # one thread.
            bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )

    def test_ghcn_fetch_called_with_all_station_ids_and_passthrough_params(self):
        p1, _ = self._patch_all()
        with p1, \
             patch.object(bts.ghcn, "fetch_ghcn_daily_bulk_concurrent", return_value={}) as mock_fetch, \
             patch.object(bts.ghcn, "align_obs_window", return_value=0), \
             patch.object(bts.ghcn, "region_from_station_id", return_value="US"), \
             patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh"):
            bts.build_rows_for_country(
                "US", _STATIONS, date(2016, 6, 15), date(2016, 6, 16), "bucket", "ls-bucket", "ls-key",
                ghcn_max_workers=16, ghcn_checkpoint_path="/tmp/some_checkpoint.jsonl",
            )
        mock_fetch.assert_called_once_with(
            ["USW00023183", "USW00003812"], date(2016, 6, 15), date(2016, 6, 16),
            max_workers=16, checkpoint_path="/tmp/some_checkpoint.jsonl",
        )

    def test_station_missing_from_ghcn_result_contributes_no_rows_others_unaffected(self):
        """One station absent from fetch_ghcn_daily_bulk_concurrent's
        returned dict (no data, matching that function's own documented
        contract) must not affect a sibling station that DID return rows in
        the same batch."""
        p1, p2 = self._patch_all(
            grid_by_station={
                "USW00023183": {"2016-06-15": {"tmax": 37.0, "tmin": 22.0}},
                "USW00003812": {"2016-06-15": {"tmax": 36.0, "tmin": 21.0}},
            },
        )
        with p1, patch.multiple(
            bts.ghcn,
            fetch_ghcn_daily_bulk_concurrent=MagicMock(return_value={
                "USW00003812": [{"date": "2016-06-15", "station_tmax_c": 35.0, "station_tmin_c": 20.0}],
            }),
            align_obs_window=MagicMock(return_value=0),
            region_from_station_id=MagicMock(return_value="US"),
            koppen_climate_zone=MagicMock(return_value="BWh"),
        ):
            rows = bts.build_rows_for_country(
                "US", _STATIONS, date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )
        assert len(rows) == 1
        assert rows[0]["station_id"] == "USW00003812"

    def test_logs_phase_timing_for_era5_and_covariates(self, caplog):
        import logging as _logging

        p1, p2 = self._patch_all()
        with caplog.at_level(_logging.INFO, logger="build_training_set"), p1, p2:
            bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )
        messages = [r.message for r in caplog.records]
        assert any("ERA5 fetch finished" in m for m in messages)
        assert any("covariate/LST snapshot finished" in m for m in messages)

    def test_nighttime_wind_missing_for_a_station_day_degrades_to_none(self):
        """No entry for a given shifted_day in nighttime_wind_by_station
        (e.g. that station-day genuinely had no nighttime hours) must
        degrade to None, not raise or default to 0.0 -- upsert_ghcn_training_
        rows then stores NULL, and Phase 3's feature-matrix build treats a
        missing covariate as an incomplete row rather than imputing it."""
        p1, p2 = self._patch_all(
            grid_by_station={"USW00023183": {"2016-06-15": {"tmax": 37.0, "tmin": 22.0}}},
            station_series=[{"date": "2016-06-15", "station_tmax_c": 38.3, "station_tmin_c": 23.3}],
        )
        with p1, p2:
            rows = bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )
        assert rows[0]["nighttime_wind_ms"] is None

    def test_applies_obs_window_shift_when_looking_up_grid_value(self):
        p1, p2 = self._patch_all(
            grid_by_station={"USW00023183": {"2016-06-16": {"tmax": 37.0, "tmin": 22.0}}},
            station_series=[{"date": "2016-06-15", "station_tmax_c": 38.3, "station_tmin_c": 23.3}],
            shift=1,
        )
        with p1, p2:
            rows = bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )

        assert len(rows) == 1
        assert rows[0]["obs_window_shift_days"] == 1
        assert rows[0]["grid_tmax_c"] == 37.0

    def test_station_day_with_no_matching_grid_value_is_skipped(self):
        p1, p2 = self._patch_all(
            grid_by_station={"USW00023183": {}},  # no grid data at all
            station_series=[{"date": "2016-06-15", "station_tmax_c": 38.3, "station_tmin_c": 23.3}],
        )
        with p1, p2:
            rows = bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )
        assert rows == []

    def test_station_with_no_ghcn_data_contributes_no_rows(self):
        p1, p2 = self._patch_all(station_series=[])
        with p1, p2:
            rows = bts.build_rows_for_country(
                "US", _STATIONS[:1], date(2016, 6, 15), date(2016, 6, 15), "bucket", "ls-bucket", "ls-key",
            )
        assert rows == []


# ---------------------------------------------------------------------------
# _group_stations_by_country / _canopy_resume_project_id
# ---------------------------------------------------------------------------


class TestSelectGeographicallyCompact:
    def test_returns_all_stations_when_under_cap(self):
        result = bts._select_geographically_compact(_STATIONS, max_count=10)
        assert result == _STATIONS

    def test_selects_cluster_nearest_centroid_not_list_order(self):
        stations = [
            {"station_id": "NEAR1", "lon": 0.0, "lat": 0.0},
            {"station_id": "NEAR2", "lon": 0.1, "lat": 0.1},
            {"station_id": "FAR", "lon": 50.0, "lat": 50.0},
        ]
        # centroid pulled toward the outlier, but the two near-each-other
        # stations are still the closer pair to that centroid than FAR is
        result = bts._select_geographically_compact(stations, max_count=2)
        ids = {s["station_id"] for s in result}
        assert ids == {"NEAR1", "NEAR2"}


class TestSelectGeographicallyStratified:
    def test_returns_all_stations_when_under_cap(self):
        result = bts._select_geographically_stratified(_STATIONS, max_count=10)
        assert result == _STATIONS

    def test_spreads_selection_across_full_extent_not_clustered(self):
        # 4 corners far apart, 1 near the middle. Stratified selection of 2
        # should pick from DIFFERENT grid cells, not the tightest cluster.
        stations = [
            {"station_id": "SW", "lon": -10.0, "lat": -10.0},
            {"station_id": "NE", "lon": 10.0, "lat": 10.0},
            {"station_id": "MID1", "lon": 0.0, "lat": 0.0},
            {"station_id": "MID2", "lon": 0.1, "lat": 0.1},
        ]
        result = bts._select_geographically_stratified(stations, max_count=2)
        lons = [s["lon"] for s in result]
        # a tight-cluster (compact) selection would pick MID1+MID2; stratified
        # must span more of the full [-10, 10] extent than that tiny pair does.
        assert (max(lons) - min(lons)) > 1.0

    def test_bounds_candidates_to_max_extent_around_centroid(self):
        # 3 stations tightly clustered near (0,0), 1 far outlier at (50,50).
        # With max_extent_deg=5, the outlier must never be selectable.
        stations = [
            {"station_id": "A", "lon": 0.0, "lat": 0.0},
            {"station_id": "B", "lon": 0.5, "lat": 0.5},
            {"station_id": "C", "lon": -0.5, "lat": -0.5},
            {"station_id": "FAR", "lon": 50.0, "lat": 50.0},
        ]
        result = bts._select_geographically_stratified(stations, max_count=10, max_extent_deg=5.0)
        ids = {s["station_id"] for s in result}
        assert "FAR" not in ids

    def test_small_extent_country_is_left_unbounded(self):
        # All within a couple degrees -- max_extent_deg=20 should not filter anything.
        result = bts._select_geographically_stratified(_STATIONS, max_count=10, max_extent_deg=20.0)
        assert result == _STATIONS

    def test_never_returns_more_than_max_count(self):
        stations = [{"station_id": f"S{i}", "lon": i * 1.0, "lat": i * 0.5} for i in range(20)]
        result = bts._select_geographically_stratified(stations, max_count=5)
        assert len(result) == 5
        assert len({s["station_id"] for s in result}) == 5  # no duplicates


class TestGroupStationsByCountry:
    def test_groups_by_fips_prefix(self):
        stations = [
            {"station_id": "USW00023183", "lon": -112.0, "lat": 33.0},
            {"station_id": "USW00003812", "lon": -111.9, "lat": 33.4},
            {"station_id": "NIM00065046", "lon": 8.5, "lat": 12.0},
        ]
        grouped = bts._group_stations_by_country(stations)
        assert {s["station_id"] for s in grouped["US"]} == {"USW00023183", "USW00003812"}
        assert {s["station_id"] for s in grouped["NI"]} == {"NIM00065046"}

    def test_empty_input_returns_empty_dict(self):
        assert bts._group_stations_by_country([]) == {}


class TestGroupStationsByZone:
    def test_groups_by_the_zone_koppen_climate_zone_returns(self):
        stations = [
            {"station_id": "A", "lat": 1.0, "lon": 1.0},
            {"station_id": "B", "lat": 2.0, "lon": 2.0},
            {"station_id": "C", "lat": 3.0, "lon": 3.0},
        ]
        zone_by_coords = {(1.0, 1.0): "Cfa", (2.0, 2.0): "Cfa", (3.0, 3.0): "BWh"}
        with patch.object(bts.ghcn, "koppen_climate_zone",
                          side_effect=lambda lat, lon: zone_by_coords[(lat, lon)]):
            grouped = bts._group_stations_by_zone(stations)
        assert {s["station_id"] for s in grouped["Cfa"]} == {"A", "B"}
        assert {s["station_id"] for s in grouped["BWh"]} == {"C"}

    def test_empty_input_returns_empty_dict(self):
        assert bts._group_stations_by_zone([]) == {}

    def test_calls_koppen_climate_zone_with_lat_then_lon(self):
        stations = [{"station_id": "A", "lat": 33.4, "lon": -112.0}]
        with patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh") as mock_zone:
            bts._group_stations_by_zone(stations)
        mock_zone.assert_called_once_with(33.4, -112.0)


class TestEra5DownloadLockAccountRotation:
    """Defect found live 2026-09-17: the slot scan ran in index order and took
    the first free slot, so at concurrency 1 -- the normal case for a single
    lane -- account 0 was chosen EVERY time and accounts 1 and 2 were never
    touched. Measured 322/111/82 jobs across three supposedly-equal accounts in
    one bastion session, with ALL 106 rejections on account 0, whose per-user
    CDS queue allowance was saturated while the other two sat idle."""

    def _isolated_lock_paths(self, tmp_path, monkeypatch, n=3):
        """Never touch the real /tmp/build_training_set_era5_download_*.lock
        files -- those are the live cross-process slots a real corpus pull
        coordinates on. Same convention as
        TestEra5DownloadLockFreeSlotFirst._patch_lock_paths."""
        paths = [str(tmp_path / f"{chr(ord('a') + i)}.lock") for i in range(n)]
        monkeypatch.setattr(bts, "_ERA5_DOWNLOAD_LOCK_PATHS", paths)
        return paths

    def _all_accounts_configured(self):
        return patch.object(bts.era5, "account_configured", return_value=True)

    def test_uses_every_account_across_repeated_uncontended_acquisitions(self, tmp_path, monkeypatch):
        """The actual regression. With no contention at all, every acquisition
        can take slot 0 -- and did. Over many acquisitions the lock must spread
        across all configured accounts, not pin to one."""
        self._isolated_lock_paths(tmp_path, monkeypatch)
        seen = set()
        with self._all_accounts_configured():
            for _ in range(200):
                with bts._era5_download_lock() as account_index:
                    seen.add(account_index)
        assert seen == {0, 1, 2}, (
            f"only accounts {sorted(seen)} were ever used; the index-order bias is back"
        )

    def test_distribution_is_not_pinned_to_one_account(self, tmp_path, monkeypatch):
        """Stronger than 'every account appears once': no account may take a
        dominant share. 3 accounts over 300 draws should sit near 100 each;
        allow generous slack for randomness but reject anything approaching the
        old 100%-on-account-0 behaviour."""
        from collections import Counter
        self._isolated_lock_paths(tmp_path, monkeypatch)
        counts = Counter()
        with self._all_accounts_configured():
            for _ in range(300):
                with bts._era5_download_lock() as account_index:
                    counts[account_index] += 1
        n_accounts = 3
        for account_index in range(n_accounts):
            assert counts[account_index] > 300 / n_accounts * 0.5, (
                f"account {account_index} got only {counts[account_index]}/300 -- too skewed"
            )

    def test_deprioritized_account_is_tried_last_not_excluded(self, tmp_path, monkeypatch):
        """A retry after a CDS 'rejected' must steer AWAY from the rejecting
        account, since a rejection is a per-account queue signal rather than a
        global one -- but the account must stay reachable, or a single-account
        deployment would have no slot to wait on."""
        self._isolated_lock_paths(tmp_path, monkeypatch)
        with self._all_accounts_configured():
            for _ in range(60):
                with bts._era5_download_lock(deprioritize_account_index=0) as account_index:
                    # slots are all free, so the deprioritized one must never win
                    assert account_index != 0

    def test_deprioritizing_the_only_account_still_makes_progress(self, tmp_path, monkeypatch):
        """Single-account deployment (ERA5_SECRET_ARN_2 unset): deprioritizing
        account 0 must not leave the caller with nothing to acquire."""
        self._isolated_lock_paths(tmp_path, monkeypatch)
        def only_account_0(account_index):
            return account_index == 0
        with patch.object(bts.era5, "account_configured", side_effect=only_account_0):
            with bts._era5_download_lock(deprioritize_account_index=0) as account_index:
                assert account_index == 0

    def test_still_yields_a_configured_account_only(self, tmp_path, monkeypatch):
        """Never hand back an account with no credential wired up."""
        self._isolated_lock_paths(tmp_path, monkeypatch)
        def only_0_and_2(account_index):
            return account_index in (0, 2)
        with patch.object(bts.era5, "account_configured", side_effect=only_0_and_2):
            for _ in range(40):
                with bts._era5_download_lock() as account_index:
                    assert account_index in (0, 2)


class TestClusterStationsByBboxCells:
    """Replaces TestChunkStationsByExtent (2026-09-17). The objective is
    INVERTED: the old grid bucketing tried to keep each request's bbox small,
    on the belief that bbox size drove CDS's per-request cost limit. Measured
    live 2026-09-17, it does not -- 4 ERA5-Land requests at an identical field
    count and bboxes from 100 to 360,000 cells were all accepted. Request
    COUNT is the real cost (~33 min of CDS queue each), so the goal is now the
    FEWEST clusters a bytes/memory budget allows."""

    def test_empty_input_returns_empty_list(self):
        assert bts._cluster_stations_by_bbox_cells([]) == []

    def test_tightly_clustered_stations_stay_in_one_cluster(self):
        stations = [
            {"station_id": "A", "lat": 32.40, "lon": -115.10},
            {"station_id": "B", "lat": 32.41, "lon": -115.11},
        ]
        clusters = bts._cluster_stations_by_bbox_cells(stations)
        assert len(clusters) == 1
        assert {s["station_id"] for s in clusters[0]} == {"A", "B"}

    def test_the_mexicali_pair_now_shares_ONE_request(self):
        """Direct reversal of the old test_widely_separated_stations_split_
        into_different_chunks. The same two real Mexicali stations (~1.4deg
        lat apart, ~150km) were deliberately SPLIT by the 0.5deg grid, on the
        belief that their padded 2.37 x 1.33deg bbox was what tripped CDS's
        'cost limits exceeded' in 2026-08-03. That was a mis-attribution --
        the same request had also cartesian-expanded 34 real days into 93
        day-slots = 13,392 fields, over the limit on the DATE axis alone.
        Their combined padded bbox is ~25 x 14 cells, nowhere near the budget,
        so they must now share a single request instead of paying two."""
        stations = [
            {"station_id": "MXM00076040", "lat": 32.4, "lon": -115.1833},
            {"station_id": "MXM00076055", "lat": 31.033, "lon": -114.85},
        ]
        clusters = bts._cluster_stations_by_bbox_cells(stations)
        assert len(clusters) == 1
        assert {s["station_id"] for s in clusters[0]} == {"MXM00076040", "MXM00076055"}

    def test_stations_too_far_apart_to_share_a_budget_do_split(self):
        """The budget is still a real bound: S. Florida and Hawaii cannot share
        a bbox (~75deg of longitude => ~46,000 cells, over the 10,000 default),
        so a genuinely dispersed set still splits rather than pulling a
        Pacific-wide grid."""
        stations = [
            {"station_id": "USFL0001", "lat": 25.5, "lon": -80.4},
            {"station_id": "USHI0001", "lat": 19.7, "lon": -155.1},
        ]
        clusters = bts._cluster_stations_by_bbox_cells(stations)
        assert len(clusters) == 2

    def test_every_station_is_kept_never_dropped(self):
        """Unlike _select_geographically_stratified/_compact (which cap and
        drop), clustering must never lose a station -- it only controls how
        many separate ERA5 requests they're split across."""
        stations = [{"station_id": f"S{i}", "lat": float(i), "lon": float(i)} for i in range(10)]
        clusters = bts._cluster_stations_by_bbox_cells(stations)
        all_ids = {s["station_id"] for c in clusters for s in c}
        assert all_ids == {f"S{i}" for i in range(10)}

    def test_every_cluster_bbox_stays_within_the_cell_budget(self):
        import random
        random.seed(20260917)
        stations = [
            {"station_id": f"S{i}", "lat": random.uniform(0, 40), "lon": random.uniform(0, 40)}
            for i in range(200)
        ]
        max_cells = 10_000
        clusters = bts._cluster_stations_by_bbox_cells(stations, max_cells=max_cells)
        for c in clusters:
            lats = [s["lat"] for s in c]
            lons = [s["lon"] for s in c]
            assert bts._bbox_cell_count(min(lons), min(lats), max(lons), max(lats)) <= max_cells

    def test_uses_far_fewer_requests_than_the_old_half_degree_grid(self):
        """The whole point of the change, asserted as a number. 50 stations
        spread over 5x5 degrees occupied ~64 distinct 0.5deg grid cells and so
        paid ~64 separate 14-segment ERA5 pulls; their combined padded bbox is
        ~61x61 = 3,721 cells, comfortably inside the budget, so they now pay
        exactly one."""
        import random
        random.seed(20260917)
        stations = [
            {"station_id": f"S{i}", "lat": random.uniform(25.0, 30.0), "lon": random.uniform(-85.0, -80.0)}
            for i in range(50)
        ]
        clusters = bts._cluster_stations_by_bbox_cells(stations)
        assert len(clusters) == 1

    def test_result_does_not_depend_on_input_order(self):
        """Narrower claim than "deterministic", deliberately. Code review
        finding: the function re-sorts its input on the first line by a total
        order (lon, lat, station_id), so a reversed-input test is guaranteed to
        pass by construction and does NOT exercise the cells < best_cells
        tie-break. Order-independence is still the property callers need --
        two invocations over the same stations must produce the same request
        plan -- so it is worth asserting under an honest name."""
        stations = [
            {"station_id": f"S{i}", "lat": 25.0 + 0.3 * i, "lon": -80.0 - 0.3 * i}
            for i in range(12)
        ]
        first = bts._cluster_stations_by_bbox_cells(stations)
        second = bts._cluster_stations_by_bbox_cells(list(reversed(stations)))
        as_ids = lambda cs: [sorted(s["station_id"] for s in c) for c in cs]
        assert as_ids(first) == as_ids(second)

    def test_hits_the_provable_lower_bound_on_the_real_us_geometry(self):
        """Round-2 review finding: greedy clustering is not guaranteed minimal.
        True in general, so pin down that it is minimal on the geometry that
        actually matters. The real 389-station US Af/Am set is two populations,
        S. Florida and Hawaii's Big Island. Their combined padded bbox is
        ~70,564 cells against a 10,000 budget, so no partition can put them
        together -- 2 is a hard lower bound, and greedy achieves it."""
        florida = [
            {"station_id": f"FL{i}", "lat": 25.32 + (1.87 * i / 20), "lon": -80.82 + (0.79 * i / 20)}
            for i in range(21)
        ]
        hawaii = [
            {"station_id": f"HI{i}", "lat": 19.18 + (0.96 * i / 10), "lon": -155.58 + (0.78 * i / 10)}
            for i in range(11)
        ]
        combined_cells = bts._bbox_cell_count(-155.58, 19.18, -80.03, 27.19)
        assert combined_cells > bts._ERA5_MAX_CHUNK_CELLS, (
            "premise of this test: the two populations must be unmergeable"
        )
        clusters = bts._cluster_stations_by_bbox_cells(florida + hawaii)
        assert len(clusters) == 2
        by_prefix = [{s["station_id"][:2] for s in c} for c in clusters]
        assert {"FL"} in by_prefix and {"HI"} in by_prefix, (
            "the two clusters must be exactly the two real populations, not a split through one"
        )

    def test_raises_when_one_station_alone_cannot_fit_the_budget(self):
        """A single station's padded bbox is the floor -- no clustering can get
        under a budget smaller than that, so it must fail loudly at clustering
        time rather than silently emitting an over-budget request."""
        stations = [{"station_id": "USW00090001", "lat": 25.0, "lon": -80.0}]
        with pytest.raises(ValueError, match="alone needs"):
            bts._cluster_stations_by_bbox_cells(stations, max_cells=50)


class TestBboxCellCount:
    def test_matches_the_area_build_era5_request_actually_sends(self):
        """The invariant that makes _ERA5_MAX_CHUNK_CELLS a real bound rather
        than an estimate: our cell count must equal the grid-point count of the
        `area` era5._build_era5_request actually puts on the wire. Code review
        finding, real: an earlier round(extent/res)+1 approximation ignored where
        the raw edges fell relative to the 0.1deg grid and could undercount the
        delivered request by several percent."""
        import random
        from datetime import date as _date
        random.seed(20260917)
        res = 0.1
        for _ in range(400):
            west = random.uniform(-179.0, 178.0)
            south = random.uniform(-88.0, 87.0)
            east = west + random.uniform(0.0, 1.0)
            north = south + random.uniform(0.0, 1.0)
            bbox = bts.stations_bbox(
                [{"lon": west, "lat": south}, {"lon": east, "lat": north}]
            )
            req = bts.era5._build_era5_request(
                bbox, _date(2023, 3, 1), _date(2023, 3, 31),
                dataset="reanalysis-era5-land",
                variables=bts._TRAINING_ERA5_VARIABLES,
            )
            n, w, s_, e = req["area"]
            actual = (round((n - s_) / res) + 1) * (round((e - w) / res) + 1)
            ours = bts._bbox_cell_count(west, south, east, north)
            assert ours == actual, (
                f"cell count {ours} != delivered {actual} for bbox {bbox} (area {req['area']})"
            )


    def test_counts_era5_land_cells_including_the_padding(self):
        # 1deg raw extent + 0.5deg pad each side = 2deg span = 21 cells at 0.1deg
        assert bts._bbox_cell_count(0.0, 0.0, 1.0, 1.0) == 21 * 21

    def test_a_single_point_still_counts_its_padded_box(self):
        # zero raw extent + 0.5deg pad each side = 1deg span = 11 cells
        assert bts._bbox_cell_count(5.0, 5.0, 5.0, 5.0) == 11 * 11


class TestCanopyResumeProjectId:
    def test_different_station_counts_get_different_project_ids(self):
        """A small test run (e.g. --max-stations-per-country 5) must not
        collide with a later full run for the same country -- their
        quadkey lists differ in size/order, so reusing one run's resume
        state for the other could skip tiles that were never actually
        visited."""
        small = bts._canopy_resume_project_id("US", _STATIONS[:1])
        full = bts._canopy_resume_project_id("US", _STATIONS)
        assert small != full

    def test_same_station_set_is_deterministic(self):
        assert bts._canopy_resume_project_id("US", _STATIONS) == bts._canopy_resume_project_id("US", _STATIONS)

    def test_different_country_label_differs_even_with_same_stations(self):
        assert bts._canopy_resume_project_id("US", _STATIONS[:1]) != bts._canopy_resume_project_id("GM", _STATIONS[:1])


# ---------------------------------------------------------------------------
# _covariate_completeness_summary
# ---------------------------------------------------------------------------


class TestCovariateCompletenessSummary:
    def test_empty_rows_returns_no_rows(self):
        assert bts._covariate_completeness_summary([]) == "no rows"

    def test_reports_non_null_fraction_per_column(self):
        rows = [
            {"elevation_mean_m": 100.0, "slope_deg": None},
            {"elevation_mean_m": None, "slope_deg": 2.0},
        ]
        summary = bts._covariate_completeness_summary(rows)
        assert "elevation_mean_m=1/2" in summary
        assert "slope_deg=1/2" in summary

    def test_koppen_main_group_code_always_shown_even_though_never_computed(self):
        """This script never computes koppen_main_group_code (only
        climate_zone) -- it should still appear in the summary, always at
        0/N, as a standing reminder of that separate gap rather than being
        silently absent from the printout."""
        summary = bts._covariate_completeness_summary([{"climate_zone": "Cfa"}])
        assert "koppen_main_group_code=0/1" in summary


# ---------------------------------------------------------------------------
# main() -- --station-ids-file / --dry-run
# ---------------------------------------------------------------------------


_US_STATION = {"station_id": "USW00023183", "lon": -112.0036, "lat": 33.4278, "elevation_m": 339.2, "name": "PHOENIX AP"}
_MX_STATION = {"station_id": "MXM00076040", "lon": -115.1833, "lat": 32.4, "elevation_m": 10.0, "name": "EJIDO NUEVO LEON"}
_UNRELATED_US_STATION = {"station_id": "USW00003812", "lon": -111.98, "lat": 33.45, "elevation_m": 350.0, "name": "TEMPE"}


class TestMainStationIdsFileAndDryRun:
    def _patch_common(self, list_ghcn_stations_return, active_ids, build_rows_return=None):
        """Common set of mocks every main() test needs -- credentials/DB-
        table-creation/inventory-fetch/active-filter/row-building/upsert,
        none of which touch real network/AWS/DB."""
        build_rows_return = build_rows_return if build_rows_return is not None else []
        return (
            patch.object(bts, "_bucket_from_credentials", return_value="vuln-bucket"),
            patch.object(bts, "_landscan_from_credentials", return_value=("ls-bucket", "ls-key")),
            patch.object(bts.ghcn, "create_ghcn_training_table"),
            patch.object(bts.ghcn, "list_ghcn_stations", return_value=list_ghcn_stations_return),
            patch.object(bts.ghcn, "active_station_ids", return_value=active_ids),
            patch.object(bts, "build_rows_for_country",
                         side_effect=lambda country, stations, *a, **k: list(build_rows_return)),
            patch.object(bts.ghcn, "upsert_ghcn_training_rows"),
            # Deterministic zone for every fixture station -- _group_stations_by_zone
            # (2026-08-03) would otherwise call the REAL koppen_climate_zone,
            # making these tests depend on real geographic raster data for a
            # behavior (call-count/pass-through) that has nothing to do with
            # which actual zone a station falls in.
            patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh"),
        )

    def test_countries_and_station_ids_file_are_mutually_exclusive(self, monkeypatch, tmp_path, capsys):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--countries", "US", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        with pytest.raises(SystemExit):
            bts.main()
        assert "exactly one of --countries or --station-ids-file" in capsys.readouterr().err

    def test_requires_one_of_countries_or_station_ids_file(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        with pytest.raises(SystemExit):
            bts.main()
        assert "exactly one of --countries or --station-ids-file" in capsys.readouterr().err

    def test_max_stations_per_country_rejected_with_station_ids_file(self, monkeypatch, tmp_path, capsys):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f), "--max-stations-per-country", "5",
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        with pytest.raises(SystemExit):
            bts.main()
        assert "not compatible with --station-ids-file" in capsys.readouterr().err

    def test_era5_cache_dir_rejected_with_openmeteo_source(self, monkeypatch, tmp_path, capsys):
        """Round-2 review finding, real: --era5-cache-dir only applies to --era5-source cds
        (build_rows_for_country's own docstring already said so, but nothing enforced or even
        warned about it) -- used to be a silent no-op, discoverable only by finding an empty
        cache dir after a killed run. Must now fail fast at argument-parsing time instead."""
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--countries", "US",
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
            "--era5-source", "openmeteo", "--era5-cache-dir", str(tmp_path / "cache"),
        ])
        with pytest.raises(SystemExit):
            bts.main()
        assert "--era5-cache-dir only applies with --era5-source cds" in capsys.readouterr().err

    def test_station_ids_file_derives_countries_and_filters_to_requested_ids(self, monkeypatch, tmp_path):
        """Pooled US+MX file: countries must be derived from the IDs'
        FIPS prefixes (not require --countries), list_ghcn_stations must be
        called with exactly those derived countries, and a station present
        in the inventory but NOT requested (_UNRELATED_US_STATION) must be
        excluded even though it shares the US country code."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183", "MXM00076040"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION, _UNRELATED_US_STATION, _MX_STATION],
            active_ids={"USW00023183", "MXM00076040", "USW00003812"},
        )
        with mocks[0], mocks[1], mocks[2], mocks[3] as mock_list, mocks[4], mocks[5] as mock_build, mocks[6] as mock_upsert, mocks[7]:
            bts.main()

        mock_list.assert_called_once_with(["MX", "US"])
        built_station_ids = {
            s["station_id"] for call in mock_build.call_args_list for s in call.args[1]
        }
        assert built_station_ids == {"USW00023183", "MXM00076040"}
        assert mock_upsert.call_count == 0  # build_rows_for_country returned [] in this fixture

    def test_station_ids_file_warns_on_ids_missing_from_inventory(self, monkeypatch, tmp_path, capsys):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183", "USW00099999"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"},
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7]:
            bts.main()

        stdout = capsys.readouterr().out
        assert "WARNING" in stdout and "USW00099999" in stdout

    def test_rows_out_flushes_after_every_chunk_not_just_at_the_end(self, monkeypatch, tmp_path):
        """Real gap found live 2026-08-20 mid-way through an actual multi-hour,
        many-chunk Spain GHCN pull: --rows-out only ever wrote once, at the
        very end of the ENTIRE station-ids-file list, so a crash/kill/bastion
        interruption after hours of real compute lost all of it with nothing
        on disk. Two stations far enough apart (different 0.5deg extent
        chunks, same mocked zone) force two separate build_rows_for_country
        calls -- the file on disk must reflect the first chunk's rows before
        the second chunk even starts, not only after both are done."""
        far_station_1 = {"station_id": "USW00023183", "lon": -112.0, "lat": 33.43, "elevation_m": 339.2, "name": "PHOENIX AP"}
        far_station_2 = {"station_id": "USW00099998", "lon": -70.0, "lat": 40.0, "elevation_m": 10.0, "name": "FAR AWAY STATION"}
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183", "USW00099998"]}))
        out = tmp_path / "rows.json"
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
            "--dry-run", "--rows-out", str(out),
        ])
        call_count = 0
        seen_row_counts_on_disk_during_run = []

        def fake_build_rows(country, stations, *a, **k):
            nonlocal call_count
            call_count += 1
            rows = [{"station_id": s["station_id"], "date": "2016-06-15"} for s in stations]
            # Read the file back INSIDE this call (before the NEXT chunk's
            # build_rows_for_country runs) -- proves the checkpoint from the
            # PREVIOUS chunk (or nothing, on the first call) already landed.
            if out.exists():
                seen_row_counts_on_disk_during_run.append(json.loads(out.read_text())["row_count"])
            else:
                seen_row_counts_on_disk_during_run.append(0)
            return rows

        mocks = self._patch_common(
            list_ghcn_stations_return=[far_station_1, far_station_2],
            active_ids={"USW00023183", "USW00099998"},
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], \
             patch.object(bts, "build_rows_for_country", side_effect=fake_build_rows), \
             mocks[6], mocks[7]:
            bts.main()

        assert call_count == 2  # confirms the two stations really did land in separate chunks
        assert seen_row_counts_on_disk_during_run == [0, 1]  # 0 before chunk 1, 1 (chunk 1's row) before chunk 2
        final = json.loads(out.read_text())
        assert final["row_count"] == 2
        assert final["complete"] is True

    def test_dry_run_skips_the_table_ddl_too_not_just_the_upsert(self, monkeypatch, tmp_path):
        """Real gap found live 2026-08-20: create_ghcn_training_table() (a real
        write-capable DB connection + DDL, idempotent but still a write) was
        called unconditionally, before dry_run was even checked -- so --dry-run
        could never actually run without DB write credentials, contradicting
        its own docstring's promise to skip "the DB write itself." Gated the
        same way the real upsert already was."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30", "--dry-run",
        ])
        mocks = self._patch_common(list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"})
        with mocks[0], mocks[1], mocks[2] as mock_create_table, mocks[3], mocks[4], mocks[5], mocks[6], mocks[7]:
            bts.main()
        mock_create_table.assert_not_called()

    def test_normal_run_still_creates_the_table(self, monkeypatch, tmp_path):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        mocks = self._patch_common(list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"})
        with mocks[0], mocks[1], mocks[2] as mock_create_table, mocks[3], mocks[4], mocks[5], mocks[6], mocks[7]:
            bts.main()
        mock_create_table.assert_called_once()

    def test_dry_run_skips_upsert_but_still_builds_rows(self, monkeypatch, tmp_path, capsys):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30", "--dry-run",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"},
            build_rows_return=[{"station_id": "USW00023183", "elevation_mean_m": 100.0}],
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5] as mock_build, mocks[6] as mock_upsert, mocks[7]:
            bts.main()

        mock_build.assert_called_once()
        mock_upsert.assert_not_called()
        out = capsys.readouterr().out
        assert "--dry-run: would write 1 row(s)" in out
        assert "--dry-run: not uploading" in out

    def test_rows_out_persists_computed_rows_alongside_dry_run(self, monkeypatch, tmp_path):
        """--rows-out + --dry-run together: the real fetch/covariate compute a
        dry-run pays for should be recoverable from disk, not thrown away with
        only a summary print to show for it (2026-08-20, real gap hit live
        staging a Spain sub-zone fit -- see this repo's own git history)."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        out = tmp_path / "rows.json"
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
            "--dry-run", "--rows-out", str(out),
        ])
        fake_rows = [{"station_id": "USW00023183", "date": "2016-06-15", "elevation_mean_m": 100.0}]
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"},
            build_rows_return=fake_rows,
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6] as mock_upsert, mocks[7]:
            bts.main()

        mock_upsert.assert_not_called()  # --rows-out is not a backdoor around --dry-run's own DB-write gate
        written = json.loads(out.read_text())
        assert written["rows"] == fake_rows
        assert written["row_count"] == 1
        assert written["countries"] == ["US"]
        assert written["dry_run"] is True

    def test_rows_out_omitted_writes_no_file(self, monkeypatch, tmp_path):
        """The default (no --rows-out) must not create any file -- confirms
        the feature is opt-in, matching every other flag in this script."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30", "--dry-run",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"},
            build_rows_return=[{"station_id": "USW00023183"}],
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7]:
            bts.main()

        assert not (tmp_path / "rows.json").exists()

    def test_normal_run_calls_upsert_with_the_built_rows(self, monkeypatch, tmp_path):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        fake_rows = [{"station_id": "USW00023183", "date": "2016-06-15"}]
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION], active_ids={"USW00023183"},
            build_rows_return=fake_rows,
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6] as mock_upsert, mocks[7]:
            bts.main()

        mock_upsert.assert_called_once_with(fake_rows)

    def test_stations_spanning_multiple_zones_in_one_country_get_separate_batches(self, monkeypatch, tmp_path):
        """Regression test for the 2026-08-03 CDS 'cost limits exceeded'
        finding: a single country-wide bbox for a geographically dispersed
        --station-ids-file set can exceed CDS's per-request cost limit.
        Two US stations in different zones must produce two separate
        build_rows_for_country calls (one per zone-batch), each with its
        own zone-scoped batch_label -- never one combined US-wide call."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00023183", "USW00003812"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[_US_STATION, _UNRELATED_US_STATION],
            active_ids={"USW00023183", "USW00003812"},
        )
        zone_by_id = {"USW00023183": "BWh", "USW00003812": "Csa"}
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], \
             mocks[5] as mock_build, mocks[6], \
             patch.object(bts.ghcn, "koppen_climate_zone",
                          side_effect=lambda lat, lon: zone_by_id[
                              "USW00023183" if (lat, lon) == (_US_STATION["lat"], _US_STATION["lon"]) else "USW00003812"
                          ]):
            bts.main()

        assert mock_build.call_count == 2
        batch_labels = sorted(call.args[0] for call in mock_build.call_args_list)
        assert batch_labels == ["US_BWh", "US_Csa"]

    def test_same_zone_stations_far_apart_now_share_ONE_era5_batch(self, monkeypatch, tmp_path):
        """Inverted premise (2026-09-17). This test previously asserted that
        the actual Mexicali pair (~1.4deg lat apart, ~150km) must be SPLIT
        into US_BWh_c0 and US_BWh_c1 by the 0.5deg spread cap. CDS was
        measured not to charge for bbox area, so splitting them bought nothing
        and doubled the ~33-min-per-request queue cost. They must now land in
        a single batch, and the label must lose its _cN suffix because the
        zone no longer splits."""
        far_station_1 = {"station_id": "USW00090001", "lon": -115.1833, "lat": 32.4, "elevation_m": 10.0, "name": "FAR1"}
        far_station_2 = {"station_id": "USW00090002", "lon": -114.85, "lat": 31.033, "elevation_m": 10.0, "name": "FAR2"}
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00090001", "USW00090002"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        mocks = self._patch_common(
            list_ghcn_stations_return=[far_station_1, far_station_2],
            active_ids={"USW00090001", "USW00090002"},
        )
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], \
             mocks[5] as mock_build, mocks[6], mocks[7]:
            bts.main()

        assert mock_build.call_count == 1
        assert mock_build.call_args_list[0].args[0] == "US_BWh"
        passed_ids = {s["station_id"] for s in mock_build.call_args_list[0].args[1]}
        assert passed_ids == {"USW00090001", "USW00090002"}

    def test_the_removed_extent_flag_errors_instead_of_being_ignored(self, monkeypatch, tmp_path):
        """A saved command line carrying --era5-max-chunk-extent-deg would
        otherwise run with a ~100x-different request-count profile than its
        author intended, at ~33 min of CDS queue per wrong request."""
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"station_ids": ["USW00090001"]}))
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--station-ids-file", str(f),
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
            "--era5-max-chunk-extent-deg", "0.5",
        ])
        with pytest.raises(SystemExit):
            bts.main()


class TestMainPerBatchIsolation:
    """issue #648: a real 2026-09-16 incident -- one CDS 'rejected' exception raised by
    build_rows_for_country while building the US batch propagated all the way out of main()'s
    country loop and killed the process before the still-queued VM/VQ countries in the same
    --countries invocation ever started. One batch's failure must not stop the rest."""

    def test_one_countrys_failure_does_not_stop_the_next_country(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--countries", "US,MX",
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])

        def fake_build_rows(batch_label, stations, *a, **k):
            if batch_label.startswith("US"):
                raise cads_api_client.processing.ProcessingFailedError("Unknown API state 'rejected'")
            return [{"station_id": s["station_id"], "date": "2016-06-15"} for s in stations]

        with patch.object(bts, "_bucket_from_credentials", return_value="vuln-bucket"), \
             patch.object(bts, "_landscan_from_credentials", return_value=("ls-bucket", "ls-key")), \
             patch.object(bts.ghcn, "create_ghcn_training_table"), \
             patch.object(bts.ghcn, "list_ghcn_stations", return_value=[_US_STATION, _MX_STATION]), \
             patch.object(bts.ghcn, "active_station_ids", return_value={"USW00023183", "MXM00076040"}), \
             patch.object(bts, "build_rows_for_country", side_effect=fake_build_rows) as mock_build, \
             patch.object(bts.ghcn, "upsert_ghcn_training_rows") as mock_upsert, \
             patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh"), \
             pytest.raises(SystemExit) as exc_info:
            bts.main()

        # a failed batch must still surface as a non-zero exit -- a caller scripting a re-run
        # of just the failed countries needs this to be machine-checkable, not just printed
        assert exc_info.value.code == 1
        # BOTH countries were attempted, in the order given -- MX was not skipped just because
        # US (processed first) failed
        built_batches = [call.args[0] for call in mock_build.call_args_list]
        assert built_batches == ["US_BWh", "MX_BWh"]
        # MX's real rows were still computed and written despite US's failure
        mock_upsert.assert_called_once_with([{"station_id": "MXM00076040", "date": "2016-06-15"}])

    def test_rows_out_confirmation_still_prints_when_a_batch_failed(self, monkeypatch, tmp_path, capsys):
        """Round-1 review finding, real: the new `if failed_batches: ... sys.exit(1)` block was
        inserted directly above the pre-existing `--rows-out: wrote ...` confirmation print
        without re-indenting it back under `if args.rows_out:` -- so it silently became
        unreachable in every case (dead if failed_batches was empty, skipped by sys.exit(1) if
        not). The file write itself was unaffected, but the confirmation message regressed
        silently with no test to catch it. --rows-out's own file write already happens
        unconditionally before this new exit path (issue #648's own requirement), so the print
        confirming it must too."""
        out = tmp_path / "rows.json"
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--countries", "US,MX",
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
            "--rows-out", str(out),
        ])

        def fake_build_rows(batch_label, stations, *a, **k):
            if batch_label.startswith("US"):
                raise cads_api_client.processing.ProcessingFailedError("Unknown API state 'rejected'")
            return [{"station_id": s["station_id"], "date": "2016-06-15"} for s in stations]

        with patch.object(bts, "_bucket_from_credentials", return_value="vuln-bucket"), \
             patch.object(bts, "_landscan_from_credentials", return_value=("ls-bucket", "ls-key")), \
             patch.object(bts.ghcn, "create_ghcn_training_table"), \
             patch.object(bts.ghcn, "list_ghcn_stations", return_value=[_US_STATION, _MX_STATION]), \
             patch.object(bts.ghcn, "active_station_ids", return_value={"USW00023183", "MXM00076040"}), \
             patch.object(bts, "build_rows_for_country", side_effect=fake_build_rows), \
             patch.object(bts.ghcn, "upsert_ghcn_training_rows"), \
             patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh"), \
             pytest.raises(SystemExit):
            bts.main()

        stdout = capsys.readouterr().out
        assert f"--rows-out: wrote 1 row(s) to {out}" in stdout
        assert json.loads(out.read_text())["complete"] is True

    def test_all_batches_succeeding_does_not_exit_nonzero(self, monkeypatch):
        """Sanity check for the new exit-code path: a clean run (no failed batches) must not
        start unconditionally exiting non-zero now that failure tracking exists."""
        monkeypatch.setattr(sys, "argv", [
            "build_training_set.py", "--countries", "US",
            "--start-date", "2016-06-01", "--end-date", "2016-06-30",
        ])
        with patch.object(bts, "_bucket_from_credentials", return_value="vuln-bucket"), \
             patch.object(bts, "_landscan_from_credentials", return_value=("ls-bucket", "ls-key")), \
             patch.object(bts.ghcn, "create_ghcn_training_table"), \
             patch.object(bts.ghcn, "list_ghcn_stations", return_value=[_US_STATION]), \
             patch.object(bts.ghcn, "active_station_ids", return_value={"USW00023183"}), \
             patch.object(bts, "build_rows_for_country", return_value=[]), \
             patch.object(bts.ghcn, "upsert_ghcn_training_rows"), \
             patch.object(bts.ghcn, "koppen_climate_zone", return_value="BWh"):
            bts.main()  # must not raise SystemExit
