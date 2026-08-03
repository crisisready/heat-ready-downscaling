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
        # calendar-month splitting instead, not this outer mechanism).
        with patch.object(bts, "_ERA5_CHUNK_DAYS", 90), \
             patch.object(bts.era5, "download_era5", return_value="/tmp/fake.nc") as mock_dl, \
             patch.object(bts.era5, "extract_era5_means", side_effect=rows_by_call), \
             patch.object(bts, "_timezones_for_stations", return_value={"USW00023183": "UTC"}), \
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
        def fake_lock():
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

    def test_yields_account_index_zero_when_both_slots_free(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ERA5_SECRET_ARN_2", "arn:aws:secretsmanager:us-east-1:123:secret:era5-2")
        monkeypatch.delenv("ERA5_SECRET_ARN_3", raising=False)
        self._patch_lock_paths(tmp_path, monkeypatch)
        with bts._era5_download_lock() as account_index:
            assert account_index == 0  # first free configured slot wins

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
        with mocks[0], mocks[1], mocks[2], mocks[3] as mock_list, mocks[4], mocks[5] as mock_build, mocks[6] as mock_upsert:
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
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6]:
            bts.main()

        stdout = capsys.readouterr().out
        assert "WARNING" in stdout and "USW00099999" in stdout

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
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5] as mock_build, mocks[6] as mock_upsert:
            bts.main()

        mock_build.assert_called_once()
        mock_upsert.assert_not_called()
        out = capsys.readouterr().out
        assert "--dry-run: would write 1 row(s)" in out
        assert "--dry-run: not uploading" in out

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
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6] as mock_upsert:
            bts.main()

        mock_upsert.assert_called_once_with(fake_rows)
