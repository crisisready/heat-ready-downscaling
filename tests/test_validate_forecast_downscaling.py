"""Unit tests for validate_forecast_downscaling.py's api_call_manager
integration -- no network calls, no real sleeping. Mirrors
test_validate_lagfill_downscaling.py's approach for the forecast band's
lead-specific fetch/checkpoint path.

Excluded from this repo's own collection (see conftest.py) -- this script
imports heat_calcs/open_meteo/api_call_manager, private-repo-only modules.
Kept as a real test file for crisisready/heat-risk-data-api's own
environment to run, same as the script it tests."""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from api_call_manager import NO_RESULT, AdaptiveThrottle

import validate_forecast_downscaling as vfd


class _ScriptedSession:
    """Same purpose as test_validate_lagfill_downscaling.py's identical
    fixture -- returns a different scripted outcome on each successive
    get_json call, for testing _fetch_lead_chunks_with_delayed_requeue's
    retry-after-a-real-delay behavior."""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.throttle = AdaptiveThrottle(max_workers=4)

    def get_json(self, url, params, **kwargs):
        self.calls += 1
        return self._outcomes.pop(0)


def _fake_lead_response(base_temp_c: float, lead_days: int):
    """24 hours of previous_dayN-suffixed hourly data, same 06:00-local-day
    window as the lag-fill test fixture (see that file's comment)."""
    times = [f"2023-06-15T{h:02d}:00" for h in range(6, 24)] + [f"2023-06-16T{h:02d}:00" for h in range(0, 6)]
    suffix = f"_previous_day{lead_days}"
    return {
        "hourly": {
            "time": times,
            f"temperature_2m{suffix}": [base_temp_c + (h % 12) for h in range(24)],
            f"dewpoint_2m{suffix}": [base_temp_c - 5 for _ in range(24)],
            f"windspeed_10m{suffix}": [3.0 for _ in range(24)],
            f"surface_pressure{suffix}": [1013.0 for _ in range(24)],
        }
    }


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.calls_params = []
        self.throttle = AdaptiveThrottle(max_workers=4)

    def get_json(self, url, params, **kwargs):
        self.calls += 1
        self.calls_params.append(params)
        return self._response


class TestProcessOneStationLead:
    def test_successful_fetch_returns_combined_payload(self):
        station = {
            "station_id": "TEST001",
            "rows": [{
                "station_id": "TEST001", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
                "grid_tmax_c": 29.0, "grid_tmin_c": 19.0,
            }],
            "tz": "UTC",
            "lead_days": 2,
            "url": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(_fake_lead_response(base_temp_c=25.0, lead_days=2))

        payload = vfd._process_one_station_lead(station, session)

        assert payload is not NO_RESULT
        assert len(payload["lead_rows"]) == 1
        assert payload["lead_rows"][0]["climate_zone"] == "Cfa"
        assert payload["lead_rows"][0]["grid_tmax_c"] != 29.0  # replaced by the lead reconstruction

    def test_pre_coverage_date_is_dropped_and_counted(self):
        station = {
            "station_id": "TEST003",
            "rows": [{
                "station_id": "TEST003", "date": "2019-01-01", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            }],
            "tz": "UTC",
            "lead_days": 1,
            "url": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(_fake_lead_response(base_temp_c=25.0, lead_days=1))

        payload = vfd._process_one_station_lead(station, session)

        # Before _COVERAGE_START (2021-03-01) -- no dates survive to fetch, so no rows.
        assert payload is NO_RESULT

    def test_no_successful_rows_returns_no_result(self, monkeypatch):
        # Same fake-clock requirement as
        # test_validate_lagfill_downscaling.py's identical test -- a
        # persistently-failing chunk now goes through
        # _fetch_lead_chunks_with_delayed_requeue's real requeue-with-delay
        # loop (#710, 2026-09-19), which would otherwise take ~24 real
        # minutes to exhaust _REQUEUE_MAX_ATTEMPTS.
        fake_now = [0.0]
        monkeypatch.setattr(vfd.time, "monotonic", lambda: fake_now[0])
        monkeypatch.setattr(vfd.time, "sleep", lambda s: fake_now.__setitem__(0, fake_now[0] + s))

        station = {
            "station_id": "TEST002",
            "rows": [{
                "station_id": "TEST002", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            }],
            "tz": "UTC",
            "lead_days": 3,
            "url": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(NO_RESULT)

        payload = vfd._process_one_station_lead(station, session)

        assert payload is NO_RESULT


class TestDelayedRequeue:
    """#710, 2026-09-19: mirrors test_validate_lagfill_downscaling.py's
    identical test class for _fetch_lead_chunks_with_delayed_requeue --
    same requeue-with-delay mechanism, threaded through lead_days, exercised
    directly against a fake clock, never a real sleep."""

    def test_chunk_failing_once_then_succeeding_is_merged_after_one_real_wait(self, monkeypatch):
        fake_now = [0.0]
        slept = []
        monkeypatch.setattr(vfd.time, "monotonic", lambda: fake_now[0])

        def fake_sleep(s):
            slept.append(s)
            fake_now[0] += s

        monkeypatch.setattr(vfd.time, "sleep", fake_sleep)

        session = _ScriptedSession([NO_RESULT, _fake_lead_response(base_temp_c=20.0, lead_days=2)])
        result = vfd._fetch_lead_chunks_with_delayed_requeue(
            "ST1", 40.0, -75.0, [[date(2023, 6, 15)]], 2, session, "UTC",
            "https://customer-previous-runs-api.open-meteo.com/v1/forecast", False,
        )

        assert session.calls == 2  # first attempt failed, requeued attempt succeeded
        assert slept == [vfd._REQUEUE_DELAY_S]  # waited the real delay, not a quick retry
        assert "2023-06-15" in result

    def test_chunk_failing_past_the_cap_gets_one_subdivided_last_attempt(self, monkeypatch):
        fake_now = [0.0]
        monkeypatch.setattr(vfd.time, "monotonic", lambda: fake_now[0])
        monkeypatch.setattr(vfd.time, "sleep", lambda s: fake_now.__setitem__(0, fake_now[0] + s))

        # A 60-day chunk (> _MIN_SUBDIVIDE_CHUNK_DAYS=46) fails every one of
        # the _REQUEUE_MAX_ATTEMPTS requeued attempts, AND the final full-
        # span last-resort attempt -- only then does the one-level
        # subdivision (2x ~30-day halves) kick in, and both halves succeed.
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(60)]
        outcomes = [NO_RESULT] * (vfd._REQUEUE_MAX_ATTEMPTS + 1) + [
            _fake_lead_response(base_temp_c=20.0, lead_days=2), _fake_lead_response(base_temp_c=21.0, lead_days=2),
        ]
        session = _ScriptedSession(outcomes)

        result = vfd._fetch_lead_chunks_with_delayed_requeue(
            "ST1", 40.0, -75.0, [dates], 2, session, "UTC",
            "https://customer-previous-runs-api.open-meteo.com/v1/forecast", False,
        )

        # 6 requeued attempts + 1 last-resort full-span attempt + 2 subdivided halves.
        assert session.calls == vfd._REQUEUE_MAX_ATTEMPTS + 3
        assert len(result) > 0  # the subdivided last resort did recover some dates

    def test_multiple_chunks_progress_independently(self, monkeypatch):
        """A failing chunk being requeued must not block an already-
        succeeding chunk from being merged -- same intent as lag_fill's
        identical test."""
        fake_now = [0.0]
        monkeypatch.setattr(vfd.time, "monotonic", lambda: fake_now[0])
        monkeypatch.setattr(vfd.time, "sleep", lambda s: fake_now.__setitem__(0, fake_now[0] + s))

        session = _ScriptedSession([
            _fake_lead_response(base_temp_c=20.0, lead_days=2),  # chunk 1 succeeds immediately
            NO_RESULT,                                            # chunk 2 fails once
            _fake_lead_response(base_temp_c=22.0, lead_days=2),  # chunk 2 succeeds on requeue
        ])
        result = vfd._fetch_lead_chunks_with_delayed_requeue(
            "ST1", 40.0, -75.0, [[date(2023, 6, 15)], [date(2023, 8, 1)]], 2, session, "UTC",
            "https://customer-previous-runs-api.open-meteo.com/v1/forecast", False,
        )

        assert session.calls == 3
        assert "2023-06-15" in result


class TestElevationNanThreading:
    """2026-08-03, gate-variant scoping -- mirrors
    test_validate_lagfill_downscaling.py's identical test class."""

    def test_disabled_by_default_no_elevation_param(self):
        station = {
            "station_id": "TEST001",
            "rows": [{
                "station_id": "TEST001", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            }],
            "tz": "UTC",
            "lead_days": 2,
            "url": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(_fake_lead_response(base_temp_c=25.0, lead_days=2))
        vfd._process_one_station_lead(station, session)
        assert "elevation" not in session.calls_params[0]

    def test_enabled_sends_bare_nan_scalar(self):
        station = {
            "station_id": "TEST001",
            "rows": [{
                "station_id": "TEST001", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            }],
            "tz": "UTC",
            "lead_days": 2,
            "url": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
            "disable_elevation_correction": True,
        }
        session = FakeSession(_fake_lead_response(base_temp_c=25.0, lead_days=2))
        vfd._process_one_station_lead(station, session)
        assert session.calls_params[0]["elevation"] == "nan"

    def test_build_paired_rows_threads_flag_to_every_station(self, tmp_path, monkeypatch):
        rows = [
            {"station_id": "A", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
             "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0},
            {"station_id": "B", "date": "2023-06-15", "lat": 51.5, "lon": -0.1,
             "climate_zone": "Cfb", "station_tmax_c": 22.0, "station_tmin_c": 14.0},
        ]
        tz_by_station = {"A": "UTC", "B": "UTC"}
        fake_session = FakeSession(_fake_lead_response(base_temp_c=20.0, lead_days=2))
        monkeypatch.setattr(vfd, "HttpSession", lambda *a, **kw: fake_session)
        checkpoint_path = str(tmp_path / "checkpoint.jsonl")

        vfd.build_paired_rows(
            rows, tz_by_station, lead_days=2, api_key=None, max_workers=2, checkpoint_path=checkpoint_path,
            disable_elevation_correction=True,
        )

        assert len(fake_session.calls_params) == 2
        assert all(p.get("elevation") == "nan" for p in fake_session.calls_params)


class TestBuildPairedRowsCheckpointing:
    def test_rerun_with_same_checkpoint_skips_already_done_stations(self, tmp_path, monkeypatch):
        rows = [
            {"station_id": "A", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
             "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0},
        ]
        tz_by_station = {"A": "UTC"}
        checkpoint_path = str(tmp_path / "checkpoint.jsonl")

        fake_session = FakeSession(_fake_lead_response(base_temp_c=20.0, lead_days=2))
        monkeypatch.setattr(vfd, "HttpSession", lambda *a, **kw: fake_session)

        vfd.build_paired_rows(rows, tz_by_station, lead_days=2, api_key=None,
                               max_workers=1, checkpoint_path=checkpoint_path)
        calls_after_first_run = fake_session.calls

        vfd.build_paired_rows(rows, tz_by_station, lead_days=2, api_key=None,
                               max_workers=1, checkpoint_path=checkpoint_path)

        assert fake_session.calls == calls_after_first_run  # resumed run made no new HTTP calls


class TestServiceConfigRetriableStatuses:
    """2026-09-18, #710: same evidence/reasoning as validate_lagfill_
    downscaling.py's own TestServiceConfigRetriableStatuses -- see that
    file for the full incident writeup."""

    def test_keyed_config_retries_400(self):
        cfg = vfd._service_config(api_key="fake-key", lead_days=1)
        assert 400 in cfg.retriable_statuses
        assert {429, 500, 502, 503, 504} <= cfg.retriable_statuses

    def test_anon_config_does_not_retry_400(self):
        cfg = vfd._service_config(api_key=None, lead_days=1)
        assert 400 not in cfg.retriable_statuses

    def test_config_name_includes_lead_days(self):
        cfg = vfd._service_config(api_key="fake-key", lead_days=3)
        assert cfg.name == "forecast_lead3"

    def test_keyed_config_retry_max_is_one(self):
        # 2026-09-19, #710: caught live -- this was missed when the delayed-
        # requeue pattern was ported from validate_lagfill_downscaling.py in
        # PR #60, and forecast_lead1 was observed doing quick in-place
        # retries immediately after launch as a result. Must match
        # validate_lagfill_downscaling._service_config's own keyed retry_max.
        cfg = vfd._service_config(api_key="fake-key", lead_days=1)
        assert cfg.retry_max == 1

    def test_anon_config_retry_max_stays_four(self):
        cfg = vfd._service_config(api_key=None, lead_days=1)
        assert cfg.retry_max == 4
