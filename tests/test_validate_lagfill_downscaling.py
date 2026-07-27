"""Unit tests for validate_lagfill_downscaling.py's api_call_manager
integration -- no network calls, no real sleeping. Focused on what's
genuinely specific to this script (fetch_fn payload shape, checkpoint/
resume, the report envelope's band_key stamp) -- score_band/fidelity_report
themselves are covered by test_score.py now that this script imports them
from heatready_downscaling.score rather than defining its own copy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from api_call_manager import NO_RESULT, AdaptiveThrottle

import validate_lagfill_downscaling as vld


def _fake_hourly_response(base_temp_c: float):
    """24 hours of Open-Meteo-shaped hourly data spanning the exact window
    heat_calcs' 06:00-local-day boundary needs for a COMPLETE "2023-06-15"
    bucket (local_day_iso: dt_local - 6h -- so local day 06-15 covers real
    UTC hours 06-15T06:00 through 06-16T05:00, not calendar-midnight)."""
    times = [f"2023-06-15T{h:02d}:00" for h in range(6, 24)] + [f"2023-06-16T{h:02d}:00" for h in range(0, 6)]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [base_temp_c + (h % 12) for h in range(24)],
            "dewpoint_2m": [base_temp_c - 5 for _ in range(24)],
            "windspeed_10m": [3.0 for _ in range(24)],
            "shortwave_radiation": [0.0 for _ in range(24)],
            "direct_radiation": [0.0 for _ in range(24)],
            "surface_pressure": [1013.0 for _ in range(24)],
        }
    }


class FakeSession:
    """Minimal stand-in for api_call_manager.HttpSession -- returns a
    scripted response regardless of params, records call count."""

    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.throttle = AdaptiveThrottle(max_workers=4)  # fetch_all reads session.throttle.stats()

    def get_json(self, url, params, **kwargs):
        self.calls += 1
        return self._response


class TestProcessOneStation:
    def test_successful_fetch_returns_combined_payload(self):
        station = {
            "station_id": "TEST001",
            "rows": [{
                "station_id": "TEST001", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
                "grid_tmax_c": 29.0, "grid_tmin_c": 19.0,
            }],
            "tz": "UTC",
            "url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(_fake_hourly_response(base_temp_c=25.0))

        payload = vld._process_one_station(station, session)

        assert payload is not NO_RESULT
        assert "nrt_rows" in payload and "fidelity_rows" in payload
        assert len(payload["nrt_rows"]) == 1
        nrt_row = payload["nrt_rows"][0]
        # grid_tmax_c/grid_tmin_c replaced by the NRT reconstruction, static
        # covariates (climate_zone, station truth) carried through unchanged.
        assert nrt_row["climate_zone"] == "Cfa"
        assert nrt_row["station_tmax_c"] == 30.0
        assert nrt_row["grid_tmax_c"] != 29.0  # replaced, not the original ERA5 value
        # fidelity_rows populated since the original row had both grid values.
        assert len(payload["fidelity_rows"]) == 1
        assert payload["fidelity_rows"][0]["era5_tmax"] == 29.0

    def test_no_successful_rows_returns_no_result(self):
        station = {
            "station_id": "TEST002",
            "rows": [{
                "station_id": "TEST002", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
                "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0,
            }],
            "tz": "UTC",
            "url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
        }
        session = FakeSession(NO_RESULT)  # every HTTP call fails

        payload = vld._process_one_station(station, session)

        assert payload is NO_RESULT


class TestBuildPairedRowsCheckpointing:
    def test_collects_and_flattens_across_stations(self, tmp_path, monkeypatch):
        rows = [
            {"station_id": "A", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
             "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0},
            {"station_id": "B", "date": "2023-06-15", "lat": 51.5, "lon": -0.1,
             "climate_zone": "Cfb", "station_tmax_c": 22.0, "station_tmin_c": 14.0},
        ]
        tz_by_station = {"A": "UTC", "B": "UTC"}

        # Patch HttpSession construction inside build_paired_rows to return
        # our fake, network-free session regardless of ServiceConfig.
        fake_session = FakeSession(_fake_hourly_response(base_temp_c=20.0))
        monkeypatch.setattr(vld, "HttpSession", lambda *a, **kw: fake_session)

        checkpoint_path = str(tmp_path / "checkpoint.jsonl")
        nrt_rows, fidelity_rows = vld.build_paired_rows(
            rows, tz_by_station, api_key=None, max_workers=2, checkpoint_path=checkpoint_path,
        )

        assert len(nrt_rows) == 2  # one per station
        assert {r["station_id"] for r in nrt_rows} == {"A", "B"}
        assert os.path.exists(checkpoint_path)  # checkpoint file actually written

    def test_rerun_with_same_checkpoint_skips_already_done_stations(self, tmp_path, monkeypatch):
        rows = [
            {"station_id": "A", "date": "2023-06-15", "lat": 40.0, "lon": -75.0,
             "climate_zone": "Cfa", "station_tmax_c": 30.0, "station_tmin_c": 20.0},
        ]
        tz_by_station = {"A": "UTC"}
        checkpoint_path = str(tmp_path / "checkpoint.jsonl")

        fake_session = FakeSession(_fake_hourly_response(base_temp_c=20.0))
        monkeypatch.setattr(vld, "HttpSession", lambda *a, **kw: fake_session)

        vld.build_paired_rows(rows, tz_by_station, api_key=None, max_workers=1, checkpoint_path=checkpoint_path)
        calls_after_first_run = fake_session.calls

        # Second run, same checkpoint -- station A already done, must not re-fetch.
        vld.build_paired_rows(rows, tz_by_station, api_key=None, max_workers=1, checkpoint_path=checkpoint_path)

        assert fake_session.calls == calls_after_first_run  # no new HTTP calls on the resumed run


class _FakeAdapter:
    """Minimal heatready_downscaling.contract.ModelAdapter stand-in --
    _build_report only needs score_band(adapter, ...) to not raise; the
    real scoring logic itself is test_score.py's responsibility."""

    model_version = "ds-test"

    def predict(self, rows, target, extra_zone_gate=None, bias_correction=None):
        return [{"applied": False} for _ in rows]


class TestBuildReportBandKey:
    """build_report (heatready_downscaling.report) must be called with
    band_key="lag_fill" so publish_band_gate.build_gate can refuse to
    publish a lag-fill report under the wrong --band-key."""

    def test_report_is_stamped_lag_fill(self):
        report = vld._build_report("ds-test", 10, 10, [], [], _FakeAdapter())
        assert report["band_key"] == "lag_fill"

    def test_report_uses_model_version_as_fold_salt_sentinel_snapshot_version(self):
        """No real Phase-2 snapshot backs a live-DB validation run -- see
        the module docstring's _SNAPSHOT_VERSION comment."""
        report = vld._build_report("ds-test", 10, 10, [], [], _FakeAdapter())
        assert report["snapshot_version"] == "ghcn_training-live"
        assert report["model_version"] == "ds-test"
