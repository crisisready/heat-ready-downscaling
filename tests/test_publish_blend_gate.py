"""Unit tests for scripts/publish_blend_gate.py -- the human-run publish step
that uploads a validate_station_blend.py --gate-out file to
s3://.../downscaling/blend_gates/{model_version}/{band_key}.json, read back
by downscaling.load_blend_gate (crisisready/heat-risk-data-api).

Schema validation now goes through heatready_downscaling.gates.
validate_blend_gate (a real jsonschema check, including the per-group
L_km/R_km/tau shape) rather than the private repo's original ad-hoc
presence check -- see test_gates.py's TestValidateBlendGate for direct
coverage of that function; these tests only exercise this script's own CLI
wiring."""

import json
import os
import subprocess
import sys

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "publish_blend_gate.py")

_VALID_GATE = {
    "tmax": {"BWh": True, "Cfb": True, "Csb": True, "Dfb": True},
    "tmin": {"BSh": True, "Cfb": True, "Csa": True},
    # Params are per broad Koppen group (A-E), not a single global triple --
    # BWh/BSh -> group B, Cfb/Csb/Csa -> group C, Dfb -> group D.
    "params": {
        "tmax": {
            "B": {"L_km": 250.0, "R_km": 25.0, "tau": 4.0},
            "C": {"L_km": 30.0, "R_km": 50.0, "tau": 1.0},
            "D": {"L_km": 50.0, "R_km": 50.0, "tau": 4.0},
        },
        "tmin": {
            "B": {"L_km": 250.0, "R_km": 25.0, "tau": 4.0},
            "C": {"L_km": 30.0, "R_km": 50.0, "tau": 1.0},
        },
    },
}


def _write_gate(tmp_path, gate):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    return str(path)


class TestPublishBlendGateCli:
    def test_dry_run_does_not_upload(self, tmp_path):
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--dry-run: not uploading" in result.stdout
        assert "Published to" not in result.stdout

    def test_dry_run_prints_zones_and_params(self, tmp_path):
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "BWh" in result.stdout and "Cfb" in result.stdout
        assert "tmax group C: L_km=30.0" in result.stdout

    def test_band_key_rejects_era5(self, tmp_path):
        """Only 'lag_fill' is a valid --band-key -- publishing under 'era5'
        or a forecast lead would falsely imply a distribution nobody has
        actually validated (see downscaling.load_blend_gate's own
        docstring). The CLI refuses this outright, not just documentation."""
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1",
             "--band-key", "era5", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_band_key_rejects_forecast_lead(self, tmp_path):
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1",
             "--band-key", "forecast_lead1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_missing_required_key_fails_fast(self, tmp_path):
        incomplete_gate = {"tmax": {"BWh": True}, "tmin": {}}  # no "params" key
        gate_path = _write_gate(tmp_path, incomplete_gate)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "params" in result.stderr

    def test_malformed_param_group_fails_schema_validation(self, tmp_path):
        """A group's params missing e.g. tau -- validate_blend_gate's real
        jsonschema check catches this; the private repo's original ad-hoc
        presence check (only "tmax"/"tmin"/"params" at the top level) did
        not."""
        malformed_gate = {
            "tmax": {"BWh": True}, "tmin": {},
            "params": {"tmax": {"B": {"L_km": 250.0, "R_km": 25.0}}},  # no "tau"
        }
        gate_path = _write_gate(tmp_path, malformed_gate)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
