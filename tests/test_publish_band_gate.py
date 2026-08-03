"""CLI-level tests for scripts/publish_band_gate.py -- the human-run publish
step that turns a validate_*_downscaling.py report into what gets uploaded
to s3://.../downscaling/band_gates/{model_version}/{band_key}.json.

build_gate itself (the actual report->gate transformation, including
bias_correction/spatial_skill/band_key-mismatch behavior) is
heatready_downscaling.gates.build_gate now, fully covered by
tests/test_gates.py -- these tests only exercise this script's own CLI
wiring (argument validation, --dry-run, schema validation via
heatready_downscaling.gates.validate_gate), matching
test_publish_blend_gate.py's style for its sibling script."""

import json
import os
import subprocess
import sys

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "publish_band_gate.py")

_VALID_REPORT = {
    "band_key": "lag_fill",
    "by_target": {
        "tmax": {
            "Cfb": {
                "qrf_beats_grid_with_margin": True, "qrf_beats_grid": True,
                "rmse_improvement_pct_debiased_cv": None, "bias_correction_c": None,
            },
        },
        "tmin": {},
    },
}


def _write_report(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return str(path)


class TestPublishBandGateCli:
    def test_dry_run_does_not_upload(self, tmp_path):
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--dry-run: not uploading" in result.stdout
        assert "Published to" not in result.stdout

    def test_dry_run_prints_passing_zones(self, tmp_path):
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "Cfb" in result.stdout

    def test_band_key_rejects_unknown_value(self, tmp_path):
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "not_a_real_band", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_mismatched_band_key_fails_fast(self, tmp_path):
        """build_gate's own band_key check (heatready_downscaling.gates) --
        a lag_fill report can't be published under --band-key forecast_lead3."""
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "forecast_lead3", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "lag_fill" in result.stderr and "forecast_lead3" in result.stderr

    def test_matching_variant_proceeds_and_prints_it(self, tmp_path):
        """2026-08-03, gate-variant scoping."""
        variant_report = {**_VALID_REPORT, "base_variant": "native_noelev"}
        report_path = _write_report(tmp_path, variant_report)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--variant", "native_noelev", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "variant=native_noelev" in result.stdout

    def test_variant_fitted_report_without_variant_flag_fails_fast(self, tmp_path):
        """The dangerous direction (see gates.build_gate's own docstring):
        a native_noelev-fitted report must not silently publish to the
        default (no-variant) key."""
        variant_report = {**_VALID_REPORT, "base_variant": "native_noelev"}
        report_path = _write_report(tmp_path, variant_report)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "native_noelev" in result.stderr

    def test_default_report_with_variant_flag_fails_fast(self, tmp_path):
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--variant", "native_noelev", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "native_noelev" in result.stderr

    def test_malformed_report_fails_validate_gate(self, tmp_path):
        """A report whose by_target values aren't real score_band metrics
        shapes still produces SOME gate dict from build_gate, but
        validate_gate (heatready_downscaling.gates, called by this CLI
        before printing/uploading) catches a genuinely malformed gate --
        the private repo's original version had no such check at all."""
        bad_report = {"band_key": "lag_fill", "by_target": "not-a-dict"}
        report_path = _write_report(tmp_path, bad_report)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
