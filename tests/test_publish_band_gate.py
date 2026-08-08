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
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import publish_band_gate  # noqa: E402

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

    def test_dry_run_never_touches_s3(self, tmp_path, monkeypatch):
        """--dry-run smoke test for the 2026-08-03 fail-closed zone-drop
        check: dry-run must stay exactly as fast/credential-free as before
        this change -- the new GET+diff+refuse logic only runs on the real
        publish path (after the dry-run early return), so a dry-run must
        never require AWS credentials or network access at all. Deliberately
        unset VULNERABILITY_DATA_BUCKET so a regression that moved the
        bucket/key resolution earlier would fail loudly here (KeyError)
        instead of silently starting to require it."""
        monkeypatch.delenv("VULNERABILITY_DATA_BUCKET", raising=False)
        report_path = _write_report(tmp_path, _VALID_REPORT)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--dry-run: not uploading" in result.stdout


def _client_error(code):
    return ClientError({"Error": {"Code": code}}, "GetObject")


class TestRefuseIfZonesWouldBeDropped:
    """Direct unit tests for _refuse_if_zones_would_be_dropped -- mocks the
    boto3 client rather than going through subprocess/CLI, matching
    test_train_downscaling.py's TestSaveModelArtifacts mocking style
    (there's no moto in this codebase)."""

    def test_no_existing_gate_proceeds(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error("NoSuchKey")
        publish_band_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise

    def test_other_client_error_reraises(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error("AccessDenied")
        try:
            publish_band_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected the ClientError to propagate"
        except ClientError as exc:
            assert exc.response["Error"]["Code"] == "AccessDenied"

    def test_same_zones_proceeds(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True}, "tmin": {}}).encode()),
        }
        publish_band_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise

    def test_added_zone_is_not_a_drop(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True}, "tmin": {}}).encode()),
        }
        publish_band_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True, "Dfa": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise -- a superset is never a drop

    def test_dropped_zone_refuses_without_confirm(self, capsys):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True, "Dfa": True}, "tmin": {}}).encode()),
        }
        try:
            publish_band_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected SystemExit"
        except SystemExit as exc:
            assert exc.code == 1
        assert "Dfa" in capsys.readouterr().err

    def test_dropped_zone_proceeds_with_confirm_drops(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True, "Dfa": True}, "tmin": {}}).encode()),
        }
        publish_band_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=True,
        )  # must not raise

    def test_drop_checked_per_target_independently(self):
        """A tmin drop must refuse even if tmax is unchanged, and vice versa."""
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(
                {"tmax": {"Cfb": True}, "tmin": {"Cfb": True}}).encode()),
        }
        try:
            publish_band_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected SystemExit for the dropped tmin zone"
        except SystemExit:
            pass


_SUBZONE_PATCH = {
    "delta_scale_subzone": {"tmax": {"Cfb": {"FR": {"scale": 0.6, "offset": 0.2}}}, "tmin": {}},
    "bias_correction_subzone": {"tmax": {}, "tmin": {}},
}


class TestDoSubzonePublish:
    """Direct unit tests for _do_subzone_publish -- mocks the boto3 client,
    same style as TestRefuseIfZonesWouldBeDropped above."""

    def test_no_current_gate_refuses(self):
        """A subzone patch refines an already-published zone-level gate --
        it must never create one out of thin air (both because the merge
        result would fail BAND_GATE_SCHEMA's own required top-level fields,
        and because a subzone-only gate is dead data nothing would ever
        read -- see _do_subzone_publish's own docstring)."""
        client = MagicMock()
        client.get_object.side_effect = _client_error("NoSuchKey")
        try:
            publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=False)
            assert False, "expected SystemExit"
        except SystemExit:
            pass
        client.put_object.assert_not_called()

    _EXISTING_GATE = {
        "tmax": {"Cfb": True}, "tmin": {"Cfb": True},
        "bias_correction": {"tmax": {}, "tmin": {}},
        "delta_scale": {"tmax": {"Cfb": {"scale": 0.415, "offset": 0.312}}, "tmin": {}},
        "spatial_skill": {"tmax": {"Cfb": True}, "tmin": {"Cfb": True}},
    }

    def test_dry_run_does_not_call_put_object(self):
        client = MagicMock()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(self._EXISTING_GATE).encode())}
        publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=True)
        client.put_object.assert_not_called()

    def test_dry_run_still_gets_the_real_current_gate(self):
        """dry-run must show what the merge would ACTUALLY produce against
        the real current state -- unlike the full-gate publish path, this
        one needs a real GET even on a dry run (see _publish_subzone_patch's
        own docstring for why)."""
        client = MagicMock()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(self._EXISTING_GATE).encode())}
        publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=True)
        client.get_object.assert_called_once()

    def test_merges_into_existing_gate_without_dropping_zone_level_fields(self):
        client = MagicMock()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(self._EXISTING_GATE).encode())}
        merged = publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=False)
        # zone-level fields untouched
        assert merged["tmax"] == {"Cfb": True}
        assert merged["delta_scale"]["tmax"]["Cfb"] == {"scale": 0.415, "offset": 0.312}
        # new subzone entry present alongside
        client.put_object.assert_called_once()
        assert merged["delta_scale_subzone"]["tmax"]["Cfb"]["FR"] == {"scale": 0.6, "offset": 0.2}

    def test_preserves_a_different_zones_existing_subzone_entry(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                **self._EXISTING_GATE,
                "delta_scale_subzone": {"tmax": {"Csa": {"ES": {"scale": 0.7, "offset": 0.0}}}, "tmin": {}},
                "bias_correction_subzone": {"tmax": {}, "tmin": {}},
            }).encode()),
        }
        merged = publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=False)
        assert merged["delta_scale_subzone"]["tmax"]["Csa"]["ES"] == {"scale": 0.7, "offset": 0.0}
        assert merged["delta_scale_subzone"]["tmax"]["Cfb"]["FR"] == {"scale": 0.6, "offset": 0.2}

    def test_other_client_error_on_get_reraises(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error("AccessDenied")
        try:
            publish_band_gate._do_subzone_publish(client, _SUBZONE_PATCH, "bucket", "key", dry_run=False)
            assert False, "expected the ClientError to propagate"
        except ClientError as exc:
            assert exc.response["Error"]["Code"] == "AccessDenied"


class TestPublishBandGateCliRegionsRouting:
    """CLI-level: a regions-stamped report must route to the subzone-patch
    path (build_subzone_patch/_do_subzone_publish), never build_gate.
    Only the missing-bucket early-exit is exercised here via subprocess
    (no AWS access needed -- it fails before ever creating an S3 client);
    the actual GET/merge/validate/PUT behavior of the subzone path is
    covered in-process with a mocked client by TestDoSubzonePublish above,
    same split as the full-gate path's own TestPublishBandGateCli
    (CLI wiring) vs. TestRefuseIfZonesWouldBeDropped (S3-interacting
    logic)."""

    def test_regions_scoped_report_does_not_crash_missing_bucket_the_same_way(self, tmp_path, monkeypatch):
        """Without a bucket, the subzone path must fail with the SAME kind
        of explicit, early SystemExit the full-gate path uses -- not an
        unhandled KeyError -- even before touching S3."""
        monkeypatch.delenv("VULNERABILITY_DATA_BUCKET", raising=False)
        regions_report = {**_VALID_REPORT, "regions": ["FR"]}
        report_path = _write_report(tmp_path, regions_report)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--report", report_path, "--model-version", "ds-test-1",
             "--band-key", "lag_fill", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "VULNERABILITY_DATA_BUCKET" in result.stderr or "bucket" in result.stderr.lower()
