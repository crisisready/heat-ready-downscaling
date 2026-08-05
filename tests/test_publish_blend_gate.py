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
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import publish_blend_gate  # noqa: E402

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

    def test_variant_printed_in_dry_run_header(self, tmp_path):
        """--variant (2026-08-04, gate-variant scoping parity with
        publish_band_gate.py) doesn't change gate validation -- it only
        changes the key key construction/print (checked here) and the
        upload key (checked below via key_band construction, since
        --dry-run never uploads)."""
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1",
             "--variant", "native_noelev", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "variant=native_noelev" in result.stdout

    def test_no_variant_omits_variant_from_dry_run_header(self, tmp_path):
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "variant=" not in result.stdout

    def test_KNOWN_LIMITATION_variant_mismatch_is_not_caught(self, tmp_path):
        """Documents a real, disclosed gap (code review, 2026-08-05), not a
        passing feature: publish_band_gate.py has TWO independent safety
        mechanisms -- (1) same-key zone-drop protection (ported here,
        covered by TestRefuseIfZonesWouldBeDropped above) and (2) a
        cross-key check that the --report's own stamped base_variant
        matches --variant (build_gate enforces this in both directions --
        see heatready_downscaling.gates.build_gate's own docstring).

        This script can only replicate (1). Mechanism (2) has no analogue
        here because validate_station_blend.py's --gate-out JSON carries no
        variant stamp at all -- there is nothing in the gate file itself to
        cross-check --variant against. This test asserts that CURRENT
        behavior -- publishing under any --variant string succeeds
        regardless of what the gate was actually validated against -- so a
        future change that silently starts enforcing this (or a refactor
        that accidentally removes intended-future enforcement) shows up as
        a deliberate, reviewed test change, not a silent behavior shift.

        The caller remains responsible for passing the --variant that
        matches whatever base distribution --gate was actually validated
        against, exactly as this script's own --help text says. Closing
        this gap for real means adding a variant stamp to
        validate_station_blend.py's own output first -- a separate,
        deliberately-not-bundled-here change (see this PR's own
        description)."""
        gate_path = _write_gate(tmp_path, _VALID_GATE)
        result = subprocess.run(
            [sys.executable, _SCRIPT, "--gate", gate_path, "--model-version", "ds-test-1",
             "--variant", "totally_unrelated_variant_the_gate_was_never_validated_against",
             "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            "if this now fails, --variant enforcement was added -- update/remove this test "
            "deliberately rather than treating the failure as a regression"
        )
        assert "variant=totally_unrelated_variant_the_gate_was_never_validated_against" in result.stdout


class TestRefuseIfZonesWouldBeDropped:
    """Direct tests of _refuse_if_zones_would_be_dropped -- near-verbatim
    clone of publish_band_gate.py's own helper (added there 2026-08-03,
    mirrored here 2026-08-04 since this script had no equivalent
    protection at all before)."""

    def test_no_prior_gate_proceeds(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error("NoSuchKey")
        publish_blend_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise -- first-ever publish to this key

    def test_other_client_error_reraises(self):
        """Only NoSuchKey/404 means 'nothing to drop' -- any other S3 error
        (e.g. AccessDenied) must propagate, not be silently swallowed as if
        it meant the same thing. Fail-closed applies to the safety check
        itself, not just the thing it's protecting. Missed in the original
        port (2026-08-04) -- publish_band_gate.py's own test suite already
        covers this for the sibling helper; this was the one gap."""
        client = MagicMock()
        client.get_object.side_effect = _client_error("AccessDenied")
        try:
            publish_blend_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected the AccessDenied ClientError to propagate"
        except Exception as exc:
            from botocore.exceptions import ClientError
            assert isinstance(exc, ClientError)
            assert exc.response["Error"]["Code"] == "AccessDenied"

    def test_same_zones_proceeds(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True}, "tmin": {}}).encode()),
        }
        publish_blend_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise

    def test_added_zone_is_not_a_drop(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True}, "tmin": {}}).encode()),
        }
        publish_blend_gate._refuse_if_zones_would_be_dropped(
            client, "bucket", "key", {"tmax": {"Cfb": True, "Csa": True}, "tmin": {}}, confirm_drops=False,
        )  # must not raise -- a superset is never a drop

    def test_dropped_zone_refuses_without_confirm(self, capsys):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True, "Csa": True}, "tmin": {}}).encode()),
        }
        try:
            publish_blend_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected SystemExit"
        except SystemExit as exc:
            assert exc.code == 1
        assert "Csa" in capsys.readouterr().err

    def test_dropped_zone_proceeds_with_confirm_drops(self):
        client = MagicMock()
        client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"tmax": {"Cfb": True, "Csa": True}, "tmin": {}}).encode()),
        }
        publish_blend_gate._refuse_if_zones_would_be_dropped(
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
            publish_blend_gate._refuse_if_zones_would_be_dropped(
                client, "bucket", "key", {"tmax": {"Cfb": True}, "tmin": {}}, confirm_drops=False,
            )
            assert False, "expected SystemExit for the dropped tmin zone"
        except SystemExit:
            pass


def _client_error(code):
    from botocore.exceptions import ClientError

    def _raise(*args, **kwargs):
        raise ClientError({"Error": {"Code": code}}, "GetObject")

    return _raise
