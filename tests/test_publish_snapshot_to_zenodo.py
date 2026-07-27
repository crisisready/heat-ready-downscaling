"""Unit tests for scripts/publish_snapshot_to_zenodo.py's pure-logic pieces
(metadata construction, CLI argument validation) -- no real Zenodo API
calls. The HTTP-calling functions (_create_new_deposition, _new_version,
_upload_file, _update_metadata, _publish) are thin wrappers around single
`requests` calls, deliberately untested against a live API here -- this
script was itself hand-verified end-to-end against the real Zenodo API
(2026-07-27: draft created, metadata/file confirmed correct via a direct
API read, then published -- DOI 10.5281/zenodo.21633231, concept DOI
10.5281/zenodo.21633230)."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import publish_snapshot_to_zenodo as psz

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "publish_snapshot_to_zenodo.py")


class TestMetadata:
    def test_license_matches_data_license(self):
        """DATA_LICENSE declares CC BY 4.0 -- this must stay in sync."""
        assert psz._metadata("v2026.07")["license"] == "cc-by-4.0"

    def test_upload_type_is_dataset(self):
        assert psz._metadata("v2026.07")["upload_type"] == "dataset"

    def test_version_field_reflects_snapshot_version(self):
        assert psz._metadata("v2026.08")["version"] == "v2026.08"

    def test_links_back_to_github_repo(self):
        m = psz._metadata("v2026.07")
        assert any(r["identifier"] == psz._GITHUB_REPO_URL for r in m["related_identifiers"])


class TestCliArgumentValidation:
    """--publish-draft is mutually exclusive with --tarball/--version-of;
    --tarball/--snapshot-version are required otherwise. No real token or
    network access needed -- these all fail argument validation before any
    HTTP call would be made."""

    def _run(self, *args):
        env = dict(os.environ)
        env.pop("ZENODO_ACCESS_TOKEN", None)
        return subprocess.run([sys.executable, _SCRIPT, *args], capture_output=True, text=True, env=env)

    def test_no_args_fails(self):
        result = self._run()
        assert result.returncode != 0

    def test_publish_draft_with_tarball_fails(self):
        result = self._run("--publish-draft", "123", "--tarball", "/tmp/x.tar.gz")
        assert result.returncode != 0
        assert "exclusive" in result.stderr

    def test_tarball_without_snapshot_version_fails(self):
        result = self._run("--tarball", "/tmp/x.tar.gz")
        assert result.returncode != 0

    def test_missing_token_fails_fast(self):
        result = self._run("--tarball", "/tmp/x.tar.gz", "--snapshot-version", "v2026.07")
        assert result.returncode != 0
        assert "ZENODO_ACCESS_TOKEN" in result.stderr
