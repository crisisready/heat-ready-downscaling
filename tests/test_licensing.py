"""Unit tests for heatready_downscaling.licensing.

The gate CONTRIBUTING.md documented in July and nothing implemented until
2026-08-25. These tests are written against the promise's own wording -- an
SPDX allowlist, a proprietary-licensed escape hatch requiring a named
licensor, and a manual-review flag -- so the implementation is checked against
the contract the repo already published rather than against itself.
"""

import pytest

from heatready_downscaling import licensing


class TestLicenseId:
    def test_an_allowlisted_identifier_clears_automatically(self):
        assert licensing.check_license_id("CC0-1.0", where="x") is False

    def test_an_unlisted_identifier_is_rejected(self):
        with pytest.raises(licensing.LicensingError, match="not on the allowlist"):
            licensing.check_license_id("WTFPL", where="x")

    def test_case_matters_and_the_error_says_so(self):
        """SPDX identifiers are case-sensitive. Silently accepting
        'cc-by-4.0' for 'CC-BY-4.0' would make the allowlist advisory rather
        than a gate, and the near-miss hint is what stops that being a
        baffling rejection."""
        with pytest.raises(licensing.LicensingError, match="case-sensitive"):
            licensing.check_license_id("cc-by-4.0", where="x")

    def test_a_non_commercial_licence_is_rejected(self):
        """HeatReady's own DATA_LICENSE carries no non-commercial
        restriction, so accepting NC data would make the published
        snapshot's stated terms wrong for part of its own contents."""
        with pytest.raises(licensing.LicensingError):
            licensing.check_license_id("CC-BY-NC-4.0", where="x")

    @pytest.mark.parametrize("empty", [None, "", "   ", 5])
    def test_a_missing_licence_is_rejected(self, empty):
        with pytest.raises(licensing.LicensingError, match="missing or empty"):
            licensing.check_license_id(empty, where="x")

    def test_the_escape_hatch_needs_a_named_licensor(self):
        with pytest.raises(licensing.LicensingError, match="requires a non-empty 'licensor'"):
            licensing.check_license_id(licensing.PROPRIETARY_LICENSE_ID, where="x")

    def test_the_escape_hatch_with_a_licensor_flags_for_review_rather_than_passing(self):
        """It is not a way past the allowlist -- it is a way to declare data
        we hold a real licence for that has no SPDX identifier, and it always
        reaches a human."""
        assert licensing.check_license_id(
            licensing.PROPRIETARY_LICENSE_ID, where="x", licensor="Ajuntament de Valencia",
        ) is True


class TestDataSource:
    def _entry(self, **over):
        base = {
            "name": "Natural Earth 1:10m coastline",
            "license": "CC0-1.0",
            "reproducible_fetch": "https://example.invalid/ne.zip",
            "redistribution_tier": "unrestricted",
        }
        base.update(over)
        return base

    def test_a_complete_permissive_entry_clears(self):
        assert licensing.check_data_source(self._entry(), where="x") is False

    @pytest.mark.parametrize("key", ["name", "license", "reproducible_fetch", "redistribution_tier"])
    def test_every_required_key_is_required(self, key):
        entry = self._entry()
        del entry[key]
        with pytest.raises(licensing.LicensingError, match="missing"):
            licensing.check_data_source(entry, where="x")

    def test_an_unknown_redistribution_tier_is_rejected(self):
        with pytest.raises(licensing.LicensingError, match="redistribution_tier"):
            licensing.check_data_source(self._entry(redistribution_tier="maybe"), where="x")

    def test_no_redistribution_data_always_reaches_a_human(self):
        """A local model may legitimately train on data we cannot republish,
        so this is not a rejection -- but it can never pass silently, because
        the published snapshot's contents depend on it."""
        assert licensing.check_data_source(
            self._entry(redistribution_tier="no-redistribution"), where="x",
        ) is True

    def test_attribution_required_contradicting_an_unrestricted_tier_is_rejected(self):
        """The snapshot's attribution notices are generated from the tier, so
        the two fields disagreeing means one of them will be wrong in
        published output."""
        with pytest.raises(licensing.LicensingError, match="contradict"):
            licensing.check_data_source(
                self._entry(attribution_required=True, redistribution_tier="unrestricted"),
                where="x",
            )

    def test_a_non_object_entry_is_rejected(self):
        with pytest.raises(licensing.LicensingError, match="must be an object"):
            licensing.check_data_source("CC0-1.0", where="x")


class TestManifestLevel:
    def _manifest(self, **method):
        return {"method": {"kind": "parameters", **method}}

    def test_a_manifest_with_no_declared_data_passes(self):
        assert licensing.check_manifest_licensing(self._manifest()) == []

    def test_a_bad_licence_anywhere_raises(self):
        m = self._manifest(extra_covariates=[{"name": "x", "license": "WTFPL"}])
        with pytest.raises(licensing.LicensingError, match="extra_covariates\\[0\\]"):
            licensing.check_manifest_licensing(m)

    def test_flagged_entries_are_returned_with_their_location(self):
        m = self._manifest(data_sources=[{
            "name": "Valencia municipal sensors",
            "license": licensing.PROPRIETARY_LICENSE_ID,
            "licensor": "Ajuntament de Valencia",
            "reproducible_fetch": "https://example.invalid/x",
            "redistribution_tier": "no-redistribution",
        }])
        flagged = licensing.check_manifest_licensing(m)
        assert len(flagged) == 1
        assert "data_sources[0]" in flagged[0]
        assert "Valencia municipal sensors" in flagged[0]

    def test_both_surfaces_are_checked_not_just_one(self):
        m = self._manifest(
            data_sources=[{
                "name": "ok", "license": "CC0-1.0",
                "reproducible_fetch": "u", "redistribution_tier": "unrestricted",
            }],
            extra_covariates=[{"name": "bad", "license": "NotAnSpdxId"}],
        )
        with pytest.raises(licensing.LicensingError, match="extra_covariates"):
            licensing.check_manifest_licensing(m)


def test_the_real_coastline_provenance_block_passes_its_own_gate():
    """coastline.NATURAL_EARTH_COASTLINE_DATA_SOURCE was written in PR #29,
    BEFORE this checker existed, with key names that were explicitly a
    proposal at the time. This asserts the two actually agree -- otherwise
    the repo's first real data-source entry would fail the repo's first
    real licensing gate."""
    from heatready_downscaling import coastline

    assert licensing.check_data_source(
        coastline.NATURAL_EARTH_COASTLINE_DATA_SOURCE, where="coastline",
    ) is False


def test_validate_manifest_enforces_licensing():
    """The rule has to hold at validate_manifest, not only in a workflow:
    score_forward_eval.py's monthly re-scoring reads merged manifests off
    disk without re-running jsonschema, so a merge-time-only check would not
    bind the official cycle."""
    from heatready_downscaling import submission

    manifest = {
        "schema_version": 1, "submission_id": "2026-08-004",
        "author": {"github": "x", "name": "X", "orcid": None, "affiliation": None},
        "track": "research", "rung": "A",
        "snapshot": {"version": "v2026.07", "manifest_sha256": "a" * 64},
        "claims": [{"model_version": "ds-2026.07-rf5", "band_key": "lag_fill",
                    "targets": ["tmax"], "zones": ["Cfb"]}],
        "method": {
            "kind": "rerun-validator", "entrypoint": "scripts/run_submission.py",
            "args": [], "package_version": "0.1.0", "code_ref": None,
            "extra_covariates": [{
                "name": "sneaky", "source": "somewhere", "license": "AllRightsReserved",
                "global": True, "reproducible_fetch": "https://example.invalid/x",
            }],
        },
        "claimed_report": "claimed_report.json",
        "tolerance": {"rmse_qrf_c": 0.005},
        "reproducibility": {"seed": None, "runtime_notes": None},
    }
    with pytest.raises(licensing.LicensingError, match="not on the allowlist"):
        submission.validate_manifest(manifest)
