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


class TestRoundOneReviewFindings:
    """Regressions for the PR #31 review. Several of these are cases where a
    check reported the WRONG thing rather than nothing, which is the failure
    mode this whole gate exists to avoid."""

    def test_every_violation_is_reported_not_just_the_first(self):
        m = {"method": {"extra_covariates": [
            {"name": "a", "license": "WTFPL"},
            {"name": "b", "license": "AlsoBad"},
        ]}}
        violations, _ = licensing.audit_manifest_licensing(m)
        assert len(violations) == 2, "a contributor should learn about all of them in one run"

    def test_a_violation_does_not_discard_the_needs_review_list(self):
        """Raising on the first violation meant the entry that must 'always
        reach a human' was never printed."""
        m = {"method": {
            "extra_covariates": [{"name": "bad", "license": "WTFPL"}],
            "data_sources": [{
                "name": "municipal", "license": licensing.PROPRIETARY_LICENSE_ID,
                "licensor": "A City", "reproducible_fetch": "u",
                "redistribution_tier": "no-redistribution",
            }],
        }}
        violations, flagged = licensing.audit_manifest_licensing(m)
        assert len(violations) == 1
        assert len(flagged) == 1 and "municipal" in flagged[0]

    def test_check_manifest_licensing_reports_all_violations_in_its_message(self):
        m = {"method": {"extra_covariates": [
            {"name": "a", "license": "WTFPL"}, {"name": "b", "license": "AlsoBad"},
        ]}}
        with pytest.raises(licensing.LicensingError) as exc:
            licensing.check_manifest_licensing(m)
        assert "WTFPL" in str(exc.value) and "AlsoBad" in str(exc.value)

    @pytest.mark.parametrize("method", ["parameters", ["a"], 7])
    def test_a_non_object_method_raises_licensing_error_not_attribute_error(self, method):
        """The module promises to 'fail loudly enough that the referee can turn
        it into a readable rejection rather than a crash'. It is reachable from
        two paths that never ran jsonschema."""
        with pytest.raises(licensing.LicensingError, match="method must be an object"):
            licensing.audit_manifest_licensing({"method": method})

    def test_a_mapping_where_a_list_belongs_raises_rather_than_iterating_keys(self):
        m = {"method": {"data_sources": {"name": "x"}}}
        with pytest.raises(licensing.LicensingError, match="must be a list"):
            licensing.audit_manifest_licensing(m)

    def test_a_non_object_manifest_raises_licensing_error(self):
        with pytest.raises(licensing.LicensingError, match="manifest must be an object"):
            licensing.audit_manifest_licensing("not a manifest")

    def test_a_trailing_space_still_gets_the_case_hint(self):
        """A trailing space inside YAML quotes is an easy typo, and comparing
        the raw value meant it missed both the allowlist and the near-miss
        hint -- producing the full allowed-list dump for the likeliest
        mistake there is."""
        assert licensing.check_license_id("CC-BY-4.0 ", where="x") is False
        with pytest.raises(licensing.LicensingError, match="case-sensitive"):
            licensing.check_license_id(" cc-by-4.0 ", where="x")


class TestScriptExitCodes:
    """The exit contract is what CI acts on, so it is tested rather than
    assumed."""

    def _run(self, tmp_path, manifest_text):
        import subprocess
        import sys
        import os

        path = tmp_path / "manifest.yaml"
        path.write_text(manifest_text)
        root = os.path.join(os.path.dirname(__file__), "..")
        env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
        return subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "check_data_licensing.py"),
             "--no-network", str(path)],
            capture_output=True, text=True, env=env,
        )

    def test_clean_manifest_exits_zero(self, tmp_path):
        r = self._run(tmp_path, "method:\n  kind: parameters\n")
        assert r.returncode == 0, r.stdout

    def test_a_violation_exits_one(self, tmp_path):
        r = self._run(tmp_path, 'method:\n  extra_covariates:\n    - name: x\n      license: WTFPL\n')
        assert r.returncode == 1
        assert "LICENSING VIOLATIONS" in r.stdout

    def test_a_flagged_entry_exits_three_not_zero(self, tmp_path):
        """Flagged entries previously exited 0, so CI showed an ordinary green
        tick a maintainer would merge straight past -- while CONTRIBUTING.md
        promised such data never passes silently."""
        r = self._run(tmp_path, (
            "method:\n  data_sources:\n"
            "    - name: municipal\n      license: proprietary-licensed\n"
            "      licensor: A City\n      reproducible_fetch: https://example.invalid/x\n"
            "      redistribution_tier: no-redistribution\n"
        ))
        assert r.returncode == 3, r.stdout
        assert "NEEDS A MAINTAINER DECISION" in r.stdout

    def test_an_empty_manifest_exits_four_not_one(self, tmp_path):
        """An empty file previously raised AttributeError, exited 1, and the
        workflow reported a YAML typo as a LICENSING VIOLATION."""
        r = self._run(tmp_path, "")
        assert r.returncode == 4, r.stdout
        assert "UNREADABLE" in r.stdout
        assert "LICENSING VIOLATIONS" not in r.stdout

    def test_a_scalar_manifest_exits_four(self, tmp_path):
        r = self._run(tmp_path, "just a string\n")
        assert r.returncode == 4, r.stdout


def test_the_referee_comment_says_so_when_licensing_cannot_be_evaluated():
    """Failing open silently erased the only contributor-visible surface for
    the manual-review flag, which is the entire routes-to-a-human mechanism."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_submission as rs
    from heatready_downscaling import report as report_mod

    manifest = {
        "submission_id": "2026-08-005", "track": "research", "rung": "A",
        "author": {"github": "x"}, "snapshot": {"version": "v2026.07"},
        "claims": [{"model_version": "m", "band_key": "lag_fill",
                    "targets": ["tmax"], "zones": ["Cfb"]}],
        "method": "not-an-object",  # forces the licensing audit to raise
    }
    body = rs.render_comment(
        manifest, [], report_mod.ToleranceResult(True, {}, []),
        {"by_target": {}}, [],
    )
    assert "Licensing could not be evaluated" in body
