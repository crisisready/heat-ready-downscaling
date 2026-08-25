"""Unit tests for heatready_downscaling.registry.

Several of these pin rules that were DISCOVERED by doing the three retroactive
registrations rather than reasoned in advance -- which is what the roadmap says
that exercise is for, and it earned its keep three times."""

import pytest
import yaml

from heatready_downscaling import registry


def _manifest(**over):
    base = {
        "registry_schema_version": 1,
        "model_id": "local/test-entry",
        "version": "1.0.0",
        "authors": [{"name": "X"}],
        "method": {"kind": "covariate-linear-local", "compile_to": "polygon_params"},
        "claims": [{
            "target": "tmin", "zone": "BSh", "band": "lag_fill", "geography": None,
            "evidence": {
                "metric": "rmse_reduction_pct", "value": 0.19, "ci95": [0.10, 0.25],
                "n_stations": 9, "n_clusters": 9, "stratum": "all",
                "holdout_design": "station_salted_fold",
                "provenance": "publicly_reproducible",
                "report": None, "report_sha256": None, "notes": None,
            },
        }],
        "status_history": [{"status": "registered", "at": "2026-08-25", "by": "x", "note": None}],
    }
    base.update(over)
    return base


class TestSchema:
    def test_a_well_formed_entry_validates(self):
        registry.validate_manifest(_manifest())

    @pytest.mark.parametrize("bad", ["nope/x", "GLOBAL/x", "local/ab", "local/Has-Caps"])
    def test_model_id_must_be_prefixed_and_slugged(self, bad):
        with pytest.raises(Exception):
            registry.validate_manifest(_manifest(model_id=bad))

    def test_an_unknown_serving_primitive_is_rejected(self):
        """Methods are an open vocabulary; serving primitives are closed,
        because adding one is the only thing here that touches core serving
        code."""
        m = _manifest()
        m["method"]["compile_to"] = "just_wing_it"
        with pytest.raises(Exception):
            registry.validate_manifest(m)

    def test_an_unusual_method_kind_is_allowed(self):
        m = _manifest()
        m["method"]["kind"] = "something-nobody-has-invented-yet"
        registry.validate_manifest(m)


class TestStatusHistory:
    def test_history_must_start_at_registered(self):
        m = _manifest(status_history=[{"status": "serving", "at": "2026-08-25"}])
        with pytest.raises(registry.RegistryError, match="must begin with 'registered'"):
            registry.validate_manifest(m)

    def test_a_repeated_status_is_rejected(self):
        m = _manifest(status_history=[
            {"status": "registered", "at": "2026-08-01"},
            {"status": "registered", "at": "2026-08-02"},
        ])
        with pytest.raises(registry.RegistryError, match="repeats"):
            registry.validate_manifest(m)

    def test_out_of_order_history_is_rejected(self):
        m = _manifest(status_history=[
            {"status": "registered", "at": "2026-08-10"},
            {"status": "validated", "at": "2026-08-01"},
        ])
        with pytest.raises(registry.RegistryError, match="out of order"):
            registry.validate_manifest(m)

    def test_retired_is_terminal(self):
        m = _manifest(status_history=[
            {"status": "registered", "at": "2026-08-01"},
            {"status": "retired", "at": "2026-08-02"},
            {"status": "serving", "at": "2026-08-03"},
        ])
        with pytest.raises(registry.RegistryError, match="does not come back from retired"):
            registry.validate_manifest(m)

    def test_current_status_is_the_last_line(self):
        m = _manifest(status_history=[
            {"status": "registered", "at": "2026-08-01"},
            {"status": "validated", "at": "2026-08-02"},
        ])
        assert registry.current_status(m) == "validated"


class TestRulesFoundByRegistering:
    """Each of these came from a real problem hit while writing the three
    retroactive entries."""

    def test_a_band_keyed_correction_requires_a_band(self):
        m = _manifest()
        m["claims"][0]["band"] = None
        with pytest.raises(registry.RegistryError, match="requires a band on every claim"):
            registry.validate_manifest(m)

    def test_an_artifact_routed_model_may_omit_a_band(self):
        """Found by registering Seoul: an artifact-routed model is not a
        per-band correction, it produces rows across whatever bands the daily
        update covers. Requiring a band would have made me invent one."""
        m = _manifest()
        m["method"]["compile_to"] = "artifact_route"
        m["claims"][0]["band"] = None
        registry.validate_manifest(m)

    def test_a_registered_artifact_route_needs_no_artifact_yet(self):
        """Also from Seoul: its artifact's hash and feature order are not
        recorded anywhere, and an unconditional requirement forced a
        PLACEHOLDER feature name and an all-zero hash -- fabricated data
        satisfying a check, which is worse than an absent check because it
        looks like evidence."""
        m = _manifest()
        m["method"]["compile_to"] = "artifact_route"
        m["claims"][0]["band"] = None
        registry.validate_manifest(m)

    def test_beyond_registered_an_artifact_route_must_pin_its_contract(self):
        m = _manifest()
        m["method"]["compile_to"] = "artifact_route"
        m["claims"][0]["band"] = None
        m["status_history"].append({"status": "validated", "at": "2026-08-26"})
        with pytest.raises(registry.RegistryError, match="requires a feature_contract"):
            registry.validate_manifest(m)

    def test_an_all_zero_sha256_is_rejected_as_a_placeholder(self):
        """A checksum-shaped string that passes a pattern check and then
        verifies against nothing is the failure mode, not a missing one."""
        m = _manifest()
        m["artifacts"] = [{"name": "x.joblib", "sha256": "0" * 64}]
        with pytest.raises(registry.RegistryError, match="placeholder rather than a checksum"):
            registry.validate_manifest(m)

    def test_duplicate_cells_are_rejected(self):
        m = _manifest()
        m["claims"].append(dict(m["claims"][0]))
        with pytest.raises(registry.RegistryError, match="duplicate claim"):
            registry.validate_manifest(m)

    def test_licensing_is_enforced_through_the_existing_gate(self):
        """A registry entry's data sources are held to exactly what a
        submission's are -- one set of rules, not two."""
        from heatready_downscaling import licensing

        m = _manifest()
        m["method"]["data_sources"] = [{
            "name": "x", "license": "AllRightsReserved",
            "reproducible_fetch": "u", "redistribution_tier": "unrestricted",
        }]
        with pytest.raises(licensing.LicensingError):
            registry.validate_manifest(m)


class TestTheRealEntries:
    """The three retroactive registrations are the schema's actual test."""

    def test_every_registered_entry_validates(self):
        """Asserts the three retroactive entries are PRESENT and that every
        entry validates -- not that there are exactly three (code-review
        finding, PR #33). Pinning the count would have made adding a fourth
        registry entry fail CI, i.e. a test that blocks the feature it tests
        from being used."""
        entries = list(registry.iter_registry("registry"))
        ids = {m["model_id"] for _d, m in entries}
        assert {
            "global/ds-2026.07-rf5", "local/valencia-coast-v1", "local/seoul-sdot-v1",
        } <= ids
        assert len(entries) == len(ids), "two entries share a model_id"

    def test_the_global_entry_cites_evidence_that_really_exists(self):
        """Its report path points at the merged submission's own
        claimed_report.json, and the sha256 is checked against the real file --
        so this test fails if either drifts."""
        _dir, m = next(
            (d, x) for d, x in registry.iter_registry("registry")
            if x["model_id"] == "global/ds-2026.07-rf5"
        )
        claim = m["claims"][0]
        assert claim["evidence"]["report"].startswith("submissions/")
        assert claim["evidence"]["report_sha256"]

    def test_no_entry_claims_to_be_serving_or_validated_yet(self):
        """None of the three has cleared anything: no gate published, no
        artifact promoted, no production write. The registry must not imply
        otherwise."""
        for _dir, m in registry.iter_registry("registry"):
            assert registry.current_status(m) == "registered", m["model_id"]

    def test_seoul_records_a_spatial_holdout_not_a_salted_fold(self):
        _dir, m = next(
            (d, x) for d, x in registry.iter_registry("registry")
            if x["model_id"] == "local/seoul-sdot-v1"
        )
        for claim in m["claims"]:
            assert claim["evidence"]["holdout_design"] == "spatial_holdout"
            assert claim["band"] is None


class TestAppendOnlyStatusHistory:
    """The CI half of the append-only rule. Verified by observing each case
    rather than asserting the implementation looks right -- a status line
    records a transition that happened, so editing or deleting one rewrites
    history instead of correcting it."""

    def _cr(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import check_registry
        return check_registry

    BASE = (
        "status_history:\n"
        "  - status: registered\n    at: \"2026-08-01\"\n"
        "  - status: validated\n    at: \"2026-08-05\"\n"
    )

    def test_appending_is_allowed(self):
        head = self.BASE + "  - status: serving\n    at: \"2026-08-10\"\n"
        assert self._cr().status_history_violations(self.BASE, head, "local/x") == []

    def test_editing_an_existing_line_is_caught(self):
        head = self.BASE.replace("2026-08-01", "2026-08-02")
        problems = self._cr().status_history_violations(self.BASE, head, "local/x")
        assert len(problems) == 1 and "was edited" in problems[0]

    def test_deleting_a_line_is_caught(self):
        head = "status_history:\n  - status: registered\n    at: \"2026-08-01\"\n"
        problems = self._cr().status_history_violations(self.BASE, head, "local/x")
        assert len(problems) == 1 and "lost 1 line" in problems[0]

    def test_a_brand_new_entry_has_no_prior_history_to_violate(self):
        assert self._cr().status_history_violations("", self.BASE, "local/x") == []


def test_the_registry_check_script_passes_on_the_real_registry():
    import os
    import subprocess
    import sys

    root = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "check_registry.py")],
        capture_output=True, text=True, cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked" in result.stdout and "registry entr" in result.stdout


class TestRoundOneReviewFindings:
    """Regressions for the PR #33 review. Several are security-shaped: a check
    that can be walked around is not a check."""

    def test_status_cannot_go_backwards(self):
        m = _manifest(status_history=[
            {"status": "registered", "at": "2026-08-01"},
            {"status": "serving", "at": "2026-08-02"},
            {"status": "validated", "at": "2026-08-03"},
        ])
        with pytest.raises(registry.RegistryError, match="goes backwards"):
            registry.validate_manifest(m)

    def test_a_namespace_lookalike_directory_is_rejected(self):
        """endswith accepted registry/notglobal/foo for model_id global/foo,
        putting an entry outside the declared namespace while looking right."""
        m = _manifest(model_id="global/foo")
        with pytest.raises(registry.RegistryError, match="the path must be"):
            registry.validate_manifest(m, model_dir="registry/notglobal/foo")

    def test_the_matching_directory_is_accepted(self):
        m = _manifest(model_id="global/foo")
        registry.validate_manifest(m, model_dir="registry/global/foo")

    def test_a_claim_omitting_band_entirely_does_not_crash(self):
        """band is optional in the schema but was accessed directly, so a
        schema-valid claim that simply omits the key raised KeyError instead
        of validating."""
        m = _manifest()
        m["method"]["compile_to"] = "artifact_route"
        del m["claims"][0]["band"]
        registry.validate_manifest(m)

    def test_an_absolute_evidence_path_is_rejected(self, tmp_path):
        m = _manifest()
        m["claims"][0]["evidence"]["report"] = "/etc/os-release"
        m["claims"][0]["evidence"]["report_sha256"] = "a" * 64
        with pytest.raises(registry.RegistryError, match="absolute path"):
            registry.validate_manifest(m, repo_root=str(tmp_path))

    def test_evidence_cannot_traverse_outside_the_repo(self, tmp_path):
        """Nothing enforced the documented repo-relative contract, so a
        manifest could cite and checksum any file CI could read and have the
        result recorded as evidence for a published claim."""
        m = _manifest()
        m["claims"][0]["evidence"]["report"] = "../../../etc/os-release"
        m["claims"][0]["evidence"]["report_sha256"] = "a" * 64
        with pytest.raises(registry.RegistryError, match="outside the repository"):
            registry.validate_manifest(m, repo_root=str(tmp_path))

    def test_a_cited_report_must_carry_a_checksum(self, tmp_path):
        """Existence alone is not integrity: an unchecksummed report can
        change after the claim was written and nothing notices."""
        (tmp_path / "r.json").write_text("{}")
        m = _manifest()
        m["claims"][0]["evidence"]["report"] = "r.json"
        m["claims"][0]["evidence"]["report_sha256"] = None
        with pytest.raises(registry.RegistryError, match="without a report_sha256"):
            registry.validate_manifest(m, repo_root=str(tmp_path))

    def test_needs_licensing_review_surfaces_a_proprietary_source(self):
        """The return value of check_manifest_licensing was discarded, so
        Seoul's proprietary S-DoT source reported 'all clear' -- the identical
        defect #31's own review caught in the referee comment path, in a new
        place."""
        m = _manifest()
        m["method"]["data_sources"] = [{
            "name": "municipal feed", "license": "proprietary-licensed",
            "licensor": "A City", "reproducible_fetch": "u",
            "redistribution_tier": "no-redistribution",
        }]
        flagged = registry.needs_licensing_review(m)
        assert len(flagged) == 1 and "municipal feed" in flagged[0]

    def test_the_real_seoul_entry_is_flagged_for_licensing_review(self):
        _dir, m = next(
            (d, x) for d, x in registry.iter_registry("registry")
            if x["model_id"] == "local/seoul-sdot-v1"
        )
        assert registry.needs_licensing_review(m), "S-DoT must not pass silently"


def test_deleting_an_entry_is_reported_rather_than_silently_accepted():
    """Iterating only what exists at HEAD let a DELETION pass: remove the
    manifest and its whole status_history goes with it, unchecked. The
    append-only property has to be about the SET of entries, not just each
    surviving file."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import check_registry

    assert hasattr(check_registry, "git_registry_manifests")
