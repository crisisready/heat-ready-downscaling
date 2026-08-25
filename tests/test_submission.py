"""Unit tests for heatready_downscaling.submission's MANIFEST_SCHEMA +
validate_manifest -- the single source of truth the lint workflow and
run_submission.py both validate a contributor's manifest.yaml against."""

import pytest

from heatready_downscaling import submission


def _manifest(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "submission_id": "2026-08-001",
        "author": {"github": "nishkishore", "name": "Nishant Kishore", "orcid": None, "affiliation": None},
        "track": "serving-ready",
        "rung": "A",
        "snapshot": {"version": "v2026.07", "manifest_sha256": "a" * 64},
        "claims": [{"model_version": "ds-2026.07-rf5", "band_key": "lag_fill", "targets": ["tmax", "tmin"], "zones": ["Cfb"]}],
        "method": {
            "kind": "rerun-validator", "entrypoint": "scripts/run_submission.py",
            "args": [], "package_version": "0.1.0", "code_ref": None, "extra_covariates": [],
        },
        "claimed_report": "claimed_report.json",
        "tolerance": {"rmse_improvement_pct_debiased_cv": 0.002, "bias_correction_c": 0.01, "rmse_qrf_c": 0.005},
        "reproducibility": {"seed": 20260721, "runtime_notes": "..."},
    }
    base.update(overrides)
    return base


class TestValidateManifest:
    def test_well_formed_manifest_passes(self):
        submission.validate_manifest(_manifest())  # must not raise

    def test_missing_required_field_raises(self):
        m = _manifest()
        del m["snapshot"]
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_bad_submission_id_shape_raises(self):
        with pytest.raises(Exception):
            submission.validate_manifest(_manifest(submission_id="not-an-id"))

    def test_bad_snapshot_sha256_shape_raises(self):
        m = _manifest()
        m["snapshot"]["manifest_sha256"] = "not-a-sha"
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_unrecognized_band_key_raises(self):
        m = _manifest()
        m["claims"][0]["band_key"] = "forecast_lead8"
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_unrecognized_target_raises(self):
        m = _manifest()
        m["claims"][0]["targets"] = ["tmax", "not_a_target"]
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_empty_claims_raises(self):
        with pytest.raises(Exception):
            submission.validate_manifest(_manifest(claims=[]))

    def test_track_must_be_serving_ready_or_research(self):
        with pytest.raises(Exception):
            submission.validate_manifest(_manifest(track="not-a-track"))

    def test_rung_c_is_schema_valid_but_flagged_elsewhere(self):
        """Rung C is a valid SHAPE (forward-compatible) -- this schema
        doesn't reject it; scripts/run_submission.py is the thing that
        actually refuses it (GOVERNANCE.md: not yet open)."""
        submission.validate_manifest(_manifest(rung="C"))  # must not raise

    def test_zero_tolerance_rejected(self):
        """A tolerance of exactly 0 can never pass (any floating-point
        noise fails it) -- exclusiveMinimum catches this, distinct from
        the too-loose ceiling check below."""
        m = _manifest()
        m["tolerance"]["rmse_qrf_c"] = 0.0
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_negative_tolerance_rejected(self):
        m = _manifest()
        m["tolerance"]["rmse_qrf_c"] = -0.01
        with pytest.raises(Exception):
            submission.validate_manifest(m)


class TestToleranceCeilings:
    """A submission cannot set a tolerance so loose that any claim
    'reproduces' regardless of correctness -- design-consult finding,
    2026-07-27."""

    def test_tolerance_exceeding_ceiling_raises_value_error(self):
        m = _manifest()
        m["tolerance"]["rmse_qrf_c"] = 999.0
        with pytest.raises(ValueError, match="exceeds the maximum"):
            submission.validate_manifest(m)

    def test_tolerance_at_exactly_the_ceiling_passes(self):
        m = _manifest()
        m["tolerance"]["rmse_qrf_c"] = submission._TOLERANCE_MAXIMA["rmse_qrf_c"]
        submission.validate_manifest(m)  # must not raise

    def test_unlisted_tolerance_metric_has_no_ceiling_enforced(self):
        """A metric this table doesn't know about isn't silently rejected
        -- only the metrics with a documented ceiling are checked."""
        m = _manifest()
        m["tolerance"]["some_future_metric"] = 12345.0
        submission.validate_manifest(m)  # must not raise


class TestRungBCandidate:
    """method.candidate -- the actual declared Rung B value -- must be
    present iff rung=='B' (2026-08-25, Rung B scoring extension)."""

    def _rung_b_manifest(self, **method_overrides):
        # Single-target claim by default -- keeps most of these tests
        # focused on the candidate SHAPE, not coverage. See
        # TestRungBCandidateCoverage below for the multi-cell case.
        m = _manifest(rung="B")
        m["claims"][0]["targets"] = ["tmax"]
        m["method"] = {
            "kind": "parameters", "entrypoint": "scripts/run_submission.py",
            "args": [], "package_version": "0.1.0", "code_ref": None, "extra_covariates": [],
            "candidate": {"tmax": {"Cfb": {"bias_correction_c": 0.8}}},
        }
        m["method"].update(method_overrides)
        return m

    def test_well_formed_rung_b_candidate_passes(self):
        submission.validate_manifest(self._rung_b_manifest())  # must not raise

    def test_affine_shape_candidate_passes(self):
        m = self._rung_b_manifest(candidate={"tmin": {"Cfb": {"scale": 0.9, "offset": 0.2}}})
        m["claims"][0]["targets"] = ["tmin"]
        submission.validate_manifest(m)  # must not raise

    def test_rung_b_with_no_candidate_raises(self):
        m = self._rung_b_manifest(candidate=None)
        del m["method"]["candidate"]
        with pytest.raises(ValueError, match="requires a non-empty method.candidate"):
            submission.validate_manifest(m)

    def test_rung_b_with_empty_candidate_raises(self):
        m = self._rung_b_manifest(candidate={})
        with pytest.raises(ValueError, match="requires a non-empty method.candidate"):
            submission.validate_manifest(m)

    def test_rung_a_with_a_candidate_raises(self):
        """A Rung A submission (evaluation coverage only) proposes no
        correction of its own -- a candidate block there is a category
        error, not a harmless extra field."""
        m = self._rung_b_manifest()
        m["rung"] = "A"
        with pytest.raises(ValueError, match="only meaningful for rung 'B'"):
            submission.validate_manifest(m)

    def test_candidate_zone_entry_mixing_both_shapes_raises(self):
        m = self._rung_b_manifest(
            candidate={"tmax": {"Cfb": {"bias_correction_c": 0.8, "scale": 0.9, "offset": 0.2}}},
        )
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_candidate_zone_entry_with_neither_shape_raises(self):
        m = self._rung_b_manifest(candidate={"tmax": {"Cfb": {"bogus": 1.0}}})
        with pytest.raises(Exception):
            submission.validate_manifest(m)

    def test_mistyped_target_key_raises(self):
        """Codex adversarial review finding, PR #24: additionalProperties
        was missing at the candidate level, so a mistyped target key like
        'Tmax' passed schema validation silently and was simply never read
        downstream -- must now be a hard reject, not a silent no-op."""
        m = self._rung_b_manifest(candidate={"Tmax": {"Cfb": {"bias_correction_c": 0.8}}})
        with pytest.raises(Exception):
            submission.validate_manifest(m)


class TestRungBCandidateCoverage:
    """Codex adversarial review finding, PR #24: method.candidate being
    non-empty is not enough -- it must cover EVERY (target, zone) the
    manifest actually claims, or an uncovered cell silently falls back to
    a Rung-A-style mechanical fit at scoring time (score_forward_eval.py's
    own docstring)."""

    def _manifest_multi_cell(self, candidate):
        m = _manifest(rung="B")
        m["claims"][0]["targets"] = ["tmax", "tmin"]
        m["claims"][0]["zones"] = ["Cfb", "BWh"]
        m["method"] = {
            "kind": "parameters", "entrypoint": "scripts/run_submission.py",
            "args": [], "package_version": "0.1.0", "code_ref": None, "extra_covariates": [],
            "candidate": candidate,
        }
        return m

    def test_full_coverage_passes(self):
        m = self._manifest_multi_cell({
            "tmax": {"Cfb": {"bias_correction_c": 0.8}, "BWh": {"bias_correction_c": 0.3}},
            "tmin": {"Cfb": {"bias_correction_c": 0.5}, "BWh": {"bias_correction_c": 0.1}},
        })
        submission.validate_manifest(m)  # must not raise

    def test_missing_one_zone_raises(self):
        m = self._manifest_multi_cell({
            "tmax": {"Cfb": {"bias_correction_c": 0.8}, "BWh": {"bias_correction_c": 0.3}},
            "tmin": {"Cfb": {"bias_correction_c": 0.5}},  # BWh missing for tmin
        })
        with pytest.raises(ValueError, match=r"missing an entry for claimed cell"):
            submission.validate_manifest(m)

    def test_missing_an_entire_target_raises(self):
        m = self._manifest_multi_cell({
            "tmax": {"Cfb": {"bias_correction_c": 0.8}, "BWh": {"bias_correction_c": 0.3}},
            # tmin entirely absent
        })
        with pytest.raises(ValueError, match=r"missing an entry for claimed cell"):
            submission.validate_manifest(m)

    def test_extra_uncovered_zone_beyond_the_claim_is_harmless(self):
        """Extra candidate entries beyond the claim are fine -- same "free
        extra feedback" spirit Rung A's own out-of-claim-zone scoring
        already has. Only MISSING coverage for a CLAIMED cell is an error."""
        m = self._manifest_multi_cell({
            "tmax": {"Cfb": {"bias_correction_c": 0.8}, "BWh": {"bias_correction_c": 0.3}, "Csa": {"bias_correction_c": 0.1}},
            "tmin": {"Cfb": {"bias_correction_c": 0.5}, "BWh": {"bias_correction_c": 0.1}},
        })
        submission.validate_manifest(m)  # must not raise


class TestParseSubmissionId:
    def test_splits_year_month_and_sequence(self):
        assert submission.parse_submission_id("2026-08-001") == ("2026-08", 1)

    def test_sequence_is_an_int_not_a_zero_padded_string(self):
        year_month, seq = submission.parse_submission_id("2026-08-042")
        assert seq == 42
        assert isinstance(seq, int)
