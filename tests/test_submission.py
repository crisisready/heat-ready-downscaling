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


class TestParseSubmissionId:
    def test_splits_year_month_and_sequence(self):
        assert submission.parse_submission_id("2026-08-001") == ("2026-08", 1)

    def test_sequence_is_an_int_not_a_zero_padded_string(self):
        year_month, seq = submission.parse_submission_id("2026-08-042")
        assert seq == 42
        assert isinstance(seq, int)
