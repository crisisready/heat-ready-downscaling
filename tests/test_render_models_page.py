"""Unit tests for scripts/render_models_page.py -- pure rendering logic over
already-validated (model_dir, manifest) pairs, plus an integration test over
this repo's own real registry/ (already CI-checked by registry.yml)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import render_models_page as rmp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _evidence(**over):
    base = {
        "metric": "rmse_c", "value": 1.0, "ci95": None, "n_stations": None, "n_clusters": None,
        "holdout_design": "station_salted_fold", "provenance": "publicly_reproducible",
    }
    base.update(over)
    return base


def _manifest(**over):
    base = {
        "registry_schema_version": 1,
        "model_id": "local/test-entry",
        "version": "1.0.0",
        "title": "Test entry",
        "authors": [{"name": "Alice", "github": "alice"}],
        "method": {"kind": "covariate-linear-local", "compile_to": "polygon_params"},
        "claims": [{
            "target": "tmin", "zone": "BSh", "band": "lag_fill", "geography": "Testville",
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


class TestFmtEvidence:
    def test_includes_metric_and_value(self):
        assert "rmse_reduction_pct=0.19" in rmp._fmt_evidence(_manifest()["claims"][0]["evidence"])

    def test_omits_null_ci95(self):
        assert "CI95" not in rmp._fmt_evidence(_evidence())

    def test_includes_ci95_when_present(self):
        evidence = _manifest()["claims"][0]["evidence"]
        assert "CI95=[0.1, 0.25]" in rmp._fmt_evidence(evidence)

    def test_pipe_in_metric_is_escaped(self):
        assert "a\\|b" in rmp._fmt_evidence(_evidence(metric="a|b"))

    def test_includes_provenance_and_holdout_design(self):
        """The whole point of these two fields (registry.py's
        EVIDENCE_PROVENANCE/HOLDOUT_DESIGNS comments) is letting a reader
        tell "anyone can check this" from "we checked this" without opening
        the manifest -- exactly what this public page is for."""
        evidence = _evidence(provenance="maintainer_attested", holdout_design="spatial_holdout")
        rendered = rmp._fmt_evidence(evidence)
        assert "provenance=maintainer_attested" in rendered
        assert "holdout=spatial_holdout" in rendered


class TestFmtCell:
    def test_includes_target_zone_band(self):
        assert rmp._fmt_cell(_manifest()["claims"][0]) == "tmin/BSh/lag_fill (Testville)"

    def test_null_band_renders_as_none_marker(self):
        claim = _manifest()["claims"][0]
        claim["band"] = None
        claim["geography"] = None
        assert rmp._fmt_cell(claim) == "tmin/BSh/(none)"

    def test_pipe_in_geography_is_escaped(self):
        """geography is a free-form string in the registry schema -- a
        literal `|` must not misalign the rendered Markdown table."""
        claim = _manifest()["claims"][0]
        claim["geography"] = "Testville | Somewhere"
        assert "\\|" in rmp._fmt_cell(claim)
        assert " | " not in rmp._fmt_cell(claim).replace("\\|", "")


class TestEscapeCell:
    def test_escapes_pipe(self):
        assert rmp._escape_cell("a|b") == "a\\|b"

    def test_leaves_plain_text_unchanged(self):
        assert rmp._escape_cell("Valencia city cluster") == "Valencia city cluster"

    def test_strips_embedded_newlines(self):
        """geography/metric have no character restriction in the registry
        schema (registry.py's _cell_schema has no pattern/length cap) -- an
        embedded newline would split a Markdown table row just like an
        unescaped `|` would."""
        assert "\n" not in rmp._escape_cell("Seoul,\nSouth Korea")
        assert "\r" not in rmp._escape_cell("a\r\nb")


class TestRenderMarkdown:
    def test_empty_registry_shows_placeholder(self):
        md = rmp.render_markdown([])
        assert "No models are registered yet" in md

    def test_populated_entry_shows_model_id_and_evidence(self):
        md = rmp.render_markdown([("registry/local/test-entry", _manifest())])
        assert "local/test-entry" in md
        assert "registered" in md
        assert "covariate-linear-local" in md
        assert "polygon_params" in md
        assert "Alice" in md
        assert "rmse_reduction_pct=0.19" in md

    def test_derived_from_shown_when_present(self):
        manifest = _manifest(lineage={"derived_from": "global/base-model", "supersedes": None, "note": None})
        md = rmp.render_markdown([("registry/local/test-entry", manifest)])
        assert "global/base-model" in md

    def test_manifest_path_is_normalized_regardless_of_dir_spelling(self):
        md = rmp.render_markdown([("./registry/local/test-entry", _manifest())])
        assert "registry/local/test-entry/manifest.yaml" in md
        assert "./registry" not in md

    def test_entries_sorted_by_model_id(self):
        manifests = [
            ("registry/local/zzz", _manifest(model_id="local/zzz")),
            ("registry/global/aaa", _manifest(model_id="global/aaa")),
        ]
        md = rmp.render_markdown(manifests)
        assert md.index("global/aaa") < md.index("local/zzz")

    def test_never_recomputes_evidence_value(self):
        """The value shown must be exactly what the manifest declares --
        this script formats, it never derives (see module docstring)."""
        manifest = _manifest()
        manifest["claims"][0]["evidence"]["value"] = 0.987654321
        md = rmp.render_markdown([("registry/local/test-entry", manifest)])
        assert "0.9877" in md  # _fmt_evidence's :.4g of the exact declared value


class TestMainIntegration:
    def test_generates_models_md_from_real_registry(self, tmp_path, monkeypatch):
        docs_dir = tmp_path / "docs"
        monkeypatch.setattr(sys, "argv", [
            "render_models_page.py",
            "--registry-dir", os.path.join(REPO_ROOT, "registry"),
            "--docs-dir", str(docs_dir),
        ])
        rmp.main()

        md = (docs_dir / "models.md").read_text()
        assert "global/ds-2026.07-rf5" in md
        assert "local/seoul-sdot-v1" in md
        assert "local/valencia-coast-v1" in md
