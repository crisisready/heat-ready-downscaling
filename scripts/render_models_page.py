"""
Regenerates docs/models.md from registry/*/*/manifest.yaml -- DERIVED, always
safe to overwrite, same convention as scripts/render_leaderboard.py for
docs/leaderboard.md. Never edit docs/models.md by hand; re-run this script
instead.

Closes the one open item in ROADMAP.md's Phase 1 acceptance criteria (private
crisisready/heat-risk-data-api repo): "the rendered models page shows real
evidence numbers matching the source reports." The registry itself
(heatready_downscaling.registry) was already real and CI-checked; nothing
rendered it publicly until this script.

Every number below is read straight from each manifest's own evidence block
-- never recomputed -- so a reader can compare this page against the
manifest (and, where `report` is set, against the cited report file) rather
than trusting a derived summary.

No AWS/network access -- reads registry/ already checked out in this repo.

Usage:
    python scripts/render_models_page.py --registry-dir registry --docs-dir docs
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heatready_downscaling import registry  # noqa: E402


def _fmt_evidence(evidence: dict) -> str:
    """One-line evidence summary: `metric value [ci_lo, ci_hi] (n=N)` --
    whatever subset of ci95/n_stations is actually present, since either can
    be null (registry.py's own schema: a null CI or station count is itself
    informative, not a gap to paper over)."""
    parts = [f"{evidence['metric']}={evidence['value']:.4g}"]
    ci = evidence.get("ci95")
    if ci is not None:
        parts.append(f"CI95=[{ci[0]:.4g}, {ci[1]:.4g}]")
    n_stations = evidence.get("n_stations")
    if n_stations is not None:
        parts.append(f"n_stations={n_stations}")
    n_clusters = evidence.get("n_clusters")
    if n_clusters is not None:
        parts.append(f"n_clusters={n_clusters}")
    return ", ".join(parts)


def _fmt_cell(claim: dict) -> str:
    band = claim.get("band") or "(none)"
    geography = claim.get("geography")
    cell = f"{claim['target']}/{claim['zone']}/{band}"
    if geography:
        cell += f" ({geography})"
    return cell


def render_markdown(entries: list[tuple[str, dict]]) -> str:
    """`entries` is (model_dir, manifest) pairs, as yielded by
    registry.iter_registry -- already validated, so this function does only
    formatting, no error handling."""
    lines = [
        "# Models",
        "",
        "*Derived from `registry/*/*/manifest.yaml` -- do not edit by hand, re-run "
        "`scripts/render_models_page.py`.*",
        "",
        "Every evidence number below is read directly from the cited manifest's own "
        "`claims[].evidence` block, never recomputed. See `registry/README.md` (if present) "
        "or `heatready_downscaling.registry`'s own docstring for what a registry entry is and "
        "is not (it does not execute anything and does not decide promotion).",
        "",
    ]

    if not entries:
        lines.append("*No models are registered yet.*")
        lines.append("")
        return "\n".join(lines)

    for model_dir, manifest in sorted(entries, key=lambda e: e[1]["model_id"]):
        manifest_path = os.path.join(model_dir, "manifest.yaml")
        method = manifest["method"]
        status = registry.current_status(manifest)

        lines.append(f"## `{manifest['model_id']}`")
        lines.append("")
        title = manifest.get("title")
        if title:
            lines.append(f"**{title}**")
            lines.append("")
        lines.append(f"- **Status**: `{status}`")
        lines.append(f"- **Method**: `{method['kind']}` -> `{method['compile_to']}`")
        authors = ", ".join(a["name"] for a in manifest["authors"])
        lines.append(f"- **Authors**: {authors}")
        lineage = manifest.get("lineage") or {}
        if lineage.get("derived_from"):
            lines.append(f"- **Derived from**: `{lineage['derived_from']}`")
        lines.append(f"- **Manifest**: `{manifest_path}`")
        lines.append("")

        lines.append("| Cell (target/zone/band) | Evidence |")
        lines.append("|---|---|")
        for claim in manifest["claims"]:
            lines.append(f"| {_fmt_cell(claim)} | {_fmt_evidence(claim['evidence'])} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-dir", default="registry")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    entries = list(registry.iter_registry(args.registry_dir))

    os.makedirs(args.docs_dir, exist_ok=True)
    with open(os.path.join(args.docs_dir, "models.md"), "w") as f:
        f.write(render_markdown(entries))


if __name__ == "__main__":
    main()
