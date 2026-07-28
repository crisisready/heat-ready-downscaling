"""
Regenerates docs/leaderboard.{md,json} from ledger/credit.jsonl and
ledger/cycles.jsonl -- these two files are DERIVED, always safe to
overwrite (plan section 7.2 point 7: "docs/leaderboard.{md,json} are
derived and safe to overwrite"). Never edit them by hand; re-run this
script instead. No AWS/network access -- reads the ledger files already
checked out in this repo.

Usage:
    python scripts/render_leaderboard.py --ledger-dir ledger --docs-dir docs
"""

from __future__ import annotations

import argparse
import json
import os

from heatready_downscaling import ledger


def active_tenures(credit_lines: list[dict]) -> list[dict]:
    """The current holder of each (model_version, band_key, target, zone)
    cell -- the latest tenure_start not yet closed by a matching
    tenure_end. credit.jsonl is append-only and therefore already
    chronologically ordered, so a single forward pass tracking the
    currently-open tenure per cell is sufficient."""
    by_cell: dict[tuple, dict] = {}
    for line in credit_lines:
        cell = line["cell"]
        cell_key = (cell["model_version"], cell["band_key"], cell["target"], cell["zone"])
        if line["event"] == "tenure_start":
            by_cell[cell_key] = line
        elif line["event"] == "tenure_end":
            existing = by_cell.get(cell_key)
            # Only clear the cell if this end actually matches the tenure
            # currently tracked as open for it (same start_month) -- a
            # defensive check against a malformed/out-of-order ledger
            # rather than blindly deleting whatever's there.
            if existing is not None and existing.get("start_month") == line.get("start_month"):
                del by_cell[cell_key]
    return sorted(by_cell.values(), key=lambda l: (l["cell"]["band_key"], l["cell"]["target"], l["cell"]["zone"]))


def credit_counts(credit_lines: list[dict]) -> dict[str, int]:
    """author_github -> total tenure_starts ever recorded for them (an
    all-time "cells won at some point" tally, not just currently-held
    ones) -- the per-contributor leaderboard ranking metric."""
    counts: dict[str, int] = {}
    for line in credit_lines:
        if line["event"] == "tenure_start":
            counts[line["author_github"]] = counts.get(line["author_github"], 0) + 1
    return counts


def recent_cycle_activity(cycle_lines: list[dict], n_cycles: int = 3) -> list[dict]:
    """The most recent `n_cycles` distinct cycle months' worth of lines --
    a small "what happened lately" section, not the full history."""
    cycles = sorted({line["cycle"] for line in cycle_lines}, reverse=True)[:n_cycles]
    return [line for line in cycle_lines if line["cycle"] in cycles]


def render_json(active: list[dict], counts: dict[str, int], recent: list[dict]) -> dict:
    return {
        "active_tenures": active,
        "credit_counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
        "recent_cycle_activity": recent,
    }


def render_markdown(active: list[dict], counts: dict[str, int]) -> str:
    lines = ["# Leaderboard", "", "*Derived from `ledger/credit.jsonl` -- do not edit by hand, re-run `scripts/render_leaderboard.py`.*", ""]

    lines.append("## Contributors")
    lines.append("")
    if not counts:
        lines.append("*No cells have been credited yet.*")
    else:
        lines.append("| Contributor | Cells won (all-time) |")
        lines.append("|---|---|")
        for author, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"| `{author}` | {count} |")
    lines.append("")

    lines.append("## Currently-credited cells")
    lines.append("")
    if not active:
        lines.append("*No cell currently has an active tenure.*")
    else:
        lines.append("| Model | Band | Target | Zone | Contributor | Since |")
        lines.append("|---|---|---|---|---|---|")
        for t in active:
            cell = t["cell"]
            lines.append(
                f"| `{cell['model_version']}` | `{cell['band_key']}` | {cell['target']} | {cell['zone']} | "
                f"`{t['author_github']}` | {t['start_month']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger-dir", default="ledger")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    with open(os.path.join(args.ledger_dir, "credit.jsonl")) as f:
        credit_lines = ledger.parse_jsonl(f.read())
    with open(os.path.join(args.ledger_dir, "cycles.jsonl")) as f:
        cycle_lines = ledger.parse_jsonl(f.read())

    active = active_tenures(credit_lines)
    counts = credit_counts(credit_lines)
    recent = recent_cycle_activity(cycle_lines)

    os.makedirs(args.docs_dir, exist_ok=True)
    with open(os.path.join(args.docs_dir, "leaderboard.json"), "w") as f:
        json.dump(render_json(active, counts, recent), f, indent=2)
    with open(os.path.join(args.docs_dir, "leaderboard.md"), "w") as f:
        f.write(render_markdown(active, counts))


if __name__ == "__main__":
    main()
