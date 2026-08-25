"""
Required CI check: every registry entry validates, and no entry's
status_history is ever rewritten.

Two separate jobs, and the second is the one that needs git.

VALIDATION is stateless -- registry.load_manifest already enforces the schema,
the path/id agreement, the licensing rules through #31's own gate, the
band/compile_to consistency, and every cited evidence file's sha256.

APPEND-ONLY is not. A status_history line records a transition that actually
happened, so editing or deleting one rewrites history rather than correcting
it; that is the same property ledger/*.jsonl has and the same shape of check
(see scripts/check_ledger_append_only.py). Without it the registry's audit
trail would only be as good as `git log`, which is precisely what having the
history inside the file is meant to avoid.

Usage (see .github/workflows/registry.yml):
    python scripts/check_registry.py --base-ref origin/main
    python scripts/check_registry.py            # validation only, no git needed
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heatready_downscaling import registry  # noqa: E402


def git_show(ref: str, path: str) -> str | None:
    """Content of `path` at `ref`, or None when it did not exist there -- a
    brand-new entry has no prior history to violate."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True,
    )
    return result.stdout if result.returncode == 0 else None


def status_history_violations(base_text: str, head_text: str, model_id: str) -> list[str]:
    """Every way a status_history can be rewritten rather than appended."""
    import yaml

    base = (yaml.safe_load(base_text) or {}).get("status_history") or []
    head = (yaml.safe_load(head_text) or {}).get("status_history") or []

    problems: list[str] = []
    if len(head) < len(base):
        problems.append(
            f"{model_id}: status_history lost {len(base) - len(head)} line(s) -- a status line "
            "records a transition that happened, so removing one rewrites history",
        )
        return problems
    for i, (was, now) in enumerate(zip(base, head)):
        if was != now:
            problems.append(
                f"{model_id}: status_history[{i}] was edited ({was} -> {now}). Append a new "
                "line instead; a correction to the record is itself part of the record",
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-ref", default=None,
        help="compare status_history against this ref; omit to validate only",
    )
    parser.add_argument("--registry-dir", default="registry")
    args = parser.parse_args()

    problems: list[str] = []
    entries = 0
    for path in sorted(glob.glob(os.path.join(args.registry_dir, "*", "*", "manifest.yaml"))):
        model_dir = os.path.dirname(path)
        entries += 1
        try:
            manifest = registry.load_manifest(model_dir, repo_root=".")
        except Exception as exc:  # noqa: BLE001 -- every failure is reported the same way
            problems.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        if args.base_ref:
            base_text = git_show(args.base_ref, path)
            if base_text is None:
                continue  # new entry
            with open(path) as fh:
                problems += status_history_violations(base_text, fh.read(), manifest["model_id"])

    print(f"checked {entries} registry entr{'ies' if entries != 1 else 'y'}")
    if problems:
        print("\nPROBLEMS:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("all clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
