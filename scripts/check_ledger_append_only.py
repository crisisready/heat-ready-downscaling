"""
Required CI check (plan section 7.3): a PR's diff to ledger/*.jsonl must
be ONLY appended lines. Compares each changed file's content at the PR's
base ref against its current (working-tree) content via
heatready_downscaling.ledger.check_append_only.

Usage (see .github/workflows/check-ledger-append.yml):
    python scripts/check_ledger_append_only.py --base-ref origin/main \\
        --paths ledger/submissions.jsonl ledger/cycles.jsonl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from heatready_downscaling import ledger


def git_show(ref: str, path: str) -> str:
    """Content of `path` at `ref`, or "" if the file didn't exist there
    yet -- a brand-new ledger/*.jsonl file being added for the first time
    is not an append-only violation, there's nothing to compare against."""
    result = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def check_paths(base_ref: str, paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        kind = os.path.splitext(os.path.basename(path))[0]  # "ledger/submissions.jsonl" -> "submissions"
        base_text = git_show(base_ref, path)
        with open(path) as f:
            head_text = f.read()
        violations.extend(f"{path}: {v}" for v in ledger.check_append_only(base_text, head_text, kind))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--paths", nargs="+", required=True, help="ledger/*.jsonl paths changed in this PR")
    args = parser.parse_args()

    violations = check_paths(args.base_ref, args.paths)
    if violations:
        print("Ledger append-only check FAILED:")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)
    print(f"Ledger append-only check passed for {len(args.paths)} file(s).")


if __name__ == "__main__":
    main()
