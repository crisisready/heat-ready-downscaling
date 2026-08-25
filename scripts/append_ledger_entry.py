"""
Append one line to ledger/submissions.jsonl after a submission PR merges to
main -- run by .github/workflows/ledger-append.yml, NEVER by the
contributor's own PR. See run_submission.py's own module docstring and the
Phase 3 design consult (2026-07-28) for why: the ledger line's own "pr"
field can only be resolved AFTER merge (GitHub's commits/{sha}/pulls API
needs a real merge commit to look up), and re-running the referee here --
against the merge commit, with a writable-token workflow that never
touched PR-authored content directly -- is what keeps "never trust the
contributor's own numbers" true even for the ledger write itself, not just
the read-only provisional score.

Deliberately does NOT reuse whatever provisional.json referee.yml produced
at PR time -- that ran against a point-in-time snapshot cache and a
read-only token; this script re-derives status/max_abs_deviation fresh
against the ACTUAL merged commit, the same discipline promote_from_public.py
(private repo, Phase 4) will apply again before anything reaches production.

Rung B (2026-08-25, Codex adversarial review finding on PR #24): this is
the AUTHORITATIVE reproduction that decides "reproduced": true/false in
the ledger, which is what makes a submission an active candidate for
score_forward_eval.py's monthly cycles at all -- so reproduce() here MUST
be called with the manifest's own method.candidate, exactly like
run_submission.py's PR-time call. Without it, a Rung B submission's
reproduced_report would carry all-None proposed_correction_* fields
(score_band never scored the declared value at all), report.compare_reports
would silently SKIP comparing those against the contributor's claimed_report
(a metric missing from either side is skipped, not a violation -- see its
own docstring), and "reproduced": true could be written to the ledger
having never actually verified the one thing a Rung B submission exists to
prove: that its declared correction generalizes.

Usage (see .github/workflows/ledger-append.yml):
    python scripts/append_ledger_entry.py \\
        --submission-dir submissions/2026-08/001-nishkishore-lagfill-cfb \\
        --repo crisisready/heat-ready-downscaling --pr-number 12 \\
        --ledger-dir ledger
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_submission as rs

from heatready_downscaling import ledger, report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("append_ledger_entry")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_submission_line(
    manifest: dict, submission_dir: str, status: str, max_abs_deviation: dict | None,
    pr_number: int, repo: str, runner_commit: str | None,
) -> dict:
    claim = manifest["claims"][0]
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "submission_id": manifest["submission_id"],
        "author_github": manifest["author"]["github"],
        "track": manifest["track"], "rung": manifest["rung"],
        "model_version": claim["model_version"], "band_key": claim["band_key"],
        "snapshot_version": manifest["snapshot"]["version"],
        "manifest_sha256": manifest["snapshot"]["manifest_sha256"],
        "claimed_report_sha256": _sha256_file(os.path.join(submission_dir, manifest["claimed_report"])),
        "reproduced": status == "pass",
        "max_abs_deviation": max_abs_deviation or {},
        "pr": f"{repo}#{pr_number}",
        "runner_commit": runner_commit,
    }


def append_line(ledger_dir: str, kind: str, line: dict) -> None:
    """Schema-validates before writing -- a malformed line must never
    reach the file at all, since ledger/*.jsonl is meant to be append-only
    and trustworthy by construction, not something a later pass cleans up."""
    ledger.validate_ledger_line(kind, line)
    path = os.path.join(ledger_dir, f"{kind}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
    logger.info("Appended one line to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--repo", required=True, help="e.g. crisisready/heat-ready-downscaling")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--pr-author", required=True, help="the merged PR's actual GitHub author")
    parser.add_argument("--ledger-dir", default="ledger")
    parser.add_argument("--cache-root", default=".cache/snapshots")
    args = parser.parse_args()

    manifest, claimed_report_data, licensing_rejects = rs.load_submission(args.submission_dir)

    # Re-derive fresh, authoritative status -- never trust whatever
    # referee.yml's PR-time run produced (see this module's own docstring).
    # licensing_rejects is folded in the same way run_submission.main does: a
    # licence violation reaching merge time is a hard reject here too, and this
    # runs on the merge commit, so it is the last chance to catch one that
    # slipped past the PR check.
    hard_rejects = list(licensing_rejects)
    hard_rejects += rs.cross_check(manifest, claimed_report_data, args.submission_dir, args.pr_author)
    if hard_rejects:
        raise SystemExit(
            f"submission {manifest['submission_id']!r} failed cross_check at merge time "
            f"(should have been caught before merge): {hard_rejects}"
        )

    snapshot_dir = rs.download_snapshot(manifest["snapshot"]["version"], args.cache_root)
    verify_violations = rs.verify_snapshot(snapshot_dir, manifest["snapshot"]["manifest_sha256"])
    if verify_violations:
        raise SystemExit(f"snapshot verification failed at merge time: {verify_violations}")

    claim = manifest["claims"][0]
    reproduced_report = rs.reproduce(
        snapshot_dir, claim["model_version"], claim["band_key"], manifest["snapshot"]["version"],
        candidate=manifest.get("method", {}).get("candidate"),
    )
    tolerance_result = report.compare_reports(claimed_report_data, reproduced_report, manifest["tolerance"])
    coverage = rs.coverage_violations(manifest, reproduced_report)

    status = "pass" if (tolerance_result.passed and not coverage) else "fail"
    runner_commit = os.environ.get("GITHUB_SHA")

    line = build_submission_line(
        manifest, args.submission_dir, status, tolerance_result.max_abs_deviation,
        args.pr_number, args.repo, runner_commit,
    )
    os.makedirs(args.ledger_dir, exist_ok=True)
    append_line(args.ledger_dir, "submissions", line)


if __name__ == "__main__":
    main()
