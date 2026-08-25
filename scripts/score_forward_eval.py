"""
The monthly official scorer (plan section 7.2). Scores every active
candidate against station-days that did NOT exist in any snapshot at that
candidate's submission time -- the only uncheatable holdout, since GHCN-Daily
is free and anyone can verify the eval set. A candidate that wins 2
consecutive monthly cycles for a cell is promoted: `tenure_start` in
ledger/credit.jsonl, closing out any prior holder's tenure first.

Runs entirely against the PUBLIC snapshot + ledger -- no AWS credentials,
same posture as run_submission.py (see that script's own module docstring).
This is a deliberate simplification of the original plan, which specced
AWS Batch compute assuming live model inference would be needed:
contract.FrozenPredictionAdapter means scoring never needs the private
model artifact, so there is no compute this job needs that a GitHub
Actions runner can't provide. Triggered by .github/workflows/
score-forward-eval.yml on a monthly cron (5th of the month, 06:00 UTC,
matching the plan's own `cron(0 6 5 * ? *)`).

"New data since submission" is derived from REAL snapshot manifests (which
months a band partition actually covers), not a fixed month-offset formula
-- comparing the current snapshot's covered months against the
candidate's own originally-claimed snapshot version's covered months for
that band, and scoring only the months absent from the latter. This is
more honest than assuming a fixed cadence the code has no way to verify,
and naturally degrades to "no new data yet" (skip the cell this cycle)
rather than accidentally scoring against data the candidate could have
already seen.

Real, deliberate v1 simplification: score.score_band's fold_salt is the
band's CURRENT snapshot_version (the one containing the new eval data),
not the candidate's original submission-time version -- matching
run_submission.py's own reproduction, and consistent with score_band's own
docstring (fold_salt should reflect the snapshot actually being scored
against).

Usage (see .github/workflows/score-forward-eval.yml):
    python scripts/score_forward_eval.py --cycle 2026-10 \\
        --current-snapshot-version v2026.10 --ledger-dir ledger \\
        --submissions-root submissions
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_submission as rs

from heatready_downscaling import contract, ledger, score, snapshot, submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_forward_eval")


def _find_submission_manifest(submissions_root: str, submission_id: str) -> str | None:
    """submissions/** is permanent -- every merged submission's own
    manifest.yaml survives forever, which is where the full targets/zones
    claim detail lives. ledger/submissions.jsonl's own line is
    deliberately flattened to one (model_version, band_key) pair per the
    plan's own schema, so this is the only place to recover which
    (target, zone) cells a submission actually claimed."""
    year_month, seq = submission.parse_submission_id(submission_id)
    # Directory convention is {NNN}-{github}-{slug}/ (CONTRIBUTING.md) --
    # the sequence number is a zero-padded PREFIX, not a substring
    # anywhere in the name (a real GitHub username can itself contain
    # digits that would otherwise false-match a naive "contains this
    # number" glob).
    matches = glob.glob(os.path.join(submissions_root, year_month, f"{seq:03d}-*", "manifest.yaml"))
    return matches[0] if matches else None


def load_active_candidates(ledger_dir: str, submissions_root: str = "submissions") -> dict[tuple, dict]:
    """{(model_version, band_key, target, zone): candidate_info} -- the
    latest reproduced=true submission covering each cell. Later
    submissions (by ts) supersede earlier ones for the same cell, matching
    "a new claim replaces an old one until it's scored," not any kind of
    permanent lock-in.

    proposed_correction_entry (2026-08-25, Rung B -- Codex adversarial
    review finding on the original design: this monthly official cycle is
    what actually decides promotion, and had NO path to the contributor's
    declared value at all before this fix): a Rung B submission's own
    manifest.yaml carries method.candidate ({target: {zone: {...}}}), the
    same shape score.score_band's proposed_correction takes per target --
    read back out here (this is the ONLY place the full manifest, as
    opposed to the flattened submissions.jsonl ledger line, is available)
    so every monthly re-score uses the SAME fixed, contributor-declared
    value every cycle, never re-fit from that cycle's own data. None for
    a Rung A candidate (no candidate block at all) -- score_cell then
    scores the mechanically-derived correction exactly as it always has."""
    with open(os.path.join(ledger_dir, "submissions.jsonl")) as f:
        submission_lines = ledger.parse_jsonl(f.read())

    active: dict[tuple, dict] = {}
    for line in sorted(submission_lines, key=lambda l: l["ts"]):
        if not line.get("reproduced"):
            continue
        manifest_path = _find_submission_manifest(submissions_root, line["submission_id"])
        if manifest_path is None:
            logger.warning("submission %s has no manifest.yaml under %s -- skipping", line["submission_id"], submissions_root)
            continue
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        # Code-review finding, PR #24: this reads a manifest straight off
        # disk with no re-validation. A merged submission predates this
        # schema, or was hand-edited post-merge, could otherwise crash
        # score_band (a candidate zone entry with neither/both shapes) or
        # this function itself (a non-dict zone value) -- which would abort
        # the ENTIRE monthly cycle for every other active cell, not just
        # this one. Skip (log + continue), matching the existing "no
        # manifest.yaml found" convention just above, never let one bad
        # manifest take down the whole cron run.
        try:
            submission.validate_manifest(manifest)
        except Exception:
            logger.warning(
                "submission %s's manifest.yaml at %s failed validate_manifest -- skipping this "
                "candidate entirely rather than risk scoring a malformed candidate",
                line["submission_id"], manifest_path, exc_info=True,
            )
            continue
        claim = manifest["claims"][0]
        manifest_candidate = manifest.get("method", {}).get("candidate") or {}
        for target in claim["targets"]:
            for zone in claim["zones"]:
                cell_key = (claim["model_version"], claim["band_key"], target, zone)
                active[cell_key] = {
                    "submission_id": line["submission_id"], "author_github": line["author_github"],
                    "snapshot_version": line["snapshot_version"],
                    "proposed_correction_entry": manifest_candidate.get(target, {}).get(zone),
                }
    return active


def band_months(snapshot_dir: str, band_key: str) -> set[str]:
    paths = glob.glob(os.path.join(snapshot_dir, "paired", f"band={band_key}", "month=*", "part-0.parquet"))
    return {os.path.basename(os.path.dirname(p)).split("=", 1)[1] for p in paths}


def new_months_since_submission(current_snapshot_dir: str, submission_snapshot_dir: str, band_key: str) -> list[str]:
    """Months the current snapshot covers for `band_key` that the
    candidate's OWN submission-time snapshot did not -- the actual
    "didn't exist yet when they submitted" data. Empty (not an error) when
    the candidate's snapshot version already covers everything current
    does -- there's simply nothing new to score this cycle yet."""
    current = band_months(current_snapshot_dir, band_key)
    at_submission = band_months(submission_snapshot_dir, band_key)
    return sorted(current - at_submission)


def score_cell(
    current_snapshot_dir: str, submission_snapshot_dir: str, model_version: str, band_key: str,
    target: str, zone: str, current_snapshot_version: str,
    proposed_correction_entry: dict | None = None,
) -> dict | None:
    """Scores ONE cell against its new-data months. Returns None (skip,
    not a loss) if there's no new data yet for this band since the
    candidate's own submission.

    proposed_correction_entry (2026-08-25, Rung B): this cell's own
    {"bias_correction_c": float} or {"scale": float, "offset": float}
    entry from the winning submission's manifest, from
    load_active_candidates -- forwarded to score.score_band so a Rung B
    candidate's win/loss status is decided by ITS OWN declared value,
    never a freshly-derived one. None (every Rung A cell, unchanged) means
    score exactly as this function always has."""
    new_months = new_months_since_submission(current_snapshot_dir, submission_snapshot_dir, band_key)
    if not new_months:
        return None

    rows = [
        r for r in snapshot.read_band_partitions(current_snapshot_dir, band_key, months=new_months)
        if r["climate_zone"] == zone
    ]
    if not rows:
        return None

    adapter = contract.FrozenPredictionAdapter.from_snapshot(current_snapshot_dir, model_version, band_key)
    proposed_correction = {zone: proposed_correction_entry} if proposed_correction_entry is not None else None
    by_zone = score.score_band(
        adapter, rows, target, fold_salt=current_snapshot_version, proposed_correction=proposed_correction,
    )
    metrics = by_zone.get(zone)
    if metrics is None:
        return None

    # Rung B: a candidate with a declared correction wins/loses on ITS
    # OWN out-of-sample performance (proposed_correction_beats_grid_with_
    # margin), never the mechanically-derived qrf_beats_grid_with_margin --
    # that field reflects a value this cell's OWN forward-eval rows would
    # produce if re-fit, which is a different question from "does the
    # contributor's specific number keep generalizing" (the whole point
    # of Rung B, and the gap Codex's adversarial review on the original
    # design caught: this cycle previously had no path to the declared
    # value at all). A Rung A cell (proposed_correction_entry is None)
    # falls through to the original qrf_beats_grid_with_margin check,
    # byte-for-byte unchanged.
    if metrics["gated_insufficient_n"]:
        status = "insufficient_n"
    elif proposed_correction_entry is not None:
        status = "win" if metrics["proposed_correction_beats_grid_with_margin"] else "loss"
    elif metrics["qrf_beats_grid_with_margin"]:
        status = "win"
    else:
        status = "loss"

    return {
        # The most RECENT new month is used as this cycle's eval_month --
        # a single representative calendar month (plan section 7.3's own
        # schema field), even though scoring itself pools ALL new-since-
        # submission months for statistical power, not just the latest one.
        "eval_month": new_months[-1],
        "n_forward": metrics["n_grid"], "n_stations": len({r["station_id"] for r in rows}),
        "rmse_grid_c": metrics["rmse_grid_c"], "rmse_qrf_c": metrics["rmse_qrf_c"],
        "rmse_debiased_cv_c": metrics["rmse_debiased_cv_c"],
        "rmse_improvement_pct_debiased_cv": metrics["rmse_improvement_pct_debiased_cv"],
        "bias_correction_c": metrics["bias_correction_c"],
        "spatial_skill": metrics.get("qrf_beats_grid"), "gated_insufficient_n": metrics["gated_insufficient_n"],
        "status": status,
        # None for every Rung A cell (proposed_correction_entry is None,
        # score_band never computed these) -- see this function's own
        # docstring for why status is decided from these, not
        # qrf_beats_grid_with_margin, whenever a candidate declared a value.
        "proposed_correction_rmse_c": metrics.get("proposed_correction_rmse_c"),
        "proposed_correction_beats_grid_with_margin": metrics.get("proposed_correction_beats_grid_with_margin"),
    }


def consecutive_wins(cycle_lines: list[dict], submission_id: str, cell_key: tuple) -> int:
    """How many of the MOST RECENT consecutive cycles this exact
    submission_id has won for this cell -- resets to 0 the moment a
    non-win (loss or insufficient_n) appears, or a different submission_id
    won a more recent cycle for the same cell (a challenger reset the
    streak, even one that hasn't itself won twice yet)."""
    model_version, band_key, target, zone = cell_key
    cell_lines = sorted(
        (
            line for line in cycle_lines
            if line["cell"]["model_version"] == model_version and line["cell"]["band_key"] == band_key
            and line["cell"]["target"] == target and line["cell"]["zone"] == zone
        ),
        key=lambda l: l["cycle"], reverse=True,
    )
    streak = 0
    for line in cell_lines:
        if line["submission_id"] != submission_id or line["status"] != "win":
            break
        streak += 1
    return streak


def build_cycle_line(cycle: str, snapshot_version: str, candidate: dict, cell_key: tuple, metrics: dict, package_version: str, runner_commit: str | None) -> dict:
    """`metrics` is score_cell's own return value -- already carries
    `eval_month` (the calendar month scored, e.g. "2026-08") distinct from
    `snapshot_version` (which snapshot release the data came from, e.g.
    "v2026.10") -- these are two genuinely different fields (plan section
    7.3's own schema), not interchangeable."""
    model_version, band_key, target, zone = cell_key
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cycle": cycle,
        "submission_id": candidate["submission_id"], "author_github": candidate["author_github"],
        "cell": {"model_version": model_version, "band_key": band_key, "target": target, "zone": zone},
        **metrics, "incumbent_submission_id": None,
        "snapshot_version": snapshot_version, "runner_commit": runner_commit, "package_version": package_version,
    }


def build_tenure_start(cycle: str, candidate: dict, cell_key: tuple, metrics: dict, cycles_won: list[str]) -> dict:
    model_version, band_key, target, zone = cell_key
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": "tenure_start",
        "cell": {"model_version": model_version, "band_key": band_key, "target": target, "zone": zone},
        "author_github": candidate["author_github"], "author_name": None, "orcid": None,
        "submission_id": candidate["submission_id"], "start_month": cycle, "end_month": None,
        "score_at_start": {"rmse_improvement_pct_debiased_cv": metrics["rmse_improvement_pct_debiased_cv"]},
        "cycles_won": cycles_won,
    }


def build_tenure_end(cell_key: tuple, prior_holder: dict, end_month: str, superseded_by: str) -> dict:
    model_version, band_key, target, zone = cell_key
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": "tenure_end",
        "cell": {"model_version": model_version, "band_key": band_key, "target": target, "zone": zone},
        "author_github": prior_holder["author_github"], "start_month": prior_holder["start_month"],
        "end_month": end_month, "superseded_by": superseded_by,
    }


def current_tenure_holder(credit_lines: list[dict], cell_key: tuple) -> dict | None:
    model_version, band_key, target, zone = cell_key
    holder = None
    for line in credit_lines:
        cell = line["cell"]
        if (cell["model_version"], cell["band_key"], cell["target"], cell["zone"]) != cell_key:
            continue
        holder = line if line["event"] == "tenure_start" else None
    return holder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycle", required=True, help="this cycle's month, e.g. 2026-10")
    parser.add_argument("--current-snapshot-version", required=True)
    parser.add_argument("--ledger-dir", default="ledger")
    parser.add_argument("--submissions-root", default="submissions")
    parser.add_argument("--cache-root", default=".cache/snapshots")
    args = parser.parse_args()

    active_candidates = load_active_candidates(args.ledger_dir, args.submissions_root)
    logger.info("Loaded %d active candidate cell(s)", len(active_candidates))

    with open(os.path.join(args.ledger_dir, "cycles.jsonl")) as f:
        cycle_lines = ledger.parse_jsonl(f.read())
    with open(os.path.join(args.ledger_dir, "credit.jsonl")) as f:
        credit_lines = ledger.parse_jsonl(f.read())

    current_snapshot_dir = rs.download_snapshot(args.current_snapshot_version, args.cache_root)
    package_version = rs._package_version()
    runner_commit = os.environ.get("GITHUB_SHA")

    new_cycle_lines: list[dict] = []
    new_credit_lines: list[dict] = []

    for cell_key, candidate in active_candidates.items():
        model_version, band_key, target, zone = cell_key
        if candidate["snapshot_version"] == args.current_snapshot_version:
            continue  # candidate submitted against the CURRENT snapshot -- nothing new to score yet

        # Code-review finding, PR #24: one cell's failure (a transient
        # download error, an unexpected data shape) must never abort the
        # WHOLE monthly cycle for every other active candidate -- isolate
        # per cell, log, and move on. load_active_candidates already
        # validates each manifest before it becomes a `candidate` here, so
        # this is defense in depth against everything ELSE that can go
        # wrong scoring one cell, not a substitute for that check.
        try:
            submission_snapshot_dir = rs.download_snapshot(candidate["snapshot_version"], args.cache_root)
            metrics = score_cell(
                current_snapshot_dir, submission_snapshot_dir, model_version, band_key, target, zone,
                args.current_snapshot_version,
                proposed_correction_entry=candidate.get("proposed_correction_entry"),
            )
        except Exception:
            logger.error(
                "Failed to score cell %s (submission %s) -- skipping this cell for cycle %s, "
                "continuing with the rest", cell_key, candidate["submission_id"], args.cycle, exc_info=True,
            )
            continue
        if metrics is None:
            continue

        line = build_cycle_line(args.cycle, args.current_snapshot_version, candidate, cell_key, metrics, package_version, runner_commit)
        new_cycle_lines.append(line)
        ledger.validate_ledger_line("cycles", line)

        all_cycle_lines = cycle_lines + new_cycle_lines
        streak = consecutive_wins(all_cycle_lines, candidate["submission_id"], cell_key)
        if metrics["status"] == "win" and streak >= 2:
            prior_holder = current_tenure_holder(credit_lines + new_credit_lines, cell_key)
            if prior_holder is not None and prior_holder["author_github"] != candidate["author_github"]:
                end_line = build_tenure_end(cell_key, prior_holder, args.cycle, candidate["submission_id"])
                new_credit_lines.append(end_line)
                ledger.validate_ledger_line("credit", end_line)
            if prior_holder is None or prior_holder["submission_id"] != candidate["submission_id"]:
                cycles_won = [c["cycle"] for c in all_cycle_lines if c["submission_id"] == candidate["submission_id"]
                              and c["cell"]["model_version"] == model_version and c["cell"]["band_key"] == band_key
                              and c["cell"]["target"] == target and c["cell"]["zone"] == zone and c["status"] == "win"]
                start_line = build_tenure_start(args.cycle, candidate, cell_key, metrics, cycles_won)
                new_credit_lines.append(start_line)
                ledger.validate_ledger_line("credit", start_line)

    for line in new_cycle_lines:
        with open(os.path.join(args.ledger_dir, "cycles.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
    for line in new_credit_lines:
        with open(os.path.join(args.ledger_dir, "credit.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")

    logger.info("Cycle %s: appended %d cycle line(s), %d credit line(s)", args.cycle, len(new_cycle_lines), len(new_credit_lines))


if __name__ == "__main__":
    main()
