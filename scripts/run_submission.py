"""
The referee (Phase 3, plan section 9.1's step 2 -- never given a detailed
spec anywhere in the original plan; designed via a dedicated consult,
2026-07-28, see PROVENANCE.md). Reproduces a contributor's submission
claim independently, compares it against their `claimed_report.json`
within the manifest's own tolerance, and writes a provisional score --
this is the "provisional" half of scoring (CONTRIBUTING.md's own "for
ranking and feedback only, never a gate decision"). The monthly official
cycle (score_forward_eval.py, a separate script) is the only thing that
can promote a candidate.

**Runs entirely inside GitHub Actions, against ONLY the public snapshot
Release asset -- zero AWS credentials, ever.** This is a deliberate
security boundary (a PR-triggered job must never hold private-account
credentials, since PRs come from arbitrary external GitHub users) as much
as a practical one: QRFModelAdapter.load needs private S3 access no
external contributor or CI job has at all, which is exactly why
contract.FrozenPredictionAdapter exists -- see that class's own docstring.
The maintainer's own independent re-derivation against the PRIVATE
snapshot copy happens later, in a completely separate script
(promote_from_public.py, private repo, Phase 4, out of scope here) --
never trust the contributor's numbers there either, but that is a later,
private-infra step.

NOT RUNNABLE STANDALONE IN THIS REPO in the sense of "needs private
modules" -- it doesn't (only requests/pyyaml/jsonschema, already this
package's own dependencies). It IS runnable standalone, by design: this is
what makes the "no AWS credentials touch a contributor-triggered event"
security property hold at all.

Real, deliberate v1 simplification: exactly ONE `claims[]` entry per
submission is supported (a submission stakes one (model_version, band_key)
claim; scoring multiple bands/models in one submission is out of scope --
open multiple submissions instead). `claimed_report.json` is a SINGLE
report (heatready_downscaling.report.build_report's own envelope, one
model_version/band_key), which would be ambiguous against more than one
claim anyway -- CONTRIBUTING.md's own example only ever shows one.

Usage (see .github/workflows/referee.yml for how this is actually invoked):
    python scripts/run_submission.py --submission-root submissions \\
        --pr-author someone --out provisional.json --comment-out comment.md --ci
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys
import tarfile
import tempfile

import requests
import yaml

from heatready_downscaling import contract, ledger, report, score, snapshot, submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_submission")

_GITHUB_API = "https://api.github.com"
_SNAPSHOT_REPO = "crisisready/heat-ready-downscaling"
# Deliberately does NOT try to capture github/slug as separate groups --
# {NNN}-{github-username}-{slug} (CONTRIBUTING.md's own convention) is
# genuinely ambiguous to split blindly, since a real GitHub username CAN
# itself contain internal hyphens (e.g. "nish-kishore") -- a generic
# greedy-vs-slug regex has no way to know where the username ends and the
# slug begins. cross_check resolves this by checking whether `rest` STARTS
# WITH the manifest's own (already-trusted) author.github value instead of
# trying to parse it out blind.
_SUBMISSION_DIR_RE = re.compile(r"submissions/(?P<month>\d{4}-\d{2})/(?P<seq>\d{3})-(?P<rest>[\w-]+)/?$")


def _package_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("heatready-downscaling")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def find_submission_dir(root: str = "submissions") -> str:
    """Exactly one submission directory must exist under `root` -- a
    referee run scores ONE PR's ONE submission. More or fewer is a hard
    reject; the CI paths-guard (.github/workflows/referee.yml) is what
    keeps a PR restricted to touching a single submissions/**/ directory
    in the first place, this is the second, defense-in-depth check."""
    candidates = [
        os.path.dirname(p) for p in sorted(glob.glob(os.path.join(root, "*", "*", "manifest.yaml")))
    ]
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one submission directory under {root}/ with a manifest.yaml, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def load_submission(submission_dir: str) -> tuple[dict, dict]:
    """Load + schema-validate manifest.yaml and claimed_report.json.
    Raises (jsonschema.ValidationError / ValueError) on a structurally
    invalid submission -- these are hard rejects, not violations to report
    alongside a provisional score."""
    with open(os.path.join(submission_dir, "manifest.yaml")) as f:
        manifest = yaml.safe_load(f)
    submission.validate_manifest(manifest)

    with open(os.path.join(submission_dir, manifest["claimed_report"])) as f:
        claimed_report = json.load(f)
    report.validate_report(claimed_report)

    return manifest, claimed_report


def cross_check(manifest: dict, claimed_report: dict, submission_dir: str, pr_author: str | None) -> list[str]:
    """Cheap, offline identity/consistency checks -- done BEFORE
    downloading anything. Returns violation strings (empty = all passed);
    these are reported in the PR comment as hard rejects, distinct from a
    tolerance violation (which still gets a full provisional score)."""
    violations: list[str] = []

    m = _SUBMISSION_DIR_RE.search(submission_dir.replace(os.sep, "/"))
    if not m:
        violations.append(
            f"submission directory path {submission_dir!r} doesn't match "
            "submissions/{YYYY-MM}/{NNN}-{github}-{slug}/"
        )
    else:
        expected_id = f"{m['month']}-{m['seq']}"
        if manifest["submission_id"] != expected_id:
            violations.append(
                f"submission_id {manifest['submission_id']!r} doesn't match its own directory "
                f"path (expected {expected_id!r})"
            )
        github = manifest["author"]["github"]
        if not m["rest"].lower().startswith(f"{github.lower()}-"):
            violations.append(
                f"directory path segment {m['rest']!r} doesn't start with author.github "
                f"{github!r} followed by a hyphen -- directory naming must be "
                "{NNN}-{author.github}-{slug}"
            )
        if pr_author and github.lower() != pr_author.lower():
            violations.append(
                f"manifest author.github {github!r} doesn't match the PR's actual author {pr_author!r}"
            )

    # Rung C (new model code) stays closed by design -- GOVERNANCE.md's own
    # unresolved-security-question gate (how untrusted model code would
    # execute safely), not a scoring limitation this referee can lift on
    # its own. Rung B (published parameters) opened 2026-08-25:
    # score.score_band's proposed_correction extension can now score a
    # CONTRIBUTOR-declared bias_correction_c/scale+offset value -- see
    # docs/plan-2026-08-25-crowdsourced-model-improvement-p0.md.
    if manifest["rung"] == "C":
        violations.append(
            "rung 'C' (new model code) is not yet open -- see GOVERNANCE.md's unresolved "
            "security question about executing untrusted model code"
        )
    elif manifest["rung"] == "B" and manifest["method"]["kind"] != "parameters":
        violations.append(
            f"rung 'B' requires method.kind == 'parameters', got {manifest['method']['kind']!r}"
        )

    # 2026-08-25: submission.MANIFEST_SCHEMA now also enforces maxItems: 1
    # on claims (Codex adversarial review finding, PR #24 round 2) -- in
    # the real pipeline, load_submission's own validate_manifest call
    # already rejects a multi-claim manifest before cross_check ever runs,
    # making this check unreachable there. Kept here anyway: cross_check
    # is called directly (bypassing schema validation) by its own test
    # suite, and this is still the more specific, contributor-readable
    # message if that ever changes.
    if len(manifest["claims"]) != 1:
        violations.append(
            f"exactly one claims[] entry is supported in v1, found {len(manifest['claims'])} -- "
            "open a separate submission per (model_version, band_key) claim"
        )
    else:
        claim = manifest["claims"][0]
        if claimed_report.get("band_key") != claim["band_key"]:
            violations.append(
                f"claimed_report band_key {claimed_report.get('band_key')!r} != manifest claim {claim['band_key']!r}"
            )
        if claimed_report.get("model_version") != claim["model_version"]:
            violations.append(
                f"claimed_report model_version {claimed_report.get('model_version')!r} != "
                f"manifest claim {claim['model_version']!r}"
            )
        if claimed_report.get("snapshot_version") != manifest["snapshot"]["version"]:
            violations.append(
                f"claimed_report snapshot_version {claimed_report.get('snapshot_version')!r} != "
                f"manifest snapshot.version {manifest['snapshot']['version']!r}"
            )

    installed_version = _package_version()
    if manifest["method"]["package_version"] != installed_version:
        violations.append(
            f"method.package_version {manifest['method']['package_version']!r} != installed "
            f"heatready_downscaling version {installed_version!r} -- pin your manifest to the "
            "version this referee actually runs"
        )

    return violations


def check_submission_id_unique(submission_id: str, ledger_dir: str = "ledger") -> list[str]:
    """submission_id must not already exist in ledger/submissions.jsonl --
    catches a concurrent-PR ID collision at PR TIME (this referee's own
    checkout already has the ledger as it stood at the PR's base commit),
    not just at merge time via check-ledger-append.yml's own duplicate
    check on the eventual bot-authored ledger PR (design-consult finding,
    2026-07-28: submission_id allocation is racy between two concurrently
    open PRs). Whichever PR merges FIRST still wins the ID either way --
    this just gives the second contributor a clear, immediate error
    instead of a confusing failure days later when their own ledger-append
    PR collides."""
    path = os.path.join(ledger_dir, "submissions.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        existing_ids = {line["submission_id"] for line in ledger.parse_jsonl(f.read())}
    if submission_id in existing_ids:
        return [f"submission_id {submission_id!r} already exists in ledger/submissions.jsonl -- choose a new one"]
    return []


def download_snapshot(version: str, cache_root: str = ".cache/snapshots") -> str:
    """Download+extract the public snapshot Release asset
    (tag snapshot-{version}, asset snapshot-{version}.tar.gz) from this
    repo's own GitHub Releases -- cached under cache_root so repeat CI runs
    for the same snapshot version don't re-download. No auth needed (public
    repo, public release asset) -- this is exactly the point: a referee
    with zero credentials can still fetch everything it needs."""
    cache_dir = os.path.join(cache_root, version)
    if os.path.exists(os.path.join(cache_dir, "MANIFEST.json")):
        logger.info("Using cached snapshot at %s", cache_dir)
        return cache_dir

    tag = f"snapshot-{version}"
    asset_name = f"snapshot-{version}.tar.gz"
    resp = requests.get(f"{_GITHUB_API}/repos/{_SNAPSHOT_REPO}/releases/tags/{tag}", timeout=30)
    resp.raise_for_status()
    release = resp.json()
    asset = next((a for a in release["assets"] if a["name"] == asset_name), None)
    if asset is None:
        raise SystemExit(
            f"release {tag!r} has no asset named {asset_name!r} -- available: "
            f"{[a['name'] for a in release['assets']]}"
        )

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    try:
        with os.fdopen(fd, "wb") as tmp, requests.get(asset["browser_download_url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        # No extraction `filter` kwarg (PEP 706, Python 3.12+ only, not
        # backported to this repo's minimum 3.11) -- acceptable here
        # because this tarball's provenance is fully controlled by this
        # same repo's own maintainer scripts/GitHub Release, never
        # contributor- or otherwise third-party-supplied content.
        with tarfile.open(tmp_path) as tar:
            tar.extractall(cache_dir)
    finally:
        os.unlink(tmp_path)
    logger.info("Downloaded and extracted snapshot %s to %s", version, cache_dir)
    return cache_dir


def verify_snapshot(snapshot_dir: str, expected_manifest_sha256: str) -> list[str]:
    """manifest_sha256 is defined as sha256 of the literal MANIFEST.json
    file bytes (not defined anywhere in the original plan -- pinned here,
    documented, and printed by build_band_paired_snapshot.py --phase pack
    so a submission's manifest.yaml can actually be built against a real
    value). Also re-verifies every partition's own sha256 via
    snapshot.verify_manifest -- a tampered or corrupted download must never
    silently score."""
    violations: list[str] = []
    manifest_path = os.path.join(snapshot_dir, "MANIFEST.json")
    with open(manifest_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    if actual_sha256 != expected_manifest_sha256:
        violations.append(
            f"snapshot MANIFEST.json sha256 mismatch: expected {expected_manifest_sha256!r}, "
            f"got {actual_sha256!r} -- refusing to score against data that doesn't match the pin"
        )
        return violations  # a manifest mismatch makes verify_manifest's own result meaningless

    try:
        snapshot.verify_manifest(snapshot_dir)
    except ValueError as exc:
        violations.append(str(exc))
    return violations


def fidelity_rows_for_band(band_rows: list[dict], era5_rows: list[dict]) -> list[dict]:
    """Join band_rows to the era5 band's own rows on (station_id, date) --
    the shape score.fidelity_report expects. Only meaningful for a
    non-era5 band (era5 IS the reference, nothing to compare it against)."""
    era5_by_key = {(r["station_id"], r["date"]): r for r in era5_rows}
    out: list[dict] = []
    for r in band_rows:
        era5_row = era5_by_key.get((r["station_id"], r["date"]))
        if era5_row is None:
            continue
        vals = (era5_row["grid_tmax_c"], era5_row["grid_tmin_c"], r["grid_tmax_c"], r["grid_tmin_c"])
        if any(v is None for v in vals):
            continue
        d = r["date"]
        out.append({
            "station_id": r["station_id"], "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
            "era5_tmax": era5_row["grid_tmax_c"], "era5_tmin": era5_row["grid_tmin_c"],
            "nrt_tmax": r["grid_tmax_c"], "nrt_tmin": r["grid_tmin_c"],
        })
    return out


def reproduce(
    snapshot_dir: str, model_version: str, band_key: str, snapshot_version: str,
    candidate: dict | None = None,
) -> dict:
    """Independently reproduce a full report for (model_version, band_key)
    against the downloaded public snapshot. Scores BOTH targets across
    EVERY zone the band's data covers -- not just the manifest's claimed
    zones, since it costs nothing extra and gives the contributor free
    feedback on adjacent zones (design-consult recommendation).

    candidate (2026-08-25, Rung B): the manifest's own `method.candidate`
    block ({target: {zone: {...}}}), forwarded to score.score_band as
    `proposed_correction` per target -- None (the default, every Rung A
    call) means no candidate to score, exactly today's behavior unchanged."""
    band_rows = snapshot.read_band_partitions(snapshot_dir, band_key)
    if not band_rows:
        raise SystemExit(f"snapshot has no rows for band={band_key!r}")

    adapter = contract.FrozenPredictionAdapter.from_snapshot(snapshot_dir, model_version, band_key)

    fidelity_check = {"n": 0}
    if band_key != "era5":
        era5_rows = snapshot.read_band_partitions(snapshot_dir, "era5")
        fidelity_check = score.fidelity_report(fidelity_rows_for_band(band_rows, era5_rows))

    by_target = {
        target: score.score_band(
            adapter, band_rows, target, fold_salt=snapshot_version,
            proposed_correction=(candidate or {}).get(target),
        )
        for target in ("tmax", "tmin")
    }

    return report.build_report(
        model_version=model_version, band_key=band_key, snapshot_version=snapshot_version,
        sample_requested=0, rows_sampled=len(band_rows), rows_paired=len(band_rows),
        fidelity_check=fidelity_check, by_target=by_target,
        generated_by={"tool": "run_submission.py", "version": _package_version(), "git_commit": os.environ.get("GITHUB_SHA")},
    )


def coverage_violations(manifest: dict, reproduced_report: dict) -> list[str]:
    """A claimed zone that's absent from the reproduction, or that scored
    zero applied rows, is a violation in its own right -- distinct from a
    tolerance mismatch on a zone that DID reproduce."""
    violations: list[str] = []
    claim = manifest["claims"][0]
    for target in claim["targets"]:
        by_zone = reproduced_report["by_target"].get(target, {})
        for zone in claim["zones"]:
            metrics = by_zone.get(zone)
            if metrics is None:
                violations.append(f"claimed zone {zone!r} (target={target}) is not present in the reproduced report at all")
            elif metrics.get("n_qrf_applied", 0) == 0:
                violations.append(f"claimed zone {zone!r} (target={target}) has zero applied rows in the reproduction")
    return violations


def render_comment(
    manifest: dict, hard_rejects: list[str], tolerance_result: "report.ToleranceResult | None",
    reproduced_report: dict | None, coverage: list[str],
) -> str:
    lines = [
        f"## Referee report -- submission `{manifest['submission_id']}`",
        "",
        f"Author: `{manifest['author']['github']}` · Track: `{manifest['track']}` · Rung: `{manifest['rung']}` · "
        f"Snapshot: `{manifest['snapshot']['version']}`",
        "",
    ]

    if hard_rejects:
        lines.append("### ❌ Rejected -- fix these before this can be scored")
        lines.extend(f"- {v}" for v in hard_rejects)
        lines.append("")
        lines.append(_DISCLAIMER)
        return "\n".join(lines)

    assert tolerance_result is not None and reproduced_report is not None
    status = "✅ Reproduced within tolerance" if tolerance_result.passed and not coverage else "⚠️ Did not fully reproduce"
    lines.append(f"### {status}")
    lines.append("")
    lines.append("Max absolute deviation per metric (claimed vs. independently reproduced):")
    for metric, dev in tolerance_result.max_abs_deviation.items():
        lines.append(f"- `{metric}`: {dev:.5f}")
    lines.append("")

    if tolerance_result.violations:
        lines.append("**Tolerance violations:**")
        for v in tolerance_result.violations:
            lines.append(
                f"- target=`{v['target']}` zone=`{v['zone']}` metric=`{v['metric']}`: "
                f"claimed={v['claimed']:.5f} reproduced={v['reproduced']:.5f} "
                f"diff={v['abs_diff']:.5f} (allowed {v['allowed']:.5f})"
            )
        lines.append("")

    if coverage:
        lines.append("**Coverage violations:**")
        lines.extend(f"- {v}" for v in coverage)
        lines.append("")

    lines.append("**Provisional per-zone results (this claim's band, all zones the snapshot covers):**")
    lines.append("")
    lines.append("| target | zone | n_qrf_applied | rmse_grid_c | rmse_qrf_c | rmse_improvement_pct_debiased_cv | qrf_beats_grid_with_margin |")
    lines.append("|---|---|---|---|---|---|---|")
    for target, by_zone in reproduced_report["by_target"].items():
        for zone, m in sorted(by_zone.items()):
            lines.append(
                f"| {target} | {zone} | {m['n_qrf_applied']} | "
                f"{_fmt(m['rmse_grid_c'])} | {_fmt(m['rmse_qrf_c'])} | "
                f"{_fmt(m['rmse_improvement_pct_debiased_cv'])} | {m['qrf_beats_grid_with_margin']} |"
            )
    lines.append("")
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


_DISCLAIMER = (
    "---\n"
    "*Provisional scores are for ranking and feedback only -- **never** a gate decision. "
    "Official promotion requires 2 consecutive winning monthly forward-eval cycles. "
    "Thin zones (below `MIN_ZONE_N`/`BIAS_CV_MIN_STATIONS`) can show a promising provisional "
    "number the official cycle will not confirm -- see `CONTRIBUTING.md`.*"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-root", default="submissions")
    parser.add_argument("--pr-author", default=None, help="the PR's actual GitHub author, for cross_check")
    parser.add_argument("--cache-root", default=".cache/snapshots")
    parser.add_argument("--out", default="provisional.json")
    parser.add_argument("--comment-out", default="comment.md")
    parser.add_argument("--ci", action="store_true", help="exit 0 even on a rejected/failed submission (referee-report.yml reads --out for the verdict)")
    parser.add_argument("--ledger-dir", default="ledger", help="for the submission_id uniqueness check")
    args = parser.parse_args()

    submission_dir = find_submission_dir(args.submission_root)
    logger.info("Scoring submission at %s", submission_dir)

    manifest, claimed_report_data = load_submission(submission_dir)
    hard_rejects = cross_check(manifest, claimed_report_data, submission_dir, args.pr_author)
    hard_rejects.extend(check_submission_id_unique(manifest["submission_id"], args.ledger_dir))

    tolerance_result = None
    reproduced_report = None
    coverage = []

    if not hard_rejects:
        snapshot_dir = download_snapshot(manifest["snapshot"]["version"], args.cache_root)
        hard_rejects.extend(verify_snapshot(snapshot_dir, manifest["snapshot"]["manifest_sha256"]))

    if not hard_rejects:
        claim = manifest["claims"][0]
        reproduced_report = reproduce(
            snapshot_dir, claim["model_version"], claim["band_key"], manifest["snapshot"]["version"],
            candidate=manifest.get("method", {}).get("candidate"),
        )
        tolerance_result = report.compare_reports(claimed_report_data, reproduced_report, manifest["tolerance"])
        coverage = coverage_violations(manifest, reproduced_report)

    status = "reject" if hard_rejects else ("pass" if (tolerance_result.passed and not coverage) else "fail")
    provisional = {
        "submission_id": manifest["submission_id"], "status": status,
        "hard_rejects": hard_rejects,
        "max_abs_deviation": tolerance_result.max_abs_deviation if tolerance_result else None,
        "tolerance_violations": tolerance_result.violations if tolerance_result else None,
        "coverage_violations": coverage,
        "reproduced_report": reproduced_report,
        "runner_commit": os.environ.get("GITHUB_SHA"), "package_version": _package_version(),
    }
    with open(args.out, "w") as f:
        json.dump(provisional, f, indent=2)

    comment = render_comment(manifest, hard_rejects, tolerance_result, reproduced_report, coverage)
    with open(args.comment_out, "w") as f:
        f.write(comment)

    logger.info("Wrote %s (status=%s) and %s", args.out, status, args.comment_out)

    if not args.ci and status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
