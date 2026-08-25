"""
check_data_licensing.py -- the network half of the licensing gate.

heatready_downscaling.licensing holds every rule that can be decided offline,
and validate_manifest enforces those, so a submission with a bad license is
rejected by the referee itself. This script adds the one check that cannot
live there: whether each declared `reproducible_fetch` URL actually resolves.
CONTRIBUTING.md has promised an "allowlist + HEAD check" since July; the
allowlist half is now real in the library, and this is the HEAD half.

Kept out of validate_manifest deliberately. A validator that reaches the
network fails for reasons that have nothing to do with the manifest -- a
flaky CDN, a runner without egress, a rate limit -- and a referee that
rejects a correct submission because a mirror blipped is worse than one that
checks reachability separately and says so.

Exit codes:
    0  everything clear
    1  a licensing RULE was violated (a real rejection)
    2  a rule passed but a URL was unreachable (advisory; the network is not
       the contributor's fault)
    3  every rule passed but an entry NEEDS A MAINTAINER DECISION
       (proprietary-licensed, or no-redistribution)
    4  a manifest could not be read or parsed at all -- NOT a licensing
       verdict, and deliberately distinct so a mis-indented YAML file is
       never reported as a licensing violation

Code 3 exists because of a code-review finding on PR #31: flagged entries
previously exited 0, so the check showed an ordinary green tick
indistinguishable from "all clear". CONTRIBUTING.md promises such data "never
passes silently", and a maintainer merging on a green check would have seen
nothing. The workflow turns 3 into a visible warning annotation.

Usage:
    python scripts/check_data_licensing.py [--no-network] [paths...]

With no paths, checks every submissions/**/manifest.yaml.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heatready_downscaling import licensing  # noqa: E402


def declared_fetch_urls(manifest: dict) -> list[tuple[str, str]]:
    """(where, url) for every reproducible_fetch in a manifest.

    Skips any entry that is not a mapping. audit_manifest_licensing already
    tolerates a non-dict entry -- it records a violation and continues -- and
    this function running afterwards on the SAME manifest used to crash with
    AttributeError instead (code-review finding, PR #31 round 2). The
    consequence was the exact mis-report exit code 4 was added to eliminate: a
    `data_sources` list of strings produced a traceback, exit 1, and a bare
    "LICENSING VIOLATION" with no message. Hardening one function and not its
    sibling on the same data is not hardening.

    Traversal routed through licensing._entries so the two stay in step rather
    than duplicating the (data_sources, extra_covariates) walk.
    """
    method = manifest.get("method")
    if not isinstance(method, dict):
        return []
    out: list[tuple[str, str]] = []
    for key in ("data_sources", "extra_covariates"):
        try:
            entries = licensing._entries(method, key)
        except licensing.LicensingError:
            continue
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("reproducible_fetch"):
                out.append((f"method.{key}[{i}]", entry["reproducible_fetch"]))
    return out


def check_reachable(url: str, timeout: float = 20.0) -> str | None:
    """None when the URL resolves, else a human-readable reason.

    HEAD first, then a Range-limited GET: plenty of data hosts answer HEAD
    with 403 or 405 while serving GET perfectly well, and failing a
    submission over that would be a false rejection.
    """
    import requests

    def _ranged_get():
        response = requests.get(
            url, timeout=timeout, allow_redirects=True, stream=True,
            headers={"Range": "bytes=0-0"},
        )
        response.close()
        return response.status_code

    head_status = None
    try:
        head_status = requests.head(url, timeout=timeout, allow_redirects=True).status_code
        if head_status < 400:
            return None
    except Exception as head_exc:  # noqa: BLE001
        # Fall through to the GET fallback rather than returning here
        # (code-review finding, PR #31): the fallback exists precisely because
        # some hosts and WAFs treat HEAD differently, and a TLS handshake or
        # connection reset on the HEAD verb specifically is exactly that case.
        # Returning on the exception skipped the fallback for the failure mode
        # it was written for.
        head_status = f"{type(head_exc).__name__}: {head_exc}"

    try:
        status = _ranged_get()
    except Exception as exc:  # noqa: BLE001 -- any transport failure is the same answer
        return f"{type(exc).__name__}: {exc} (HEAD: {head_status})"
    if status < 400:
        return None
    return f"HTTP {status} (HEAD: {head_status})"


def check_one(path: str, *, network: bool):
    """(violations, flagged_for_review, unreachable, unreadable) for one
    manifest.

    `unreadable` is its own category on purpose (code-review finding, PR #31):
    yaml.safe_load returns None for an empty file and a scalar or list for a
    mis-indented one, and the first version only caught LicensingError -- so an
    empty manifest.yaml raised AttributeError, exited 1, and the workflow
    reported it as a LICENSING VIOLATION. A contributor with a YAML typo does
    not have a licensing problem and should not be told they do.
    """
    import yaml

    violations: list[str] = []
    flagged: list[str] = []
    unreachable: list[str] = []
    unreadable: list[str] = []

    try:
        with open(path) as fh:
            manifest = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001
        return violations, flagged, unreachable, [f"{path}: could not parse YAML -- {exc}"]

    if not isinstance(manifest, dict):
        kind = "empty file" if manifest is None else f"a {type(manifest).__name__}, not a mapping"
        return violations, flagged, unreachable, [f"{path}: {kind}"]

    try:
        found_violations, found_flagged = licensing.audit_manifest_licensing(manifest)
    except licensing.LicensingError as exc:
        # Structural: method is not an object, or a list field is not a list.
        return violations, flagged, unreachable, [f"{path}: {exc}"]
    violations += [f"{path}: {v}" for v in found_violations]
    flagged += [f"{path}: {f}" for f in found_flagged]

    if network:
        for where, url in declared_fetch_urls(manifest):
            reason = check_reachable(url)
            if reason:
                unreachable.append(f"{path}: {where} -> {url} ({reason})")

    return violations, flagged, unreachable, unreadable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="manifest.yaml paths; default: every submission")
    parser.add_argument(
        "--no-network", action="store_true",
        help="skip the reachability check (offline rule checks only)",
    )
    args = parser.parse_args()

    # Default to every submission, but a CI caller should pass only the
    # manifests its PR actually touched: network-checking every historical
    # submission serially means one dead upstream in an old merged entry
    # produces exit-2 warnings on unrelated PRs forever (noted in the PR #31
    # review).
    paths = args.paths or sorted(glob.glob("submissions/*/*/manifest.yaml"))
    if not paths:
        print("no manifests found -- nothing to check")
        return 0

    violations: list[str] = []
    flagged: list[str] = []
    unreachable: list[str] = []
    unreadable: list[str] = []
    for path in paths:
        v, f, u, bad = check_one(path, network=not args.no_network)
        violations += v
        flagged += f
        unreachable += u
        unreadable += bad

    print(f"checked {len(paths)} manifest(s)")
    if unreadable:
        print("\nUNREADABLE manifest(s) -- a parsing/structure problem, NOT a licensing verdict:")
        for item in unreadable:
            print(f"  - {item}")
    if violations:
        print("\nLICENSING VIOLATIONS -- these reject the submission:")
        for item in violations:
            print(f"  - {item}")
    if flagged:
        print("\nNEEDS A MAINTAINER DECISION (admissible, but never automatic):")
        for item in flagged:
            print(f"  - {item}")
    if unreachable:
        print("\nUnreachable reproducible_fetch URL(s) -- advisory, not a rejection:")
        for item in unreachable:
            print(f"  - {item}")
    if not (violations or flagged or unreachable or unreadable):
        print("all clear")

    # Order matters: a real violation outranks everything, then an unreadable
    # manifest, then a decision a human owes, then mere reachability.
    if violations:
        return 1
    if unreadable:
        return 4
    if flagged:
        return 3
    if unreachable:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
