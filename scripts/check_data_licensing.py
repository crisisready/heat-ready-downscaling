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
    0  every manifest passes, including reachability
    1  a licensing RULE was violated (a real rejection)
    2  a rule passed but a URL was unreachable (advisory: reported, and the
       workflow surfaces it, but it is not treated as a licensing violation
       because the network is not the contributor's fault)

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
    """(where, url) for every reproducible_fetch in a manifest."""
    method = manifest.get("method") or {}
    out: list[tuple[str, str]] = []
    for i, entry in enumerate(method.get("data_sources") or []):
        if entry.get("reproducible_fetch"):
            out.append((f"method.data_sources[{i}]", entry["reproducible_fetch"]))
    for i, entry in enumerate(method.get("extra_covariates") or []):
        if entry.get("reproducible_fetch"):
            out.append((f"method.extra_covariates[{i}]", entry["reproducible_fetch"]))
    return out


def check_reachable(url: str, timeout: float = 20.0) -> str | None:
    """None when the URL resolves, else a human-readable reason.

    HEAD first, then a Range-limited GET: plenty of data hosts answer HEAD
    with 403 or 405 while serving GET perfectly well, and failing a
    submission over that would be a false rejection.
    """
    import requests

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return None
        response = requests.get(
            url, timeout=timeout, allow_redirects=True, stream=True,
            headers={"Range": "bytes=0-0"},
        )
        response.close()
        if response.status_code < 400:
            return None
        return f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 -- any transport failure is the same answer here
        return f"{type(exc).__name__}: {exc}"


def check_one(path: str, *, network: bool) -> tuple[list[str], list[str], list[str]]:
    """(violations, flagged_for_review, unreachable) for one manifest."""
    import yaml

    with open(path) as fh:
        manifest = yaml.safe_load(fh)

    violations: list[str] = []
    flagged: list[str] = []
    unreachable: list[str] = []

    try:
        flagged = [f"{path}: {f}" for f in licensing.check_manifest_licensing(manifest)]
    except licensing.LicensingError as exc:
        violations.append(f"{path}: {exc}")
        # Still worth reporting reachability, so a contributor fixing one
        # problem sees both rather than discovering the second on the next run.

    if network:
        for where, url in declared_fetch_urls(manifest):
            reason = check_reachable(url)
            if reason:
                unreachable.append(f"{path}: {where} -> {url} ({reason})")

    return violations, flagged, unreachable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="manifest.yaml paths; default: every submission")
    parser.add_argument(
        "--no-network", action="store_true",
        help="skip the reachability check (offline rule checks only)",
    )
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob("submissions/*/*/manifest.yaml"))
    if not paths:
        print("no manifests found -- nothing to check")
        return 0

    violations: list[str] = []
    flagged: list[str] = []
    unreachable: list[str] = []
    for path in paths:
        v, f, u = check_one(path, network=not args.no_network)
        violations += v
        flagged += f
        unreachable += u

    print(f"checked {len(paths)} manifest(s)")
    if violations:
        print("\nLICENSING VIOLATIONS -- these reject the submission:")
        for item in violations:
            print(f"  - {item}")
    if flagged:
        print("\nNeeds a maintainer decision (admissible, not automatic):")
        for item in flagged:
            print(f"  - {item}")
    if unreachable:
        print("\nUnreachable reproducible_fetch URL(s) -- advisory, not a rejection:")
        for item in unreachable:
            print(f"  - {item}")
    if not (violations or flagged or unreachable):
        print("all clear")

    if violations:
        return 1
    if unreachable:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
