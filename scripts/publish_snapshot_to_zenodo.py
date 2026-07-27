"""
Publish (or version-bump) a band-paired snapshot as a Zenodo dataset deposition.

Deliberately SEPARATE from Zenodo's own GitHub-repository integration
(zenodo.org account settings -> GitHub): that integration archives this
repository's git source tree at a tagged release, not a release's uploaded
binary asset -- confirmed live 2026-07-27, the first archived release
produced a 198KB zip of the repo's own files, not the ~55MB snapshot tarball
that was actually attached to that GitHub Release. A dataset DOI meant to be
cited by academic contributors needs to point at the actual data, so this
script creates/versions a DEDICATED Zenodo deposition for the snapshot
tarball itself, using Zenodo's REST API directly.

Three, deliberately DISTINCT modes -- kept separate rather than overloading
one --deposition-id flag, after a real mix-up publishing the very first
snapshot (2026-07-27): passing the freshly-created DRAFT's own id back in
as if it were "the existing published record to version" 404'd, because
Zenodo's "new version" action only applies to an ALREADY-PUBLISHED record,
never to a draft that was never published. That is a different id and a
different action from "just publish the draft you already prepared" --
this script now names them differently so the mistake can't recur:

  (no flags beyond --tarball/--snapshot-version): creates a BRAND-NEW
    deposition -- the very first snapshot publish ever. Prints the
    resulting concept id -- SAVE THIS, pass it as --version-of on every
    future call so subsequent snapshots become new versions of the same
    dataset record rather than unrelated ones.
  --version-of CONCEPT_ID: creates a new version of that ALREADY-PUBLISHED
    deposition (Zenodo's own "new version" action), uploads the new
    snapshot, and updates metadata -- this is what a monthly snapshot
    regeneration (plan section 6.4) should call, using the concept id
    Zenodo assigned the very first time.
  --publish-draft DRAFT_ID: ONLY publishes an already-prepared draft (no
    upload, no metadata, no versioning) -- for finishing a review you
    started with a prior invocation of one of the two modes above, which
    both leave their result as an unpublished draft unless --publish was
    also passed in that same call.

Does NOT publish by default -- a Zenodo publish is effectively permanent
(the resulting DOI must resolve forever; Zenodo does not support deleting a
published record). Creates and populates a DRAFT, prints its review URL,
and requires an explicit --publish flag (in the SAME invocation) or a
separate --publish-draft call to actually publish. Verify the draft's
metadata/files via the printed URL or Zenodo's own API before publishing.

Usage:
    export ZENODO_ACCESS_TOKEN=...

    # First publish ever -- creates a new deposition, leaves it as a draft.
    python scripts/publish_snapshot_to_zenodo.py \\
        --tarball /tmp/snapshot-v2026.07.tar.gz --snapshot-version v2026.07

    # ... inspect the draft at the printed URL, then publish that SAME draft:
    python scripts/publish_snapshot_to_zenodo.py --publish-draft <id from the first call>

    # A future monthly snapshot -- new version of the same record, using
    # the CONCEPT id (not any prior version's own draft/record id):
    python scripts/publish_snapshot_to_zenodo.py \\
        --tarball /tmp/snapshot-v2026.08.tar.gz --snapshot-version v2026.08 \\
        --version-of <the ORIGINAL concept id> --publish
"""

from __future__ import annotations

import argparse
import os

import requests

_ZENODO_API = "https://zenodo.org/api/deposit/depositions"
_LICENSE_ID = "cc-by-4.0"  # Zenodo's SPDX-ish license identifier for CC BY 4.0, matching DATA_LICENSE
_GITHUB_REPO_URL = "https://github.com/crisisready/heat-ready-downscaling"


def _access_token() -> str:
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        raise SystemExit(
            "ZENODO_ACCESS_TOKEN not set -- generate one at zenodo.org account settings "
            "(scopes: deposit:write, deposit:actions) and export it before running this script."
        )
    return token


def _metadata(snapshot_version: str) -> dict:
    return {
        "title": f"HeatReady downscaling band-paired snapshot {snapshot_version}",
        "upload_type": "dataset",
        "description": (
            "Band-paired training/validation snapshot for the HeatReady neighborhood-resolution "
            "downscaling model (crisisready/heat-ready-downscaling) -- one row per station-day-band "
            "across 9 bands (era5, lag_fill, forecast_lead1..7), 907 GHCN-Daily stations. "
            "See MANIFEST.json inside the archive for per-partition checksums, and the GitHub "
            f"repository ({_GITHUB_REPO_URL}) for the code that generated it and the full "
            "F3/F7 data-quality disclosures."
        ),
        "creators": [{"name": "Kishore, Nishant", "affiliation": "CrisisReady"}],
        "license": _LICENSE_ID,
        "keywords": ["heat risk", "downscaling", "climate data", "GHCN-Daily", "ERA5-Land"],
        "related_identifiers": [{
            "identifier": _GITHUB_REPO_URL, "relation": "isSupplementTo", "resource_type": "software",
        }],
        "version": snapshot_version,
    }


def _create_new_deposition(token: str) -> dict:
    resp = requests.post(_ZENODO_API, params={"access_token": token}, json={})
    resp.raise_for_status()
    return resp.json()


def _new_version(token: str, deposition_id: str) -> dict:
    """Zenodo's own "new version" action: clones the latest version of
    `deposition_id` into a fresh draft, carrying over its metadata (which
    _metadata's own values below then overwrite) but NOT its files -- the
    draft starts empty and the caller uploads the new snapshot into it."""
    resp = requests.post(f"{_ZENODO_API}/{deposition_id}/actions/newversion", params={"access_token": token})
    resp.raise_for_status()
    draft_url = resp.json()["links"]["latest_draft"]
    draft = requests.get(draft_url, params={"access_token": token})
    draft.raise_for_status()
    return draft.json()


def _clear_existing_files(token: str, deposition: dict) -> None:
    """A new-version draft carries over the PREVIOUS version's files by
    default -- remove them before uploading this version's snapshot, or the
    new version would contain both the old and new tarballs."""
    for f in deposition.get("files", []):
        resp = requests.delete(f["links"]["self"], params={"access_token": token})
        resp.raise_for_status()


def _upload_file(token: str, deposition: dict, tarball_path: str) -> None:
    bucket_url = deposition["links"]["bucket"]
    filename = os.path.basename(tarball_path)
    with open(tarball_path, "rb") as f:
        resp = requests.put(f"{bucket_url}/{filename}", params={"access_token": token}, data=f)
    resp.raise_for_status()


def _update_metadata(token: str, deposition_id: str, metadata: dict) -> dict:
    resp = requests.put(
        f"{_ZENODO_API}/{deposition_id}", params={"access_token": token}, json={"metadata": metadata},
    )
    resp.raise_for_status()
    return resp.json()


def _publish(token: str, deposition_id: str) -> dict:
    resp = requests.post(f"{_ZENODO_API}/{deposition_id}/actions/publish", params={"access_token": token})
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tarball", default=None, help="path to the packed public snapshot .tar.gz")
    parser.add_argument("--snapshot-version", default=None)
    parser.add_argument("--version-of", default=None,
                         help="an ALREADY-PUBLISHED deposition's concept id -- creates a new version of it. "
                              "Omit entirely for the very first publish ever (creates a brand-new deposition).")
    parser.add_argument("--publish-draft", default=None,
                         help="ONLY publish an already-prepared draft id -- no upload, no metadata, no "
                              "versioning. Mutually exclusive with --tarball/--version-of.")
    parser.add_argument("--publish", action="store_true",
                         help="publish immediately after creating/versioning in THIS SAME invocation "
                              "(default: leave as an unpublished draft for review)")
    args = parser.parse_args()

    # Argument validation before requiring a token -- a CLI mistake should
    # surface as a CLI error, not a confusing "no token" message that has
    # nothing to do with what was actually wrong.
    if args.publish_draft and (args.tarball or args.version_of):
        raise SystemExit("--publish-draft is exclusive with --tarball/--version-of -- see this script's own docstring")
    if not args.publish_draft and (not args.tarball or not args.snapshot_version):
        raise SystemExit("--tarball and --snapshot-version are required unless using --publish-draft")

    token = _access_token()

    if args.publish_draft:
        published = _publish(token, args.publish_draft)
        print(f"Published: DOI={published['doi']} concept_doi={published.get('conceptdoi')}")
        print(f"Record: {published['links']['html']}")
        return

    if args.version_of:
        deposition = _new_version(token, args.version_of)
        _clear_existing_files(token, deposition)
    else:
        deposition = _create_new_deposition(token)

    draft_id = deposition["id"]
    _upload_file(token, deposition, args.tarball)
    deposition = _update_metadata(token, draft_id, _metadata(args.snapshot_version))

    print(f"Draft deposition ready: id={draft_id}")
    print(f"Review at: {deposition['links']['html']}")
    print(f"Files: {[f['filename'] if 'filename' in f else f.get('key') for f in deposition.get('files', [])]}")

    if not args.publish:
        print(f"Not publishing (pass --publish-draft {draft_id} once you've verified the draft above).")
        return

    published = _publish(token, draft_id)
    print(f"Published: DOI={published['doi']} concept_doi={published.get('conceptdoi')}")
    print(f"Record: {published['links']['html']}")


if __name__ == "__main__":
    main()
