"""
coast_dist_km -- distance from a point to the nearest ocean coastline.

Exists because Valencia's real validated local correction is
`corrected = raw_grid + intercept + slope * coast_dist_km` (see
crisisready/heat-risk-data-api's research/valencia-local-model-evaluation/
PHASE6_REAL_LOCAL_MODEL.md: 31.0% hot-day tmin RMSE reduction, 95% cluster
bootstrap CI [20.4%, 39.2%]), and until this module existed the snapshot had
no coast-distance column for that correction to be scored against. It was the
one covariate score.STATIC_COVARIATE_ALLOWLIST could not offer.

DATA SOURCE (decided by Nishant, 2026-08-25): Natural Earth 1:10m physical
coastline. Public domain, no attribution obligation, no redistribution
restriction -- the cleanest possible licensing precedent for this pipeline's
first data input beyond the published snapshot. See NATURAL_EARTH_* below for
the pinned URL and checksum, and DATA_LICENSE for how the program treats
third-party data generally.

WHAT THIS MEASURES, AND WHAT IT DOES NOT -- read before using it as a
covariate anywhere but a maritime city. Natural Earth's `coastline` layer is
OCEAN coastline. It does not include lakes. A large-lake city therefore gets a
distance to the sea, which is not the quantity its own microclimate responds
to. Measured directly, not assumed:

    Valencia city         4.1 km      correct, Mediterranean
    Madrid              305.0 km      correct, deep interior
    Seoul                21.2 km      correct, Yellow Sea
    Chicago             951.5 km      TECHNICALLY correct, PHYSICALLY USELESS
                                      -- Chicago is on Lake Michigan, and its
                                      real lake-breeze effect is a few km away

Chicago is not a hypothetical: it is the negative case in this program's own
research record (research/seoul-local-sensor-validation/
CHICAGO_NEGATIVE_FINDING.md). A lake-inclusive `water_dist_km` is a real and
probably worthwhile follow-up, deliberately NOT folded in here under the name
`coast_dist_km`: silently including lakes under that name would make the
covariate mean one thing at fitting time and another at serving time, which is
the exact class of mismatch that already forced pop_density_per_km2 off the
allowlist (see score.COVARIATE_EXCLUSIONS).

METHOD, and its stated approximation: distance to the nearest coastline
VERTEX, not to the nearest point on the coastline segment, computed as a
great-circle distance via a haversine BallTree. The error is bounded by the
local vertex spacing, which for Natural Earth 10m is well under a kilometre in
populated coastal areas (410,957 vertices over 4,133 line features). That is
comfortably inside what this covariate is used for -- Valencia's own fitted
slope is per-kilometre over a 0-25 km range -- and it buys a pure
numpy+scikit-learn implementation with no geometry-library dependency in the
scoring path at all. A segment-exact version would need shapely per query and
would move the number by less than the coastline's own cartographic
generalisation at 1:10m. Stated rather than silently accepted.
"""

import hashlib
import logging
import struct

logger = logging.getLogger(__name__)

# Pinned source. The checksum is verified on every fetch: Natural Earth
# republishes at the same URL across releases, so an unpinned fetch would
# silently change every station's covariate value between snapshot builds.
NATURAL_EARTH_COASTLINE_URL = (
    "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
)
NATURAL_EARTH_COASTLINE_SHA256 = (
    "bfa04cdbcbef07ef90dfca1dabb48062eca29900a113df0f389303e255484017"
)
NATURAL_EARTH_COASTLINE_BYTES = 3069451
# Machine-readable provenance for this data source. Intended to be the first
# entry the roadmap's data_sources licensing CI consumes -- that checker and
# its schema do NOT exist yet (they are a separate, later PR), so these key
# names are this module's proposal, not conformance to an existing contract.
# Stated plainly because the alternative is a comment citing a schema nobody
# can find (code-review finding, PR #29).
NATURAL_EARTH_COASTLINE_DATA_SOURCE = {
    "name": "Natural Earth 1:10m physical coastline",
    "version": "10m",
    "license": "CC0-1.0",
    "license_note": (
        "Natural Earth is public domain: 'no permission needed', no attribution "
        "obligation, no redistribution restriction. Treated as CC0-1.0 for the "
        "SPDX allowlist."
    ),
    "attribution_required": False,
    "redistribution_tier": "unrestricted",
    "reproducible_fetch": NATURAL_EARTH_COASTLINE_URL,
    "sha256": NATURAL_EARTH_COASTLINE_SHA256,
}

_SHAPEFILE_HEADER_BYTES = 100
_SHAPE_TYPE_POLYLINE = 3
EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius, matching what haversine assumes


def _check_ranges(lats, lons, *, what: str) -> None:
    """Raise if latitudes/longitudes are out of range.

    This is the cheap guard against the one mistake this module's own
    docstring says it fears, and which a caller WILL eventually make:
    CoastlineIndex takes vertices as (lon, lat) while distance_km takes
    (lats, lons). Swapping them produces a confidently plausible wrong
    number, not an error -- measured on the real archive, Seoul is 21.2 km
    correct and 608.0 km swapped. Since the caller is a snapshot builder that
    bakes one number per station into a published column, a wrong-but-
    plausible value is the worst possible outcome (code-review finding, PR
    #29). Only |lat| > 90 is strictly detectable, but that catches the
    realistic swap for every station outside the tropics.
    """
    import numpy as np

    lat_arr, lon_arr = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    bad_lat = np.abs(lat_arr) > 90.0
    bad_lon = np.abs(lon_arr) > 180.0
    if bad_lat.any() or bad_lon.any():
        raise ValueError(
            f"{what} out of range: {int(bad_lat.sum())} with |lat| > 90 and "
            f"{int(bad_lon.sum())} with |lon| > 180. Note that CoastlineIndex takes "
            "vertices as (lon, lat) while distance_km takes (lats, lons) -- a swap here "
            "otherwise yields a plausible but wrong distance rather than an error.",
        )


def parse_shapefile_polyline_vertices(shp_bytes: bytes):
    """Every vertex of every PolyLine record in an ESRI .shp, as an (N, 2)
    float64 array of (lon, lat).

    Hand-parsed rather than read through geopandas/pyogrio, deliberately.
    The PolyLine record layout is fixed and trivial (little-endian: shape
    type, bounding box, part count, point count, part offsets, then packed
    x/y doubles), it needs no GDAL stack, and keeping it dependency-free
    means the one place this repo touches a shapefile cannot break on a
    GDAL/pyogrio packaging change in CI. Non-PolyLine records are skipped
    rather than raising -- a future Natural Earth release adding another
    geometry type to this layer should degrade to "fewer vertices", not to a
    hard failure mid-snapshot-build.
    """
    import numpy as np

    parts: list = []
    offset = _SHAPEFILE_HEADER_BYTES
    total = len(shp_bytes)
    while offset < total:
        # Record header is big-endian (record number, content length in
        # 16-bit words); the record body that follows is little-endian.
        _record_number, content_len_words = struct.unpack_from(">ii", shp_bytes, offset)
        body = offset + 8
        shape_type = struct.unpack_from("<i", shp_bytes, body)[0]
        record_end = body + content_len_words * 2
        if record_end > total or content_len_words <= 0:
            raise ValueError(
                f"shapefile record at offset {offset} claims {content_len_words} 16-bit "
                f"words, which runs past the {total}-byte file -- truncated or malformed",
            )
        if shape_type == _SHAPE_TYPE_POLYLINE:
            num_parts, num_points = struct.unpack_from("<ii", shp_bytes, body + 4 + 32)
            points_at = body + 4 + 32 + 8 + 4 * num_parts
            # Bounds-checked rather than trusted (code-review finding, PR #29):
            # a record whose num_points overstates its own content would
            # otherwise read the FOLLOWING record's bytes as coordinates and
            # report them as coastline, with no error. Unreachable through
            # fetch_coastline_vertices, whose checksum is verified first, but
            # this is a public function and a direct caller has no such guard.
            if num_points < 0 or points_at + 16 * num_points > record_end:
                raise ValueError(
                    f"shapefile PolyLine at offset {offset} claims {num_points} points, "
                    "which runs past the end of its own record -- malformed",
                )
            flat = np.frombuffer(
                shp_bytes, dtype="<f8", count=2 * num_points, offset=points_at,
            )
            parts.append(flat.reshape(num_points, 2))
        offset = record_end

    if not parts:
        raise ValueError(
            "no PolyLine records found in this .shp -- either the wrong layer was "
            "downloaded, or Natural Earth changed this layer's geometry type",
        )
    return np.concatenate(parts)


def fetch_coastline_vertices(cache_path: str | None = None):
    """Download the pinned Natural Earth coastline and return its vertices as
    an (N, 2) (lon, lat) array. Verifies size AND sha256 before parsing --
    Natural Earth republishes at the same URL, so an unverified fetch would
    silently shift every station's covariate between snapshot builds.

    cache_path, if given, is read instead of the network when it already
    matches the pinned checksum, and written after a successful fetch.
    """
    import os

    payload = None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            candidate = fh.read()
        if hashlib.sha256(candidate).hexdigest() == NATURAL_EARTH_COASTLINE_SHA256:
            payload = candidate
        else:
            logger.warning(
                "cached coastline at %s does not match the pinned checksum -- refetching",
                cache_path,
            )

    if payload is None:
        import requests

        response = requests.get(NATURAL_EARTH_COASTLINE_URL, timeout=120)
        response.raise_for_status()
        payload = response.content
        # Verify BEFORE writing the cache (code-review finding, PR #29). The
        # motivating failure this checksum exists for -- an HTML error page
        # served with 200 -- would otherwise be persisted to disk and, in CI,
        # uploaded by any cache-save step, before being rejected.
        verify_coastline_archive(payload)
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as fh:
                fh.write(payload)
    else:
        verify_coastline_archive(payload)

    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        shp_bytes = archive.read("ne_10m_coastline.shp")
    return parse_shapefile_polyline_vertices(shp_bytes)


def verify_coastline_archive(payload: bytes) -> None:
    """Raise unless `payload` is byte-for-byte the pinned Natural Earth
    archive. Size is checked first purely because it produces a clearer
    message for the common failure (an HTML error page served with 200)."""
    if len(payload) != NATURAL_EARTH_COASTLINE_BYTES:
        raise ValueError(
            f"coastline archive is {len(payload)} bytes, expected "
            f"{NATURAL_EARTH_COASTLINE_BYTES} -- refusing to build a covariate from "
            "an unexpected payload",
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != NATURAL_EARTH_COASTLINE_SHA256:
        raise ValueError(
            f"coastline archive sha256 is {digest}, expected "
            f"{NATURAL_EARTH_COASTLINE_SHA256} -- Natural Earth republishes at the "
            "same URL, so this is either a new release or a corrupted download. "
            "Bump the pin deliberately (and rebuild every affected snapshot) rather "
            "than loosening this check.",
        )


class CoastlineIndex:
    """Nearest-coastline-vertex lookup. Pure numpy + scikit-learn, no
    geometry library, so this is importable and usable anywhere in the
    scoring path."""

    def __init__(self, vertices):
        import numpy as np
        from sklearn.neighbors import BallTree

        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(f"vertices must be (N, 2) (lon, lat), got {vertices.shape}")
        if not len(vertices):
            raise ValueError("vertices is empty -- nothing to measure distance to")
        _check_ranges(vertices[:, 1], vertices[:, 0], what="vertices")
        self.n_vertices = len(vertices)
        # BallTree's haversine metric expects (lat, lon) IN RADIANS and
        # returns angular distance, hence the column swap and the radius
        # multiply in distance_km. Getting either wrong yields plausible-
        # looking numbers off by a constant factor, so both are covered by
        # tests against known city distances.
        self._tree = BallTree(np.radians(vertices[:, ::-1]), metric="haversine")

    def distance_km(self, lats, lons):
        """Great-circle km from each (lat, lon) to the nearest coastline
        vertex. Inputs may be scalars or sequences; a None/NaN coordinate
        yields NaN rather than raising, so one bad station cannot fail a whole
        snapshot build."""
        import numpy as np

        def _coords(values):
            # None -> NaN so a single unusable station cannot fail a whole
            # snapshot build; atleast_1d so scalars and sequences take the
            # same path.
            return np.array(
                [np.nan if v is None else float(v) for v in np.atleast_1d(values)],
                dtype=float,
            )

        lat_arr = _coords(lats)
        lon_arr = _coords(lons)
        if lat_arr.shape != lon_arr.shape:
            raise ValueError(
                f"lats and lons must be the same length, got {lat_arr.shape} and {lon_arr.shape}",
            )

        _check_ranges(lat_arr, lon_arr, what="query coordinates")
        out = np.full(lat_arr.shape, np.nan)
        usable = np.isfinite(lat_arr) & np.isfinite(lon_arr)
        if usable.any():
            query = np.radians(np.column_stack([lat_arr[usable], lon_arr[usable]]))
            angular, _ = self._tree.query(query, k=1)
            out[usable] = angular[:, 0] * EARTH_RADIUS_KM
        return out
