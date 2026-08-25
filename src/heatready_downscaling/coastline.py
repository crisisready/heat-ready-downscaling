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
# Machine-readable provenance, the same shape a submission's own
# method.data_sources entry must take (see submission.DATA_SOURCE_SCHEMA).
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
        if shape_type == _SHAPE_TYPE_POLYLINE:
            num_parts, num_points = struct.unpack_from("<ii", shp_bytes, body + 4 + 32)
            points_at = body + 4 + 32 + 8 + 4 * num_parts
            flat = np.frombuffer(
                shp_bytes, dtype="<f8", count=2 * num_points, offset=points_at,
            )
            parts.append(flat.reshape(num_points, 2))
        offset = body + content_len_words * 2

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
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as fh:
                fh.write(payload)

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

        lat_arr = np.atleast_1d(np.asarray(
            [np.nan if v is None else float(v) for v in np.atleast_1d(lats)], dtype=float,
        ))
        lon_arr = np.atleast_1d(np.asarray(
            [np.nan if v is None else float(v) for v in np.atleast_1d(lons)], dtype=float,
        ))
        if lat_arr.shape != lon_arr.shape:
            raise ValueError(
                f"lats and lons must be the same length, got {lat_arr.shape} and {lon_arr.shape}",
            )

        out = np.full(lat_arr.shape, np.nan)
        usable = np.isfinite(lat_arr) & np.isfinite(lon_arr)
        if usable.any():
            query = np.radians(np.column_stack([lat_arr[usable], lon_arr[usable]]))
            angular, _ = self._tree.query(query, k=1)
            out[usable] = angular[:, 0] * EARTH_RADIUS_KM
        return out
