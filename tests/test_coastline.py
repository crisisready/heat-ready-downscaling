"""Unit tests for heatready_downscaling.coastline.

Two layers, deliberately. Everything about the INDEX runs offline against a
tiny hand-built vertex set, so the suite never touches the network. The
real-city distance checks are marked and skipped unless the pinned Natural
Earth archive is already cached locally -- they are the only thing that can
catch a lat/lon column swap or a missing radius multiply, both of which
produce confidently plausible wrong numbers rather than an error.
"""

import math
import os
import struct

import numpy as np
import pytest

from heatready_downscaling import coastline


def _polyline_shp(*lines) -> bytes:
    """A minimal, valid ESRI .shp containing the given PolyLine features, so
    the parser is tested against bytes it did not itself produce assumptions
    about."""
    out = bytearray(b"\x00" * 100)
    for i, pts in enumerate(lines, start=1):
        body = struct.pack("<i", 3)
        body += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)  # bounding box
        body += struct.pack("<ii", 1, len(pts))          # numParts, numPoints
        body += struct.pack("<i", 0)                      # part offsets
        for lon, lat in pts:
            body += struct.pack("<2d", lon, lat)
        out += struct.pack(">ii", i, len(body) // 2) + body
    return bytes(out)


class TestShapefileParsing:
    def test_reads_every_vertex_of_every_feature(self):
        shp = _polyline_shp(
            [(0.0, 0.0), (1.0, 1.0)],
            [(10.0, 10.0), (11.0, 11.0), (12.0, 12.0)],
        )
        verts = coastline.parse_shapefile_polyline_vertices(shp)
        assert verts.shape == (5, 2)
        assert verts[0].tolist() == [0.0, 0.0]
        assert verts[-1].tolist() == [12.0, 12.0]

    def test_non_polyline_records_are_skipped_not_fatal(self):
        """A future Natural Earth release adding another geometry type to this
        layer should lose vertices, not fail a snapshot build."""
        shp = bytearray(_polyline_shp([(1.0, 2.0), (3.0, 4.0)]))
        point_body = struct.pack("<i", 1) + struct.pack("<2d", 9.0, 9.0)
        shp += struct.pack(">ii", 99, len(point_body) // 2) + point_body
        verts = coastline.parse_shapefile_polyline_vertices(bytes(shp))
        assert verts.shape == (2, 2)

    def test_a_shapefile_with_no_polylines_raises(self):
        with pytest.raises(ValueError, match="no PolyLine records"):
            coastline.parse_shapefile_polyline_vertices(bytes(100))


class TestArchiveVerification:
    def test_wrong_size_is_rejected_with_a_size_message(self):
        with pytest.raises(ValueError, match="bytes, expected"):
            coastline.verify_coastline_archive(b"<html>404</html>")

    def test_right_size_wrong_digest_is_rejected(self):
        payload = b"x" * coastline.NATURAL_EARTH_COASTLINE_BYTES
        with pytest.raises(ValueError, match="sha256"):
            coastline.verify_coastline_archive(payload)

    def test_the_data_source_block_is_complete_and_permissive(self):
        """This is the pipeline's first data input beyond the snapshot, so its
        provenance block is the precedent every later one follows."""
        block = coastline.NATURAL_EARTH_COASTLINE_DATA_SOURCE
        for key in ("name", "license", "reproducible_fetch", "sha256", "redistribution_tier"):
            assert block.get(key), f"missing {key}"
        assert block["license"] == "CC0-1.0"
        assert block["attribution_required"] is False
        assert block["sha256"] == coastline.NATURAL_EARTH_COASTLINE_SHA256


class TestCoastlineIndex:
    def test_distance_to_a_vertex_on_top_of_the_query_is_zero(self):
        idx = coastline.CoastlineIndex([[10.0, 50.0], [11.0, 51.0]])
        assert idx.distance_km(50.0, 10.0)[0] == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_of_latitude_is_about_111_km(self):
        """Pins BOTH the (lat, lon) column order and the radius multiply. Get
        either wrong and every number stays plausible but is off by a factor,
        which no amount of code review reliably catches."""
        idx = coastline.CoastlineIndex([[0.0, 0.0]])
        km = idx.distance_km(1.0, 0.0)[0]
        assert km == pytest.approx(111.19, abs=0.5)

    def test_a_degree_of_longitude_shrinks_with_latitude(self):
        """The column-swap check that latitude alone cannot make: at 60N a
        degree of longitude is about half a degree of latitude."""
        idx = coastline.CoastlineIndex([[0.0, 60.0]])
        km = idx.distance_km(60.0, 1.0)[0]
        assert km == pytest.approx(111.19 * math.cos(math.radians(60.0)), abs=1.0)

    def test_it_picks_the_nearest_of_many_vertices(self):
        idx = coastline.CoastlineIndex([[0.0, 0.0], [0.0, 10.0], [0.0, 0.5]])
        assert idx.distance_km(0.4, 0.0)[0] == pytest.approx(11.1, abs=1.0)

    def test_vector_input_returns_one_distance_per_point(self):
        idx = coastline.CoastlineIndex([[0.0, 0.0]])
        out = idx.distance_km([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
        assert out.shape == (3,)
        assert out[0] < out[1] < out[2]

    def test_a_none_coordinate_yields_nan_rather_than_raising(self):
        """One unusable station must not fail a whole snapshot build, and NaN
        is what score_band's own missing-covariate path already handles."""
        idx = coastline.CoastlineIndex([[0.0, 0.0]])
        out = idx.distance_km([0.0, None], [0.0, 0.0])
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(out[1])

    def test_mismatched_input_lengths_raise(self):
        idx = coastline.CoastlineIndex([[0.0, 0.0]])
        with pytest.raises(ValueError, match="same length"):
            idx.distance_km([0.0, 1.0], [0.0])

    @pytest.mark.parametrize("bad", [[], [[1.0, 2.0, 3.0]]])
    def test_malformed_vertices_raise(self, bad):
        with pytest.raises(ValueError):
            coastline.CoastlineIndex(bad)


_CACHE = os.environ.get("HEATREADY_COASTLINE_CACHE", "")


@pytest.mark.skipif(
    not (_CACHE and os.path.exists(_CACHE)),
    reason="set HEATREADY_COASTLINE_CACHE to the pinned ne_10m_coastline.zip to run",
)
class TestAgainstRealCoastline:
    """Real-world sanity, against the actual pinned archive. These are the
    numbers that were checked before any of this was wired into the snapshot
    schema; the Chicago case is kept as a test precisely because it is the
    documented LIMITATION, not a bug to fix quietly."""

    @pytest.fixture(scope="class")
    @classmethod
    def index(cls):
        return coastline.CoastlineIndex(coastline.fetch_coastline_vertices(_CACHE))

    @pytest.mark.parametrize("name, lat, lon, low, high", [
        ("Valencia city",  39.4699,  -0.3763,    1.0,   10.0),
        ("Madrid",         40.4168,  -3.7038,  250.0,  350.0),
        ("Seoul",          37.5665, 126.9780,   10.0,   60.0),
    ])
    def test_known_city_distances(self, index, name, lat, lon, low, high):
        km = index.distance_km(lat, lon)[0]
        assert low <= km <= high, f"{name}: {km:.1f} km outside [{low}, {high}]"

    def test_chicago_reads_as_far_inland_because_lakes_are_excluded(self, index):
        """Documented limitation, pinned as a test so it cannot be discovered
        by surprise later: Natural Earth's coastline layer is ocean-only, and
        Chicago is this program's own negative-case city. Anyone who wants a
        lake-aware covariate needs a new one, not a redefinition of this."""
        km = index.distance_km(41.8781, -87.6298)[0]
        assert km > 500, (
            f"Chicago measured {km:.1f} km; if this is now small, the coastline "
            "layer has started including lakes and coast_dist_km's meaning changed"
        )
