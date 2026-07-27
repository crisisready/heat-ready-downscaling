"""
Köppen-Geiger climate-zone classification -- extracted from
crisisready/heat-risk-data-api's src/ghcn.py (lines ~298-428 as of
2026-07-27, commit 57479e5). See PROVENANCE.md.

Climate-zone labeling uses `kgcpy` (real Köppen-Geiger classification, an
offline raster lookup bundled in the pip package -- no network call).
"""

import logging

logger = logging.getLogger(__name__)

# ndownloader host WAF-blocked; a dedicated red-team search (2026-07-18,
# in the private repo) found the actual fix -- the WAF only fronts the
# apex host, and ndownloader.figshare.com (the subdomain the Figshare API's
# own download_url points at) serves the same files with zero challenge.
# kgcpy bundles a downsampled (~3km/px) version of that raster, which is
# what lookupCZ/nearbyCZ read below; verified live against known stations
# (Phoenix -> BWh, Dhaka -> Aw, Mexico City -> Cwb, Miami -> Am) plus a full
# md5-verified download of the source archive.
_CLIMATE_ZONE_BANDS = (
    (23.5, "tropical"),
    (35.0, "subtropical"),
    (55.0, "temperate"),
    (90.0, "polar"),
)

# Neighbor-search radius (in raster pixels, ~3km/px) tried when a station's
# own pixel resolves to "Ocean" -- real for a coastal station whose pixel
# center falls just offshore. A single fixed size, not an escalating ladder:
# kgcpy.nearbyCZ returns neighbor zones in raster-scan order, not sorted by
# distance (confirmed by reading its source), so trying progressively wider
# searches and taking the first hit from each does NOT get progressively
# "nearer" results -- it just costs more raster re-reads for no accuracy
# gain. One search at this radius is equally reliable. Not a substitute for
# true small-island resolution: verified live that tiny islands/atolls (Key
# West, Diego Garcia) stay "Ocean" even at this radius, since kgcpy's bundled
# raster genuinely has no land pixel there at ~3km resolution -- those fall
# through to the latitude-band fallback below, not a bug to fix.
_OCEAN_FALLBACK_SEARCH_SIZE = 15


def koppen_climate_zone(lat: float, lon: float) -> str:
    """
    Real Köppen-Geiger climate zone (e.g. 'BWh', 'Cfb', 'Aw') for a station
    location, via kgcpy's offline raster lookup. Falls back to the coarse
    latitude-band proxy (_latitude_band_fallback) only when the lookup
    itself errors, or when the station's pixel -- and a neighbor search
    around it -- all resolve to open ocean (small islands/atolls the
    bundled ~3km/px raster can't resolve to land; a documented, acceptable
    trade-off, not the common case for GHCN stations, which are land-based
    weather stations).
    """
    from kgcpy import lookupCZ, nearbyCZ

    try:
        zone = lookupCZ(lat, lon)
    except Exception:
        logger.warning("kgcpy lookupCZ failed for (%s, %s); using latitude-band fallback", lat, lon, exc_info=True)
        return _latitude_band_fallback(lat)

    if zone != "Ocean":
        return zone

    try:
        _center, _uncertainty, neighbors = nearbyCZ(lat, lon, size=_OCEAN_FALLBACK_SEARCH_SIZE)
        land_neighbors = [z for z in neighbors if z != "Ocean"]
        if land_neighbors:
            return land_neighbors[0]
    except Exception:
        logger.warning("kgcpy nearbyCZ failed for (%s, %s); using latitude-band fallback", lat, lon, exc_info=True)

    logger.warning(
        "kgcpy resolved (%s, %s) as Ocean even after a %d-px neighbor search; using latitude-band fallback",
        lat, lon, _OCEAN_FALLBACK_SEARCH_SIZE,
    )
    return _latitude_band_fallback(lat)


def _latitude_band_fallback(lat: float) -> str:
    """Coarse latitude-band climate-zone proxy, used only when
    koppen_climate_zone can't resolve a real classification (see there)."""
    abs_lat = abs(lat)
    for threshold, label in _CLIMATE_ZONE_BANDS:
        if abs_lat < threshold:
            return label
    return "polar"


# Koppen main-group letter -> ordinal code. Ordinal, not one-hot: only 5
# main groups, and a tree ensemble (QRF) does not need true numeric
# ordering to split on this cheaply -- one-hot would add sparse columns a
# training set this size does not support well. Coarse main group (A-E),
# not the full subtype code (e.g. "Cfa" has dozens of variants) -- the
# goal is conditioning on broad climate REGIME, and the full subtype
# cardinality would be far too sparse to learn from.
_KOPPEN_MAIN_GROUP_CODE = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# koppen_climate_zone's rare fallback path (kgcpy lookup failure, or an
# unresolvable ocean pixel) returns a _CLIMATE_ZONE_BANDS label instead of a
# real Köppen code -- approximate-mapped to the closest main group so this
# feature always has SOME value on that rare path rather than silently
# becoming a missing-covariate gap. Not an exact equivalence (the label
# bands are pure latitude cutoffs, Köppen's main groups also depend on
# precipitation/temperature), just a reasonable nearest match.
_LATITUDE_BAND_TO_KOPPEN_GROUP = {"tropical": "A", "subtropical": "C", "temperate": "D", "polar": "E"}


def koppen_broad_group_letter_from_zone(zone: str) -> str:
    """The Köppen main-group LETTER (A-E) for an already-resolved zone
    string -- the same resolution/fallback koppen_main_group_code_from_zone
    uses for its ordinal code, split out so a caller that keys by group
    NAME rather than ordinal (e.g. the station-blend validation harness's
    per-broad-group L_km/R_km/tau parameters) doesn't have to invert the
    ordinal or duplicate this fallback logic. Falls back to Köppen group
    "C" (a neutral middle group -- note this is Köppen's own "C", not this
    module's _CLIMATE_ZONE_BANDS latitude-band label "temperate", which
    maps to group "D") in the unreachable case of an unrecognized zone
    string."""
    letter = zone[0] if zone and zone[0] in _KOPPEN_MAIN_GROUP_CODE else _LATITUDE_BAND_TO_KOPPEN_GROUP.get(zone)
    return letter if letter in _KOPPEN_MAIN_GROUP_CODE else "C"


def koppen_main_group_code_from_zone(zone: str) -> int:
    """Ordinal 0-4 code for the Köppen main climate group (A-E) given an
    already-resolved zone string (from koppen_climate_zone) -- split out of
    koppen_main_group_code so a caller that also needs the zone string
    itself (e.g. for a separate CV-gate/conformal-CI lookup) can compute it
    ONCE and derive both values from that single kgcpy raster lookup,
    rather than calling koppen_climate_zone a second time."""
    return _KOPPEN_MAIN_GROUP_CODE[koppen_broad_group_letter_from_zone(zone)]


def koppen_main_group_code(lat: float, lon: float) -> int:
    """Ordinal 0-4 code for the Köppen main climate group (A-E) at (lat,
    lon) -- see koppen_main_group_code_from_zone above. Callers that also
    need the raw zone string should call koppen_climate_zone once and pass
    its result to koppen_main_group_code_from_zone directly, rather than
    calling this function (which repeats that same lookup)."""
    return koppen_main_group_code_from_zone(koppen_climate_zone(lat, lon))
