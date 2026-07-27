"""Unit tests for heatready_downscaling.koppen -- no network calls (kgcpy's
raster lookup is offline)."""

from unittest.mock import patch

import pytest

from heatready_downscaling import koppen


class TestKoppenClimateZone:
    def test_real_land_stations_return_koppen_codes(self):
        # Phoenix -> BWh (desert), confirmed against a known station.
        assert koppen.koppen_climate_zone(33.4278, -112.0036) == "BWh"

    def test_lookup_failure_falls_back_to_latitude_band(self):
        with patch("kgcpy.lookupCZ", side_effect=RuntimeError("boom")):
            zone = koppen.koppen_climate_zone(10.0, 0.0)
        assert zone == "tropical"

    def test_ocean_pixel_resolved_via_neighbor_search(self):
        with patch("kgcpy.lookupCZ", return_value="Ocean"), \
             patch("kgcpy.nearbyCZ", return_value=(None, None, ["Ocean", "Cfb", "Ocean"])):
            zone = koppen.koppen_climate_zone(40.0, -70.0)
        assert zone == "Cfb"

    def test_persistent_ocean_falls_back_to_latitude_band(self):
        with patch("kgcpy.lookupCZ", return_value="Ocean"), \
             patch("kgcpy.nearbyCZ", return_value=(None, None, ["Ocean", "Ocean"])):
            zone = koppen.koppen_climate_zone(60.0, -70.0)
        assert zone == "polar"


class TestLatitudeBandFallback:
    @pytest.mark.parametrize("lat,expected", [
        (0.0, "tropical"), (23.4, "tropical"), (23.5, "subtropical"),
        (34.9, "subtropical"), (35.0, "temperate"), (54.9, "temperate"),
        (55.0, "polar"), (-60.0, "polar"),
    ])
    def test_latitude_bands(self, lat, expected):
        assert koppen._latitude_band_fallback(lat) == expected


class TestKoppenMainGroupCode:
    def test_koppen_broad_group_letter_from_zone(self):
        assert koppen.koppen_broad_group_letter_from_zone("BWh") == "B"
        assert koppen.koppen_broad_group_letter_from_zone("Cfb") == "C"

    def test_unrecognized_zone_falls_back_to_c(self):
        assert koppen.koppen_broad_group_letter_from_zone("unrecognized") == "C"

    def test_latitude_band_label_maps_to_koppen_group(self):
        # "temperate" (a latitude-band label) maps to Koppen group D, not
        # its own letter "C" (a real, easy-to-get-backwards distinction).
        assert koppen.koppen_broad_group_letter_from_zone("temperate") == "D"

    def test_ordinal_code_from_zone(self):
        assert koppen.koppen_main_group_code_from_zone("BWh") == 1
        assert koppen.koppen_main_group_code_from_zone("Dfc") == 3

    def test_main_group_code_end_to_end(self):
        # Phoenix -> BWh -> group B -> code 1
        assert koppen.koppen_main_group_code(33.4278, -112.0036) == 1
