"""Unit tests for prewarm_lagfill_chunks.py's next_window() -- the only pure
logic in that module (everything else needs network/DB access, see the
module's own docstring). Excluded from this repo's own collection (see
conftest.py) since importing the module at all requires open_meteo/
api_call_manager/db, private-repo-only modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prewarm_lagfill_chunks import next_window


class TestNextWindow:
    def test_skips_done_and_lookahead_then_takes_window(self):
        station_order = [f"S{i}" for i in range(20)]
        done = {"S0", "S1", "S2"}  # real job has finished 3 stations
        result = next_window(station_order, done, already_prewarmed=set(), lookahead=5, window=4)
        # not_done = S3..S19 (17 stations); skip 5 (S3..S7, "too close, real job will get there
        # before a prewarm could help"); take next 4: S8..S11.
        assert result == ["S8", "S9", "S10", "S11"]

    def test_already_prewarmed_stations_are_not_repeated(self):
        station_order = [f"S{i}" for i in range(20)]
        result = next_window(station_order, done=set(), already_prewarmed={"S5", "S6"},
                              lookahead=5, window=4)
        # not_done = all 20; skip 5 (S0..S4); window would be S5..S8, but S5/S6 already prewarmed.
        assert result == ["S7", "S8"]

    def test_empty_when_lookahead_exceeds_remaining_stations(self):
        station_order = [f"S{i}" for i in range(5)]
        result = next_window(station_order, done=set(), already_prewarmed=set(), lookahead=10, window=4)
        assert result == []

    def test_window_smaller_than_available_still_returns_only_window_size(self):
        station_order = [f"S{i}" for i in range(100)]
        result = next_window(station_order, done=set(), already_prewarmed=set(), lookahead=0, window=3)
        assert result == ["S0", "S1", "S2"]
