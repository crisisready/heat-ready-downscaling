"""Unit tests for scripts/gsod_homogeneity.py -- pure Python, no network."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import gsod_homogeneity as h


def _rows(sid, lat, lon, offset_by_year, noise, rng):
    rows = []
    for y, off in offset_by_year.items():
        for m in (4, 5, 6, 7, 8, 9):
            for d in range(1, 29):
                rows.append({"station_id": sid, "lat": lat, "lon": lon, "date": f"{y}-{m:02d}-{d:02d}",
                             "delta_tmax_c": off + rng.gauss(0, noise)})
    return rows


def _network(local_break, shared_drift):
    rng = random.Random(1)
    rows = []
    for i in range(6):
        sid = f"S{i}"
        common = {y: rng.gauss(0, 0.15) + (shared_drift if y >= 2021 else 0.0) for y in range(2016, 2026)}
        offs = {y: 1.5 + v - (local_break if (i == 0 and y >= 2023) else 0.0) for y, v in common.items()}
        rows += _rows(sid, 22.0 + 0.3 * i, 72.0 + 0.3 * i, offs, 1.0, rng)
    return rows


def test_local_break_is_flagged_and_pre_break_years_dropped():
    tests, first_kept = h.screen(_network(local_break=1.7, shared_drift=0.0))
    assert first_kept == {"S0": 2023}
    flagged = [t for t in tests if t.get("flagged")]
    assert {t["station_id"] for t in flagged} == {"S0"}
    assert all(t["shift_c"] < -1.0 for t in flagged)
    rows = h.apply_mode(_network(1.7, 0.0), "break", first_kept)
    assert min(int(r["date"][:4]) for r in rows if r["station_id"] == "S0") == 2023
    assert min(int(r["date"][:4]) for r in rows if r["station_id"] == "S1") == 2016


def test_shared_drift_is_not_a_break():
    # Every station moves +0.8 C from 2021 (e.g. drift in the ERA5-Land reference): the neighbour
    # composite cancels it, so nothing is flagged.
    _, first_kept = h.screen(_network(local_break=0.0, shared_drift=0.8))
    assert first_kept == {}


def test_benjamini_hochberg_matches_known_values():
    q = h.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert [round(x, 4) for x in q] == [0.04, 0.0533, 0.0533, 0.5]


def test_mc_pvalue_is_small_for_large_statistic_and_large_for_zero():
    splits = h._allowed_splits(list(range(2016, 2026)))
    assert h.mc_pvalue(50.0, 10, splits, n_sim=2000) < 0.01
    assert h.mc_pvalue(0.0, 10, splits, n_sim=2000) > 0.99


def test_recent_mode_keeps_only_recent_years():
    rows = [{"station_id": "A", "date": "2022-05-01"}, {"station_id": "A", "date": "2023-05-01"}]
    assert [r["date"] for r in h.apply_mode(rows, "recent", {}, 2023)] == ["2023-05-01"]
