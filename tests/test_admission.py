"""Tests for the station-record admission rule and the siting form -- pure Python, no network.

The Ahmedabad fixtures use the airport's annual record-minus-record gaps from the hand-applied
rule's own worked example (GSOD minus METAR daily max: +1.03 C in 2024, +1.53 C in 2025) and the
third-record comparison of the 2026-10-01 level investigation (against Gandhinagar 42654: GSOD
-0.11 / -0.10 C, METAR -1.21 / -1.69 C in 2024 / 2025). The daily series are synthetic, shaped to
those annual gaps; the real records are run through scripts/admit_records.py.
"""
import json
import math
import os
import random
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import gsod_homogeneity as gh
from heatready_downscaling import admission as ad
from heatready_downscaling import siting

SERVING_YEAR = 2026  # window 2024-2025


def _base(d: date) -> float:
    return 30.0 + 6.0 * math.sin(2 * math.pi * (d.timetuple().tm_yday - 100) / 365.25)


def _days(first_year, last_year, offset_by_year, keep=lambda d: True):
    out, d = {}, date(first_year, 1, 1)
    while d <= date(last_year, 12, 31):
        if keep(d):
            out[d.isoformat()] = round(_base(d) + offset_by_year(d.year), 4)
        d += timedelta(days=1)
    return out


def _rec(rid, station, days, lat=23.07, lon=72.63, kind="gsod", freq="daily", tested=True, flagged=False,
         first_kept=None, siting_form=None):
    return ad.Record(rid, ad.sha256_of_days(days), station, lat, lon, kind, freq, days,
                     ad.BreakResult(tested=tested, flagged=flagged, first_kept_year=first_kept), siting_form)


def _form(rid, **over):
    f = {"form_version": 1, "record_id": rid, "latitude": 23.07, "longitude": 72.63, "shielded": True,
         "height_m": 1.5, "surface": "grass", "placement": "outdoor", "units": "C", "time_basis": "local_day",
         "instrument_changes": []}
    f.update(over)
    return f


# Ahmedabad airport: GSOD, and METAR daily max sitting below it by the E46 gaps.
GSOD_OFFSET = {2024: 0.0, 2025: 0.0}
METAR_OFFSET = {2024: -1.03, 2025: -1.53}


def _airport_pair():
    gsod = _rec("gsod-42647", "42647", _days(2016, 2025, lambda y: GSOD_OFFSET.get(y, 0.0)), kind="gsod")
    metar = _rec("metar-VAAH", "VAAH", _days(2016, 2025, lambda y: METAR_OFFSET.get(y, 0.0)), kind="metar",
                 freq="hourly")
    return gsod, metar


def _gandhinagar(**kw):
    # The third record sits where GSOD sits in the window and METAR sits 1.1-1.6 C below.
    return _rec("synop-42654", "42654", _days(2016, 2025, lambda y: 0.1 if y in (2024, 2025) else 0.0),
                lat=23.22, lon=72.68, kind="synop", freq="synoptic", **kw)


# ---- single-record rules -------------------------------------------------------------------

def test_clean_single_record_is_admitted_as_chosen():
    r = ad.admit("loc", [_rec("a", "s1", _days(2016, 2025, lambda y: 0.0))], SERVING_YEAR)
    v = r.verdicts[0]
    assert (v.verdict, v.rule, v.role) == ("admitted", "R3_choice", "chosen")
    assert r.window == (2024, 2025) and r.chosen_record_id == "a"
    assert "siting_assumed_standard" in v.caveats


def test_days_after_the_window_are_never_read():
    days = _days(2016, 2025, lambda y: 0.0)
    clean = ad.season_means(_rec("a", "s1", dict(days)), ad.anchor_window(SERVING_YEAR))
    days["2026-03-01"] = 99.0
    for d in ("2026-03-02", "2026-03-03", "2026-03-04"):
        days[d] = 99.0
    for d in ("2023-03-01", "2023-03-02", "2023-03-03"):  # before the window too
        days[d] = 99.0
    dirty = ad.season_means(_rec("a", "s1", days), ad.anchor_window(SERVING_YEAR))
    assert dirty == clean


def test_timestamped_day_keys_are_read_by_their_date():
    days = {f"{k}T00:00": v for k, v in _days(2016, 2025, lambda y: 0.0).items()}
    assert min(ad.season_coverage(_rec("a", "s1", days), (2024, 2025)).values()) == 1.0


def test_duplicate_record_ids_are_refused():
    r = _rec("d", "s1", _days(2016, 2025, lambda y: 0.0))
    with pytest.raises(ValueError):
        ad.admit("loc", [r, _rec("d", "s2", _days(2016, 2025, lambda y: 0.0))], SERVING_YEAR)


def test_siting_label_needs_a_conflict_that_involves_an_eligible_record():
    a = _rec("a", "s1", _days(2016, 2025, lambda y: 1.0), kind="contributed", siting_form=_form("a"))
    i1 = _rec("i1", "s2", _days(2016, 2025, lambda y: 0.6), kind="contributed",
              siting_form=_form("i1", placement="indoor"))
    i2 = _rec("i2", "s3", _days(2016, 2025, lambda y: 1.5), kind="contributed",
              siting_form=_form("i2", placement="indoor"))
    assert ad.admit("loc", [a, i1, i2], SERVING_YEAR).verdicts[0].rule == "R3_choice"


def test_coverage_below_70_percent_in_any_season_rejects():
    days = _days(2016, 2025, lambda y: 0.0, keep=lambda d: not (d.month in (6, 7, 8, 9) and d.day % 2 == 0
                                                                 and d.day % 3 != 0))
    v = ad.admit("loc", [_rec("a", "s1", days)], SERVING_YEAR).verdicts[0]
    assert v.verdict == "rejected" and v.rule == "R1_coverage" and "JJAS" in v.evidence["coverage_short"]


def test_75_percent_coverage_passes():
    v = ad.admit("loc", [_rec("a", "s1", _days(2016, 2025, lambda y: 0.0, lambda d: d.toordinal() % 4 != 0))],
                 SERVING_YEAR).verdicts[0]
    assert all(0.70 <= c < 0.80 for c in v.evidence["coverage"].values()), v.evidence["coverage"]
    assert v.verdict == "admitted"


def test_homogeneity_flag_rejects_and_untested_is_admitted_with_caveat():
    days = _days(2016, 2025, lambda y: 0.0)
    flagged = ad.admit("loc", [_rec("a", "s1", days, flagged=True, first_kept=2023)], SERVING_YEAR).verdicts[0]
    assert (flagged.verdict, flagged.rule) == ("rejected", "R1_homogeneity_break")
    untested = ad.admit("loc", [_rec("a", "s1", days, tested=False)], SERVING_YEAR).verdicts[0]
    assert untested.verdict == "admitted" and "homogeneity_untestable" in untested.caveats


def test_no_eligible_record_means_no_chosen_record():
    r = ad.admit("loc", [_rec("a", "s1", _days(2016, 2025, lambda y: 0.0), flagged=True, first_kept=2023)],
                 SERVING_YEAR)
    assert r.chosen_record_id is None


# ---- choice rule --------------------------------------------------------------------------

def test_choice_longest_span_then_frequency():
    long = _rec("long", "s1", _days(2016, 2025, lambda y: 0.0), freq="daily")
    short = _rec("short", "s2", _days(2020, 2025, lambda y: 0.0), freq="hourly")
    assert ad.admit("loc", [long, short], SERVING_YEAR).chosen_record_id == "long"
    tie = _rec("tie", "s3", _days(2016, 2025, lambda y: 0.0), freq="hourly")
    r = ad.admit("loc", [long, tie], SERVING_YEAR)
    assert r.chosen_record_id == "tie"
    roles = {v.record_id: v.role for v in r.verdicts}
    assert roles == {"long": "corroborating", "tie": "chosen"}


def test_instrument_change_before_window_shortens_the_span():
    a = _rec("a", "s1", _days(2016, 2025, lambda y: 0.0), siting_form=_form("a", instrument_changes=[
        {"date": "2022-05-01", "change": "sensor"}]), kind="contributed")
    b = _rec("b", "s2", _days(2019, 2025, lambda y: 0.0))
    r = ad.admit("loc", [a, b], SERVING_YEAR)
    assert r.chosen_record_id == "b"  # a's span starts 2022-05-01, b's at 2019-01-01


# ---- siting --------------------------------------------------------------------------------

@pytest.mark.parametrize("over, rule", [
    ({"placement": "indoor"}, "R0_indoor_lived_exposure_target"),
    ({"placement": "semi_outdoor"}, "R0_semi_outdoor_not_admissible"),
    ({"shielded": False}, "R0_unshielded_needs_bias_model"),
    ({"instrument_changes": [{"date": "2025-03-01", "change": "relocation"}]}, "R0_instrument_change_in_window"),
])
def test_siting_rejections(over, rule):
    rec = _rec("a", "s1", _days(2016, 2025, lambda y: 0.0), kind="contributed", siting_form=_form("a", **over))
    assert ad.admit("loc", [rec], SERVING_YEAR).verdicts[0].rule == rule


def test_contributed_record_needs_a_valid_form():
    days = _days(2016, 2025, lambda y: 0.0)
    assert ad.admit("loc", [_rec("a", "s1", days, kind="contributed")], SERVING_YEAR).verdicts[0].rule == \
        "R0_siting_form_missing"
    bad = _form("a")
    del bad["shielded"]
    assert ad.admit("loc", [_rec("a", "s1", days, kind="contributed", siting_form=bad)],
                    SERVING_YEAR).verdicts[0].rule == "R0_siting_form_invalid"


def test_siting_form_resolves_a_disagreement():
    ok = _rec("ok", "s1", _days(2016, 2025, lambda y: 0.0), kind="contributed", siting_form=_form("ok"))
    hot_indoors = _rec("indoor", "s2", _days(2016, 2025, lambda y: 3.0), kind="contributed",
                       siting_form=_form("indoor", placement="indoor"))
    r = ad.admit("loc", [ok, hot_indoors], SERVING_YEAR)
    by = {v.record_id: v for v in r.verdicts}
    assert (by["ok"].verdict, by["ok"].rule) == ("admitted", "R4_resolved_by_siting")
    assert by["indoor"].verdict == "rejected" and by["indoor"].rule == "R0_indoor_lived_exposure_target"
    assert by["ok"].evidence["conflicts_before_siting"]


def test_siting_rejection_without_a_disagreement_is_not_labelled_a_resolution():
    ok = _rec("ok", "s1", _days(2016, 2025, lambda y: 0.0), kind="contributed", siting_form=_form("ok"))
    same = _rec("indoor", "s2", _days(2016, 2025, lambda y: 0.1), kind="contributed",
                siting_form=_form("indoor", placement="indoor"))
    assert ad.admit("loc", [ok, same], SERVING_YEAR).verdicts[0].rule == "R3_choice"


# ---- the E46 stations: the original rule's verdicts ---------------------------------------

def test_airport_pair_is_unresolved_by_the_disagreement_stop_as_the_hand_rule_found():
    gsod, metar = _airport_pair()
    r = ad.admit("ahmedabad", [gsod, metar], SERVING_YEAR)
    assert r.chosen_record_id is None
    assert [(v.verdict, v.rule) for v in r.verdicts] == [("unresolved", "R4_disagreement")] * 2
    gaps = r.verdicts[0].evidence["conflicts"][0][2]
    assert all(g > ad.DISAGREEMENT_C for g in gaps.values())
    assert gaps["JJAS"] == pytest.approx(1.28, abs=0.01)  # mean of the 2024 and 2025 gaps


# ---- the new rule: a third record resolves it, with no tuning ------------------------------

def test_with_gandhinagar_present_gsod_is_admitted_and_metar_contradicted():
    gsod, metar = _airport_pair()
    r = ad.admit("ahmedabad", [gsod, metar], SERVING_YEAR, resolvers=[_gandhinagar()])
    by = {v.record_id: v for v in r.verdicts}
    assert (by["gsod-42647"].verdict, by["gsod-42647"].rule, by["gsod-42647"].role) == (
        "admitted", "R4_resolved_by_third_record", "chosen")
    assert (by["metar-VAAH"].verdict, by["metar-VAAH"].rule) == ("rejected", "R4_contradicted_by_third_record")
    per = by["gsod-42647"].evidence["per_record_vs_third_record"]
    assert per["gsod-42647"]["status"] == "supported" and per["metar-VAAH"]["status"] == "contradicted"
    assert r.chosen_record_id == "gsod-42647"


def test_the_third_record_rule_has_no_per_record_input_that_could_favour_a_station():
    # Swapping which station is called GSOD or METAR, and their order, cannot change the verdict.
    gsod, metar = _airport_pair()
    a = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[_gandhinagar()])
    b = ad.admit("x", [metar, gsod], SERVING_YEAR, resolvers=[_gandhinagar()])
    assert {v.record_id: v.verdict for v in a.verdicts} == {v.record_id: v.verdict for v in b.verdicts}


def test_a_resolver_needs_paired_days_not_the_coverage_floor():
    # Gandhinagar reports its 24 h max on roughly half the days; that is enough to compare.
    gsod, metar = _airport_pair()
    half = _gandhinagar()
    half.days = {d: v for d, v in half.days.items() if int(d[8:10]) % 2 == 0}
    assert min(ad.season_coverage(half, (2024, 2025)).values()) < ad.MIN_SEASON_COVERAGE
    r = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[half])
    assert [v.verdict for v in r.verdicts] == ["admitted", "rejected"]


def test_a_resolver_with_too_few_paired_days_leaves_both_unresolved():
    gsod, metar = _airport_pair()
    few = _gandhinagar()
    few.days = {d: v for d, v in few.days.items() if not ("2024-01-01" <= d <= "2025-12-31" and d[8:10] != "01")}
    r = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[few])
    assert [v.verdict for v in r.verdicts] == ["unresolved", "unresolved"]


def test_a_record_failing_only_coverage_still_blocks_the_other_from_being_admitted_alone():
    # The 2026 real-data case: the GSOD record ends mid-2025 (ONDJF coverage 68%), METAR is complete.
    gsod, metar = _airport_pair()
    gsod.days = {d: v for d, v in gsod.days.items() if d <= "2025-08-24"}
    assert ad.season_coverage(gsod, (2024, 2025))["ONDJF"] < ad.MIN_SEASON_COVERAGE <= \
        min(ad.season_coverage(gsod, (2024, 2025))["MAM"], ad.season_coverage(gsod, (2024, 2025))["JJAS"])
    r = ad.admit("ahmedabad", [gsod, metar], SERVING_YEAR)
    by = {v.record_id: v for v in r.verdicts}
    assert r.chosen_record_id is None
    assert (by["gsod-42647"].verdict, by["gsod-42647"].rule) == ("rejected", "R1_coverage")
    assert (by["metar-VAAH"].verdict, by["metar-VAAH"].rule) == ("unresolved", "R4_disagreement")
    assert by["gsod-42647"].evidence["conflicts"]


def test_a_coverage_failed_record_supported_by_a_third_record_is_still_not_admitted():
    gsod, metar = _airport_pair()
    gsod.days = {d: v for d, v in gsod.days.items() if d <= "2025-08-24"}
    r = ad.admit("ahmedabad", [gsod, metar], SERVING_YEAR, resolvers=[_gandhinagar()])
    by = {v.record_id: v for v in r.verdicts}
    assert r.chosen_record_id is None
    assert (by["gsod-42647"].verdict, by["gsod-42647"].rule) == ("rejected", "R1_coverage")
    assert (by["metar-VAAH"].verdict, by["metar-VAAH"].rule) == ("rejected", "R4_contradicted_by_third_record")
    ev = by["gsod-42647"].evidence["per_record_vs_third_record"]["gsod-42647"]
    assert ev["status"] == "supported" and ev["below_coverage_floor"]


def test_a_coverage_failed_record_that_agrees_does_not_block_admission():
    a = _rec("a", "s1", _days(2016, 2025, lambda y: 0.0))
    b = _rec("b", "s2", _days(2016, 2025, lambda y: 0.1, keep=lambda d: d.toordinal() % 2 == 0 or d.year < 2024))
    r = ad.admit("loc", [a, b], SERVING_YEAR)
    by = {v.record_id: v for v in r.verdicts}
    assert by["a"].verdict == "admitted" and by["b"].rule == "R1_coverage"


def test_a_flagged_resolver_is_not_used():
    gsod, metar = _airport_pair()
    r = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[_gandhinagar(flagged=True, first_kept=2022)])
    assert [v.verdict for v in r.verdicts] == ["unresolved", "unresolved"]


def test_a_resolver_at_the_same_station_or_too_far_is_not_used():
    gsod, metar = _airport_pair()
    same = _gandhinagar()
    same.station_id = "VAAH"
    far = _gandhinagar()
    far.lat, far.lon = 24.5, 73.7
    for t, why in ((same, "same station"), (far, "farther")):
        r = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[t])
        assert [v.verdict for v in r.verdicts] == ["unresolved", "unresolved"]
        assert why in r.verdicts[0].evidence["resolvers_considered"][0]["unused"]


def test_a_third_record_that_supports_neither_leaves_both_unresolved():
    gsod, metar = _airport_pair()
    odd = _rec("odd", "99999", _days(2016, 2025, lambda y: 4.0 if y in (2024, 2025) else 0.0), lat=23.2, lon=72.7)
    r = ad.admit("x", [gsod, metar], SERVING_YEAR, resolvers=[odd])
    assert [(v.verdict, v.rule) for v in r.verdicts] == [("unresolved", "R4_disagreement")] * 2


def test_too_few_paired_days_is_no_evidence():
    # Two records each above the coverage floor overlap on at least 40% of days, so this floor binds
    # only on short windows; the comparison itself must still refuse thin overlap.
    gsod, _ = _airport_pair()
    sparse = _gandhinagar()
    sparse.days = {d: v for d, v in sparse.days.items() if not ("2024-01-01" <= d <= "2025-12-31"
                                                                 and d[8:10] != "01")}
    gaps = ad._paired_gap(gsod, sparse, (2024, 2025))
    assert all(g is None for g in gaps.values())


# ---- output and the real break test ---------------------------------------------------------

def test_log_result_writes_one_line_per_record_with_id_and_sha(tmp_path):
    gsod, metar = _airport_pair()
    r = ad.admit("ahmedabad", [gsod, metar], SERVING_YEAR, resolvers=[_gandhinagar()])
    path = tmp_path / "verdicts.jsonl"
    ad.log_result(r, path)
    ad.log_result(r, path)  # append-only
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 4
    assert {(x["record_id"], x["sha256"], x["verdict"]) for x in lines[:2]} == {
        ("gsod-42647", gsod.sha256, "admitted"), ("metar-VAAH", metar.sha256, "rejected")}
    assert all(x["rule_version"] == ad.RULE_VERSION and x["window"] == [2024, 2025] for x in lines)


def _network_rows(local_break):
    rng = random.Random(1)
    rows = []
    for i in range(6):
        sid = f"S{i}"
        common = {y: rng.gauss(0, 0.15) for y in range(2016, 2026)}
        for y, v in common.items():
            off = 1.5 + v - (local_break if (i == 0 and y >= 2023) else 0.0)
            for m in (4, 5, 6, 7, 8, 9):
                for d in range(1, 29):
                    rows.append({"station_id": sid, "lat": 22.0 + 0.3 * i, "lon": 72.0 + 0.3 * i,
                                 "date": f"{y}-{m:02d}-{d:02d}", "delta_tmax_c": off + rng.gauss(0, 1.0)})
    return rows


def test_break_result_reads_the_real_screen_output_and_flows_into_the_verdict():
    tests, first_kept = gh.screen(_network_rows(local_break=1.7))
    broken = ad.break_result_from_screen(tests, first_kept, "S0")
    clean = ad.break_result_from_screen(tests, first_kept, "S1")
    assert broken.flagged and broken.first_kept_year == 2023 and broken.tested and broken.evidence
    assert clean.tested and not clean.flagged
    days = _days(2016, 2025, lambda y: 0.0)
    r = ad.admit("loc", [_rec("s0", "S0", days), _rec("s1", "S1", days)], SERVING_YEAR)
    assert {v.verdict for v in r.verdicts} == {"admitted"}  # both clean until the screen result is attached
    s0 = _rec("s0", "S0", days)
    s0.break_result = broken
    s1 = _rec("s1", "S1", days)
    s1.break_result = clean
    by = {v.record_id: v for v in ad.admit("loc", [s0, s1], SERVING_YEAR).verdicts}
    assert (by["s0"].verdict, by["s0"].rule) == ("rejected", "R1_homogeneity_break")
    assert by["s1"].verdict == "admitted"


# ---- siting-form schema ---------------------------------------------------------------------

def test_valid_form_and_minimum_contribution():
    assert siting.validate_siting_form(_form("a")) == []
    contribution = {"record_id": "a", "latitude": 23.07, "longitude": 72.63,
                    "daily": [{"date": "2025-01-01", "tmax": 28.1, "tmin": 12.0}], "siting_form": _form("a")}
    assert siting.validate_contribution(contribution) == []


@pytest.mark.parametrize("field", ["shielded", "height_m", "surface", "placement", "units", "time_basis",
                                   "instrument_changes", "latitude", "longitude", "record_id"])
def test_every_siting_field_is_required(field):
    f = _form("a")
    del f[field]
    assert any(field in p for p in siting.validate_siting_form(f))


def test_form_rejects_bad_values():
    assert siting.validate_siting_form(_form("a", units="Rankine"))
    assert siting.validate_siting_form(_form("a", latitude=123.0))
    assert siting.validate_siting_form(_form("a", height_m=0))
    assert siting.validate_siting_form(_form("a", surface="lava"))
    assert siting.validate_siting_form(_form("a", instrument_changes=[{"date": "2024-02-30", "change": "sensor"}]))
    assert siting.validate_siting_form(_form("a", unknown_field=1))


def test_contribution_needs_daily_values_and_a_consistent_form():
    base = {"record_id": "a", "latitude": 23.07, "longitude": 72.63,
            "daily": [{"date": "2025-01-01", "tmax": 28.1}], "siting_form": _form("a")}
    assert siting.validate_contribution(base) == []
    assert siting.validate_contribution({**base, "daily": []})
    assert siting.validate_contribution({**base, "daily": [{"date": "2025-01-01"}]})
    assert any("differs" in p for p in siting.validate_contribution({**base, "latitude": 10.0}))
    no_form = {k: v for k, v in base.items() if k != "siting_form"}
    assert siting.validate_contribution(no_form)
