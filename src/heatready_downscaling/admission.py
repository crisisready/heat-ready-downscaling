"""
The station-record admission rule: one rule every station record passes before it can touch the model.

Generalises research/ahmedabad-bsh-refinement/levelfix/ANCHOR_SOURCE_RULE.md (heat-risk-data-api,
rules 1-6, written 2026-09-30 and applied by hand) into code, per the 2026-10-01 global-model
development plan (D3, chunk C2). It consumes the existing homogeneity screen's output
(scripts/gsod_homogeneity.py, unchanged) rather than reimplementing it: the caller runs
`gsod_homogeneity.screen` over the record's network and hands the result in as a `BreakResult`
(see `break_result_from_screen`). scripts/admit_records.py is that wiring.

Rule, per location (a group of records describing the same place), for a serving year Y:

  Window     The anchor window is the two complete calendar years before Y. Records are never
             read past it.
  R0 siting  A contributed record needs a valid siting form (siting.py); open-network records
             (GSOD, GHCN, SYNOP, METAR) are assumed standard. Indoor, unshielded and semi-outdoor
             records are rejected (routed to the lived-exposure target and the bias model, which
             run elsewhere). An instrument change inside the window rejects the record; one in
             the 3 years before the window starts its homogeneous span at the change.
  R1 break   The record must not be flagged by the homogeneity break test, run over every row up
             to the end of the window (never later). The hand-applied rule said "window plus the 3
             preceding years", but that is 5 years and the screen needs 6 per station-season, so it
             could test nothing; the full earlier record is used instead.
  R1 cover   At least 70% day coverage in each of MAM, JJAS and ONDJF over the window.
  R3 choice  Of the eligible records, the one with the longest homogeneous span ending in the
             window is chosen; a tie goes to the higher-frequency record (hourly, then synoptic,
             then daily).
  R4 stop    If two records differ by more than 0.5 C in any season's mean over the
             window, nobody is admitted yet: "record unresolved", served at the model's own level.
             A record that fails only the coverage floor cannot be chosen but still counts here as a
             witness (decided 2026-10-01 after the first real-data run, where the airport GSOD record
             ended mid-2025 and its coverage failure would otherwise have left METAR admitted alone).
             NEW (plan D3): the disagreement is resolved, and nothing else resolves it, by
               (a) siting: when the siting form or instrument history disqualifies all but the
                   records that agree (R0 above) the survivors are admitted; or
               (b) a third independent record: a different station within RESOLVER_MAX_KM that
                   itself passes R0-R1 (no coverage floor: it needs MIN_PAIRED_DAYS_PER_SEASON same-day
             pairs in every season instead), compared with each conflicting record on same-day pairs
                   over the window. A conflicting record is supported if its mean difference from
                   the third record is within 0.5 C in every season, contradicted if not. When
                   every conflicting record is supported or contradicted, at least one is
                   supported and the supported ones agree with each other, the supported ones that
                   pass the coverage floor are admitted and the contradicted are rejected. Otherwise all stay unresolved.
  R6 judging Loggers never choose or judge anything. The module has no input for them.

Every threshold below is the one the hand-applied rule used, inherited rather than chosen for
any particular record. The module never reads a model output, so no verdict can depend on how a
candidate scores.

Output: one `Verdict` per record (admitted / unresolved / rejected, the deciding rule, the
evidence), wrapped in an `AdmissionResult`, and `log_result` appends them as JSON lines with the
record id and sha256 so the level_anchor job and the pooled residual layer can consume them.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from heatready_downscaling.siting import OFFICIAL_SOURCE_KINDS, validate_siting_form

RULE_VERSION = "2026-10-01.1"

# Calendar-month assignment of the three anchor seasons (the same ones the per-season offset uses).
ANCHOR_SEASONS = {"MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ONDJF": (10, 11, 12, 1, 2)}
WINDOW_YEARS = 2
BREAK_TEST_PRECEDING_YEARS = 3
MIN_SEASON_COVERAGE = 0.70
DISAGREEMENT_C = 0.5
RESOLVER_TOLERANCE_C = DISAGREEMENT_C
RESOLVER_MAX_KM = 50.0
# Fewest same-day pairs per season for a third-record comparison to count as evidence; the same
# figure gsod_homogeneity uses as its minimum days per season-year (MIN_DAYS_PER_SEASON_YEAR).
MIN_PAIRED_DAYS_PER_SEASON = 45
FREQUENCY_RANK = {"daily": 1, "synoptic": 2, "hourly": 3}

ADMITTED, UNRESOLVED, REJECTED = "admitted", "unresolved", "rejected"


@dataclass
class BreakResult:
    """What the homogeneity screen said about one record."""
    tested: bool
    flagged: bool = False
    first_kept_year: int | None = None
    evidence: list = field(default_factory=list)


@dataclass
class Record:
    record_id: str
    sha256: str
    station_id: str
    lat: float
    lon: float
    source_kind: str  # gsod, ghcn, synop, metar, or "contributed"
    frequency: str  # hourly, synoptic, daily
    days: dict  # ISO date -> daily tmax in C (None values are ignored)
    break_result: BreakResult
    siting: dict | None = None


@dataclass
class Verdict:
    record_id: str
    sha256: str
    verdict: str
    rule: str
    role: str | None = None  # "chosen" or "corroborating" when admitted
    evidence: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)


@dataclass
class AdmissionResult:
    location: str
    serving_year: int
    window: tuple
    rule_version: str
    chosen_record_id: str | None
    verdicts: list


def break_result_from_screen(tests, first_kept, station_id) -> BreakResult:
    """Read one station's outcome from `gsod_homogeneity.screen(rows)` -> (tests, first_kept)."""
    mine = [t for t in tests if t["station_id"] == station_id]
    return BreakResult(
        tested=any(t.get("tested") for t in mine),
        flagged=station_id in first_kept,
        first_kept_year=first_kept.get(station_id),
        evidence=[{k: t.get(k) for k in ("season", "break_year", "shift_c", "p", "q", "passes", "flagged")}
                  for t in mine if t.get("tested")],
    )


def anchor_window(serving_year: int) -> tuple:
    return (serving_year - WINDOW_YEARS, serving_year - 1)


def _km(a_lat, a_lon, b_lat, b_lon):
    la1, lo1, la2, lo2 = map(math.radians, (a_lat, a_lon, b_lat, b_lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _season_of(month):
    for name, months in ANCHOR_SEASONS.items():
        if month in months:
            return name
    return None


def _season_day_counts(window):
    counts = {s: 0 for s in ANCHOR_SEASONS}
    d, end = date(window[0], 1, 1), date(window[1], 12, 31)
    while d <= end:
        s = _season_of(d.month)
        if s:
            counts[s] += 1
        d += timedelta(days=1)
    return counts


def _window_values(record, window):
    """{season: {iso date: tmax}} for the days of the record inside the window."""
    out = {s: {} for s in ANCHOR_SEASONS}
    lo, hi = f"{window[0]}-01-01", f"{window[1]}-12-31"
    for d, v in record.days.items():
        d = d[:10]
        if v is None or not lo <= d <= hi:
            continue
        s = _season_of(int(d[5:7]))
        if s:
            out[s][d] = v
    return out


def season_coverage(record, window) -> dict:
    counts = _season_day_counts(window)
    vals = _window_values(record, window)
    return {s: len(vals[s]) / counts[s] for s in ANCHOR_SEASONS}


def season_means(record, window) -> dict:
    vals = _window_values(record, window)
    return {s: (sum(v.values()) / len(v) if len(v) >= MIN_PAIRED_DAYS_PER_SEASON else None)
            for s, v in vals.items()}


def _max_season_gap(a, b, window):
    """Largest absolute difference between two records' season means over the window."""
    ma, mb = season_means(a, window), season_means(b, window)
    gaps = {s: abs(ma[s] - mb[s]) for s in ANCHOR_SEASONS if ma[s] is not None and mb[s] is not None}
    return gaps, (max(gaps.values()) if gaps else None)


def _first_date(record):
    ds = [d for d, v in record.days.items() if v is not None]
    return min(ds) if ds else None


def _siting_check(record, window):
    """(rejection rule or None, evidence, caveats, span-start override ISO date or None)."""
    caveats, evidence = [], {}
    form = record.siting
    if form is None:
        if record.source_kind in OFFICIAL_SOURCE_KINDS:
            return None, {"siting": "assumed_standard_open_network"}, ["siting_assumed_standard"], None
        return "R0_siting_form_missing", {"siting": "missing"}, caveats, None
    problems = validate_siting_form(form)
    if problems:
        return "R0_siting_form_invalid", {"siting_problems": problems}, caveats, None
    evidence["siting"] = {k: form[k] for k in ("shielded", "placement", "height_m", "surface", "time_basis")}
    if form["placement"] == "indoor":
        return "R0_indoor_lived_exposure_target", evidence, caveats, None
    if form["placement"] != "outdoor":
        return "R0_semi_outdoor_not_admissible", evidence, caveats, None
    if not form["shielded"]:
        return "R0_unshielded_needs_bias_model", evidence, caveats, None
    w_lo, w_hi = f"{window[0]}-01-01", f"{window[1]}-12-31"
    pre_lo = f"{window[0] - BREAK_TEST_PRECEDING_YEARS}-01-01"
    changes = sorted(c["date"] for c in form["instrument_changes"])
    in_window = [c for c in changes if w_lo <= c <= w_hi]
    if in_window:
        evidence["instrument_changes_in_window"] = in_window
        return "R0_instrument_change_in_window", evidence, caveats, None
    before = [c for c in changes if pre_lo <= c < w_lo]
    if before:
        evidence["instrument_change_starts_span"] = before[-1]
        return None, evidence, caveats, before[-1]
    return None, evidence, caveats, None


def _span_start(record, override_iso):
    start = _first_date(record)
    if override_iso and start:
        start = max(start, override_iso)
    return start


def _paired_gap(rec, resolver, window):
    """Per season: (mean(rec - resolver) over same-day pairs, n pairs) or None when too few pairs."""
    a, b = _window_values(rec, window), _window_values(resolver, window)
    out = {}
    for s in ANCHOR_SEASONS:
        common = sorted(set(a[s]) & set(b[s]))
        out[s] = ((sum(a[s][d] - b[s][d] for d in common) / len(common), len(common))
                  if len(common) >= MIN_PAIRED_DAYS_PER_SEASON else None)
    return out


def _eligibility(record, window, check_coverage=True):
    """R0, R1 break, R1 coverage (skipped for a third record, which only needs paired days). Returns (Verdict if rejected else None, evidence, caveats, span start)."""
    ev = {"record_id": record.record_id}
    rej_rule, s_ev, caveats, override = _siting_check(record, window)
    ev.update(s_ev)
    br = record.break_result
    ev["homogeneity"] = {"tested": br.tested, "flagged": br.flagged, "first_kept_year": br.first_kept_year,
                         "tests": br.evidence}
    cov = season_coverage(record, window)
    ev["coverage"] = {s: round(c, 3) for s, c in cov.items()}
    ev["season_means_c"] = {s: (None if m is None else round(m, 3)) for s, m in season_means(record, window).items()}
    if not br.tested:
        caveats = caveats + ["homogeneity_untestable"]
    if rej_rule:
        return Verdict(record.record_id, record.sha256, REJECTED, rej_rule, evidence=ev, caveats=caveats), ev, caveats, None
    if br.flagged:
        return Verdict(record.record_id, record.sha256, REJECTED, "R1_homogeneity_break", evidence=ev,
                       caveats=caveats), ev, caveats, None
    low = {s: round(c, 3) for s, c in cov.items() if c < MIN_SEASON_COVERAGE}
    if low and check_coverage:
        ev["coverage_short"] = low
        return Verdict(record.record_id, record.sha256, REJECTED, "R1_coverage", evidence=ev,
                       caveats=caveats), ev, caveats, None
    return None, ev, caveats, _span_start(record, override)


def _conflicts(recs, window):
    """Pairs of records whose season means differ by more than DISAGREEMENT_C, with the gaps."""
    out = []
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            gaps, worst = _max_season_gap(a, b, window)
            if worst is not None and worst > DISAGREEMENT_C:
                out.append((a.record_id, b.record_id, {s: round(g, 3) for s, g in gaps.items()}))
    return out


def _choose(recs, spans):
    return min(recs, key=lambda r: (spans[r.record_id], -FREQUENCY_RANK[r.frequency], r.record_id))


def _only_coverage_failed(verdict):
    return verdict is not None and verdict.rule == "R1_coverage"


def admit(location, records, serving_year, resolvers=()) -> AdmissionResult:
    """Run the admission rule over the records of one location. `resolvers` are candidate third
    records (other stations nearby); they are screened here and never receive a verdict."""
    window = anchor_window(serving_year)
    ids = [r.record_id for r in list(records) + list(resolvers)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate record_id among records and resolvers: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    verdicts, eligible, witnesses, spans, evid, cav = {}, [], [], {}, {}, {}
    pre_siting_pool = []
    for r in records:
        v, ev, caveats, span = _eligibility(r, window)
        evid[r.record_id], cav[r.record_id] = ev, caveats
        if v:
            verdicts[r.record_id] = v
            if _only_coverage_failed(v):
                witnesses.append(r)  # cannot be chosen, still counts in the disagreement stop
            # A record that failed only on siting still counts as having been in disagreement.
            elif v.rule.startswith("R0_") and not br_or_cov_fail(r, window):
                pre_siting_pool.append(r)
        else:
            eligible.append(r)
            spans[r.record_id] = span
            pre_siting_pool.append(r)

    def admit_group(group, rule, extra):
        chosen = _choose(group, {r.record_id: spans[r.record_id] for r in group})
        for r in group:
            verdicts[r.record_id] = Verdict(
                r.record_id, r.sha256, ADMITTED, rule, role="chosen" if r is chosen else "corroborating",
                evidence={**evid[r.record_id], "span_start": spans[r.record_id], **extra}, caveats=cav[r.record_id])
        return chosen

    chosen = None
    conflicts = _conflicts(eligible + witnesses, window)
    if eligible and not conflicts:
        eligible_ids = {r.record_id for r in eligible}
        before = [c for c in _conflicts(pre_siting_pool, window) if eligible_ids & {c[0], c[1]}]
        chosen = admit_group(eligible, "R4_resolved_by_siting" if before else "R3_choice",
                             {"conflicts_before_siting": before} if before else {})
    elif conflicts:
        chosen = _resolve_with_third_record(eligible, witnesses, conflicts, resolvers, window, verdicts, evid, cav,
                                            admit_group)
    return AdmissionResult(location, serving_year, window, RULE_VERSION, chosen.record_id if chosen else None,
                           [verdicts[r.record_id] for r in records])


def br_or_cov_fail(record, window):
    """True when the record fails the homogeneity or coverage test regardless of siting."""
    cov = season_coverage(record, window)
    return record.break_result.flagged or any(c < MIN_SEASON_COVERAGE for c in cov.values())


def _resolve_with_third_record(eligible, witnesses, conflicts, resolvers, window, verdicts, evid, cav, admit_group):
    in_conflict = sorted({rid for a, b, _ in conflicts for rid in (a, b)})
    pool = eligible + witnesses
    conf_recs = [r for r in pool if r.record_id in in_conflict]
    clear = [r for r in eligible if r.record_id not in in_conflict]
    third_ok, third_ev = [], []
    for t in resolvers:
        info = {"resolver": t.record_id, "station_id": t.station_id}
        if any(t.station_id == r.station_id for r in conf_recs):
            info["unused"] = "same station as a conflicting record"
        else:
            km = min(_km(t.lat, t.lon, r.lat, r.lon) for r in conf_recs)
            info["km"] = round(km, 1)
            tv, _, _, _ = _eligibility(t, window, check_coverage=False)
            if km > RESOLVER_MAX_KM:
                info["unused"] = f"farther than {RESOLVER_MAX_KM:g} km"
            elif tv:
                info["unused"] = f"resolver fails {tv.rule}"
            else:
                third_ok.append(t)
        third_ev.append(info)

    status = {}
    for r in conf_recs:
        per_resolver = {}
        for t in third_ok:
            gap = _paired_gap(r, t, window)
            per_resolver[t.record_id] = {s: (None if g is None else {"mean_diff_c": round(g[0], 3), "n": g[1]})
                                         for s, g in gap.items()}
        cells = [g for p in per_resolver.values() for g in p.values()]
        if cells and all(g is not None for g in cells):
            verdict_for = "contradicted" if any(abs(g["mean_diff_c"]) > RESOLVER_TOLERANCE_C for g in cells) \
                else "supported"
        else:
            verdict_for = "no_evidence"
        status[r.record_id] = {"status": verdict_for, "vs_resolvers": per_resolver,
                               "below_coverage_floor": r in witnesses}

    supported = [r for r in conf_recs if status[r.record_id]["status"] == "supported"]
    decisive = all(status[r.record_id]["status"] in ("supported", "contradicted") for r in conf_recs)
    agree = not supported or not _conflicts(supported, window)
    extra = {"conflicts": conflicts, "resolvers_considered": third_ev, "per_record_vs_third_record": status}
    for w in witnesses:  # a coverage-failed record keeps its rejection and carries the disagreement evidence
        verdicts[w.record_id].evidence.update({k: v for k, v in extra.items() if w.record_id in in_conflict})
    if decisive and supported and agree:
        for r in conf_recs:
            if status[r.record_id]["status"] == "contradicted" and r in eligible:
                verdicts[r.record_id] = Verdict(r.record_id, r.sha256, REJECTED, "R4_contradicted_by_third_record",
                                                evidence={**evid[r.record_id], **extra}, caveats=cav[r.record_id])
        group = [r for r in supported if r in eligible] + clear
        return admit_group(group, "R4_resolved_by_third_record", extra) if group else None
    for r in conf_recs + clear:
        if r in eligible:
            verdicts[r.record_id] = Verdict(r.record_id, r.sha256, UNRESOLVED, "R4_disagreement",
                                            evidence={**evid[r.record_id], **extra}, caveats=cav[r.record_id])
    return None


def result_to_dict(result: AdmissionResult) -> dict:
    return {"location": result.location, "serving_year": result.serving_year, "window": list(result.window),
            "rule_version": result.rule_version, "chosen_record_id": result.chosen_record_id,
            "verdicts": [v.__dict__ for v in result.verdicts]}


def log_result(result: AdmissionResult, path) -> None:
    """Append one JSON line per verdict, keyed by record id and sha256."""
    with open(path, "a") as f:
        for v in result.verdicts:
            f.write(json.dumps({"location": result.location, "serving_year": result.serving_year,
                                "window": list(result.window), "rule_version": result.rule_version,
                                "record_id": v.record_id, "sha256": v.sha256, "verdict": v.verdict, "rule": v.rule,
                                "role": v.role, "caveats": v.caveats, "evidence": v.evidence},
                               sort_keys=True, default=str) + "\n")


def sha256_of_days(days: dict) -> str:
    """A stable content hash for a record built in memory (file-backed records hash their file)."""
    return hashlib.sha256(json.dumps(sorted(days.items()), default=str).encode()).hexdigest()
