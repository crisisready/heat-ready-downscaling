"""
replay_downscaling.py -- the tier-mix/delta-distribution diff GOVERNANCE.md
requires as the human-in-the-loop promotion-review step, before
promote_from_public.py's merge-only S3 gate write ever runs. See
docs/plan-2026-08-25-crowdsourced-model-improvement-p0.md (PR #23) for the
full design and its own two open-question resolutions.

Read-only: never writes anywhere, never publishes anything -- matching
this repo's "compute offline, publish is a separate deliberate step"
discipline (contract.FrozenPredictionAdapter, qc_sdot_anomalies.py). This
tool informs the human judgment call GOVERNANCE.md requires; it is never
itself a pass/fail gate -- the mechanical pass/fail already happened in
score_forward_eval.py/report.compare_reports.

Usage:
    python scripts/replay_downscaling.py \\
        --snapshot-dir snapshots/v2026.08 --band-key lag_fill \\
        --model-version ds-2026.07-rf5 \\
        --new candidate_gate.json [--old current_gate.json] [--out diff.json]

--old omitted means "no correction, nothing gated" -- the same fail-closed
baseline heat-risk-data-api's own load_band_gate already falls back to
when a band has no gate published yet; the right comparison for lighting
a dark cell for the first time.

--model-version is always required and SHARED by both --old/--new -- an
earlier design sketch let each side independently be "either a
model_version or a gate-json," which doesn't work: a gate JSON carries no
model_version field at all (BAND_GATE_SCHEMA/BLEND_GATE_SCHEMA have none),
so it alone can never say which frozen-predictions partition to replay
against (Codex adversarial review finding, PR #23). Both arms replay the
SAME model's frozen predictions with two different published corrections
on top -- comparing two different MODELS at once is a different, harder
question this tool does not attempt.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from heatready_downscaling import contract, gates, snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replay_downscaling")

_CONFIDENCE_TIERS = ("high", "medium", "low")
_PERCENTILES = (10, 50, 90)
_EARTH_R_KM = 6371.0
_DEFAULT_BOUNDARY_MAX_KM = 50.0


def load_gate(path: str | None) -> dict:
    """None (no --old given) means the fail-closed "nothing published
    yet" baseline -- the same default heat-risk-data-api's own
    load_band_gate already falls back to in production when a band has
    no gate file yet, per contract.py's own docstring."""
    empty = {
        "tmax": {}, "tmin": {},
        "bias_correction": {"tmax": {}, "tmin": {}},
        "delta_scale": {"tmax": {}, "tmin": {}},
    }
    if path is None:
        return empty
    with open(path) as f:
        gate = json.load(f)
    gates.validate_gate(gate)  # never replay against a malformed gate file
    return gate


def subzone_code(station_id) -> str | None:
    """GHCN station_id country prefix -- the same convention gates.py's
    own delta_scale_subzone/bias_correction_subzone fields already use
    (see that module's own docstring on build_subzone_patch)."""
    if not station_id or len(str(station_id)) < 2:
        return None
    return str(station_id)[:2]


def resolve_delta_scale(gate: dict, target: str, zone: str, subzone: str | None) -> dict | None:
    """A delta_scale_subzone[target][zone][subzone] entry, when present,
    REPLACES the flat delta_scale[target][zone] entry wholesale for a
    matching row -- mirrors heat-risk-data-api's own real, live
    manager._apply_subzone_delta_scale_override (Paris/France,
    2026-08-08), the one actual serving-side precedent for this
    precedence. Falls back to the flat entry (or None) otherwise.

    bias_correction_subzone has NO analogous serving-side consumer as of
    this writing (confirmed directly: no _apply_subzone_bias_correction_
    override-shaped function exists anywhere in manager.py) -- it is
    therefore deliberately never applied by this tool either (see
    subzone_status below); doing so would show a "what would change" that
    doesn't match what production actually does today."""
    if subzone is not None:
        sub_entry = gate.get("delta_scale_subzone", {}).get(target, {}).get(zone, {}).get(subzone)
        if sub_entry is not None:
            return sub_entry
    return gate.get("delta_scale", {}).get(target, {}).get(zone)


def correction_keys_for_gate(rows: list[dict], gate: dict, target: str) -> list:
    """One hashable "effective correction" signature per row, resolved
    via resolve_delta_scale -- feeds boundary_discontinuity's
    correction_keys param so a cross-subzone boundary pair is classified
    by whether the correction ACTUALLY differs between two rows, not by
    whether their raw subzone_code strings happen to differ (see that
    function's own docstring)."""
    keys = []
    for r in rows:
        resolved = resolve_delta_scale(gate, target, r.get("climate_zone"), subzone_code(r.get("station_id")))
        keys.append(tuple(sorted(resolved.items())) if resolved is not None else None)
    return keys


def subzone_status(gate: dict, target: str) -> dict:
    """Informational only -- which (zone, subzone) cells this gate
    publishes a delta_scale_subzone entry for (APPLIED by this tool,
    matching real serving), and which publish a bias_correction_subzone
    entry (NOT applied -- no serving-side consumer exists yet). Surfacing
    this distinction explicitly so a human reader never assumes both
    subzone fields are equally live."""
    delta_cells = sorted(
        f"{zone}.{sub}" for zone, by_sub in gate.get("delta_scale_subzone", {}).get(target, {}).items() for sub in by_sub
    )
    bias_cells = sorted(
        f"{zone}.{sub}" for zone, by_sub in gate.get("bias_correction_subzone", {}).get(target, {}).items() for sub in by_sub
    )
    return {
        "delta_scale_subzone_cells_applied": delta_cells,
        "bias_correction_subzone_cells_published_but_not_applied": bias_cells,
    }


def predict_with_gate(adapter, rows: list[dict], target: str, gate: dict) -> list[dict]:
    """One predict() call applying a gate's zone-level extra_zone_gate +
    bias_correction, PLUS a per-row subzone-aware delta_scale resolution
    -- never a second implementation of the prediction path, just the
    subzone patching predict() itself has no concept of at all (its own
    delta_scale parameter is one flat {target: {zone: {...}}} dict for
    the whole batch, with no subzone dimension).

    Rows needing a DIFFERENT value than their own zone's flat one (i.e.
    resolve_delta_scale would return the subzone fit, not the flat one)
    are re-scored in a second, smaller batch with that one zone's flat
    entry swapped to the subzone fit, and spliced back into the base
    result at the matching indices -- cheap here since
    FrozenPredictionAdapter.predict is a dict lookup, not live inference."""
    extra_zone_gate = {target: gate.get(target, {})}
    bias_correction = {target: gate.get("bias_correction", {}).get(target, {})}
    flat_delta_scale = {target: gate.get("delta_scale", {}).get(target, {})}

    base = adapter.predict(
        rows, target, extra_zone_gate=extra_zone_gate, bias_correction=bias_correction,
        delta_scale=flat_delta_scale,
    )

    # Group rows whose resolve_delta_scale() result differs from the FLAT
    # value `base` was already scored with -- resolve_delta_scale is the
    # single source of truth for subzone-vs-flat precedence (code-review
    # finding, PR #25: an earlier draft re-implemented this same decision
    # inline here instead of calling it, a real risk of the two silently
    # drifting apart on a future edit to one but not the other).
    by_cell: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        zone = r.get("climate_zone")
        sc = subzone_code(r.get("station_id"))
        resolved = resolve_delta_scale(gate, target, zone, sc)
        if resolved is not None and resolved != flat_delta_scale[target].get(zone):
            by_cell.setdefault((zone, tuple(sorted(resolved.items()))), []).append(i)

    if not by_cell:
        return base

    results = list(base)
    for (zone, resolved_items), idxs in by_cell.items():
        patched_delta_scale = {target: dict(flat_delta_scale[target])}
        patched_delta_scale[target][zone] = dict(resolved_items)
        sub_rows = [rows[i] for i in idxs]
        sub_preds = adapter.predict(
            sub_rows, target, extra_zone_gate=extra_zone_gate, bias_correction=bias_correction,
            delta_scale=patched_delta_scale,
        )
        for i, p in zip(idxs, sub_preds):
            results[i] = p
    return results


def tier_mix(preds: list[dict]) -> dict:
    """Confidence histogram + applied/cv_gate_passed/out_of_distribution
    rates -- the exact fields contract.ModelAdapter.predict already
    returns per row, aggregated once for one arm (old or new). A shift
    toward higher confidence with no OOD-rate spike is the expected good
    case; a shift where previously-applied rows go not-applied, or the
    OOD rate spikes, is GOVERNANCE.md's own named "genuinely bad diff"
    example."""
    n = len(preds)
    applied = [p for p in preds if p["applied"]]
    n_applied = len(applied)
    confidence_counts = {tier: sum(1 for p in applied if p["confidence"] == tier) for tier in _CONFIDENCE_TIERS}
    n_cv_gate_passed = sum(1 for p in preds if p.get("cv_gate_passed"))
    n_ood = sum(1 for p in applied if p.get("out_of_distribution"))
    return {
        "n": n, "n_applied": n_applied,
        "applied_rate": (n_applied / n) if n else None,
        "confidence_counts": confidence_counts,
        "cv_gate_passed_rate": (n_cv_gate_passed / n) if n else None,
        "out_of_distribution_rate": (n_ood / n_applied) if n_applied else None,
    }


def _percentile_summary(values: list[float]) -> dict:
    if not values:
        return {**{f"p{p}": None for p in _PERCENTILES}, "mean": None, "std": None, "n": 0}
    arr = np.array(values, dtype=float)
    out = {f"p{p}": float(np.percentile(arr, p)) for p in _PERCENTILES}
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std())
    out["n"] = len(arr)
    return out


def delta_distribution(preds: list[dict]) -> dict:
    """delta_c's own mean/std/p10/p50/p90 over applied rows -- catches a
    correction that shifts the central estimate by an implausible amount
    even when RMSE technically improved."""
    return _percentile_summary([p["delta_c"] for p in preds if p["applied"] and p["delta_c"] is not None])


def paired_delta_diff(old_preds: list[dict], new_preds: list[dict]) -> dict:
    """new_delta_c - old_delta_c for rows applied under BOTH arms --
    catches a correction that moves the median acceptably but blows up
    the tails, which the two arms' own separate distributions (above)
    would not surface on their own."""
    diffs = [
        new["delta_c"] - old["delta_c"]
        for old, new in zip(old_preds, new_preds)
        if old["applied"] and new["applied"] and old["delta_c"] is not None and new["delta_c"] is not None
    ]
    return _percentile_summary(diffs)


def _diff_metrics(old_preds: list[dict], new_preds: list[dict]) -> dict:
    """The tier_mix/delta_distribution/paired_delta_diff triple, computed
    once and reused for the overall, per-zone, AND per-subzone breakouts
    -- one shared implementation rather than three copies that could
    silently diverge."""
    return {
        "tier_mix": {"old": tier_mix(old_preds), "new": tier_mix(new_preds)},
        "delta_distribution": {"old": delta_distribution(old_preds), "new": delta_distribution(new_preds)},
        "paired_delta_diff": paired_delta_diff(old_preds, new_preds),
    }


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def boundary_discontinuity(
    rows: list[dict], old_served: list[float | None], new_served: list[float | None],
    max_km: float = _DEFAULT_BOUNDARY_MAX_KM, subzoned_zones: frozenset[str] = frozenset(),
    correction_keys: list | None = None,
) -> dict:
    """Mean absolute jump in the served value between geographically
    nearby (same date, within max_km) rows that straddle a correction's
    OWN scope boundary -- a direct generalization of Valencia's own
    boundary-discontinuity diagnostic (PHASE3_BOUNDARY_DIAGNOSTIC.md),
    which found a candidate correction can amplify a pre-existing
    raw-grid discontinuity at a boundary even when its own zone's
    aggregate metrics look fine. old_served/new_served: the served value
    (grid + corrected delta, or grid alone if not applied) per row, same
    order as rows.

    Two DISTINCT boundary types, reported separately (Codex adversarial
    review finding, PR #25: an earlier draft only ever checked
    cross-ZONE pairs, which finds zero pairs for a delta_scale_subzone
    correction confined to one country within a single Köppen zone --
    e.g. French Cfb vs. neighboring German Cfb straddle exactly the
    boundary a subzone-scoped fit actually introduces, and the original
    "different zone" definition can never see it):
      - cross-zone: different climate_zone entirely.
      - cross-subzone: SAME climate_zone, that zone is in
        `subzoned_zones` (i.e. either gate publishes a delta_scale_subzone
        entry for it), AND the two rows actually resolve to a DIFFERENT
        effective correction. Restricted to subzoned_zones so a zone with
        no subzone-scoped correction at all doesn't generate meaningless
        noise from an arbitrary country split that was never a real
        correction boundary.

    correction_keys (round-2 Codex adversarial review finding, PR #25):
    one hashable value per row (same order as `rows`) -- the row's
    resolved delta_scale signature under whichever gate is under review,
    typically built via resolve_delta_scale + subzone_code. Two same-zone
    rows are only counted as a cross-subzone pair if their keys actually
    DIFFER -- comparing raw subzone_code strings alone (the fallback when
    this is omitted, kept for callers/tests with no gate handy) would
    wrongly count e.g. a DE/US pair in a zone where only FR has an
    override: both DE and US fall back to the identical flat correction,
    so they never cross a boundary the correction actually introduced,
    and counting them dilutes the real FR signal with unrelated noise.

    O(n^2) within each date group -- acceptable for a human-run review
    tool over realistic station-day counts, not intended for a snapshot
    with an enormous number of same-day rows; this is never on a hot
    serving path."""
    by_date: dict = {}
    for i, r in enumerate(rows):
        by_date.setdefault(r.get("date"), []).append(i)

    cross_zone_old, cross_zone_new, n_cross_zone = [], [], 0
    cross_subzone_old, cross_subzone_new, n_cross_subzone = [], [], 0
    for idxs in by_date.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            ia = idxs[a]
            if old_served[ia] is None or rows[ia].get("lat") is None or rows[ia].get("lon") is None:
                continue
            zone_a = rows[ia].get("climate_zone")
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                if old_served[ib] is None or rows[ib].get("lat") is None or rows[ib].get("lon") is None:
                    continue
                zone_b = rows[ib].get("climate_zone")
                same_zone = zone_a == zone_b
                if same_zone:
                    if zone_a not in subzoned_zones:
                        continue
                    if correction_keys is not None:
                        if correction_keys[ia] == correction_keys[ib]:
                            continue
                    else:
                        sub_a = subzone_code(rows[ia].get("station_id")) or "other"
                        sub_b = subzone_code(rows[ib].get("station_id")) or "other"
                        if sub_a == sub_b:
                            continue
                dist = _haversine_km(rows[ia]["lat"], rows[ia]["lon"], rows[ib]["lat"], rows[ib]["lon"])
                if dist > max_km:
                    continue
                old_jump = abs(old_served[ia] - old_served[ib])
                new_jump = (
                    abs(new_served[ia] - new_served[ib])
                    if new_served[ia] is not None and new_served[ib] is not None else None
                )
                if same_zone:
                    n_cross_subzone += 1
                    cross_subzone_old.append(old_jump)
                    if new_jump is not None:
                        cross_subzone_new.append(new_jump)
                else:
                    n_cross_zone += 1
                    cross_zone_old.append(old_jump)
                    if new_jump is not None:
                        cross_zone_new.append(new_jump)
    return {
        "max_km": max_km,
        "cross_zone": {
            "n_pairs": n_cross_zone,
            "old_mean_abs_jump_c": float(np.mean(cross_zone_old)) if cross_zone_old else None,
            "new_mean_abs_jump_c": float(np.mean(cross_zone_new)) if cross_zone_new else None,
        },
        "cross_subzone": {
            "n_pairs": n_cross_subzone,
            "old_mean_abs_jump_c": float(np.mean(cross_subzone_old)) if cross_subzone_old else None,
            "new_mean_abs_jump_c": float(np.mean(cross_subzone_new)) if cross_subzone_new else None,
        },
        # Backward-compatible aliases (pre-split shape) pointed at the
        # cross-zone numbers specifically, since that was this function's
        # entire scope before this fix.
        "n_cross_zone_pairs_within_km": n_cross_zone,
        "old_mean_abs_jump_c": float(np.mean(cross_zone_old)) if cross_zone_old else None,
        "new_mean_abs_jump_c": float(np.mean(cross_zone_new)) if cross_zone_new else None,
    }


def _served_values(rows: list[dict], preds: list[dict], grid_col: str) -> list[float | None]:
    return [
        (r[grid_col] + p["delta_c"]) if (p["applied"] and p["delta_c"] is not None) else r.get(grid_col)
        for r, p in zip(rows, preds)
    ]


def check_partition_coverage(rows: list[dict], n_covered: int, model_version: str, band_key: str) -> None:
    """A mistyped --model-version (or --band-key), or one that names a
    real but WRONG partition (a different month, a disjoint station
    set), produces a FrozenPredictionAdapter that returns not-applied for
    every row in THIS batch -- indistinguishable, from the output alone,
    from "the model genuinely applies nowhere in this batch." This tool
    would otherwise happily print a valid-looking report claiming
    "nothing changes" (Codex adversarial review finding, PR #25).

    n_covered is adapter.coverage(rows) -- how many of `rows` have ANY
    matching frozen prediction key at all, NOT just whether the
    partition is non-empty (round-2 finding on the same review: a
    non-empty partition for the wrong month/stations would still pass a
    bare emptiness check while covering zero of THESE rows). Hard stop
    when coverage is zero -- this can never be a legitimate real-world
    result for a real band (SOME rows always have complete covariates),
    only a wrong model_version/band_key pairing or a partition/row
    mismatch."""
    if rows and n_covered == 0:
        raise SystemExit(
            f"snapshot has {len(rows)} row(s) for band={band_key!r} but ZERO of them have any matching "
            f"frozen prediction for model_version={model_version!r} -- this almost certainly means "
            "model_version or band_key is wrong, or the predictions partition covers a disjoint set of "
            "stations/months, not that the model genuinely applies nowhere. Refusing to produce a replay "
            "diff that would otherwise misleadingly show 'nothing changes' under both arms."
        )


def replay_band(
    snapshot_dir: str, model_version: str, band_key: str, old_gate: dict, new_gate: dict,
    boundary_max_km: float = _DEFAULT_BOUNDARY_MAX_KM,
) -> dict:
    """The full diff for one (model_version, band_key) across both
    targets -- per-target overall + per-zone tier-mix/delta-distribution/
    paired-delta diffs, plus one boundary-discontinuity check per target."""
    band_rows = snapshot.read_band_partitions(snapshot_dir, band_key)
    if not band_rows:
        raise SystemExit(f"snapshot has no rows for band={band_key!r}")
    adapter = contract.FrozenPredictionAdapter.from_snapshot(snapshot_dir, model_version, band_key)
    # len(adapter) (round-2 code-review finding, PR #25: an earlier draft
    # called snapshot.read_predictions_partitions a SECOND time here just
    # for this count, re-globbing and re-parsing the identical partition
    # from_snapshot had just read one line above) -- see
    # check_partition_coverage's own docstring for why this check exists.
    check_partition_coverage(band_rows, adapter.coverage(band_rows), model_version, band_key)

    result: dict = {"model_version": model_version, "band_key": band_key, "by_target": {}}
    for target in ("tmax", "tmin"):
        grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
        old_preds = predict_with_gate(adapter, band_rows, target, old_gate)
        new_preds = predict_with_gate(adapter, band_rows, target, new_gate)
        old_served = _served_values(band_rows, old_preds, grid_col)
        new_served = _served_values(band_rows, new_preds, grid_col)

        # A zone with a delta_scale_subzone entry in EITHER gate must be
        # diffed at (target, zone, subzone_code), not just (target, zone)
        # -- Codex adversarial review finding, PR #25: a regression
        # confined to one subzone (e.g. a French-only Cfb fit) can
        # otherwise be diluted/hidden by every OTHER country's rows in
        # that same zone's aggregate. Only zones that actually publish a
        # subzone entry get this finer breakout, to avoid noise from
        # meaningless country splits in a zone with no subzone-scoped
        # correction at all.
        subzoned_zones = set(old_gate.get("delta_scale_subzone", {}).get(target, {})) | set(
            new_gate.get("delta_scale_subzone", {}).get(target, {}),
        )

        zones = sorted({r.get("climate_zone") for r in band_rows if r.get("climate_zone")})
        by_zone: dict[str, dict] = {}
        for zone in zones:
            idxs = [i for i, r in enumerate(band_rows) if r.get("climate_zone") == zone]
            zone_old = [old_preds[i] for i in idxs]
            zone_new = [new_preds[i] for i in idxs]
            zone_entry = {**_diff_metrics(zone_old, zone_new)}
            if zone in subzoned_zones:
                by_subzone: dict[str, dict] = {}
                subzone_values = sorted({subzone_code(band_rows[i].get("station_id")) or "other" for i in idxs})
                for sv in subzone_values:
                    sub_idxs = [
                        i for i in idxs if (subzone_code(band_rows[i].get("station_id")) or "other") == sv
                    ]
                    by_subzone[sv] = _diff_metrics(
                        [old_preds[i] for i in sub_idxs], [new_preds[i] for i in sub_idxs],
                    )
                zone_entry["by_subzone"] = by_subzone
            by_zone[zone] = zone_entry

        result["by_target"][target] = {
            "overall": _diff_metrics(old_preds, new_preds),
            "by_zone": by_zone,
            "boundary_discontinuity": boundary_discontinuity(
                band_rows, old_served, new_served, max_km=boundary_max_km, subzoned_zones=frozenset(subzoned_zones),
                # Combined (old_key, new_key) per row: a same-zone pair is
                # boundary-worthy if EITHER arm's correction actually
                # differs between them -- a pre-existing discontinuity
                # under `old` that `new` fixes is just as worth surfacing
                # as one `new` introduces that `old` didn't have.
                correction_keys=list(zip(
                    correction_keys_for_gate(band_rows, old_gate, target),
                    correction_keys_for_gate(band_rows, new_gate, target),
                )),
            ),
            "subzone_status": {"old": subzone_status(old_gate, target), "new": subzone_status(new_gate, target)},
        }
    return result


def render_summary(result: dict) -> str:
    """Human-readable summary for the maintainer's own review -- never a
    pass/fail verdict, GOVERNANCE.md's own human judgment call stays with
    the reader, not this tool."""
    lines = [f"## Replay diff -- {result['model_version']} / {result['band_key']}", ""]
    for target, t in result["by_target"].items():
        lines.append(f"### {target}")
        overall = t["overall"]
        old_tm, new_tm = overall["tier_mix"]["old"], overall["tier_mix"]["new"]
        lines.append(
            f"- applied_rate: {old_tm['applied_rate']} -> {new_tm['applied_rate']}, "
            f"OOD_rate: {old_tm['out_of_distribution_rate']} -> {new_tm['out_of_distribution_rate']}, "
            f"confidence: {old_tm['confidence_counts']} -> {new_tm['confidence_counts']}"
        )
        old_dd, new_dd = overall["delta_distribution"]["old"], overall["delta_distribution"]["new"]
        lines.append(f"- delta_c mean: {old_dd['mean']} -> {new_dd['mean']} (n={old_dd['n']} -> {new_dd['n']})")
        bd = t["boundary_discontinuity"]
        cz, cs = bd["cross_zone"], bd["cross_subzone"]
        lines.append(
            f"- boundary discontinuity, cross-zone (within {bd['max_km']}km, n_pairs={cz['n_pairs']}): "
            f"{cz['old_mean_abs_jump_c']} -> {cz['new_mean_abs_jump_c']}"
        )
        if cs["n_pairs"]:
            lines.append(
                f"- boundary discontinuity, cross-subzone (n_pairs={cs['n_pairs']}): "
                f"{cs['old_mean_abs_jump_c']} -> {cs['new_mean_abs_jump_c']}"
            )
        for zone, z in sorted(t["by_zone"].items()):
            old_zt, new_zt = z["tier_mix"]["old"], z["tier_mix"]["new"]
            lines.append(
                f"  - {zone}: applied_rate {old_zt['applied_rate']} -> {new_zt['applied_rate']}, "
                f"n={old_zt['n']}"
            )
            # Round-2 code-review finding, PR #25: the whole point of the
            # by_subzone breakdown is to surface a regression the
            # zone-level aggregate above can dilute/hide -- it must
            # actually appear in the DEFAULT human-readable summary, not
            # only in the JSON a maintainer would have to know to pass
            # --out and inspect manually to ever see it.
            for sub, s in sorted(z.get("by_subzone", {}).items()):
                old_st, new_st = s["tier_mix"]["old"], s["tier_mix"]["new"]
                paired_mean = s["paired_delta_diff"]["mean"]
                lines.append(
                    f"    - {zone}.{sub}: applied_rate {old_st['applied_rate']} -> {new_st['applied_rate']}, "
                    f"n={old_st['n']}, paired delta_c mean change={paired_mean}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--band-key", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--old", default=None, help="path to the currently-published gate JSON; omit for 'nothing published yet'")
    parser.add_argument("--new", required=True, help="path to the candidate gate JSON")
    parser.add_argument("--boundary-max-km", type=float, default=_DEFAULT_BOUNDARY_MAX_KM)
    parser.add_argument("--out", default=None, help="write the full JSON diff here (always printed to stdout as a summary)")
    args = parser.parse_args()

    old_gate = load_gate(args.old)
    new_gate = load_gate(args.new)
    result = replay_band(
        args.snapshot_dir, args.model_version, args.band_key, old_gate, new_gate,
        boundary_max_km=args.boundary_max_km,
    )

    print(render_summary(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Full JSON diff written to %s", args.out)


if __name__ == "__main__":
    main()
