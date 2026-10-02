"""
Out-of-time scorecard: one versioned definition of "better" for every candidate change (model
version, level anchor, residual layer, bias model, new data source), scored against the model we
serve today. Spec: scorecard/v1/spec.yaml; method and history: scorecard/v1/README.md.

This module is pure scoring: it takes per-row served predictions that the runner
(scripts/scorecard.py) produced and returns metrics, the ship decision and the report. It never
fits a model, reads a database or touches S3, so the ship rule is testable on synthetic rows.

Row frame columns (one row per station-day-target in the manifest):
  station_id, date (YYYY-MM-DD), year (int), target ("tmax"/"tmin"), zone, delta_obs (station minus
  ERA5-Land grid, C), grid_c (the ERA5-Land grid value for that target), inc_delta / cand_delta
  (served delta: model delta where the serving path applied the model, 0.0 where it fell back to
  the grid), inc_ci95 / cand_ci95 (served 95% half-width, NaN where not applied), dist_km (distance
  to the incumbent's nearest training station at that row's cutoff).
Station frame columns: station_id, zone, zone_group, region, airport (yes/no/unknown),
  setting (urban/rural/unknown), visibility (public/private).

Errors are served value minus station: (grid + served delta) - station = served delta - delta_obs.
Raw ERA5-Land's error is -delta_obs. Every comparison is paired: candidate and incumbent on the
identical rows. RMSE, bias and MAE are pooled over station-days (station-day weighted); the
equal-weight zone mean is reported alongside, never instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGETS = ("tmax", "tmin")
METRICS = ("rmse_tmax", "rmse_tmin", "hot_mae_tmax")
_BOOTSTRAP_SEED = 20261001


# ---------------------------------------------------------------- spec + manifest

def load_spec(path: str) -> dict:
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    for key in ("scorecard_version", "zone_tolerance_rmse_c", "min_stations_per_zone", "zone_groups",
                "tiers", "hot_day_quantile", "manifest"):
        if key not in spec:
            raise ValueError(f"scorecard spec missing {key!r}")
    if not spec["zone_tolerance_rmse_c"] > 0:
        raise ValueError("zone_tolerance_rmse_c must be positive")
    if int(spec["min_stations_per_zone"]) < 1:
        raise ValueError("min_stations_per_zone must be >= 1")
    seen: dict[str, str] = {}
    for group, zones in spec["zone_groups"].items():
        for z in zones:
            if z in seen:
                raise ValueError(f"zone {z!r} is in both {seen[z]!r} and {group!r}")
            seen[z] = group
    for tier in ("fast", "full"):
        if not spec["tiers"].get(tier, {}).get("cutoffs"):
            raise ValueError(f"tier {tier!r} has no cutoffs")


def zone_group(zone: str, spec: dict) -> str:
    """The fixed zone group a zone pools into. A zone not listed falls to its Koppen main letter
    (A tropical, B arid, C temperate, D/E cold); anything else is 'unassigned' and never pooled
    silently into a real group."""
    for group, zones in spec["zone_groups"].items():
        if zone in zones:
            return group
    letter = (zone or "?")[:1]
    return {"A": "tropical", "B": "arid", "C": "temperate", "D": "cold", "E": "cold"}.get(letter, "unassigned")


def score_year(cutoff: str) -> int:
    """A cutoff YYYY-12-31 scores the following calendar year."""
    if len(cutoff) != 10 or cutoff[4:] != "-12-31":
        raise ValueError(f"cutoff must be a year end YYYY-12-31, got {cutoff!r}")
    return int(cutoff[:4]) + 1


def manifest_sha256(keys) -> str:
    """sha256 of the station-day manifest: one 'station_id,date,target' line per key, sorted,
    newline-terminated. Order- and duplicate-insensitive by construction (duplicates are an error)."""
    lines = [f"{s},{str(d)[:10]},{t}" for s, d, t in keys]
    if len(set(lines)) != len(lines):
        raise ValueError("manifest has duplicate station-day-target keys")
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode() + b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- row preparation

def flag_hot_days(rows: pd.DataFrame, quantile: float) -> pd.Series:
    """Hot days: tmax rows whose ERA5-Land grid value is at or above that station's own `quantile`
    of grid tmax within the scored year. A fixed input shared by candidate and incumbent: selecting
    days by the observed station value would favour a warm-biased model through regression to the
    mean (plan section 4)."""
    hot = pd.Series(False, index=rows.index)
    tmax = rows[rows["target"] == "tmax"]
    if tmax.empty:
        return hot
    thr = tmax.groupby(["station_id", "year"])["grid_c"].transform(lambda s: s.quantile(quantile))
    hot.loc[tmax.index] = tmax["grid_c"] >= thr
    return hot


def _errors(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["err_inc"] = out["inc_delta"] - out["delta_obs"]
    out["err_cand"] = out["cand_delta"] - out["delta_obs"]
    out["err_grid"] = -out["delta_obs"]
    return out


def _metric_block(r: pd.DataFrame) -> dict:
    """Paired metrics over one subset of error rows. None where the subset has no rows for a metric."""
    out: dict = {}
    for t in TARGETS:
        s = r[r["target"] == t]
        n = len(s)
        out[f"n_rows_{t}"] = int(n)
        out[f"n_stations_{t}"] = int(s["station_id"].nunique())
        for who in ("inc", "cand", "grid"):
            e = s[f"err_{who}"].to_numpy()
            out[f"rmse_{t}_{who}"] = float(np.sqrt(np.mean(e ** 2))) if n else None
            out[f"bias_{t}_{who}"] = float(np.mean(e)) if n else None
        out[f"d_rmse_{t}"] = (out[f"rmse_{t}_cand"] - out[f"rmse_{t}_inc"]) if n else None
        out[f"d_bias_abs_{t}"] = (abs(out[f"bias_{t}_cand"]) - abs(out[f"bias_{t}_inc"])) if n else None
    h = r[(r["target"] == "tmax") & r["hot"]]
    out["n_rows_hot"] = int(len(h))
    for who in ("inc", "cand", "grid"):
        out[f"hot_mae_tmax_{who}"] = float(np.mean(np.abs(h[f"err_{who}"]))) if len(h) else None
    out["d_hot_mae_tmax"] = (out["hot_mae_tmax_cand"] - out["hot_mae_tmax_inc"]) if len(h) else None
    return out


# ---------------------------------------------------------------- units (zones, pooled thin zones, groups)

@dataclass
class Unit:
    name: str
    kind: str            # "zone" | "pooled" | "group"
    zones: list[str]
    gating: bool
    n_stations: dict = field(default_factory=dict)
    note: str = ""


def build_units(rows: pd.DataFrame, spec: dict) -> list[Unit]:
    """Gating units for the zone-tolerance check. A zone with at least min_n unseen stations (for a
    target) is its own unit; zones under min_n pool, per zone group, into one unit; a pooled unit
    still under min_n is reported but does not gate ("Point estimate, pooled", Nishant 2026-10-01).
    Station counts are per target, so a tmax-only station set never props up a tmin unit."""
    min_n = int(spec["min_stations_per_zone"])
    units: list[Unit] = []
    by_group: dict[str, dict[str, list[str]]] = {}
    for t in TARGETS:
        counts = rows[rows["target"] == t].groupby("zone")["station_id"].nunique()
        for zone, n in counts.items():
            g = zone_group(zone, spec)
            if n >= min_n:
                units.append(Unit(f"{zone}", "zone", [zone], True, {t: int(n)}))
            else:
                by_group.setdefault(t, {}).setdefault(g, []).append(zone)
    for t, groups in by_group.items():
        for g, zones in groups.items():
            s = rows[(rows["target"] == t) & rows["zone"].isin(zones)]
            n = int(s["station_id"].nunique())
            units.append(Unit(f"{g} (pooled: {'+'.join(sorted(zones))})", "pooled", sorted(zones), n >= min_n,
                              {t: n}, "" if n >= min_n else f"{n} stations < min {min_n}; reported, not gating"))
    # merge per-target duplicates of the same unit name into one record
    merged: dict[str, Unit] = {}
    for u in units:
        if u.name in merged:
            merged[u.name].n_stations.update(u.n_stations)
            merged[u.name].gating = merged[u.name].gating or u.gating
        else:
            merged[u.name] = u
    return list(merged.values())


def _unit_target_gates(u: Unit, target: str, spec: dict) -> bool:
    return u.n_stations.get(target, 0) >= int(spec["min_stations_per_zone"])


# ---------------------------------------------------------------- scoring

def score(rows: pd.DataFrame, stations: pd.DataFrame, spec: dict) -> dict:
    """Metrics for the public scorecard. Private (contributor) stations are scored separately and
    never enter the public blocks (plan section 4 item 9)."""
    rows = rows.merge(stations[["station_id", "zone_group", "region", "airport", "setting", "visibility"]],
                      on="station_id", how="left", validate="many_to_one")
    if rows["visibility"].isna().any():
        missing = sorted(rows.loc[rows["visibility"].isna(), "station_id"].unique())[:5]
        raise ValueError(f"rows for stations not in the truth station list: {missing}")
    rows["hot"] = flag_hot_days(rows, float(spec["hot_day_quantile"]))
    rows = _errors(rows)
    public = rows[rows["visibility"] == "public"]
    private = rows[rows["visibility"] != "public"]

    out: dict = {"scorecard_version": spec["scorecard_version"], "global": _metric_block(public)}
    units = build_units(public, spec)
    out["units"] = []
    for u in sorted(units, key=lambda u: (u.kind != "zone", u.name)):
        block = _metric_block(public[public["zone"].isin(u.zones)])
        out["units"].append({"name": u.name, "kind": u.kind, "zones": u.zones, "gating": u.gating,
                             "gating_targets": [t for t in TARGETS if _unit_target_gates(u, t, spec)],
                             "note": u.note, **block})
    out["groups"] = []
    for g in sorted(public["zone_group"].unique()):
        out["groups"].append({"name": g, **_metric_block(public[public["zone_group"] == g])})
    out["report"] = _report(public, spec, units)
    out["private"] = ({sid: _metric_block(s) for sid, s in private.groupby("station_id")}
                      if len(private) else {})
    return out


def _report(r: pd.DataFrame, spec: dict, units: list[Unit]) -> dict:
    """Report-only blocks: never gate a ship (plan section 4 item 8)."""
    rep: dict = {}
    rep["bootstrap_ci95"] = bootstrap_ci(r, spec)
    rep["per_year"] = {int(y): _sign_row(_metric_block(s)) for y, s in r.groupby("year")}
    bands = spec.get("distance_bands_km", [10, 50, 200])
    edges = [0.0, *bands, float("inf")]
    rep["distance_bands"] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = r[(r["dist_km"] >= lo) & (r["dist_km"] < hi)]
        if len(s):
            rep["distance_bands"][f"{lo:g}-{hi:g} km"] = _compact(_metric_block(s))
    rep["strata"] = {}
    for col in ("airport", "setting"):
        for val, s in r.groupby(col):
            rep["strata"][f"{col}={val}"] = _compact(_metric_block(s))
    zone_units = [u for u in units if u.kind == "zone"]
    ew = {}
    for t in TARGETS:
        ds = [_metric_block(r[r["zone"].isin(u.zones)])[f"d_rmse_{t}"]
              for u in zone_units if _unit_target_gates(u, t, spec)]
        ds = [d for d in ds if d is not None]
        ew[f"d_rmse_{t}"] = float(np.mean(ds)) if ds else None
        ew[f"n_zones_{t}"] = len(ds)
    rep["equal_weight_zone_mean"] = ew
    rep["within_city_anomaly_corr"] = within_city_anomaly_corr(r, spec)
    rep["interval_coverage"] = {
        who: (float(np.mean(np.abs(s[f"err_{who}"]) <= s[f"{who}_ci95"])) if len(s) else None)
        for who in ("inc", "cand") for s in [r[r[f"{who}_ci95"].notna()]]}
    return rep


def _compact(block: dict) -> dict:
    keep = ("n_stations_tmax", "n_rows_tmax", "n_stations_tmin", "n_rows_tmin", "rmse_tmax_inc", "rmse_tmax_cand",
            "rmse_tmax_grid", "d_rmse_tmax", "rmse_tmin_inc", "rmse_tmin_cand", "rmse_tmin_grid", "d_rmse_tmin",
            "d_hot_mae_tmax")
    return {k: block.get(k) for k in keep}


def _sign_row(block: dict) -> dict:
    return {m: (None if block.get(f"d_{m}") is None else
                ("better" if block[f"d_{m}"] < 0 else "worse" if block[f"d_{m}"] > 0 else "equal"))
            for m in METRICS} | {f"d_{m}": block.get(f"d_{m}") for m in METRICS}


def bootstrap_ci(r: pd.DataFrame, spec: dict) -> dict:
    """Station-resampled 95% intervals for the global paired deltas (report only: thin zones are
    gated on the point estimate, per the 2026-10-01 decision)."""
    n_boot = int(spec.get("bootstrap_reps", 1000))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    out: dict = {}
    for t in TARGETS:
        s = r[r["target"] == t]
        if s.empty:
            out[f"d_rmse_{t}"] = None
            continue
        g = s.assign(se_i=s["err_inc"] ** 2, se_c=s["err_cand"] ** 2).groupby("station_id")[["se_i", "se_c"]]
        sums, counts = g.sum().to_numpy(), g.size().to_numpy()
        idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
        n = counts[idx].sum(axis=1)
        d = np.sqrt(sums[idx, 1].sum(axis=1) / n) - np.sqrt(sums[idx, 0].sum(axis=1) / n)
        out[f"d_rmse_{t}"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    h = r[(r["target"] == "tmax") & r["hot"]]
    if len(h):
        g = h.assign(ae_i=h["err_inc"].abs(), ae_c=h["err_cand"].abs()).groupby("station_id")[["ae_i", "ae_c"]]
        sums, counts = g.sum().to_numpy(), g.size().to_numpy()
        idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
        n = counts[idx].sum(axis=1)
        d = (sums[idx, 1].sum(axis=1) - sums[idx, 0].sum(axis=1)) / n
        out["d_hot_mae_tmax"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    else:
        out["d_hot_mae_tmax"] = None
    return out


def city_clusters(stations: pd.DataFrame, radius_km: float) -> dict[str, int]:
    """Single-linkage clusters of truth stations within radius_km of each other: station_id ->
    cluster id. Used only for the report-only within-city anomaly correlation."""
    ids = stations["station_id"].tolist()
    lat = np.radians(stations["lat"].to_numpy(float))
    lon = np.radians(stations["lon"].to_numpy(float))
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(ids)):
        dlat = lat[i + 1:] - lat[i]
        dlon = lon[i + 1:] - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[i + 1:]) * np.sin(dlon / 2) ** 2
        for j in np.nonzero(6371.0 * 2 * np.arcsin(np.sqrt(a)) <= radius_km)[0]:
            parent[find(i + 1 + j)] = find(i)
    return {sid: find(i) for i, sid in enumerate(ids)}


def within_city_anomaly_corr(r: pd.DataFrame, spec: dict) -> dict:
    """For cities (clusters) with at least `min_city_stations` unseen stations: anomaly = value minus
    the city's mean that day (over the city's stations reporting that day, 3 or more); correlation of
    served anomalies with observed anomalies, tmax, pooled over city-station-days. Raw ERA5-Land is
    shown for reference: shared cells give it little within-city variance by construction."""
    min_city = int(spec.get("min_city_stations", 5))
    if "city" not in r.columns:
        return {"cities": 0}
    s = r[(r["target"] == "tmax") & r["city"].notna()]
    sizes = s.groupby("city")["station_id"].nunique()
    s = s[s["city"].isin(sizes[sizes >= min_city].index)]
    if s.empty:
        return {"cities": 0}
    s = s.assign(obs=s["grid_c"] + s["delta_obs"], inc=s["grid_c"] + s["inc_delta"],
                 cand=s["grid_c"] + s["cand_delta"], grid=s["grid_c"])
    day = s.groupby(["city", "date"])
    s = s[day["station_id"].transform("count") >= 3]
    day = s.groupby(["city", "date"])
    out = {"cities": int(s["city"].nunique()), "station_days": int(len(s))}
    an = {k: s[k] - day[k].transform("mean") for k in ("obs", "inc", "cand", "grid")}
    for k in ("inc", "cand", "grid"):
        a, b = an[k].to_numpy(), an["obs"].to_numpy()
        out[f"corr_{k}"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None
    return out


# ---------------------------------------------------------------- ship rule

def validate_declaration(decl: dict) -> None:
    """The aim is declared in the PR before the scorer runs (plan section 4)."""
    for key in ("candidate", "incumbent", "declared_at", "aim"):
        if key not in decl:
            raise ValueError(f"declaration missing {key!r}")
    aim = decl["aim"]
    if aim.get("metric") not in METRICS:
        raise ValueError(f"aim.metric must be one of {METRICS}, got {aim.get('metric')!r}")
    if not (isinstance(aim.get("min_effect_c"), (int, float)) and aim["min_effect_c"] > 0):
        raise ValueError("aim.min_effect_c must be a positive number (C of improvement)")
    sel = aim.get("subset") or {}
    if not any(sel.get(k) for k in ("zones", "zone_groups", "regions", "station_ids", "airport", "setting")):
        raise ValueError("aim.subset must name at least one of zones/zone_groups/regions/station_ids/airport/setting")


def aim_mask(rows: pd.DataFrame, subset: dict) -> pd.Series:
    m = pd.Series(True, index=rows.index)
    if subset.get("zones"):
        m &= rows["zone"].isin(subset["zones"])
    if subset.get("zone_groups"):
        m &= rows["zone_group"].isin(subset["zone_groups"])
    if subset.get("regions"):
        m &= rows["region"].astype(str).str.startswith(tuple(subset["regions"]))
    if subset.get("station_ids"):
        m &= rows["station_id"].isin(subset["station_ids"])
    for col in ("airport", "setting"):
        if subset.get(col):
            m &= rows[col].isin(subset[col])
    return m


def score_aim(rows: pd.DataFrame, stations: pd.DataFrame, spec: dict, decl: dict) -> dict:
    validate_declaration(decl)
    r = rows.merge(stations[["station_id", "zone_group", "region", "airport", "setting", "visibility"]],
                   on="station_id", how="left", validate="many_to_one")
    r["hot"] = flag_hot_days(r, float(spec["hot_day_quantile"]))
    r = _errors(r[r["visibility"] == "public"])
    block = _metric_block(r[aim_mask(r, decl["aim"]["subset"])])
    metric = decl["aim"]["metric"]
    return {"subset": decl["aim"]["subset"], "metric": metric, "min_effect_c": decl["aim"]["min_effect_c"],
            "delta": block.get(f"d_{metric}"), "n_stations_tmax": block["n_stations_tmax"],
            "n_stations_tmin": block["n_stations_tmin"], "block": _compact(block)}


def ship_decision(result: dict, aim: dict, spec: dict) -> dict:
    """The ship rule as code (plan section 4 item 7). Returns {"pass": bool, "reasons": [...],
    "checks": [...]}; every check that fails is a reason. A check with no rows to evaluate is
    recorded as not applicable, never as a pass in disguise."""
    tol = float(spec["zone_tolerance_rmse_c"])
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": ok, "detail": detail})

    g = result["global"]
    for m in METRICS:
        d = g.get(f"d_{m}")
        add(f"global {m} not worse", None if d is None else d <= 0,
            "no rows" if d is None else f"candidate minus incumbent {d:+.4f} C")
    for u in result["units"]:
        for t in u["gating_targets"]:
            d = u.get(f"d_rmse_{t}")
            add(f"{u['name']} rmse_{t} within tolerance", None if d is None else d <= tol,
                f"{d:+.4f} C vs tolerance +{tol:.2f} C ({u[f'n_stations_{t}']} stations)" if d is not None else "no rows")
    for grp in result["groups"]:
        for t in TARGETS:
            c, raw = grp.get(f"rmse_{t}_cand"), grp.get(f"rmse_{t}_grid")
            add(f"{grp['name']} group rmse_{t} not worse than raw ERA5-Land", None if c is None else c <= raw,
                "no rows" if c is None else f"candidate {c:.4f} vs ERA5-Land {raw:.4f} C")
    d = aim.get("delta")
    add(f"aim {aim['metric']} improves by >= {aim['min_effect_c']:.3f} C", None if d is None else d <= -aim["min_effect_c"],
        "no rows in the aimed subset" if d is None else f"candidate minus incumbent {d:+.4f} C on the aimed subset")
    failed = [c for c in checks if c["ok"] is False]
    aim_missing = checks[-1]["ok"] is None
    reasons = [f"{c['check']}: {c['detail']}" for c in failed]
    if aim_missing:
        reasons.append("aim has no rows to score; a ship needs a measurable aim")
    return {"pass": not failed and not aim_missing, "reasons": reasons, "checks": checks}


# ---------------------------------------------------------------- publication

def _f(v, fmt="{:.3f}"):
    return "" if v is None else fmt.format(v)


def render_zone_table(result: dict, decision: dict, title: str) -> str:
    """Public markdown: zone-level results and pass/fail only (no per-station rows)."""
    lines = [f"# {title}", "", f"Scorecard {result['scorecard_version']}. Error is served value minus station, C. "
             "Deltas are candidate minus incumbent on identical station-days (negative is better).", "",
             f"**Ship rule: {'PASS' if decision['pass'] else 'FAIL'}**", ""]
    if decision["reasons"]:
        lines += ["Reasons:", *[f"- {r}" for r in decision["reasons"]], ""]
    hdr = ("| Unit | Gating | tmax stations | tmax RMSE inc | tmax RMSE cand | ERA5-Land | delta | "
           "tmin stations | tmin RMSE inc | tmin RMSE cand | ERA5-Land | delta | hot-day MAE delta |")
    sep = "|" + "---|" * 13
    lines += [hdr, sep]

    def row(name, gating, b):
        return (f"| {name} | {gating} | {b.get('n_stations_tmax', '')} | {_f(b.get('rmse_tmax_inc'))} | "
                f"{_f(b.get('rmse_tmax_cand'))} | {_f(b.get('rmse_tmax_grid'))} | {_f(b.get('d_rmse_tmax'), '{:+.3f}')} | "
                f"{b.get('n_stations_tmin', '')} | {_f(b.get('rmse_tmin_inc'))} | {_f(b.get('rmse_tmin_cand'))} | "
                f"{_f(b.get('rmse_tmin_grid'))} | {_f(b.get('d_rmse_tmin'), '{:+.3f}')} | {_f(b.get('d_hot_mae_tmax'), '{:+.3f}')} |")
    lines.append(row("**Global**", "yes", result["global"]))
    for u in result["units"]:
        lines.append(row(u["name"], ", ".join(u["gating_targets"]) or "no", u))
    lines += ["", "Zone groups (raw ERA5-Land floor):", "", hdr, sep]
    for grp in result["groups"]:
        lines.append(row(grp["name"], "floor", grp))
    rep = result["report"]
    lines += ["", "## Report only (not gating)", "",
              f"- Bootstrap 95% CI (station resample), global: {rep['bootstrap_ci95']}",
              f"- Equal-weight zone mean: {rep['equal_weight_zone_mean']}",
              f"- Within-city anomaly correlation: {rep['within_city_anomaly_corr']}",
              f"- Served 95% interval coverage: {rep['interval_coverage']}", "",
              "Per year:", "", "| Year | d rmse_tmax | d rmse_tmin | d hot_mae_tmax |", "|---|---|---|---|"]
    for y, s in sorted(rep["per_year"].items()):
        lines.append(f"| {y} | {_f(s['d_rmse_tmax'], '{:+.3f}')} | {_f(s['d_rmse_tmin'], '{:+.3f}')} | "
                     f"{_f(s['d_hot_mae_tmax'], '{:+.3f}')} |")
    lines += ["", "Distance to the incumbent's nearest training station:", "",
              "| Band | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |", "|---|---|---|---|---|"]
    for k, b in rep["distance_bands"].items():
        lines.append(f"| {k} | {b['n_stations_tmax']} | {_f(b['d_rmse_tmax'], '{:+.3f}')} | {b['n_stations_tmin']} | "
                     f"{_f(b['d_rmse_tmin'], '{:+.3f}')} |")
    lines += ["", "Strata:", "", "| Stratum | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |", "|---|---|---|---|---|"]
    for k, b in rep["strata"].items():
        lines.append(f"| {k} | {b['n_stations_tmax']} | {_f(b['d_rmse_tmax'], '{:+.3f}')} | {b['n_stations_tmin']} | "
                     f"{_f(b['d_rmse_tmin'], '{:+.3f}')} |")
    return "\n".join(lines) + "\n"
