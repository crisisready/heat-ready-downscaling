"""
Station homogeneity screen for build_gsod_bsh_rows.py output, and the corpus variants built from it.

Why: the downscaling model has no time feature, so a station whose relationship to ERA5-Land
changed partway through the record (a site move, an instrument change, real local change) is
learned as the average of its two regimes. Ahmedabad (WMO 42647) is the case that prompted this:
its April-June station-minus-ERA5-Land offset fell from about +2.2 C (2016-2022) to +0.5
(2023-2025), and a model trained across the break predicts roughly the decade mean.

The rule (fixed here before any candidate built with it is scored against CommHATS, the
evaluation set; see research/ahmedabad-bsh-refinement/RESULTS.md in heat-risk-data-api):

  1. For each station and season (AMJ = Apr-Jun, JAS = Jul-Sep), take the annual mean of
     delta_tmax_c (station minus ERA5-Land) over years with >= MIN_DAYS_PER_SEASON_YEAR days.
  2. Reference: subtract a composite of the K_NEIGHBOURS nearest stations' annual anomalies
     (each series minus its own mean), weighted by their correlation with the station
     (non-positive correlations get no weight). This removes change the neighbours share,
     e.g. drift in ERA5-Land itself, which a station-vs-ERA5-Land test would read as a break.
  3. On that difference series, SNHT: the maximum over allowed break years of
     k*mean(z_pre)^2 + (n-k)*mean(z_post)^2 on the standardised series. Allowed breaks leave
     >= MIN_SEGMENT_YEARS on each side and fall in BREAK_YEARS.
  4. p-value by Monte Carlo: the same maximum, same n and same allowed positions, over
     N_MONTE_CARLO iid normal series (SNHT's null), fixed seed.
  5. Benjamini-Hochberg at FDR_Q over the whole family of station-season tests (m = every tested
     station-season, flagged ones included at the p-value that flagged them), plus a physical
     floor: |post-break mean - pre-break mean| of the difference series >= MIN_SHIFT_C.
  6. Iterate: flag only the single most significant station-season that passes 5, take that
     station out of every other station's reference composite, and re-test the remaining
     stations, until nothing new passes. Without this, one station's break leaks into its neighbours' references and
     shows up as a mirror-image "break" at each of them.
  7. A flagged station keeps only its post-break years (all seasons, both targets); the earlier
     segment is dropped, not adjusted, since adjusting means choosing which regime is "true".

Preview, 2026-09-30 (GSOD tmax minus Open-Meteo era5_land at each of the 28 IN/PK BSh stations,
before any candidate was scored): nothing is flagged. Ahmedabad's AMJ shift against its four
Gujarat neighbours (133-344 km away) is about -1 C from 2023 but p = 0.15 (JAS p = 0.13), because
the neighbour-referenced series is noisy at n = 10. The rule was left as specified rather than
loosened after seeing that, so "break" and "none" give the same corpus for this station set.

Known limits: the network is sparse (Gujarat and Delhi are the only dense neighbourhoods), n is
~10 annual values, and a coordinated instrument rollout across neighbours would cancel out of
step 2 and go unflagged.

Variants (--mode):
  none    all rows unchanged
  break   flagged stations keep their post-break years (the rule above)
  recent  every station keeps only years >= --recent-from (default 2023)

Usage:
    python3 scripts/gsod_homogeneity.py --rows gsod_bsh_rows.json --mode break --out rows_break.json
      (always writes the full per-station test table next to --out as *_homogeneity.json)
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict

SEASONS = {"AMJ": (4, 5, 6), "JAS": (7, 8, 9)}
MIN_DAYS_PER_SEASON_YEAR = 45
MIN_YEARS = 6
MIN_SEGMENT_YEARS = 3
BREAK_YEARS = range(2019, 2024)
K_NEIGHBOURS = 5
N_MONTE_CARLO = 20000
MC_SEED = 20260930
FDR_Q = 0.05
MIN_SHIFT_C = 0.5


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _mean(xs):
    return sum(xs) / len(xs)


def seasonal_annual_means(rows):
    """{station_id: {season: {year: mean delta_tmax_c}}} over years with enough days."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("delta_tmax_c") is None:
            continue
        m, y = int(r["date"][5:7]), int(r["date"][:4])
        for season, months in SEASONS.items():
            if m in months:
                acc[(r["station_id"], season)][y].append(r["delta_tmax_c"])
    out = defaultdict(dict)
    for (sid, season), by_year in acc.items():
        series = {y: _mean(v) for y, v in by_year.items() if len(v) >= MIN_DAYS_PER_SEASON_YEAR}
        if len(series) >= MIN_YEARS:
            out[sid][season] = series
    return out


def _corr(a, b):
    ma, mb = _mean(a), _mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def difference_series(sid, season, series_by_sid, coords, k=K_NEIGHBOURS, exclude=frozenset()):
    """The station's annual anomaly minus a correlation-weighted neighbour composite, over the
    years where the station and at least one weighted neighbour both have data. Returns
    ({year: value}, [neighbour ids used])."""
    own = series_by_sid[sid][season]
    others = [o for o in series_by_sid if o != sid and o not in exclude and season in series_by_sid[o]]
    nearest = sorted(others, key=lambda o: _km(coords[sid], coords[o]))[:k]
    own_mean = _mean(list(own.values()))
    weights, anomalies = {}, {}
    # Each neighbour's anomaly is centred on the years it shares with the station, so a neighbour
    # whose record starts late can't shift the composite by where its own mean happens to fall.
    for o in nearest:
        s = series_by_sid[o][season]
        common = sorted(set(own) & set(s))
        if len(common) < MIN_YEARS:
            continue
        c = _corr([own[y] for y in common], [s[y] for y in common])
        if c > 0:
            weights[o] = c
            mo = _mean([s[y] for y in common])
            anomalies[o] = {y: s[y] - mo for y in common}
    diff = {}
    for y, v in own.items():
        w = [(weights[o], anomalies[o][y]) for o in weights if y in anomalies[o]]
        if w:
            diff[y] = (v - own_mean) - sum(wi * a for wi, a in w) / sum(wi for wi, _ in w)
    return diff, sorted(weights)


def _allowed_splits(years):
    """Indices k (first post-break position) allowed for a sorted year list."""
    n = len(years)
    return [k for k in range(MIN_SEGMENT_YEARS, n - MIN_SEGMENT_YEARS + 1) if years[k] in BREAK_YEARS]


def snht_max(values, splits):
    """Max SNHT statistic over the given split positions, on the standardised series."""
    n = len(values)
    m = _mean(values)
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    if sd == 0:
        return 0.0, None
    z = [(x - m) / sd for x in values]
    best, best_k = 0.0, None
    for k in splits:
        t = k * _mean(z[:k]) ** 2 + (n - k) * _mean(z[k:]) ** 2
        if t > best:
            best, best_k = t, k
    return best, best_k


_null_cache = {}


def mc_pvalue(t_obs, n, splits, n_sim=N_MONTE_CARLO, seed=MC_SEED):
    """P(max SNHT >= t_obs) under iid normal noise with the same n and split positions."""
    key = (n, tuple(splits), n_sim, seed)
    if key not in _null_cache:
        rng = random.Random(seed)
        _null_cache[key] = sorted(snht_max([rng.gauss(0, 1) for _ in range(n)], splits)[0] for _ in range(n_sim))
    null = _null_cache[key]
    lo, hi = 0, len(null)
    while lo < hi:  # first index with null >= t_obs
        mid = (lo + hi) // 2
        if null[mid] < t_obs:
            lo = mid + 1
        else:
            hi = mid
    return (len(null) - lo + 1) / (len(null) + 1)


def benjamini_hochberg(pvalues):
    """q-values (BH-adjusted p-values), same order as the input."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    q = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        q[i] = running
    return q


def _test_all(series, coords, exclude):
    """SNHT + Monte Carlo p for every station-season not in `exclude` (no BH here)."""
    tests = []
    for sid in sorted(series):
        if sid in exclude:
            continue
        for season in sorted(series[sid]):
            diff, neighbours = difference_series(sid, season, series, coords, exclude=exclude)
            years = sorted(diff)
            splits = _allowed_splits(years)
            test = {"station_id": sid, "season": season, "years": years, "neighbours": neighbours}
            if len(years) < MIN_YEARS or not splits:
                tests.append({**test, "tested": False})
                continue
            values = [diff[y] for y in years]
            t, k = snht_max(values, splits)
            shift = _mean(values[k:]) - _mean(values[:k]) if k else 0.0
            tests.append({**test, "tested": True, "snht": round(t, 3), "break_year": years[k] if k else None,
                          "shift_c": round(shift, 3), "p": mc_pvalue(t, len(values), splits)})
    return tests


def _bh_passes(flagged_tests, tests):
    """BH over the whole family (already-flagged tests at their flagging p, plus the current
    round's tested ones). Sets q on the current tests; returns the ones that pass step 5."""
    current = [t for t in tests if t["tested"]]
    family = flagged_tests + current
    q = benjamini_hochberg([t["p"] for t in family])
    for t, qi in zip(family[len(flagged_tests):], q[len(flagged_tests):]):
        t["q"] = qi
        t["passes"] = qi <= FDR_Q and abs(t["shift_c"]) >= MIN_SHIFT_C
    return [t for t in current if t["passes"]]


def screen(rows):
    """Run the rule. Returns (tests, {station_id: first kept year}): the last round's tests for
    unflagged stations plus, for each flagged station, the one test that flagged it."""
    series = seasonal_annual_means(rows)
    coords = {}
    for r in rows:
        coords.setdefault(r["station_id"], (r["lat"], r["lon"]))
    exclude, flagged_tests = set(), []
    while True:
        tests = _test_all(series, coords, frozenset(exclude))
        passing = _bh_passes(flagged_tests, tests)
        if not passing:
            break
        top = min(passing, key=lambda t: (t["p"], -t["snht"]))
        top["flagged"] = True
        top["flag_round"] = len(exclude) + 1
        flagged_tests.append(top)
        exclude.add(top["station_id"])
    for t in tests:
        t.setdefault("flagged", False)
    first_kept = {t["station_id"]: t["break_year"] for t in flagged_tests}
    return tests + flagged_tests, first_kept


def apply_mode(rows, mode, first_kept, recent_from=2023):
    if mode == "none":
        return list(rows)
    if mode == "break":
        return [r for r in rows if int(r["date"][:4]) >= first_kept.get(r["station_id"], 0)]
    if mode == "recent":
        return [r for r in rows if int(r["date"][:4]) >= recent_from]
    raise ValueError(mode)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help="build_gsod_bsh_rows.py output (gsod_bsh_rows.json)")
    ap.add_argument("--mode", choices=("none", "break", "recent"), required=True)
    ap.add_argument("--recent-from", type=int, default=2023)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.rows, "rb") as f:
        raw = f.read()
    payload = json.loads(raw)
    if payload.get("complete") is not True:
        raise SystemExit(f"{args.rows} is not marked complete")
    tests, first_kept = screen(payload["rows"])
    rows = apply_mode(payload["rows"], args.mode, first_kept, args.recent_from)
    table = args.out.rsplit(".json", 1)[0] + "_homogeneity.json"
    with open(table, "w") as f:
        json.dump({"rule": {k: (list(v) if isinstance(v, range) else v) for k, v in globals().items()
                            if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, dict, range))},
                   "tests": tests, "flagged_first_kept_year": first_kept}, f, indent=1, default=list)
    by_station = defaultdict(int)
    for r in rows:
        by_station[r["station_id"]] += 1
    out = {k: v for k, v in payload.items() if k not in ("rows", "row_count", "rows_by_station")}
    out.update(rows=rows, row_count=len(rows), rows_by_station=dict(by_station), homogeneity_mode=args.mode,
               homogeneity_source_sha256=hashlib.sha256(raw).hexdigest(),
               homogeneity_first_kept_year=first_kept if args.mode == "break" else None,
               recent_from=args.recent_from if args.mode == "recent" else None)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"{args.mode}: {len(rows)} of {len(payload['rows'])} rows kept; flagged {first_kept}; table {table}")


if __name__ == "__main__":
    main()
