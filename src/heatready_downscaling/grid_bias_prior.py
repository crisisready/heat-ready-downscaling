"""
Grid-bias prior: what nearby stations say about how far ERA5-Land runs from the station record at a place, by
calendar month. Built for the Aw/Dwa tmax refinement (heat-risk-data-api chunk C-ref, 2026-10-04): the served
rf6 adds a near-constant warm delta everywhere because nothing in FEATURE_ORDER tells it where the grid already
runs warm (NE China) and where it runs cool (Thailand, India). This gives the forest that signal as two features:

  grid_bias_prior_c     kernel-weighted mean of neighbouring stations' mean (station - grid) delta for the row's
                        month, shrunk toward 0 by SHRINK_WEIGHT: sum(w d) / (sum(w) + SHRINK_WEIGHT)
  grid_bias_support     sum(w), how much station evidence stands behind the prior (0 = none in range)

w = exp(-0.5 (km / LENGTH_KM)^2) for neighbours within RADIUS_KM. A station never informs its own prior: every
neighbour within SAME_SITE_KM of the query point is skipped, which also covers one site carried under two ids. A
station-month counts only with >= MIN_DAYS days. Far from any station the prior is 0 with support 0, and the forest
learns how much to trust it from the support column.

Leakage discipline is the caller's: the table must be built from the rows the model is fit on (in leave-region-out
CV, the fold's training rows only; in the scorecard, the cutoff corpus without truth stations).
"""
import math

import numpy as np

LENGTH_KM = 150.0
RADIUS_KM = 600.0
SAME_SITE_KM = 2.0
SHRINK_WEIGHT = 1.0
MIN_DAYS = 10
FEATURES = ("grid_bias_prior_c", "grid_bias_support")
_R = 6371.0


def _month(d):
    return int(d[5:7]) if isinstance(d, str) else d.month


def station_month_table(station_ids, lats, lons, dates, deltas):
    """Per (station, month) mean delta. Returns dict month -> (lat array, lon array, mean array, station ids)."""
    acc = {}
    for sid, la, lo, d, v in zip(station_ids, lats, lons, dates, deltas):
        if v is None or not math.isfinite(v):
            continue
        k = (sid, _month(d))
        a = acc.get(k)
        if a is None:
            acc[k] = [la, lo, v, 1]
        else:
            a[2] += v
            a[3] += 1
    table = {}
    for (sid, m), (la, lo, s, n) in acc.items():
        if n >= MIN_DAYS:
            table.setdefault(m, []).append((la, lo, s / n, sid))
    return {m: (np.array([t[0] for t in v]), np.array([t[1] for t in v]), np.array([t[2] for t in v]),
                np.array([t[3] for t in v])) for m, v in table.items()}


def to_json(table):
    return {str(m): {"lat": a.tolist(), "lon": b.tolist(), "mean_delta": c.tolist(), "station_id": s.tolist()}
            for m, (a, b, c, s) in table.items()}


def from_json(obj):
    return {int(m): (np.array(v["lat"]), np.array(v["lon"]), np.array(v["mean_delta"]), np.array(v["station_id"]))
            for m, v in obj.items()}


def _km(lat, lon, lats, lons):
    p1, p2 = math.radians(lat), np.radians(lats)
    dp, dl = p2 - p1, np.radians(lons - lon)
    h = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _R * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def prior_at(table, lat, lon, month):
    """(grid_bias_prior_c, grid_bias_support) at one point and month."""
    t = table.get(month)
    if t is None or lat is None or lon is None:
        return 0.0, 0.0
    la, lo, mean, _ = t
    km = _km(lat, lon, la, lo)
    use = (km > SAME_SITE_KM) & (km <= RADIUS_KM)
    if not use.any():
        return 0.0, 0.0
    w = np.exp(-0.5 * (km[use] / LENGTH_KM) ** 2)
    sw = float(w.sum())
    return float((w * mean[use]).sum() / (sw + SHRINK_WEIGHT)), sw


def priors_for_rows(table, lats, lons, dates):
    """Prior and support arrays for many rows, computed once per distinct (lat, lon, month)."""
    cache = {}
    prior = np.zeros(len(lats))
    support = np.zeros(len(lats))
    for i, (la, lo, d) in enumerate(zip(lats, lons, dates)):
        k = (la, lo, _month(d))
        if k not in cache:
            cache[k] = prior_at(table, la, lo, k[2])
        prior[i], support[i] = cache[k]
    return prior, support
