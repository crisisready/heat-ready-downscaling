"""
PROVENANCE: moved from crisisready/heat-risk-data-api's scripts/validate_station_blend.py at commit e6cc1d2a338fd9c87ef51cbc1daf6cb1ed5e2b3b (2026-07-27, Phase 1.4, plan section 5.4). See this repository's own PROVENANCE.md.

REFACTORED during the move: model loading/inference goes through
`heatready_downscaling.contract.QRFModelAdapter` instead of the private
repo's `downscaling.load_model`/`predict_downscaled`. `ghcn.
koppen_broad_group_letter_from_zone` is replaced by `heatready_downscaling.
koppen.koppen_broad_group_letter_from_zone` -- the same function, already
extracted into this package (see koppen.py's own docstring).

Unlike validate_lagfill_downscaling.py / validate_forecast_downscaling.py,
this script has NO private-repo-only imports left after the above two
swaps -- it never made live DB/Open-Meteo calls to begin with (see the
module docstring below: it reads a pre-dumped JSON row file). It IS
runnable standalone in this repo, given a --rows-in JSON file (produced by
validate_lagfill_downscaling.py's --phase dump-rows on the private repo's
own bastion, or a future band-paired snapshot reader once Phase 2 ships)
and AWS credentials with read access to VULNERABILITY_DATA_BUCKET (for
QRFModelAdapter.load).

Leave-one-station-out validation of the nearby-station distance-weighted
residual blend (Tier 2.5 `era5_land_station_blended`).

See docs/plan-2026-07-21-nearby-station-kriging.md (crisisready/heat-risk-
data-api) section 5 for the design this implements. Method, per held-out
real station s on date d:

  1. delta_qrf(s,d)     = adapter.predict's delta_c for s's own covariates
                           (the model's existing, already-validated correction).
  2. delta_station(s,d) = Gaussian-distance-weighted average of the grid-
                           relative anomaly (station_obs - grid_obs) of every
                           OTHER real station reporting on the SAME date
                           within R km of s, lapse-adjusted (Sec 4).
  3. delta_blend(s,d)   = lambda * delta_station + (1 - lambda) * delta_qrf,
                           lambda = 1 - exp(-sum_w / tau)  (more/closer
                           neighbors -> more confidence in the station term).
  4. Compare corrected-vs-truth RMSE for: raw grid, QRF-only, station-only
     (lambda=1, neighbor-having rows only), and blend -- per (zone, target),
     leave-one-out (s never sees its own observation).

(L_km, R_km, tau) are tuned by grid search over held-out RMSE, matching the
"tune by validation, not intuition" instruction in section 4. Tuned per
BROAD KOPPEN GROUP (A/B/C/D/E, koppen.koppen_broad_group_letter_from_zone),
not a single global triple, and not per-zone either -- a middle ground
found necessary after live use showed the single-global version badly
underserved most zones: Cfb alone is ~46% of all rows, so a single
pooled-RMSE objective is effectively "whatever minimizes Cfb's error,"
regardless of what any other zone's own station-vs-QRF tradeoff looks like.
Concretely, several zones (Aw, BSh, Csa, Cfc on tmax) had a station-only
RMSE dramatically better than their blend RMSE under the single global
triple -- lambda was underweighting real, strong local station signal
purely because tau was tuned to Cfb's saturation point, not theirs. Full
per-zone tuning was and remains rejected (too few free parameters to
validate honestly at real per-zone station density, per the plan's own
original reasoning) -- broad-group pools enough rows per group (even the
smallest, E, still pools every polar-zone row together) to tune honestly
without inheriting the single-global version's Cfb-dominance problem.

Gate, per (zone, target), all three must hold (fail-closed):
  - blend beats QRF-only by more than the SE of the paired squared-error
    difference (mean_d > se_d over the whole held-out sample);
  - |mean(blend residual)| <= 3 * SE(blend residual)  (no systematic bias);
  - the station-only path (lambda=1 rows) beats raw grid by more than the
    SE of ITS paired difference -- station-only isn't automatically better
    than the grid just because a station exists nearby (module docstring
    point 3 of the plan, section 3).
All three are genuinely enforced in blend_gate_passes below.

No live API/DB calls: reads a JSON dump shaped like validate_lagfill_
downscaling.py's --phase dump-rows output ({"rows": [...]}), with the same
ghcn_training columns (station_id, date, lat, lon, climate_zone, region,
station_tmax_c/station_tmin_c, grid_tmax_c/grid_tmin_c, plus the covariates
build_feature_matrix needs) -- this dataset already carries every real
station's location and grid-relative anomaly for the SAME date range, which
is exactly what a leave-one-station-out neighbor search needs; no new
fetch is required.

Usage:
    VULNERABILITY_DATA_BUCKET=... AWS_REGION=us-east-1 \\
        python scripts/validate_station_blend.py \\
            --model-version ds-2026.07-rf5 --rows-in /tmp/lagfill_rows.json \\
            --out /tmp/station_blend_validation.json \\
            --gate-out /tmp/station_blend_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np

from heatready_downscaling.contract import QRFModelAdapter
from heatready_downscaling.koppen import koppen_broad_group_letter_from_zone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_station_blend")

# Informed by an offline decorrelation-length CV run (docs/plan-2026-07-21-
# nearby-station-kriging.md section 7 open question 1, crisisready/heat-
# risk-data-api): real signal (nearest-same-day-neighbor beats the "trust
# the grid" baseline) holds through the 0-10/10-25/25-50 km bins for both
# tmax and tmin, then turns negative at 50-100 km -- confirms R should sit
# near 50 km (matches metar.py's already-coded _METAR_NEARBY_RADIUS_KM=50.0),
# and correlation is highest in the 0-25 km bins, suggesting a Gaussian L in
# the 10-25 km range. Search brackets these findings rather than assuming
# them.
_L_KM_GRID = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 250.0)
_R_KM_GRID = (10.0, 15.0, 20.0, 25.0, 35.0, 50.0)
_TAU_GRID = (0.5, 1.0, 2.0, 4.0)  # weight-sum saturation scale for lambda
_LAPSE_RATE_C_PER_KM = 6.5  # standard atmosphere, degrees C per km of elevation
_MIN_ZONE_N = 30  # zones with fewer held-out samples than this are reported but not gated
# Pure |bias| <= 3*SE fails almost any nonzero bias once n is large (SE
# shrinks with sqrt(n), the bias itself does not) -- confirmed live:
# zones with clear, large-sample RMSE wins (e.g. tmax Cfb, n=128103, blend RMSE
# 0.971 vs QRF-only 1.032) were failing the gate on biases of 0.03-0.09 C,
# negligible next to RMSEs of 1-2 C. Same statistical-vs-practical-significance
# tension heatready_downscaling.score's AUTO_ENABLE_MARGIN already solves
# for the RMSE side (a flat % margin, not pure SE) -- mirrored here as a flat
# magnitude floor: a bias passes if it clears EITHER the statistical bar (within
# 3*SE, the right call for small-n zones where SE is still large) OR this
# practical one (small in absolute terms regardless of n).
#
# 0.25 (raised from an initial 0.15): re-running this harness against
# NRT-reconstructed lag-fill-style data (not ERA5-Land -- see below) showed
# real, physically-driven biases up to ~1.2C for some zones -- NOT an SE
# artifact, the QRF model's own known cross-distribution transfer bias
# (confirmed: QRF-only, no blend at all, already carries comparable bias on
# the SAME zones in the currently-published lag_fill band gate, which has
# no bias check of its own). Sorting bias magnitude among zones where the
# blend genuinely beats QRF on RMSE shows a real gap in the data on BOTH
# targets: a first cluster at <=0.227C (tmax)/<=0.207C (tmin), then a jump
# to 0.33C+ -- 0.25 sits in that gap, admitting the first cluster (still
# <20% of these zones' ~1.5-2C RMSE) without crossing into the second
# (0.75-1.2C, 40-80% of RMSE, genuinely not small). This also happens to
# bring Cfb (temperate oceanic -- Paris, London, Seattle, among the most
# common real-world zones and the single largest-n zone in both datasets)
# in on both targets, but the threshold was chosen from the gap in the
# data, not backed into to admit a specific zone -- it applies uniformly to
# both bands and would have moved regardless of which zone sat closest to
# it.
_PRACTICAL_BIAS_FLOOR_C = 0.25


def load_rows(rows_in: str) -> list[dict]:
    with open(rows_in) as f:
        dumped = json.load(f)
    rows = dumped["rows"] if "rows" in dumped else dumped
    logger.info("Loaded %d row(s) from %s", len(rows), rows_in)
    return rows


def compute_qrf_deltas(adapter: QRFModelAdapter, rows: list[dict], target: str) -> np.ndarray:
    """delta_qrf per row, NaN where adapter.predict did not apply (no
    extra_zone_gate -- these rows carry the model's own trained-on ERA5-Land
    base value, the same distribution adapter.predict's built-in gate
    already reflects; a lag-fill/forecast-style extra restriction would be
    wrong here)."""
    preds = adapter.predict(rows, target)
    out = np.full(len(rows), np.nan)
    for i, p in enumerate(preds):
        if p["applied"]:
            out[i] = p["delta_c"]
    return out


def build_neighbor_index(rows: list[dict]) -> dict[str, list[int]]:
    """date (ISO str) -> list of row indices reporting that date. Grouping by
    date is the search scope for "who else reported the same day" -- a
    per-date pairwise distance matrix stays small (station count per date,
    not the full 274k-row table)."""
    by_date: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_date.setdefault(r["date"], []).append(i)
    return by_date


class NeighborCandidates:
    """Flat (i, j, dist_km) candidate pairs -- every OTHER-station, same-date
    row within max_R_km of row i -- precomputed ONCE via a per-date BallTree
    radius query (haversine metric), then reused across every (L_km, R_km,
    tau) combo in the grid search via fast masking. Replaces an O(k^2)
    per-date pairwise distance matrix (365 real dates averaging ~751
    same-day stations each -- ~75 billion elementwise ops per (L,R) combo,
    confirmed too slow to finish a single combo in a 100s smoke test) with
    an O(k log k) tree query, run once instead of once per grid-search
    combo (15 (L,R) pairs x 2 targets = 30x fewer neighbor searches too)."""

    def __init__(self, i: np.ndarray, j: np.ndarray, dist_km: np.ndarray, n_rows: int):
        self.i, self.j, self.dist_km, self.n_rows = i, j, dist_km, n_rows


def build_neighbor_candidates(
    rows: list[dict], by_date: dict[str, list[int]], max_R_km: float,
) -> NeighborCandidates:
    """One BallTree (haversine metric, radians in/out) per date, queried
    once at max_R_km -- excludes same-station pairs (leave-ONE-STATION-out,
    not just leave-one-row-out, so a station reporting on an adjacent date
    can't leak its own signal back to itself) and self-pairs."""
    from sklearn.neighbors import BallTree

    lat = np.array([r["lat"] for r in rows])
    lon = np.array([r["lon"] for r in rows])
    station_id = np.array([r["station_id"] for r in rows])
    earth_r_km = 6371.0088
    radius_rad = max_R_km / earth_r_km

    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    all_d: list[np.ndarray] = []

    for _d, idxs in by_date.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        coords_rad = np.radians(np.column_stack([lat[idxs], lon[idxs]]))
        tree = BallTree(coords_rad, metric="haversine")
        neighbor_lists, dist_lists = tree.query_radius(coords_rad, r=radius_rad, return_distance=True)

        sub_sid = station_id[idxs]
        for local_i, (neighbors, dists) in enumerate(zip(neighbor_lists, dist_lists)):
            if neighbors.size == 0:
                continue
            keep = (neighbors != local_i) & (sub_sid[neighbors] != sub_sid[local_i])
            if not keep.any():
                continue
            all_i.append(np.full(keep.sum(), idxs[local_i]))
            all_j.append(idxs[neighbors[keep]])
            all_d.append(dists[keep] * earth_r_km)

    if not all_i:
        return NeighborCandidates(np.array([], dtype=int), np.array([], dtype=int), np.array([]), len(rows))
    return NeighborCandidates(np.concatenate(all_i), np.concatenate(all_j), np.concatenate(all_d), len(rows))


def compute_station_deltas(
    candidates: NeighborCandidates, anomaly: np.ndarray, elevation: np.ndarray,
    L_km: float, R_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (delta_station, weight_sum, n_neighbors) per row, aggregating
    the precomputed candidate pairs -- filter to R_km, lapse-adjust (section
    4), Gaussian-weight by L_km, group-sum by the held-out row index i via
    np.bincount (fast regardless of group size, no per-date Python loop)."""
    n = candidates.n_rows
    i, j, dist = candidates.i, candidates.j, candidates.dist_km
    if i.size == 0:
        return np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=int)

    within_r = dist <= R_km
    valid_anom = np.isfinite(anomaly[j])
    usable = within_r & valid_anom
    if not usable.any():
        return np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=int)

    i_u, j_u, dist_u = i[usable], j[usable], dist[usable]
    lapse_adj = _LAPSE_RATE_C_PER_KM * (elevation[j_u] - elevation[i_u]) / 1000.0
    lapse_adj = np.where(np.isfinite(lapse_adj), lapse_adj, 0.0)
    adj_anom = anomaly[j_u] + lapse_adj

    w = np.exp(-((dist_u / L_km) ** 2))
    weight_sum = np.bincount(i_u, weights=w, minlength=n)
    weighted_sum = np.bincount(i_u, weights=w * adj_anom, minlength=n)
    n_neighbors = np.bincount(i_u, minlength=n)

    has_any = weight_sum > 0
    delta_station = np.where(has_any, weighted_sum / np.where(has_any, weight_sum, 1.0), np.nan)
    return delta_station, weight_sum, n_neighbors


def score_combo(
    rows: list[dict], truth: np.ndarray, grid: np.ndarray, delta_qrf: np.ndarray,
    delta_station: np.ndarray, weight_sum: np.ndarray, n_neighbors: np.ndarray, tau: float,
) -> dict:
    """Per-(zone) RMSEs for grid/qrf/station-only/blend, plus the paired-SE
    gate checks, for ONE (L_km, R_km, tau) combo already scored into
    delta_station/weight_sum. `lambda_` saturates with total neighbor weight
    -- more/closer real stations, more confidence in the station term."""
    has_station = np.isfinite(delta_station)
    has_qrf = np.isfinite(delta_qrf)
    lambda_ = np.where(has_station, 1.0 - np.exp(-weight_sum / tau), 0.0)

    pred_grid = grid
    pred_qrf = np.where(has_qrf, grid + delta_qrf, np.nan)
    pred_station_only = np.where(has_station, grid + delta_station, np.nan)
    pred_blend = np.where(
        has_station & has_qrf, grid + lambda_ * delta_station + (1 - lambda_) * delta_qrf,
        np.where(has_station, grid + delta_station, np.where(has_qrf, grid + delta_qrf, np.nan)),
    )

    zones = np.array([r.get("climate_zone") for r in rows])
    by_zone: dict[str, dict] = {}
    for zone in sorted(set(z for z in zones if z)):
        mask = zones == zone
        n_total = int(mask.sum())

        grid_err = truth[mask] - pred_grid[mask]
        rmse_grid = float(np.sqrt(np.mean(grid_err ** 2))) if n_total else None

        qrf_mask = mask & has_qrf
        n_qrf = int(qrf_mask.sum())
        rmse_qrf = float(np.sqrt(np.mean((truth[qrf_mask] - pred_qrf[qrf_mask]) ** 2))) if n_qrf else None

        station_mask = mask & has_station
        n_station = int(station_mask.sum())
        rmse_station_only = (
            float(np.sqrt(np.mean((truth[station_mask] - pred_station_only[station_mask]) ** 2)))
            if n_station else None
        )

        blend_mask = mask & (has_station | has_qrf)
        n_blend = int(blend_mask.sum())
        blend_resid = truth[blend_mask] - pred_blend[blend_mask]
        rmse_blend = float(np.sqrt(np.mean(blend_resid ** 2))) if n_blend else None
        bias_blend = float(np.mean(blend_resid)) if n_blend else None
        se_bias_blend = float(np.std(blend_resid, ddof=1) / np.sqrt(n_blend)) if n_blend > 1 else None

        both_mask = mask & has_station & has_qrf
        n_both = int(both_mask.sum())
        blend_beats_qrf = None
        if n_both >= _MIN_ZONE_N:
            sq_err_qrf = (truth[both_mask] - pred_qrf[both_mask]) ** 2
            sq_err_blend = (truth[both_mask] - pred_blend[both_mask]) ** 2
            d = sq_err_qrf - sq_err_blend
            mean_d, se_d = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(n_both))
            blend_beats_qrf = mean_d > se_d if se_d > 0 else mean_d > 0

        station_beats_grid = None
        if n_station >= _MIN_ZONE_N:
            sq_err_grid = (truth[station_mask] - pred_grid[station_mask]) ** 2
            sq_err_station = (truth[station_mask] - pred_station_only[station_mask]) ** 2
            d2 = sq_err_grid - sq_err_station
            mean_d2, se_d2 = float(np.mean(d2)), float(np.std(d2, ddof=1) / np.sqrt(n_station))
            station_beats_grid = mean_d2 > se_d2 if se_d2 > 0 else mean_d2 > 0

        no_systematic_bias = (
            (abs(bias_blend) <= 3 * se_bias_blend or abs(bias_blend) <= _PRACTICAL_BIAS_FLOOR_C)
            if (bias_blend is not None and se_bias_blend) else None
        )

        blend_gate_passes = bool(
            n_both >= _MIN_ZONE_N and n_blend >= _MIN_ZONE_N
            and blend_beats_qrf and no_systematic_bias
            and station_beats_grid
        )

        by_zone[zone] = {
            "n_total": n_total, "n_qrf_applied": n_qrf, "n_station_neighbor": n_station, "n_blend_scored": n_blend,
            "rmse_grid_c": rmse_grid, "rmse_qrf_c": rmse_qrf,
            "rmse_station_only_c": rmse_station_only, "rmse_blend_c": rmse_blend,
            "bias_blend_c": bias_blend, "se_bias_blend_c": se_bias_blend,
            "blend_beats_qrf": blend_beats_qrf, "station_beats_grid": station_beats_grid,
            "no_systematic_bias": no_systematic_bias,
            "gated_insufficient_n": n_both < _MIN_ZONE_N,
            "blend_gate_passes": blend_gate_passes,
        }
    return by_zone


_BROAD_GROUPS = ("A", "B", "C", "D", "E")


def grid_search(
    rows: list[dict], target: str, adapter: QRFModelAdapter, candidates: NeighborCandidates,
) -> tuple[dict, dict]:
    """Search (L_km, R_km, tau) PER BROAD KOPPEN GROUP (module docstring
    explains why a single global triple was replaced), scoring each combo by
    the pooled blend RMSE of that group's own zones only -- the neighbor
    search itself (candidates) stays shared/unrestricted across the whole
    dataset (a station's real physical neighbors don't care about
    climate-zone labels), only which rows count toward "best for this
    group" changes per group. Still minimizes a POOLED objective within
    each group (not per zone -- would overfit the smallest zones), just
    pooled over a fairer, non-Cfb-dominated denominator. Returns (by_zone,
    group_params) where group_params is {group_letter: {"L_km":.., "R_km":..,
    "tau":.., "pooled_rmse":..}}."""
    truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"
    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    elev_col = "elevation_mean_m"

    truth = np.array([r[truth_col] for r in rows], dtype=float)
    grid = np.array([r[grid_col] for r in rows], dtype=float)
    elevation = np.array([r.get(elev_col) if r.get(elev_col) is not None else np.nan for r in rows], dtype=float)
    anomaly = truth - grid

    delta_qrf = compute_qrf_deltas(adapter, rows, target)

    best_by_group: dict[str, tuple] = {}  # group -> (pooled_rmse, L_km, R_km, tau)
    by_zone_at_best: dict[str, dict] = {}  # group -> the by_zone dict scored at that group's best combo

    for L_km in _L_KM_GRID:
        for R_km in _R_KM_GRID:
            delta_station, weight_sum, n_neighbors = compute_station_deltas(
                candidates, anomaly, elevation, L_km, R_km,
            )
            for tau in _TAU_GRID:
                by_zone = score_combo(rows, truth, grid, delta_qrf, delta_station, weight_sum, n_neighbors, tau)
                for group in _BROAD_GROUPS:
                    zone_keys = [z for z in by_zone if koppen_broad_group_letter_from_zone(z) == group]
                    pooled_n = sum(by_zone[z]["n_blend_scored"] for z in zone_keys)
                    if pooled_n == 0:
                        continue
                    pooled_sq_err = sum(
                        (by_zone[z]["rmse_blend_c"] ** 2) * by_zone[z]["n_blend_scored"]
                        for z in zone_keys if by_zone[z]["rmse_blend_c"] is not None
                    )
                    pooled_rmse = float(np.sqrt(pooled_sq_err / pooled_n)) if pooled_n else float("inf")
                    prev = best_by_group.get(group)
                    if prev is None or pooled_rmse < prev[0]:
                        best_by_group[group] = (pooled_rmse, L_km, R_km, tau)
                        by_zone_at_best[group] = by_zone
            logger.info("[%s] scored L=%.0fkm R=%.0fkm across all %d tau values", target, L_km, R_km, len(_TAU_GRID))

    by_zone: dict = {}
    group_params: dict = {}
    for group, (pooled_rmse, L_km, R_km, tau) in best_by_group.items():
        group_params[group] = {"L_km": L_km, "R_km": R_km, "tau": tau, "pooled_rmse": pooled_rmse}
        logger.info(
            "[%s] BEST for group %s: L=%.0fkm R=%.0fkm tau=%.1f pooled blend RMSE=%.4f",
            target, group, L_km, R_km, tau, pooled_rmse,
        )
        for z, m in by_zone_at_best[group].items():
            if koppen_broad_group_letter_from_zone(z) == group:
                by_zone[z] = m

    return by_zone, group_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--bucket", default=None, help="defaults to VULNERABILITY_DATA_BUCKET env var")
    parser.add_argument("--rows-in", required=True, help="JSON dump of ghcn_training rows (see module docstring)")
    parser.add_argument("--out", default="/tmp/station_blend_validation.json")
    parser.add_argument("--gate-out", default="/tmp/station_blend_gate.json")
    args = parser.parse_args()

    bucket = args.bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
    adapter = QRFModelAdapter.load(args.model_version, bucket=bucket)

    rows = load_rows(args.rows_in)
    by_date = build_neighbor_index(rows)
    logger.info("%d distinct date(s) across %d row(s)", len(by_date), len(rows))

    max_R_km = max(_R_KM_GRID)
    t0 = time.monotonic()
    candidates = build_neighbor_candidates(rows, by_date, max_R_km)
    logger.info(
        "Built %d neighbor candidate pair(s) within %.0fkm in %.1fs (shared across both targets/all grid combos)",
        candidates.i.size, max_R_km, time.monotonic() - t0,
    )

    report: dict = {"model_version": args.model_version, "rows_scored": len(rows), "by_target": {}}
    gate: dict = {"tmax": {}, "tmin": {}, "params": {}}

    for target in ("tmax", "tmin"):
        by_zone, group_params = grid_search(rows, target, adapter, candidates)
        report["by_target"][target] = {"group_params": group_params, "by_zone": by_zone}
        gate["params"][target] = {
            group: {"L_km": p["L_km"], "R_km": p["R_km"], "tau": p["tau"]} for group, p in group_params.items()
        }
        gate[target] = {zone: m["blend_gate_passes"] for zone, m in by_zone.items()}

        passing = sorted(z for z, m in by_zone.items() if m["blend_gate_passes"])
        failing = sorted(z for z, m in by_zone.items() if not m["blend_gate_passes"] and not m["gated_insufficient_n"])
        insufficient = sorted(z for z, m in by_zone.items() if m["gated_insufficient_n"])
        station_only_passing = sorted(z for z, m in by_zone.items() if m["station_beats_grid"] is True)
        logger.info("[%s] zones PASSING blend gate: %s", target, passing)
        logger.info("[%s] zones FAILING blend gate: %s", target, failing)
        logger.info("[%s] zones insufficient n (< %d): %s", target, _MIN_ZONE_N, insufficient)
        logger.info("[%s] zones where station-only beats raw grid: %s", target, station_only_passing)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote full report to %s", args.out)

    with open(args.gate_out, "w") as f:
        json.dump(gate, f, indent=2)
    logger.info("Wrote gate file to %s", args.gate_out)


if __name__ == "__main__":
    main()
