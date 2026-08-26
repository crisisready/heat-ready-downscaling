"""
Worked Rung B/L1 example (roadmap Phase 2's "contributor front door" -- see
examples/README.md): fit a covariate-linear correction on a real station
subset, then independently reproduce its score the same way the referee does,
producing a real `claimed_report.json` you could open a Rung B submission PR
with (see CONTRIBUTING.md's "What Rung B actually asks of you").

TWO HALVES, DELIBERATELY SEPARATE, because the handover for this ticket
originally conflated them (see the correction in
docs/handover-2026-08-26-crowdsourcing-next-slice.md):

1. FITTING (this script's own job -- score.py never does this). A plain
   least-squares intercept+slope on (covariate, station-minus-grid residual)
   over a named station subset. Nothing sophisticated: Valencia's own real
   validated result (registry/local/valencia-coast-v1) is exactly this shape,
   one term, fit the same way.
2. SCORING (score.score_band's `covariate_linear` proposed_correction path,
   via the SAME `run_submission.reproduce()` the referee calls) --
   independently evaluates the fitted intercept/slope against every row in
   the zone, with a station-cluster bootstrap CI on the RMSE improvement.
   This half never re-fits anything; it only checks whether the declared
   value generalizes.

THE NAMED EXAMPLE: tmax/Am/lag_fill, ds-2026.07-rf5, corrected on
`wc_tree_frac` (per-location tree-canopy land-cover fraction; one of
`score.STATIC_COVARIATE_ALLOWLIST`'s admitted static covariates) -- a real, interpretable
urban/canopy-cooling signal (higher canopy fraction -> the grid overstates
tmax relative to station truth) fit on all 12 GHCN stations snapshot v2026.07
carries for the Am zone's lag_fill band. As of that snapshot this clears the
Rung B/L1 bar (CI excludes zero, covariate earns its keep over a flat
constant) on both the whole-year and hot-day strata -- like the Rung A
example, a real result from real data, not a fixture, and can come back
differently against a future snapshot.

Usage:
    python examples/rung_b_l1_covariate_linear_fit.py
    python examples/rung_b_l1_covariate_linear_fit.py --zone BSk --target tmin --covariate elevation_mean_m
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import run_submission  # noqa: E402

from heatready_downscaling import report, score, snapshot  # noqa: E402

MODEL_VERSION = "ds-2026.07-rf5"
BAND_KEY = "lag_fill"
SNAPSHOT_VERSION = "v2026.07"

DEFAULT_TARGET = "tmax"
DEFAULT_ZONE = "Am"
DEFAULT_COVARIATE = "wc_tree_frac"


def fit_covariate_linear(
    band_rows: list[dict], zone: str, target: str, covariate: str,
) -> tuple[dict, int]:
    """Plain least-squares intercept+slope of (station_truth - grid) on
    `covariate`, over every row in `zone` with both a paired truth/grid value
    and a non-null covariate. Returns (correction_entry, n_stations) in the
    exact shape score.score_band's `covariate_linear` proposed_correction
    expects for one zone (registry.py/score.py's own schema) -- `basis:
    raw_grid` so the correction is scoreable even where the shipped model's
    own CV gate does not apply (CONTRIBUTING.md's own basis guidance).

    Deliberately unweighted OLS on raw station-days, not a per-station mean
    first -- with enough stations this is the same choice Valencia's own fit
    made (see registry/local/valencia-coast-v1/manifest.yaml)."""
    import numpy as np

    truth_col, grid_col = f"station_{target}_c", f"grid_{target}_c"
    rows = [
        r for r in band_rows
        if r["climate_zone"] == zone and r.get(truth_col) is not None
        and r.get(grid_col) is not None and r.get(covariate) is not None
    ]
    stations = sorted({r["station_id"] for r in rows})
    if len(stations) < 2:
        raise SystemExit(
            f"{zone}/{target}/{covariate} has only {len(stations)} station(s) with usable rows -- "
            "at least 2 distinct stations are needed for the bootstrap CI to mean anything "
            "(see CONTRIBUTING.md's Rung B section)",
        )

    residual = np.array([r[truth_col] - r[grid_col] for r in rows])
    cov_values = np.array([r[covariate] for r in rows])
    design = np.vstack([cov_values, np.ones_like(cov_values)]).T
    slope, intercept = np.linalg.lstsq(design, residual, rcond=None)[0]

    entry = {
        "basis": "raw_grid",
        "intercept": float(intercept),
        "terms": [{"covariate": covariate, "slope": float(slope)}],
        # The exact range the fit is evidence over -- score_band excludes any
        # row outside it rather than extrapolating (see its own docstring on
        # why: Valencia's wider-cluster test degraded badly past its fit range).
        "valid_range": [[float(cov_values.min()), float(cov_values.max())]],
    }
    return entry, len(stations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=["tmax", "tmin"])
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--covariate", default=DEFAULT_COVARIATE, choices=list(score.STATIC_COVARIATE_ALLOWLIST))
    parser.add_argument("--snapshot-version", default=SNAPSHOT_VERSION)
    parser.add_argument("--cache-root", default=".cache/snapshots")
    parser.add_argument("--out", default="examples/output/rung_b_l1_claimed_report.json")
    parser.add_argument(
        "--require-pass", action="store_true", default=True,
        help="fail (exit 1) unless the fitted correction's CI excludes zero on the whole-year "
        "stratum -- keeps this example honest about what it claims when run in CI (default: on)",
    )
    args = parser.parse_args()

    print(f"Downloading public snapshot {args.snapshot_version} (cached under {args.cache_root})...")
    snapshot_dir = run_submission.download_snapshot(args.snapshot_version, args.cache_root)
    band_rows = snapshot.read_band_partitions(snapshot_dir, BAND_KEY)

    print(f"Fitting a covariate-linear correction on {args.zone}/{args.target}/{args.covariate}...")
    entry, n_stations = fit_covariate_linear(band_rows, args.zone, args.target, args.covariate)
    print(f"  fit over {n_stations} distinct stations: intercept={entry['intercept']:.4f}, "
          f"slope={entry['terms'][0]['slope']:.4f}, valid_range={entry['valid_range'][0]}")

    print("Reproducing the score the referee would compute for this DECLARED candidate "
          "(never re-fit -- see score_band's own docstring)...")
    candidate = {args.target: {args.zone: entry}}
    reproduced = run_submission.reproduce(
        snapshot_dir, MODEL_VERSION, BAND_KEY, args.snapshot_version, candidate=candidate,
    )
    report.validate_report(reproduced)

    zone_result = reproduced["by_target"][args.target][args.zone]
    all_stratum = zone_result["proposed_correction_by_stratum"]["all"]
    hot_stratum = zone_result["proposed_correction_by_stratum"]["hot_day"]

    print(f"\n{args.target}/{args.zone}/{BAND_KEY}, correction on {args.covariate}, "
          f"snapshot {args.snapshot_version}:")
    for name, stratum in (("all", all_stratum), ("hot_day", hot_stratum)):
        ci = stratum["rmse_improvement_ci95_pct"]
        ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]" if ci else "None (fewer than 2 resampleable stations)"
        print(f"  [{name:>7}] rmse_improvement={stratum['rmse_improvement_pct']:.1%}, "
              f"CI95={ci_str}, verdict={stratum['verdict']}, "
              f"covariate_earns_keep={stratum['covariate_earns_keep']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reproduced, f, indent=2, default=str)
    print(f"\nWrote {args.out} -- this IS a real claimed_report.json. Your submission's "
          "manifest.yaml would declare method.candidate = "
          f"{json.dumps(candidate, indent=2)}")

    if args.require_pass and all_stratum["verdict"] != "pass":
        raise SystemExit(
            f"{args.target}/{args.zone}/{args.covariate}'s whole-year verdict is "
            f"{all_stratum['verdict']!r}, not 'pass', as of snapshot {args.snapshot_version} -- "
            "this example's claim is stale (pass --zone/--target/--covariate to explore a new one)",
        )


if __name__ == "__main__":
    main()
