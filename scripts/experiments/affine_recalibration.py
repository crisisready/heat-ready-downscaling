"""
Phase 1 (plan-2026-07-28-lagfill-base-mismatch-fix.md, section 5) -- Option R.

For every (target, climate_zone) in the `lag_fill` band, compares three
correction variants against the frozen ds-2026.07-rf5 predictions:

  - no_correction  : raw QRF delta, unmodified
  - debias_only    : today's shipped correction (score_band's bias_correction_c,
                      an offset added to grid+delta), out-of-fold via
                      station-grouped CV
  - affine         : e ~= alpha*d + b (scale + offset applied to delta),
                      out-of-fold via the same station-grouped CV folds
  - grid_debias_only : e ~= b (no model at all -- discards the QRF's delta
                      entirely, just bias-corrects the raw grid), out-of-fold
                      via the same folds. Added on an Opus scoping-consult
                      finding (2026-07-28): without this baseline, Option R's
                      Cfb win looks like "the model's correction, rescaled"
                      when for tmax it is mostly "the Open-Meteo grid's own
                      bias, corrected" -- affine only adds ~2 points of a
                      17-point gain over this baseline. Must be in the
                      selection set, not just the writeup, or the choice
                      between affine/debias_only silently overstates what
                      the QRF model is contributing.

Selection is per-cell, out-of-fold-RMSE-driven (plan section 5 step 7) --
never a blanket "always use affine" rule. Reuses score.score_band verbatim
for the no_correction/debias_only numbers (rather than reimplementing them)
so this script's incumbent baseline is byte-identical to the real gate; the
affine arm is computed here since score_band has no scale term.

Run: /root/projects/crisisready/heat-ready-downscaling/.venv/bin/python \
     scripts/experiments/affine_recalibration.py
Writes: /tmp/lagfill_exp/affine_recalibration.json
"""

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from heatready_downscaling import contract, snapshot  # noqa: E402
from heatready_downscaling.score import (  # noqa: E402
    AUTO_ENABLE_MARGIN,
    BIAS_CV_FOLDS,
    BIAS_CV_MIN_STATIONS,
    MIN_ZONE_N,
    PRACTICAL_BIAS_FLOOR_C,
    score_band,
)

SNAPSHOT_DIR = "/tmp/release_recheck"
MODEL_VERSION = "ds-2026.07-rf5"
BAND = "lag_fill"
FOLD_SALT = "v2026.07"
OUTPUT_PATH = Path("/tmp/lagfill_exp/affine_recalibration.json")


def fold_of_station(station_id: str) -> int:
    return int(hashlib.md5(f"{FOLD_SALT}:{station_id}".encode()).hexdigest(), 16) % BIAS_CV_FOLDS


def bias_ok(bias_val, se_val):
    if bias_val is None:
        return None
    if se_val is not None:
        return abs(bias_val) <= 3 * se_val or abs(bias_val) <= PRACTICAL_BIAS_FLOOR_C
    return abs(bias_val) <= PRACTICAL_BIAS_FLOOR_C


def candidate_gate(rmse_selected, rmse_grid, bias_val, se_val, n_qrf, cv_eligible):
    if rmse_selected is None or rmse_grid is None or rmse_grid <= 0:
        return False
    if n_qrf < MIN_ZONE_N:
        return False
    margin = (rmse_grid - rmse_selected) / rmse_grid
    margin_ok = margin >= AUTO_ENABLE_MARGIN
    ok_bias = bias_ok(bias_val, se_val)
    if cv_eligible:
        return bool(margin_ok and ok_bias)
    # thin-station fallback mirrors score_band's qrf_beats_grid_with_margin
    # fallback: raw margin AND bias_bounded_uncorrected, never margin alone.
    return bool(margin_ok and ok_bias)


def main():
    rows = snapshot.read_band_partitions(SNAPSHOT_DIR, BAND)
    adapter = contract.FrozenPredictionAdapter.from_snapshot(SNAPSHOT_DIR, MODEL_VERSION, BAND)

    output = {
        "snapshot_dir": SNAPSHOT_DIR,
        "model_version": MODEL_VERSION,
        "band": BAND,
        "fold_salt": FOLD_SALT,
        "cells": {},
    }

    for target in ("tmax", "tmin"):
        grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
        truth_col = "station_tmax_c" if target == "tmax" else "station_tmin_c"

        preds = adapter.predict(rows, target)
        incumbent = score_band(adapter, rows, target, fold_salt=FOLD_SALT)

        by_zone = {}
        for r, p in zip(rows, preds):
            zone = r.get("climate_zone")
            if zone is None or not p["applied"]:
                continue
            b = by_zone.setdefault(zone, {"station_ids": [], "d": [], "e": []})
            b["station_ids"].append(str(r.get("station_id")))
            b["d"].append(p["delta_c"])
            b["e"].append(r[truth_col] - r[grid_col])

        for zone, inc in incumbent.items():
            n_qrf = inc["n_qrf_applied"]
            rmse_grid = inc["rmse_grid_c"]
            rmse_qrf_raw = inc["rmse_qrf_c"]
            bias_qrf_raw = inc["bias_qrf_c"]
            se_bias_qrf_raw = inc["se_bias_qrf_c"]
            rmse_debiased_cv = inc["rmse_debiased_cv_c"]
            bias_debiased_cv = inc["bias_debiased_cv_c"]

            b = by_zone.get(zone, {"station_ids": [], "d": [], "e": []})
            station_ids = np.array(b["station_ids"])
            d = np.array(b["d"], dtype=float)
            e = np.array(b["e"], dtype=float)
            unique_stations = np.unique(station_ids)
            n_stations = int(len(unique_stations))
            cv_eligible = n_stations >= BIAS_CV_MIN_STATIONS

            rmse_affine_cv = bias_affine_cv = se_bias_affine_cv = None
            alpha_publish = b_publish = None
            rmse_grid_debias_cv = bias_grid_debias_cv = se_bias_grid_debias_cv = None
            b_grid_debias_publish = None
            debiased_cv_check = None  # internal consistency check against score_band
            if cv_eligible and len(d) > 0:
                fold_lookup = {sid: fold_of_station(sid) for sid in unique_stations}
                row_folds = np.array([fold_lookup[sid] for sid in station_ids])

                oof_affine = np.empty(len(d))
                oof_debias = np.empty(len(d))
                oof_grid_debias = np.empty(len(d))
                for k in range(BIAS_CV_FOLDS):
                    train_mask, test_mask = row_folds != k, row_folds == k
                    if not test_mask.any():
                        continue
                    # affine: e ~= alpha*d + b, fit on other folds' stations
                    X_train = np.column_stack([d[train_mask], np.ones(train_mask.sum())])
                    coef, *_ = np.linalg.lstsq(X_train, e[train_mask], rcond=None)
                    alpha_k, b_k = float(coef[0]), float(coef[1])
                    oof_affine[test_mask] = e[test_mask] - (alpha_k * d[test_mask] + b_k)

                    # debias-only, recomputed here (not just taken from score_band)
                    # to get per-row residuals for the bias/SE check below; the
                    # aggregate RMSE is cross-checked against score_band's own
                    # rmse_debiased_cv_c as an integrity check.
                    qrf_err_train = e[train_mask] - d[train_mask]
                    fold_bias = float(np.mean(qrf_err_train)) if train_mask.any() else 0.0
                    oof_debias[test_mask] = (e[test_mask] - d[test_mask]) - fold_bias

                    # grid-debias-only: e ~= b, no d at all -- discards the QRF
                    # delta entirely and just bias-corrects the raw grid.
                    b_grid_k = float(np.mean(e[train_mask])) if train_mask.any() else 0.0
                    oof_grid_debias[test_mask] = e[test_mask] - b_grid_k

                rmse_affine_cv = float(np.sqrt(np.mean(oof_affine ** 2)))
                bias_affine_cv = float(np.mean(oof_affine))
                se_bias_affine_cv = float(np.std(oof_affine, ddof=1) / np.sqrt(len(oof_affine))) if len(oof_affine) > 1 else None

                debiased_cv_check = float(np.sqrt(np.mean(oof_debias ** 2)))

                rmse_grid_debias_cv = float(np.sqrt(np.mean(oof_grid_debias ** 2)))
                bias_grid_debias_cv = float(np.mean(oof_grid_debias))
                se_bias_grid_debias_cv = (
                    float(np.std(oof_grid_debias, ddof=1) / np.sqrt(len(oof_grid_debias))) if len(oof_grid_debias) > 1 else None
                )

                # publish-time fits: refit on the full zone (CV validates, the
                # shipped correction uses everything -- same pattern as
                # score_band's bias_correction_c).
                X_full = np.column_stack([d, np.ones(len(d))])
                coef_full, *_ = np.linalg.lstsq(X_full, e, rcond=None)
                alpha_publish, b_publish = float(coef_full[0]), float(coef_full[1])
                b_grid_debias_publish = float(np.mean(e))

            candidates = {
                "no_correction": {
                    "rmse_c": rmse_qrf_raw,
                    "bias_c": bias_qrf_raw,
                    "se_bias_c": se_bias_qrf_raw,
                },
                "debias_only": {
                    "rmse_c": rmse_debiased_cv,
                    "bias_c": bias_debiased_cv,
                    "se_bias_c": None,
                },
                "affine": {
                    "rmse_c": rmse_affine_cv,
                    "bias_c": bias_affine_cv,
                    "se_bias_c": se_bias_affine_cv,
                    "alpha": alpha_publish,
                    "b": b_publish,
                },
                "grid_debias_only": {
                    "rmse_c": rmse_grid_debias_cv,
                    "bias_c": bias_grid_debias_cv,
                    "se_bias_c": se_bias_grid_debias_cv,
                    "b": b_grid_debias_publish,
                },
            }
            for name, c in candidates.items():
                c["passes_gate"] = candidate_gate(
                    c["rmse_c"], rmse_grid, c["bias_c"], c["se_bias_c"], n_qrf, cv_eligible,
                )
                c["improvement_pct_vs_grid"] = (
                    (rmse_grid - c["rmse_c"]) / rmse_grid if (c["rmse_c"] is not None and rmse_grid) else None
                )

            # Selection is lowest OOF RMSE AMONG PASSING candidates, not lowest
            # RMSE full stop -- picking the RMSE-optimal candidate regardless of
            # whether IT clears the bias floor understates how many zones are
            # fine, because production always ships at least the debias offset
            # once CV is possible (a raw uncorrected delta with a large bias can
            # have a lower point-estimate RMSE than the debiased version while
            # still being undeployable on its own). best_rmse_variant is kept
            # separately, purely informational, so losers are still visible.
            valid_rmse = {name: c["rmse_c"] for name, c in candidates.items() if c["rmse_c"] is not None}
            best_rmse_variant = min(valid_rmse, key=valid_rmse.get) if valid_rmse else None
            passing_rmse = {name: rmse for name, rmse in valid_rmse.items() if candidates[name]["passes_gate"]}
            selected_variant = min(passing_rmse, key=passing_rmse.get) if passing_rmse else None
            selected = candidates.get(selected_variant, {}) if selected_variant else {}

            beats_incumbent = (
                selected_variant not in (None, "debias_only")
                and rmse_debiased_cv is not None
                and selected.get("rmse_c") is not None
                and selected["rmse_c"] < rmse_debiased_cv
            )

            output["cells"].setdefault(target, {})[zone] = {
                "n_grid": inc["n_grid"],
                "n_qrf_applied": n_qrf,
                "n_stations": n_stations,
                "rmse_grid_c": rmse_grid,
                "candidates": candidates,
                "best_rmse_variant": best_rmse_variant,
                "selected_variant": selected_variant,
                "selected_gate_pass": selected_variant is not None,
                "recommended_change_from_incumbent": bool(beats_incumbent),
                "spatial_skill": bool(inc.get("qrf_beats_grid")) if inc.get("qrf_beats_grid") is not None else None,
                "gated_insufficient_n": inc["gated_insufficient_n"],
                "debiased_cv_rmse_internal_check_c": debiased_cv_check,
                "debiased_cv_rmse_scoreband_c": rmse_debiased_cv,
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {OUTPUT_PATH}")

    # Quick console summary for the run log.
    for target, zones in output["cells"].items():
        print(f"\n=== {target} ===")
        for zone in sorted(zones):
            c = zones[zone]
            print(
                f"{zone:>5} n_qrf={c['n_qrf_applied']:>6} stations={c['n_stations']:>4} "
                f"grid={c['rmse_grid_c']:.3f} selected={c['selected_variant']} "
                f"pass={c['selected_gate_pass']} change={c['recommended_change_from_incumbent']}"
            )


if __name__ == "__main__":
    main()
