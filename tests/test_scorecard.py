"""Scorecard v1: manifest hashing, units and pooling, the ship rule, report blocks, and the runner's
corpus selection and served-prediction path (real quantile_forest fits on tiny synthetic data)."""
import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest

from heatready_downscaling import scorecard as sc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = sc.load_spec(os.path.join(REPO, "scorecard/v1/spec.yaml"))


def _spec(**kw):
    return {**SPEC, "bootstrap_reps": 50, **kw}


def _frame(stations_by_zone, n_days=40, inc_err=1.0, cand_err=None, seed=0, year=2025, targets=("tmax", "tmin")):
    """Synthetic paired rows. cand_err: {zone: error scale} (default equal to the incumbent)."""
    rng = np.random.default_rng(seed)
    rows, st = [], []
    for zone, n in stations_by_zone.items():
        for i in range(n):
            sid = f"{zone}-{i:02d}"
            st.append({"station_id": sid, "zone": zone, "zone_group": sc.zone_group(zone, SPEC), "region": "XX",
                       "airport": "no", "setting": "rural", "visibility": "public", "lat": 10.0 + i, "lon": 20.0})
            for t in targets:
                obs = rng.normal(-1.0, 1.5, n_days)
                noise = rng.normal(0, 1, n_days)
                ce = (cand_err or {}).get(zone, inc_err)
                for d in range(n_days):
                    rows.append({"station_id": sid, "date": f"{year}-01-{d % 28 + 1:02d}" if d < 28 else f"{year}-02-{d - 27:02d}",
                                 "year": year, "target": t, "zone": zone, "delta_obs": obs[d],
                                 "grid_c": 30.0 + rng.normal(0, 3), "inc_delta": obs[d] + inc_err * noise[d],
                                 "cand_delta": obs[d] + ce * noise[d], "inc_ci95": 2.0, "cand_ci95": 2.0,
                                 "dist_km": 25.0})
    return pd.DataFrame(rows), pd.DataFrame(st)


def _decl(zones=("BSh",), metric="rmse_tmax", min_effect=0.02):
    return {"candidate": "c", "incumbent": "i", "declared_at": "2026-10-02", "kind": "layer",
            "aim": {"subset": {"zones": list(zones)}, "metric": metric, "min_effect_c": min_effect}}


def _run(rows, st, spec, decl):
    res = sc.score(rows, st, spec)
    aim = sc.score_aim(rows, st, spec, decl)
    return res, aim, sc.ship_decision(res, aim, spec)


# ---------------------------------------------------------------- spec + manifest

def test_committed_spec_is_valid_and_frozen_values():
    assert SPEC["zone_tolerance_rmse_c"] == 0.10
    assert SPEC["min_stations_per_zone"] == 8
    assert SPEC["tiers"]["fast"]["cutoffs"] == ["2024-12-31"]


def test_manifest_hash_is_order_insensitive_and_refuses_duplicates():
    a = [("S1", "2025-01-01", "tmax"), ("S2", "2025-01-02", "tmin")]
    assert sc.manifest_sha256(a) == sc.manifest_sha256(list(reversed(a)))
    assert sc.manifest_sha256(a) != sc.manifest_sha256(a[:1])
    with pytest.raises(ValueError):
        sc.manifest_sha256(a + a[:1])


def test_score_year_requires_year_end_cutoff():
    assert sc.score_year("2024-12-31") == 2025
    with pytest.raises(ValueError):
        sc.score_year("2024-06-30")


def test_zone_group_listed_and_fallback():
    assert sc.zone_group("BSh", SPEC) == "arid"
    assert sc.zone_group("Dwd", SPEC) == "cold"
    assert sc.zone_group("Aq", SPEC) == "tropical"   # unlisted, Koppen letter fallback
    assert sc.zone_group("Ocean", SPEC) == "unassigned"


def test_spec_rejects_zone_in_two_groups():
    bad = {**SPEC, "zone_groups": {"arid": ["BSh"], "tropical": ["BSh"]}}
    with pytest.raises(ValueError):
        sc.validate_spec(bad)


# ---------------------------------------------------------------- units

def test_thin_zones_pool_and_pooled_below_min_n_does_not_gate():
    rows, _ = _frame({"BSh": 10, "BWh": 3, "Cfc": 4, "temperate": 2}, n_days=5)
    units = {u.name: u for u in sc.build_units(rows, SPEC)}
    assert units["BSh"].gating and units["BSh"].kind == "zone"
    pooled = [u for u in units.values() if u.kind == "pooled"]
    assert {tuple(u.zones) for u in pooled} == {("BWh",), ("Cfc", "temperate")}
    assert all(not u.gating for u in pooled)   # 3 and 6 stations, both below 8


def test_station_counts_are_per_target():
    rows, _ = _frame({"BSh": 9}, n_days=3, targets=("tmax",))
    rows2, _ = _frame({"BSh": 4}, n_days=3, targets=("tmin",), seed=1)
    units = sc.build_units(pd.concat([rows, rows2]), SPEC)
    bsh = [u for u in units if u.name == "BSh"][0]
    assert bsh.n_stations == {"tmax": 9}       # 4 tmin stations do not make BSh a tmin unit
    assert any(u.kind == "pooled" and u.n_stations.get("tmin") == 4 for u in units)


# ---------------------------------------------------------------- ship rule

def test_identical_candidate_fails_only_on_aim():
    rows, st = _frame({"BSh": 10, "Csa": 9})
    res, aim, dec = _run(rows, st, _spec(), _decl())
    assert res["global"]["d_rmse_tmax"] == pytest.approx(0.0)
    assert not dec["pass"]
    assert len(dec["reasons"]) == 1 and dec["reasons"][0].startswith("aim rmse_tmax")


def test_better_everywhere_passes():
    rows, st = _frame({"BSh": 10, "Csa": 9}, cand_err={"BSh": 0.7, "Csa": 0.9})
    res, aim, dec = _run(rows, st, _spec(), _decl())
    assert aim["delta"] < -0.02
    assert dec["pass"], dec["reasons"]


def test_zone_worse_than_tolerance_blocks_even_if_global_better():
    rows, st = _frame({"BSh": 30, "Csa": 9}, cand_err={"BSh": 0.5, "Csa": 1.3})
    res, aim, dec = _run(rows, st, _spec(), _decl())
    assert res["global"]["d_rmse_tmax"] < 0
    assert not dec["pass"]
    assert any(r.startswith("Csa rmse_tmax within tolerance") for r in dec["reasons"])


def test_group_worse_than_raw_era5_blocks():
    rows, st = _frame({"BSh": 10}, inc_err=4.0, cand_err={"BSh": 3.5})
    res, aim, dec = _run(rows, st, _spec(), _decl())
    assert any("not worse than raw ERA5-Land" in r for r in dec["reasons"])


def test_aim_with_no_rows_fails():
    rows, st = _frame({"BSh": 10}, cand_err={"BSh": 0.5})
    res, aim, dec = _run(rows, st, _spec(), _decl(zones=("Af",)))
    assert aim["delta"] is None and not dec["pass"]
    assert any("aim has no rows" in r for r in dec["reasons"])


def test_declaration_validation():
    with pytest.raises(ValueError):
        sc.validate_declaration(_decl(min_effect=0))
    with pytest.raises(ValueError):
        sc.validate_declaration(_decl(metric="r2"))
    bad = _decl()
    bad["aim"]["subset"] = {}
    with pytest.raises(ValueError):
        sc.validate_declaration(bad)


# ---------------------------------------------------------------- report and privacy

def test_hot_days_follow_the_grid_not_the_station():
    rows = pd.DataFrame({"station_id": ["A"] * 10, "year": 2025, "target": "tmax",
                         "grid_c": np.arange(10.0), "delta_obs": -np.arange(10.0)})
    hot = sc.flag_hot_days(rows, 0.9)
    assert hot.tolist() == [False] * 9 + [True]


def test_private_stations_never_enter_public_blocks():
    rows, st = _frame({"BSh": 10}, cand_err={"BSh": 0.5})
    st.loc[st["station_id"] == "BSh-00", "visibility"] = "private"
    res = sc.score(rows, st, _spec())
    assert res["global"]["n_stations_tmax"] == 9
    assert list(res["private"]) == ["BSh-00"]


def test_rows_for_unknown_station_are_refused():
    rows, st = _frame({"BSh": 3}, n_days=2)
    with pytest.raises(ValueError):
        sc.score(rows, st[st["station_id"] != "BSh-00"], _spec())


def test_bootstrap_is_deterministic_and_brackets_point():
    rows, st = _frame({"BSh": 12}, cand_err={"BSh": 0.8})
    a = sc.score(rows, st, _spec())["report"]["bootstrap_ci95"]
    b = sc.score(rows, st, _spec())["report"]["bootstrap_ci95"]
    assert a == b
    pt = sc.score(rows, st, _spec())["global"]["d_rmse_tmax"]
    assert a["d_rmse_tmax"][0] <= pt <= a["d_rmse_tmax"][1]


def test_within_city_correlation_needs_five_stations():
    rows, st = _frame({"BSh": 6}, n_days=10, cand_err={"BSh": 0.5})
    st["lat"], st["lon"] = 10.0 + np.arange(6) * 0.01, 20.0
    rows["city"] = rows["station_id"].map(sc.city_clusters(st, 30))
    out = sc.within_city_anomaly_corr(sc._errors(rows), _spec())
    assert out["cities"] == 1 and out["corr_cand"] > out["corr_inc"]
    st2 = st.assign(lat=10.0 + np.arange(6) * 5.0)
    rows["city"] = rows["station_id"].map(sc.city_clusters(st2, 30))
    assert sc.within_city_anomaly_corr(sc._errors(rows), _spec())["cities"] == 0


def test_zone_table_renders():
    rows, st = _frame({"BSh": 10, "Cfc": 4}, cand_err={"BSh": 0.8})
    res, aim, dec = _run(rows, st, _spec(), _decl())
    md = sc.render_zone_table(res, dec, "t")
    assert "**Ship rule: " in md and "| BSh |" in md and "pooled: Cfc" in md


# ---------------------------------------------------------------- committed truth set

def test_committed_truth_set_matches_holdout_and_dedupe():
    st = pd.read_csv(os.path.join(REPO, "scorecard/v1/truth_stations.csv"))
    with open(os.path.join(REPO, "scorecard/v1/holdout_stations.txt")) as f:
        hold = [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]
    assert len(hold) == 8
    assert set(st.loc[st["origin"] == "moved_2026-10-01", "station_id"]) == set(hold)
    assert st["station_id"].is_unique
    removed = pd.read_csv(os.path.join(REPO, "scorecard/v1/dedupe_removed.csv"))
    assert not set(removed["station_id"]) & set(st["station_id"])
    assert (st["zone_group"] == st["zone"].map(lambda z: sc.zone_group(z, SPEC))).all()


# ---------------------------------------------------------------- runner

def test_recipe_corpus_cutoff_sources_and_truth_exclusion():
    import scorecard as runner
    rows = [{"station_id": "T1", "date": "2023-05-01", "_src": "ghcn_training"},
            {"station_id": "A", "date": "2023-05-01", "_src": "ghcn_training"},
            {"station_id": "A", "date": "2025-05-01", "_src": "ghcn_training"},
            {"station_id": "G", "date": "2020-05-01", "_src": "rows_break.json"}]
    rf6 = {"extra_rows": []}
    rf8b = {"extra_rows": [{"name": "rows_break.json"}]}
    assert [r["station_id"] for r in runner.recipe_corpus(rows, rf6, "2024-12-31", {"T1"})] == ["A"]
    assert {r["station_id"] for r in runner.recipe_corpus(rows, rf8b, "2024-12-31", {"T1"})} == {"A", "G"}


def test_manifest_keys_and_check(tmp_path):
    import scorecard as runner
    rows = [{"station_id": "T1", "date": "2025-01-01", "delta_tmax_c": 1.0, "grid_tmax_c": 30.0,
             "delta_tmin_c": None, "grid_tmin_c": 20.0},
            {"station_id": "T1", "date": "2024-01-01", "delta_tmax_c": 1.0, "grid_tmax_c": 30.0,
             "delta_tmin_c": 0.5, "grid_tmin_c": 20.0},
            {"station_id": "X", "date": "2025-01-01", "delta_tmax_c": 1.0, "grid_tmax_c": 30.0,
             "delta_tmin_c": 0.5, "grid_tmin_c": 20.0}]
    keys = runner.manifest_keys(rows, {"T1"}, {2025})
    assert keys == [("T1", "2025-01-01", "tmax")]
    p = tmp_path / "m.csv.gz"
    runner.write_manifest(str(p), keys)
    spec = {"manifest": {"path": str(p), "sha256": sc.manifest_sha256(keys)}}
    assert runner.check_manifest(spec, "") == keys
    spec["manifest"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        runner.check_manifest(spec, "")


def _synthetic_rows(n_st, n_days, zone, rng, year=2023):
    out = []
    for s in range(n_st):
        lat, lon = 20 + rng.uniform(-5, 5), 70 + rng.uniform(-5, 5)
        urban = rng.uniform(0, 1)
        for d in range(n_days):
            g = 30 + 5 * np.sin(d / 10) + rng.normal(0, 1)
            delta = 1.5 * urban + rng.normal(0, 0.3)
            out.append({"station_id": f"{zone}{s}", "date": f"{year}-{1 + d // 28:02d}-{1 + d % 28:02d}", "lat": lat,
                        "lon": lon, "region": "IN", "climate_zone": zone, "station_tmax_c": g + delta,
                        "station_tmin_c": g - 10, "grid_tmax_c": g, "grid_tmin_c": g - 10, "delta_tmax_c": delta,
                        "delta_tmin_c": 0.0, "lst_warm_season_anomaly_c": urban, "canopy_height_mean_m": 2.0,
                        "canopy_frac_over_3m": 0.1, "wc_built_frac": urban, "wc_tree_frac": 0.1, "wc_water_frac": 0.0,
                        "ghsl_urban_fraction": urban, "pop_density_per_km2": 100.0, "elevation_rel_to_gridcell_m": 0.0,
                        "elevation_mean_m": 200.0, "slope_deg": 1.0, "aspect_deg": 90.0,
                        "grid_specific_humidity_kgkg": 0.01, "koppen_main_group_code": 2, "nighttime_wind_ms": 2.0})
    return out


def test_served_predictions_apply_gate_and_fall_back_to_grid():
    pytest.importorskip("quantile_forest")
    import scorecard as runner
    rng = np.random.default_rng(0)
    train = _synthetic_rows(8, 40, "BSh", rng)
    test = _synthetic_rows(2, 5, "BSh", rng, year=2025) + _synthetic_rows(1, 5, "BWh", rng, year=2025)
    test[0] = {**test[0], "nighttime_wind_ms": None}   # incomplete covariates: grid fallback
    meta = {"model_version": "m", "cv": {"leave_region_out": {
        "tmax": {"by_zone": {"BSh": {"qrf_beats_grid": True}, "BWh": {"qrf_beats_grid": False}}},
        "tmin": {"by_zone": {}}}}}
    out = runner.served_predictions(train, test, "tmax", meta, n_jobs=1)
    assert len(out) == len(test)
    assert out[0]["applied"] is False and out[0]["covariates_missing"]
    assert all(o["applied"] for o in out[1:10])                  # BSh, complete, gate passes
    assert all(o["applied"] is False and o["cv_gate_passed"] is False for o in out[10:])   # BWh gate fails
    assert np.corrcoef([o["delta_c"] for o in out[1:10]], [t["delta_tmax_c"] for t in test[1:10]])[0, 1] > 0.5


def test_trainer_holdout_helpers(tmp_path):
    import train_downscaling as td
    p = tmp_path / "h.txt"
    p.write_text("# c\nA  # x\n\nB\n")
    assert td.read_holdout_stations(str(p)) == ["A", "B"]
    rows = [{"station_id": "A"}, {"station_id": "C"}, {"station_id": "A"}]
    kept, dropped = td.exclude_stations(rows, ["A", "B"])
    assert kept == [{"station_id": "C"}] and dropped == {"A": 2, "B": 0}
    p.write_text("A\nA\n")
    with pytest.raises(SystemExit):
        td.read_holdout_stations(str(p))


def test_truth_set_builder_dedupes_by_distance_and_wmo(tmp_path):
    import scorecard_build_truth_set as b
    spans = tmp_path / "spans.csv"
    spans.write_text("station_id,lat,lon,d_min,d_max,n_rows\n"
                     "U1,41.6617,-1.0081,2025-01-01,2025-12-31,364\n"   # truth, WMO 08160
                     "U2,41.6667,-1.0333,2025-01-01,2025-01-14,14\n"    # truth, same WMO, 2.2 km
                     "U3,10.0,10.0,2025-01-01,2025-12-31,300\n"         # truth, 0.5 km from training T1
                     "T1,10.0045,10.0,2023-01-01,2023-12-31,365\n"
                     "T2,30.0,30.0,2023-01-01,2023-12-31,365\n")
    ghcnd = tmp_path / "ghcnd.txt"
    ghcnd.write_text("".join(f"{sid:<11} {'':<29}{name:<30}{'':<9}{w:<5}\n" for sid, name, w in
                             (("U1", "ZARAGOZA AEROPUERTO", "08160"), ("U2", "ZARAGOZA", "08160"),
                              ("U3", "SOMEWHERE", ""), ("T1", "X", ""), ("T2", "Y", ""))))
    hold = tmp_path / "h.txt"
    hold.write_text("T2\n")
    rows, removed, _ = b.build(str(spans), [], str(ghcnd), str(hold), 2025)
    assert {r["station_id"] for r in rows} == {"U1", "T2"}
    why = {r["station_id"]: r["reason"] for r in removed}
    assert "WMO" in why["U2"] and "training" in why["U3"]
    assert {r["station_id"]: r["airport"] for r in rows}["U1"] == "yes"
    assert {r["station_id"]: r["origin"] for r in rows}["T2"] == "moved_2026-10-01"


def test_untested_global_metric_blocks_ship():
    rows, st = _frame({"BSh": 10}, cand_err={"BSh": 0.5}, targets=("tmax",))
    res, aim, dec = _run(rows, st, _spec(), _decl())
    assert not dec["pass"]
    assert any("global rmse_tmin not worse: no rows" in r for r in dec["reasons"])


def test_model_version_needs_full_tier():
    rows, st = _frame({"BSh": 10, "Csa": 9}, cand_err={"BSh": 0.7, "Csa": 0.9})
    res = sc.score(rows, st, _spec())
    aim = sc.score_aim(rows, st, _spec(), _decl())
    assert not sc.ship_decision(res, aim, _spec(), kind="model_version", tier="fast")["pass"]
    assert sc.ship_decision(res, aim, _spec(), kind="model_version", tier="full")["pass"]


def test_empty_zone_groups_are_listed():
    rows, st = _frame({"BSh": 10}, n_days=3)
    names = {g["name"] for g in sc.score(rows, st, _spec())["groups"]}
    assert {"arid", "tropical", "temperate", "cold"} <= names


def test_yaml_booleans_in_subset_are_refused():
    d = _decl()
    d["aim"]["subset"] = {"airport": [True]}
    with pytest.raises(ValueError):
        sc.validate_declaration(d)
