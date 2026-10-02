# Out-of-time scorecard, version 2

Version 2 is version 1 plus 101 GHCN-D stations from the C1 thin-zone pull (2026-10-02), so the tropical and cold
zones are scored: 338 truth stations, manifest sha256 in `spec.yaml`. The 101 stations' rows are
`thin_zone_rows.json` (built by `scripts/build_thin_zone_rows.py`, listed under `truth_extra_rows`); everything else
below is unchanged from v1. First result: `results/2026-10-rf8b-vs-rf6/` (fast tier, both recipes rescored).


One standing definition of "better" for the global downscaling model. Every candidate change (a new
model version, a level anchor, the pooled residual layer, a logger bias model, a new station source)
is scored here against the model we serve today, on stations the model has never seen, in a year it
has not seen. It replaces the per-case station sets and metrics used through September 2026. The
trainer's leave-region-out CV stays as its own internal check.

Frozen values live in `spec.yaml`. Changing any of them, or rolling the scored year forward, is a
version bump (`scorecard/v2/`), and the incumbent is rescored first.

## Truth set

`truth_stations.csv`: 237 stations, built by `scripts/scorecard_build_truth_set.py`.

- 229 stations whose first row in the corpus is in 2025. They have no row before the fast tier's
  cutoff, so they are unseen in space and time without moving anything.
- 8 South Asian GSOD stations moved out of training on 2026-10-01 (`holdout_stations.txt`), chosen
  for geographic spread across the 27 India and Pakistan stations. Nishant approved this set over a
  15-station set and a full 15-per-zone rebuild, because moving stations whose only rows are in 2023
  costs 8% to 100% of a thin zone's training rows and still leaves nothing to score until a 2025
  pull lands. The 8 cost 4.9% of BSh training rows (2.5% of the corpus).
- Deduplicated (`dedupe_removed.csv`): of two truth stations within 1 km of each other, or sharing
  a WMO id, the one with more 2025 rows stays; a truth station that duplicates a training station is
  dropped from truth.
- Tagged airport (from the station name; `unknown` where no name exists), urban or rural (mean GHSL
  urban fraction at the station, threshold 0.5, filled by `build-manifest`), and region.
- Contributor stations held out for their own place would carry `visibility: private` and are scored
  only into a private file. None exist yet.

Zone coverage today (unseen stations, v2): BSh 88, Csa 47, Cfb 33, BSk 25, Csb 18, Cfa 17, Af 15, Am 15, Cwa 15,
Dfb 13, Dwa 13, Dfa 12, Aw 10, Cwb 8, Cfc 4, BWh 3, "temperate" 2. The 101 thin-zone stations are GHCN-D
(public domain), with 2023 and 2025 rows, so the full tier can score them too. (v1 had no tropical or cold
station and reported those groups as unscored.)

## How a candidate is scored

1. Declare the aim in the PR before anything runs: `declarations/<name>.yaml` names the aimed subset
   (zones, zone groups, regions, station tags or ids), the metric, and the minimum effect in C.
2. The runner checks the manifest's sha256 against the spec, then refits each recipe
   (`scorecard/recipes/*.yaml`: corpus and feature set) on rows dated on or before each cutoff, with
   every truth station removed at every date.
3. Predictions go through the serving contract (`contract.QRFModelAdapter.predict`, the mirror of
   the API's own serving function): the version's own CV gate per zone and target, and the raw
   ERA5-Land value wherever the model does not apply (a failed gate or a missing covariate). That
   is the value a user would get. In v1 the per-zone CV gate, conformal widths and AOA threshold
   come from the served version's own metadata.json, which was fit on all of its data, so the gate
   a refit uses was chosen with the full corpus in view. The gate is pass or fail per zone, and rf6
   and rf8b agree on it everywhere (tmin falls back to the grid in As, BWh and Cwa for both).
   Refitting the gate at each cutoff is a v2 item.
4. Fast tier (every candidate): cutoff 2024-12-31, score 2025. The incumbent's predictions are
   cached, so a layer candidate supplies its own served deltas (fit before the cutoff) and scores in
   minutes.
5. Full tier (model versions and new training sources, which ship only on this tier): cutoffs 2021 to 2024, a refit at each, the
   paired deltas pooled over every scored year. A year where either side has no training rows before
   the cutoff is listed as unscoreable in the provenance block. Today the corpus holds 2023 and 2025
   for most stations, so an incumbent trained on the main table alone can only be scored in 2024
   (BSh tmax at the moved South Asian stations) and 2025.

## Metrics and the ship rule

Errors are served value minus station. Metrics are pooled over station-days: tmax RMSE and bias,
tmin RMSE and bias, and tmax MAE on hot days. A hot day is a day whose ERA5-Land grid tmax is at or
above that station's own 90th percentile for the year, so candidate and incumbent are judged on the
same days. Every number in the ship rule is a paired delta, candidate minus incumbent, on identical
station-days.

`heatready_downscaling.scorecard.ship_decision` returns pass or fail with every reason. A candidate
ships only when all of these hold:

- Global: not worse on tmax RMSE, tmin RMSE, or hot-day tmax MAE. A global metric with no rows
  blocks the ship.
- Every zone with at least 8 unseen stations for a target, and every pool of thin zones within a
  zone group that reaches 8 together, is no more than 0.10 C worse on RMSE (point estimate). A pool
  still under 8 is reported and does not gate.
- No zone group (arid, tropical, temperate, cold) is worse than raw ERA5-Land.
- The declared aim improves by at least the declared minimum effect.

Reported and never gating: station-bootstrap 95% intervals, the per-year table, distance to the
incumbent's nearest training station in bands (under 10, 10 to 50, 50 to 200, over 200 km), the
airport and urban/rural strata, the equal-weight zone mean, within-city anomaly correlation for
clusters of 5 or more unseen stations within 30 km, and served 95% interval coverage.

Zone-level results and pass or fail are public (`results/`). Results at a contributor's private
stations go only to that contributor and to us.

## Running it

Both subcommands read `ghcn_training` (SELECT only) through the trainer's own loader, so they run
where the trainer runs, with the extra rows files named in the spec and recipes in `--data-dir`
(checked by sha256).

    python scripts/scorecard.py build-manifest --spec scorecard/v1/spec.yaml --data-dir DIR --out-dir OUT
    python scripts/scorecard.py run --spec scorecard/v1/spec.yaml --tier fast \
        --declaration scorecard/v1/declarations/2026-10-rf8b-vs-rf6.yaml \
        --incumbent-recipe scorecard/recipes/ds-2026.09-rf6.yaml --incumbent-metadata rf6/metadata.json \
        --candidate-recipe scorecard/recipes/ds-2026.09-rf8b.yaml --candidate-metadata rf8b/metadata.json \
        --data-dir DIR --cache-dir CACHE --out-dir OUT --n-jobs 8

A production retrain that should stay comparable with the scorecard passes
`--holdout-stations scorecard/v1/holdout_stations.txt` to `train_downscaling.py`.

## Open work

- New stations for the zones without unseen coverage (tropical and cold groups, BWh, Cwa, Cwb,
  Cfc), with India, Africa and Mexico as the sourcing targets.
- Rows for 2022 and 2024 at the main-table stations, so the full tier covers four years everywhere.
- Station tagging is a first pass (name keywords for airports, one GHSL threshold for urban). Better
  airport matching (ICAO/WMO metadata) and WMO ids for the AEMET-coded stations are good
  `colleague-ok` tickets.
