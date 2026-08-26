# Models

*Derived from `registry/*/*/manifest.yaml` -- do not edit by hand, re-run `scripts/render_models_page.py`.*

Every evidence number below is read directly from the cited manifest's own `claims[].evidence` block, never recomputed. See `registry/README.md` (if present) or `heatready_downscaling.registry`'s own docstring for what a registry entry is and is not (it does not execute anything and does not decide promotion).

## `global/ds-2026.07-rf5`

**HeatReady global downscaling model (QRF)**

- **Status**: `registered`
- **Method**: `qrf-global` -> `zone_params`
- **Authors**: Nishant Kishore
- **Manifest**: `registry/global/ds-2026.07-rf5/manifest.yaml`

| Cell (target/zone/band) | Evidence |
|---|---|
| tmin/BWk/lag_fill | rmse_improvement_pct=0.2343, provenance=publicly_reproducible, holdout=station_salted_fold, n_stations=2 |

## `local/seoul-sdot-v1`

**Seoul local downscaling model (S-DoT)**

- **Status**: `registered`
- **Method**: `local-retrain` -> `artifact_route`
- **Authors**: Nishant Kishore
- **Derived from**: `global/ds-2026.07-rf5`
- **Manifest**: `registry/local/seoul-sdot-v1/manifest.yaml`

| Cell (target/zone/band) | Evidence |
|---|---|
| tmax/Cwa/(none) (Seoul, 63 held-out dong) | rmse_c=1.14, provenance=maintainer_attested, holdout=spatial_holdout, n_clusters=63 |
| tmin/Cwa/(none) (Seoul, 63 held-out dong) | rmse_c=1.13, provenance=maintainer_attested, holdout=spatial_holdout, n_clusters=63 |

## `local/valencia-coast-v1`

**Valencia coast-distance local correction**

- **Status**: `registered`
- **Method**: `covariate-linear-local` -> `polygon_params`
- **Authors**: Nishant Kishore
- **Manifest**: `registry/local/valencia-coast-v1/manifest.yaml`

| Cell (target/zone/band) | Evidence |
|---|---|
| tmin/BSh/lag_fill (Valencia city cluster (9 ECA&D stations)) | rmse_reduction_pct_hot_day=0.31, CI95=[0.204, 0.392], provenance=publicly_reproducible, holdout=station_salted_fold, n_stations=9, n_clusters=9 |
| tmax/BSh/lag_fill (Valencia city cluster (9 ECA&D stations)) | rmse_reduction_pct=0.118, CI95=[-0.118, 0.294], provenance=publicly_reproducible, holdout=station_salted_fold, n_stations=9, n_clusters=9 |
