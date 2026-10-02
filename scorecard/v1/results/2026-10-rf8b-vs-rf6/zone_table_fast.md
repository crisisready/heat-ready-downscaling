# ds-2026.09-rf8b vs ds-2026.09-rf6, fast tier

Scorecard 1.0.0. Error is served value minus station, C. Deltas are candidate minus incumbent on identical station-days (negative is better).

**Ship rule: FAIL**

Reasons:
- aim rmse_tmax improves by >= 0.020 C: candidate minus incumbent -0.0008 C on the aimed subset
- a model_version candidate ships only on the full tier (this is the fast tier)

| Unit | Gating | tmax stations | tmax RMSE inc | tmax RMSE cand | ERA5-Land | delta | tmin stations | tmin RMSE inc | tmin RMSE cand | ERA5-Land | delta | hot-day MAE delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Global** | yes | 237 | 2.245 | 2.231 | 2.877 | -0.015 | 229 | 2.267 | 2.267 | 2.472 | +0.000 | -0.006 |
| BSh | tmax, tmin | 88 | 2.350 | 2.350 | 2.843 | -0.001 | 80 | 2.695 | 2.695 | 2.885 | +0.000 | -0.012 |
| BSk | tmax, tmin | 25 | 1.671 | 1.657 | 2.240 | -0.013 | 25 | 1.908 | 1.908 | 2.085 | +0.000 | +0.007 |
| Cfa | tmax, tmin | 17 | 1.964 | 1.964 | 3.070 | -0.000 | 17 | 1.937 | 1.937 | 1.968 | +0.000 | +0.010 |
| Cfb | tmax, tmin | 33 | 2.870 | 2.826 | 3.814 | -0.043 | 33 | 2.240 | 2.240 | 2.529 | +0.000 | -0.024 |
| Csa | tmax, tmin | 47 | 1.561 | 1.569 | 2.342 | +0.008 | 47 | 1.865 | 1.865 | 2.065 | +0.000 | +0.022 |
| Csb | tmax, tmin | 18 | 2.219 | 2.235 | 2.791 | +0.016 | 18 | 2.003 | 2.003 | 2.407 | +0.000 | -0.019 |
| arid (pooled: BWh) | no | 3 | 2.189 | 2.330 | 4.047 | +0.141 | 3 | 1.330 | 1.330 | 1.330 | +0.000 | +0.099 |
| temperate (pooled: Cfc+temperate) | no | 6 | 3.922 | 3.716 | 2.979 | -0.206 | 6 | 2.775 | 2.775 | 2.878 | +0.000 | -0.166 |

Zone groups (raw ERA5-Land floor):

| Unit | Gating | tmax stations | tmax RMSE inc | tmax RMSE cand | ERA5-Land | delta | tmin stations | tmin RMSE inc | tmin RMSE cand | ERA5-Land | delta | hot-day MAE delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arid | floor | 116 | 2.204 | 2.203 | 2.740 | -0.000 | 108 | 2.497 | 2.497 | 2.681 | +0.000 | -0.005 |
| cold | floor | 0 |  |  |  |  | 0 |  |  |  |  |  |
| temperate | floor | 121 | 2.282 | 2.254 | 2.993 | -0.027 | 121 | 2.054 | 2.054 | 2.282 | +0.000 | -0.008 |
| tropical | floor | 0 |  |  |  |  | 0 |  |  |  |  |  |

## Report only (not gating)

- Bootstrap 95% CI (station resample), global: {'d_rmse_tmax': [-0.040685530761557086, 0.009602306651168797], 'd_rmse_tmin': [0.0, 0.0], 'd_hot_mae_tmax': [-0.028018057596038395, 0.015598709213574955]}
- Equal-weight zone mean: {'d_rmse_tmax': -0.005544753621965308, 'n_zones_tmax': 6, 'd_rmse_tmin': 0.0, 'n_zones_tmin': 6}
- Within-city anomaly correlation: {'cities': 7, 'station_days': 40206, 'corr_inc': 0.8940244112681389, 'corr_cand': 0.8993611365984605, 'corr_grid': 0.8337892416515781}
- Served 95% interval coverage: {'inc': 0.9638095114866307, 'cand': 0.9678464912564453}

Per year:

| Year | d rmse_tmax | d rmse_tmin | d hot_mae_tmax |
|---|---|---|---|
| 2025 | -0.015 | +0.000 | -0.006 |

Distance to the incumbent's nearest training station:

| Band | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |
|---|---|---|---|---|
| 0-10 km | 6 | -0.008 | 6 | +0.000 |
| 10-50 km | 77 | -0.035 | 76 | +0.000 |
| 50-200 km | 93 | -0.010 | 88 | +0.000 |
| 200-inf km | 61 | +0.006 | 59 | +0.000 |

Strata:

| Stratum | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |
|---|---|---|---|---|
| airport=no | 196 | -0.020 | 188 | +0.000 |
| airport=unknown | 13 | -0.056 | 13 | +0.000 |
| airport=yes | 28 | +0.042 | 28 | +0.000 |
| setting=rural | 134 | -0.023 | 131 | +0.000 |
| setting=urban | 103 | -0.002 | 98 | +0.000 |
