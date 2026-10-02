# ds-2026.09-rf8b vs ds-2026.09-rf6, fast tier

Scorecard 2.0.0. Error is served value minus station, C. Deltas are candidate minus incumbent on identical station-days (negative is better).

**Ship rule: FAIL**

Reasons:
- aim rmse_tmax improves by >= 0.020 C: candidate minus incumbent -0.0008 C on the aimed subset
- a model_version candidate ships only on the full tier (this is the fast tier)

| Unit | Gating | tmax stations | tmax RMSE inc | tmax RMSE cand | ERA5-Land | delta | tmin stations | tmin RMSE inc | tmin RMSE cand | ERA5-Land | delta | hot-day MAE delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Global** | yes | 338 | 2.324 | 2.308 | 2.921 | -0.016 | 330 | 2.416 | 2.416 | 2.620 | +0.000 | -0.003 |
| Af | tmax, tmin | 15 | 2.227 | 2.239 | 3.197 | +0.011 | 15 | 1.570 | 1.570 | 1.730 | +0.000 | -0.004 |
| Am | tmax, tmin | 15 | 1.603 | 1.565 | 2.730 | -0.038 | 15 | 2.097 | 2.097 | 2.248 | +0.000 | -0.031 |
| Aw | tmax, tmin | 10 | 2.482 | 2.381 | 2.115 | -0.101 | 10 | 1.305 | 1.305 | 1.473 | +0.000 | -0.086 |
| BSh | tmax, tmin | 88 | 2.350 | 2.350 | 2.843 | -0.001 | 80 | 2.695 | 2.695 | 2.885 | +0.000 | -0.012 |
| BSk | tmax, tmin | 25 | 1.671 | 1.657 | 2.240 | -0.013 | 25 | 1.908 | 1.908 | 2.085 | +0.000 | +0.007 |
| Cfa | tmax, tmin | 17 | 1.964 | 1.964 | 3.070 | -0.000 | 17 | 1.937 | 1.937 | 1.968 | +0.000 | +0.010 |
| Cfb | tmax, tmin | 33 | 2.870 | 2.826 | 3.814 | -0.043 | 33 | 2.240 | 2.240 | 2.529 | +0.000 | -0.024 |
| Csa | tmax, tmin | 47 | 1.561 | 1.569 | 2.342 | +0.008 | 47 | 1.865 | 1.865 | 2.065 | +0.000 | +0.022 |
| Csb | tmax, tmin | 18 | 2.219 | 2.235 | 2.791 | +0.016 | 18 | 2.003 | 2.003 | 2.407 | +0.000 | -0.019 |
| Cwa | tmax, tmin | 15 | 2.014 | 1.975 | 2.609 | -0.039 | 15 | 2.500 | 2.500 | 2.500 | +0.000 | +0.033 |
| Cwb | tmax, tmin | 8 | 3.907 | 3.960 | 4.822 | +0.053 | 8 | 4.438 | 4.438 | 4.613 | +0.000 | +0.119 |
| Dfa | tmax, tmin | 12 | 2.257 | 2.280 | 2.425 | +0.024 | 12 | 2.851 | 2.851 | 3.168 | +0.000 | +0.069 |
| Dfb | tmax, tmin | 13 | 2.633 | 2.577 | 3.749 | -0.056 | 13 | 3.194 | 3.194 | 3.486 | +0.000 | -0.010 |
| Dwa | tmax, tmin | 13 | 2.679 | 2.668 | 2.315 | -0.011 | 13 | 2.967 | 2.967 | 3.248 | +0.000 | -0.021 |
| arid (pooled: BWh) | no | 3 | 2.189 | 2.330 | 4.047 | +0.141 | 3 | 1.330 | 1.330 | 1.330 | +0.000 | +0.099 |
| temperate (pooled: Cfc+temperate) | no | 6 | 3.922 | 3.716 | 2.979 | -0.206 | 6 | 2.775 | 2.775 | 2.878 | +0.000 | -0.166 |

Zone groups (raw ERA5-Land floor):

| Unit | Gating | tmax stations | tmax RMSE inc | tmax RMSE cand | ERA5-Land | delta | tmin stations | tmin RMSE inc | tmin RMSE cand | ERA5-Land | delta | hot-day MAE delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arid | floor | 116 | 2.204 | 2.203 | 2.740 | -0.000 | 108 | 2.497 | 2.497 | 2.681 | +0.000 | -0.005 |
| cold | floor | 38 | 2.533 | 2.517 | 2.913 | -0.017 | 38 | 3.010 | 3.010 | 3.306 | +0.000 | +0.012 |
| temperate | floor | 144 | 2.385 | 2.364 | 3.099 | -0.020 | 144 | 2.286 | 2.286 | 2.485 | +0.000 | +0.002 |
| tropical | floor | 40 | 2.150 | 2.100 | 2.661 | -0.051 | 40 | 1.678 | 1.678 | 1.834 | +0.000 | -0.044 |

## Report only (not gating)

- Bootstrap 95% CI (station resample), global: {'d_rmse_tmax': [-0.03672341100134461, 0.005511898840910942], 'd_rmse_tmin': [0.0, 0.0], 'd_hot_mae_tmax': [-0.02209307053429698, 0.016248232516388995]}
- Equal-weight zone mean: {'d_rmse_tmax': -0.01363214852854341, 'n_zones_tmax': 14, 'd_rmse_tmin': 0.0, 'n_zones_tmin': 14}
- Within-city anomaly correlation: {'cities': 7, 'station_days': 40206, 'corr_inc': 0.8940244112681389, 'corr_cand': 0.8993611365984605, 'corr_grid': 0.8337892416515781}
- Served 95% interval coverage: {'inc': 0.9540153085788962, 'cand': 0.9565552749746986}

Per year:

| Year | d rmse_tmax | d rmse_tmin | d hot_mae_tmax |
|---|---|---|---|
| 2025 | -0.016 | +0.000 | -0.003 |

Distance to the incumbent's nearest training station:

| Band | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |
|---|---|---|---|---|
| 0-10 km | 9 | +0.013 | 9 | +0.000 |
| 10-50 km | 85 | -0.032 | 84 | +0.000 |
| 50-200 km | 109 | -0.007 | 104 | +0.000 |
| 200-inf km | 135 | -0.012 | 133 | +0.000 |

Strata:

| Stratum | tmax stations | d rmse_tmax | tmin stations | d rmse_tmin |
|---|---|---|---|---|
| airport=no | 287 | -0.019 | 279 | +0.000 |
| airport=unknown | 13 | -0.056 | 13 | +0.000 |
| airport=yes | 38 | +0.030 | 38 | +0.000 |
| setting=rural | 184 | -0.021 | 181 | +0.000 |
| setting=urban | 154 | -0.008 | 149 | +0.000 |
