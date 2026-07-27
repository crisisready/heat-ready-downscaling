"""
FEATURE_ORDER + build_feature_matrix -- the single source of truth for the
downscaling model's feature contract. Extracted from
crisisready/heat-risk-data-api's src/downscaling.py (lines ~439-608 as of
2026-07-27, commit 57479e5). See PROVENANCE.md.

Order is load-bearing -- must match a model's own metadata.json
"feature_order" exactly (see downscaling.contract.validate_feature_order).
Index 0 is a stand-in name: the Tmax model's column 0 is the ERA5-Land grid
daily Tmax, the Tmin model's is grid daily Tmin -- same physical role (the
raw grid value being corrected), different metric.

The 6 features below elevation_rel_to_gridcell_m were added after a
red-team review of the original feature set found real gaps -- each is
either already-extracted data that was never wired into FEATURE_ORDER, or a
"free rider" derivable from data already pulled with zero new ingestion:
  - koppen_main_group_code: the model's ONLY explicit climate-regime input
    before this was latitude + a same-day grid specific-humidity value -- a
    weak, weather-confounded proxy. koppen.koppen_main_group_code is a pure
    offline lookup (kgcpy), already computed for CV-fold reporting but
    never fed to the model itself.
  - elevation_mean_m: the *strength* of a given relative-elevation effect
    plausibly depends on absolute altitude too (thinner air, different
    boundary-layer depth at 2,000m+ vs. near sea level), an interaction the
    model can't learn if only the relative offset is visible.
  - slope_deg/aspect_sin/aspect_cos: a spatial gradient computed from the
    same elevation raster window already read for the stats above --
    near-zero marginal cost. Aspect is circular (compass bearing) so it
    gets the same sin/cos treatment as day-of-year below, not a raw degree
    value.
  - grid_diurnal_range_c: grid_tmax_c - grid_tmin_c, both already pulled
    from the same ERA5 request that provides grid_daily_value_c -- zero
    new ingestion. A day with a large diurnal range indicates clear/dry
    conditions; a small range indicates humid/cloudy conditions -- the same
    clear-sky/calm-night regime the urban-heat-island literature (Oke 1982)
    ties to maximum UHI intensity.

nighttime_wind_ms: nighttime wind-driven mixing -- Oke 1982's other
UHI-intensity modulator alongside clear skies. grid_diurnal_range_c (added
earlier as a proxy) was checked against the literature and found NOT to
stand in for wind (DTR tracks clear-sky radiative cooling, a distinct
mechanism from mechanical mixing).
"""

FEATURE_ORDER = (
    "grid_daily_value_c", "lst_warm_season_anomaly_c", "canopy_height_mean_m",
    "canopy_frac_over_3m", "wc_built_frac", "wc_tree_frac", "wc_water_frac",
    "ghsl_urban_fraction", "log1p_pop_density", "elevation_rel_to_gridcell_m",
    "koppen_main_group_code", "elevation_mean_m", "slope_deg", "aspect_sin", "aspect_cos",
    "grid_diurnal_range_c",
    "doy_sin", "doy_cos", "latitude", "grid_specific_humidity_kgkg",
    "nighttime_wind_ms",
)


def _doy_trig(d) -> tuple[float, float]:
    """Cyclic day-of-year encoding -- a raw integer would put Dec-31 and
    Jan-1 maximally apart; (sin, cos) keeps them adjacent."""
    import math
    doy = d.timetuple().tm_yday
    angle = 2 * math.pi * doy / 365.25
    return math.sin(angle), math.cos(angle)


def build_feature_matrix(
    rows: list[dict], target: str,
) -> tuple["np.ndarray", list[bool], list[list[str]]]:
    """
    Build the feature matrix for a batch of polygon-day/station-day
    covariate dicts, for either the "tmax" or "tmin" target.

    Each row is expected to carry: grid_tmax_c/grid_tmin_c, lat, lon, date (a
    date object or ISO string), pop_density_per_km2, and the covariate
    columns named identically to the training snapshot's own columns
    (lst_warm_season_anomaly_c, canopy_height_mean_m, canopy_frac_over_3m,
    wc_built_frac, wc_tree_frac, wc_water_frac, ghsl_urban_fraction,
    elevation_rel_to_gridcell_m, elevation_mean_m, slope_deg, aspect_deg,
    grid_specific_humidity_kgkg, nighttime_wind_ms).

    Returns (X, complete_mask, missing_by_row):
      - X: (n, len(FEATURE_ORDER)) array. A row missing ANY feature gets a
        0.0 placeholder at the missing slots -- NEVER trust X[i] unless
        complete_mask[i] is True (no imputation: a real gap always falls
        back to the raw grid value, never a filled-in guess).
      - complete_mask[i]: True only if every one of the features was
        present (not None) for row i.
      - missing_by_row[i]: the feature names that were None for row i
        (empty list when complete).
    """
    import math
    from datetime import date as _date

    import numpy as np

    from heatready_downscaling.koppen import koppen_main_group_code

    grid_col = "grid_tmax_c" if target == "tmax" else "grid_tmin_c"
    X = np.zeros((len(rows), len(FEATURE_ORDER)), dtype=float)
    complete_mask: list[bool] = []
    missing_by_row: list[list[str]] = []

    for i, r in enumerate(rows):
        d = r.get("date")
        if isinstance(d, str):
            d = _date.fromisoformat(d)
        doy_sin, doy_cos = _doy_trig(d) if d is not None else (None, None)
        pop = r.get("pop_density_per_km2")
        log1p_pop = math.log1p(pop) if pop is not None and pop >= 0 else None

        lat, lon = r.get("lat"), r.get("lon")
        koppen_code = r.get("koppen_main_group_code")
        if koppen_code is None and lat is not None and lon is not None:
            # Live fallback for callers that don't already carry a stored
            # value -- training/snapshot rows always do (backfilled), so
            # this branch is never hit during scoring against a snapshot
            # and avoids re-running kgcpy's raster lookup per row.
            koppen_code = koppen_main_group_code(lat, lon)

        aspect_deg = r.get("aspect_deg")
        aspect_sin, aspect_cos = (
            (math.sin(math.radians(aspect_deg)), math.cos(math.radians(aspect_deg)))
            if aspect_deg is not None else (None, None)
        )

        grid_tmax, grid_tmin = r.get("grid_tmax_c"), r.get("grid_tmin_c")
        diurnal_range = (
            grid_tmax - grid_tmin if grid_tmax is not None and grid_tmin is not None else None
        )

        values = {
            "grid_daily_value_c": r.get(grid_col),
            "lst_warm_season_anomaly_c": r.get("lst_warm_season_anomaly_c"),
            "canopy_height_mean_m": r.get("canopy_height_mean_m"),
            "canopy_frac_over_3m": r.get("canopy_frac_over_3m"),
            "wc_built_frac": r.get("wc_built_frac"),
            "wc_tree_frac": r.get("wc_tree_frac"),
            "wc_water_frac": r.get("wc_water_frac"),
            "ghsl_urban_fraction": r.get("ghsl_urban_fraction"),
            "log1p_pop_density": log1p_pop,
            "elevation_rel_to_gridcell_m": r.get("elevation_rel_to_gridcell_m"),
            "koppen_main_group_code": koppen_code,
            "elevation_mean_m": r.get("elevation_mean_m"),
            "slope_deg": r.get("slope_deg"),
            "aspect_sin": aspect_sin,
            "aspect_cos": aspect_cos,
            "grid_diurnal_range_c": diurnal_range,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "latitude": r.get("lat"),
            "grid_specific_humidity_kgkg": r.get("grid_specific_humidity_kgkg"),
            "nighttime_wind_ms": r.get("nighttime_wind_ms"),
        }
        missing = [c for c in FEATURE_ORDER if values[c] is None]
        missing_by_row.append(missing)
        complete_mask.append(not missing)
        if not missing:
            X[i] = [values[c] for c in FEATURE_ORDER]

    return X, complete_mask, missing_by_row
