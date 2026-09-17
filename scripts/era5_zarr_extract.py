"""
Reads station-level ERA5-Land hourly values out of a whole-country Zarr tile
(written by pull_era5_tile_to_zarr.py) instead of downloading a per-batch
NetCDF from CDS. Same (name, datetime, t2m, d2m, sp, [wind_ms], [solar_wm2])
row shape era5.extract_era5_means produces, so every downstream consumer
(heat_calcs.aggregate_hourly_to_daily, _daily_mean_specific_humidity,
_daily_mean_nighttime_wind) works unchanged regardless of which source
produced the rows.

Real risk this module exists to avoid (found in the 2026-09-17 architecture
consult, docs/diagnosis-2026-09-17-cds-acquisition-request-shape.md §5.1):
era5.extract_era5_means calls `ds[var].values[...]` -- materializing the
WHOLE variable array -- which is fine for a small per-batch bounding box but
OOMs on a country-sized tile (Indonesia's full archipelago is ~102 GB
uncompressed). Every function here does a LAZY xarray selection (.sel/.isel,
never full-array .values) to narrow down to ONE station's time series BEFORE
materializing anything, so memory use is bounded by (stations x timesteps),
not (tile cells x timesteps) regardless of tile size.

Interpolation: nearest-cell only (this module's whole reason to exist is a
one-time bulk backfill, not a fresh per-request precision study -- era5.py's
own bilinear mode stays the CDS-download path's job). The same land-mask
rescue era5.extract_era5_means does (substitute the nearest unmasked cell
when the exact-nearest one is NaN, i.e. open water) is preserved, but checks
only the ONE candidate station's own first-timestep value, never a whole
array, for the same OOM reason.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import era5  # noqa: E402  (for _find_var/_find_var_optional, kept as the single source of truth)


def open_era5_zarr_store(store_uri: str):
    """Opens a Zarr tile lazily -- no data is read into memory by this call,
    only metadata/coordinates. `store_uri` may be a local path or an s3://
    URI (xarray dispatches to s3fs automatically when the zarr/s3fs packages
    are installed and the URI has an s3:// scheme)."""
    import xarray as xr

    ds = xr.open_zarr(store_uri, consolidated=True)
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    if float(ds[lon_coord].max()) > 180:
        ds = ds.assign_coords({lon_coord: ((ds[lon_coord] + 180) % 360) - 180})
        ds = ds.sortby(lon_coord)
    return ds


def _nearest_index(coord_values, target: float) -> int:
    """Plain-Python nearest-index search over a 1-D coordinate array -- avoids
    pulling numpy's argmin over the FULL coordinate array into a dependency
    on numpy being imported here; coord_values is already small (grid extent
    in one dimension, not the full 2-D/3-D tile) so this is cheap regardless
    of tile size."""
    best_i, best_d = 0, None
    for i, v in enumerate(coord_values):
        d = abs(float(v) - target)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def extract_era5_means_from_zarr(
    ds, stations: list[dict],
) -> list[dict]:
    """Returns rows shaped exactly like era5.extract_era5_means's output, for
    `stations` (each a dict with at least "name"/"lat"/"lon" keys -- the same
    shape build_training_set.py already carries per station) sliced out of
    the already-open lazy Zarr dataset `ds` (see open_era5_zarr_store).

    The land-mask rescue below deliberately narrows to ONE station's own
    (lat_idx, lon_idx) neighborhood before ever calling .values -- unlike
    era5.extract_era5_means, which can afford ds[var].values[0] once because
    its own NetCDF is already bounded to one batch's small bbox. A tile-sized
    equivalent of that same call would materialize the whole country."""
    t2m_var = era5._find_var(ds, ["t2m", "2m_temperature", "VAR_2T"])
    d2m_var = era5._find_var(ds, ["d2m", "2m_dewpoint_temperature", "VAR_2D"])
    sp_var = era5._find_var(ds, ["sp", "surface_pressure"])
    u10_var = era5._find_var_optional(ds, ["u10", era5._WIND_U_VARIABLE, "VAR_10U"])
    v10_var = era5._find_var_optional(ds, ["v10", era5._WIND_V_VARIABLE, "VAR_10V"])
    ssrd_var = era5._find_var_optional(ds, ["ssrd", era5._SOLAR_RADIATION_VARIABLE, "VAR_SSRD"])

    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    lat_arr = ds[lat_coord].values  # 1-D, small (grid extent), safe to materialize
    lon_arr = ds[lon_coord].values

    rows: list[dict] = []
    for station in stations:
        name = station["name"]
        li = _nearest_index(lat_arr, station["lat"])
        lj = _nearest_index(lon_arr, station["lon"])

        # Land-mask rescue: check only THIS station's own nearest-cell first
        # timestep (a lazy scalar select), not a whole array.
        t2m_first = ds[t2m_var].isel({lat_coord: li, lon_coord: lj, time_dim: 0}).values
        if t2m_first != t2m_first:  # NaN check without importing numpy/math for one scalar
            li, lj = _rescue_nearest_unmasked(ds, t2m_var, lat_coord, lon_coord, time_dim, li, lj)

        point = ds.isel({lat_coord: li, lon_coord: lj})  # still lazy -- one grid column
        t2m = point[t2m_var].values  # NOW materializes -- but only this station's timeseries
        d2m = point[d2m_var].values
        sp = point[sp_var].values
        times = point[time_dim].values
        u10 = point[u10_var].values if u10_var else None
        v10 = point[v10_var].values if v10_var else None
        ssrd = point[ssrd_var].values if ssrd_var else None

        for i, t in enumerate(times):
            row = {
                "name": name, "datetime": str(t)[:19],
                "t2m": float(t2m[i]) - 273.15 if abs(float(t2m[i])) > 200 else float(t2m[i]),
                "d2m": float(d2m[i]) - 273.15 if abs(float(d2m[i])) > 200 else float(d2m[i]),
                "sp": float(sp[i]),
            }
            if u10 is not None and v10 is not None:
                row["wind_ms"] = (float(u10[i]) ** 2 + float(v10[i]) ** 2) ** 0.5
            if ssrd is not None:
                row["solar_wm2"] = max(float(ssrd[i]) / 3600.0, 0.0)
            rows.append(row)
    return rows


def _rescue_nearest_unmasked(ds, var, lat_coord, lon_coord, time_dim, li, lj, radius: int = 3):
    """Same intent as era5.extract_era5_means's masked-cell rescue (a coastal
    station's exact-nearest cell can be open water/NaN), scoped to a small
    (2*radius+1)^2 neighborhood around (li, lj) instead of the whole tile --
    materializing a 7x7 patch is negligible regardless of tile size, unlike
    materializing the tile itself."""
    lat_size = ds.sizes[lat_coord]
    lon_size = ds.sizes[lon_coord]
    lat_lo, lat_hi = max(0, li - radius), min(lat_size, li + radius + 1)
    lon_lo, lon_hi = max(0, lj - radius), min(lon_size, lj + radius + 1)
    patch = ds[var].isel(
        {lat_coord: slice(lat_lo, lat_hi), lon_coord: slice(lon_lo, lon_hi), time_dim: 0}
    ).values
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i, j] == patch[i, j]:  # not NaN
                return lat_lo + i, lon_lo + j
    # Nothing unmasked nearby -- return the original (caller's downstream
    # values will be NaN, matching era5.extract_era5_means's own fallback
    # behavior when its rescue also finds nothing).
    return li, lj
