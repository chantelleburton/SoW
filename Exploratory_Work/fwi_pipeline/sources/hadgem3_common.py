"""Shared low-level helper FUNCTIONS for HadGEM3-A sources (historical and
attribution). Deliberately just functions, not a shared class — the two
Source classes stay separate/independent (see session plan)."""
from __future__ import annotations

import xarray as xr


def fix_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Fix files where time is a coordinate on an anonymous dim rather than a
    dimension itself, and drop within-file duplicate timestamps. Also
    normalise the historical tree's lat/lon dim names to latitude/longitude
    (the attribution tree already uses latitude/longitude), so both trees
    are consistent for regrid_to_tracer and the shared chunk dicts."""
    rename = {k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds.dims}
    if rename:
        ds = ds.rename(rename)
    if "time" in ds.coords and "time" not in ds.indexes:
        anon_dim = ds["time"].dims[0]
        ds = ds.swap_dims({anon_dim: "time"})
        ds = ds.drop_duplicates("time")
    return ds


def regrid_to_tracer(da: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Regrid a variable on the staggered wind grid onto the tracer grid via
    linear interpolation. Longitude is padded periodically so the wrap-around
    column is not NaN. Confirmed necessary for both the attribution tree and
    the historical tree (sfcWind lat=325 vs tasmax lat=324)."""
    if da.latitude.equals(target.latitude) and da.longitude.equals(target.longitude):
        return da
    lon = da.longitude
    left = da.isel(longitude=[-1]).assign_coords(longitude=[lon.values[-1] - 360.0])
    right = da.isel(longitude=[0]).assign_coords(longitude=[lon.values[0] + 360.0])
    da_ext = xr.concat([left, da, right], dim="longitude")
    return da_ext.interp(latitude=target.latitude, longitude=target.longitude, method="linear")
