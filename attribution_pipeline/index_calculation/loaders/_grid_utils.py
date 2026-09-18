"""Small helpers shared by loaders (not by the core calculator, which stays
dataset-agnostic)."""

import xarray as xr


def regrid_to_tracer(
    da: xr.DataArray,
    target: xr.DataArray,
    lat_name="latitude",
    lon_name="longitude",
):
    """Regrid a variable on the staggered wind grid onto the tracer grid via linear
    interpolation. Longitude is padded periodically so the wrap-around column is not NaN."""
    lat, lon = da[lat_name], da[lon_name]
    if lat.equals(target[lat_name]) and lon.equals(target[lon_name]):
        return da
    left = da.isel({lon_name: [-1]}).assign_coords(
        {lon_name: [lon.values[-1] - 360.0]}
    )
    right = da.isel({lon_name: [0]}).assign_coords(
        {lon_name: [lon.values[0] + 360.0]}
    )
    da_ext = xr.concat([left, da, right], dim=lon_name)
    return da_ext.interp(
        **{lat_name: target[lat_name], lon_name: target[lon_name]},
        method="linear",
    )


def fix_anonymous_time_dim(ds):
    """Fix files where time is a coordinate on an anonymous dim (e.g. dim0) rather
    than a dimension itself, and drop cross-file duplicate timestamps."""
    if "time" in ds.coords and "time" not in ds.indexes:
        anon_dim = ds["time"].dims[0]
        ds = ds.swap_dims({anon_dim: "time"})
        ds = ds.drop_duplicates("time")
    return ds
