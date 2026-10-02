"""
Cube/shapefile/temporal helper functions used across attribution_pipeline
stages (bias_correction, metrics).
"""

import os

import geopandas as gpd
import iris
import iris.coord_categorisation as icc
import numpy as np
from shapely.geometry import MultiPolygon


def CountryMean(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    grid_weights = iris.analysis.cartography.area_weights(cube)
    cube = cube.collapsed(coords, iris.analysis.MEAN, weights=grid_weights)
    return cube


def CountryMax(cube):
    coords = ('longitude', 'latitude')
    cube = cube.collapsed(coords, iris.analysis.MAX)
    return cube


def CountryPercentile(cube, percentile):
    coords = ('longitude', 'latitude')
    cube = cube.collapsed(coords, iris.analysis.PERCENTILE, percent=percentile)
    return cube


def ConstrainToYear(cube, target_year):
    year_constraint = iris.Constraint(time=lambda cell: cell.point.year == target_year)
    out = cube.extract(year_constraint)
    if out is None:
        t = cube.coord('time')
        dts = t.units.num2date(t.points)
        years_present = sorted({dt.year for dt in dts})
        raise ValueError(f"No data for year {target_year}. Years present: {years_present[:5]} ... {years_present[-5:]}")
    return out


def sub_year_months(cube, months_of_year):
    """Selects months of a year from data
    Arguments:
        data -- iris cube with time array we can add add_month_number too.
        months_of_year -- numeric, month of the year you are interested in
                from 0 (Jan) to 11 (Dec)
    Returns:
        cube of just months we are interested in.
    """
    try:
        icc.add_month_number(cube, 'time')
    except Exception:
        pass

    months_of_year = np.array(months_of_year) + 1
    season = iris.Constraint(month_number=lambda cell, mnths=months_of_year:
                              np.any(np.abs(mnths - cell[0]) < 0.5))
    return cube.extract(season)


def constrain_cube_to_months(cube, months):
    """Filter a cube's time axis down to specific calendar month(s).

    Thin wrapper around sub_year_months that accepts 1-indexed months
    (1=Jan ... 12=Dec) for readability at call sites, since sub_year_months
    itself expects 0-indexed months.

    NOTE: any baseline regression CSVs used alongside this correction must be
    generated from the same month window, or the correction becomes
    scientifically inconsistent.

    Arguments:
        cube -- iris cube with a 'time' coordinate.
        months -- int, or tuple/list of ints, 1-indexed (1=Jan, ..., 12=Dec).
    Returns:
        cube constrained to the given month(s).
    """
    if isinstance(months, int):
        months = (months,)
    months_0idx = [m - 1 for m in months]
    return sub_year_months(cube, months_0idx)


def contrain_coords(cube, extent):
    # CB added this to convert 0-360 cube to -180to+180)
    cube = cube.intersection(longitude=(-180, 180))

    longitude_constraint = iris.Constraint(longitude=lambda cell: extent[0] <= cell.point <= extent[1])
    latitude_constraint = iris.Constraint(latitude=lambda cell: extent[2] <= cell.point <= extent[3])

    return cube.extract(longitude_constraint & latitude_constraint)


def apply_shapefile_inclusive(shp_file, shape_name, cube, mainland_only=False,
                               name_column='name'):

    shapefile = gpd.read_file(shp_file)

    # Set coordinate system for iris.util.mask_cube_from_shape
    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

    # Get geometry for this region
    if name_column not in shapefile.columns:
        raise KeyError(
            f"Column {name_column!r} not found in {shp_file!r}. "
            f"Available columns: {shapefile.columns.tolist()}"
        )
    region_gdf = shapefile[shapefile[name_column] == shape_name]
    if region_gdf.empty:
        raise ValueError(
            f"No feature with {name_column}={shape_name!r} found in "
            f"{shp_file!r}."
        )
    region_geom = region_gdf['geometry'].values[0]

    # Optionally drop offshore islands: keep only the single largest polygon
    # (the mainland). Default off -- masking is otherwise unchanged.
    if mainland_only:
        if isinstance(region_geom, MultiPolygon):
            total_area = region_geom.area
            mainland_geom = max(region_geom.geoms, key=lambda g: g.area)
            kept_area = mainland_geom.area
            dropped_area = total_area - kept_area
            dropped_pct = 100 * dropped_area / total_area
            print(
                f"mainland_only: kept 1 of {len(region_geom.geoms)} polygons "
                f"for '{shape_name}' -- kept area {kept_area:.2f} deg^2, "
                f"dropped area {dropped_area:.2f} deg^2 "
                f"({dropped_pct:.1f}% of total area)"
            )
            region_geom = mainland_geom
        else:
            print(
                f"mainland_only: '{shape_name}' is a single polygon -- "
                "nothing to drop"
            )

    # Step 1: Crop to bounding box to reduce data volume
    minx, miny, maxx, maxy = region_geom.bounds
    cube = contrain_coords(cube, (minx, maxx, miny, maxy))

    # contrain_coords() wraps 0-360 -> -180/180 via cube.intersection(), which
    # reassembles the (possibly dask-backed) data along the longitude axis and
    # can leave the array's lazy .chunks metadata out of sync with its real
    # shape. That desync doesn't surface here -- it silently propagates and
    # later breaks mask_cube's dask broadcast/rechunk below ("Chunks do not
    # add up to shape").
    cube.data
    # Step 2: Apply inclusive mask via iris
    masked_cube = iris.util.mask_cube_from_shape(cube, region_geom)

    return masked_cube
