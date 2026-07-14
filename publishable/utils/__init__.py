"""
Utility functions for StateOfFires project.
"""

from .cubefuncs import (
    CountryMean,
    TimeMean,
    CountryMax,
    CountryPercentile,
    TimeMax,
    TimePercentile,
)

from .constrain_cubes_standard import (
    constrain_to_data,
    ar6_region,
    add_lat_lon_bounds,
    annual_average,
    make_time_series,
    sub_year_range,
    sub_year_months,
    constrain_cube_by_cube_and_numericIDs,
    constrain_GFED,
    constrain_olson,
    contrain_to_shape,
    contrain_to_sow_shapefile,
    constrain_natural_earth,
    mask_data_with_geometry,
    constrain_region,
    contrain_coords,
    constrain_BR_biomes,
)

__all__ = [
    "CountryMean",
    "TimeMean",
    "CountryMax",
    "CountryPercentile",
    "TimeMax",
    "TimePercentile",
    "constrain_to_data",
    "ar6_region",
    "add_lat_lon_bounds",
    "annual_average",
    "make_time_series",
    "sub_year_range",
    "sub_year_months",
    "constrain_cube_by_cube_and_numericIDs",
    "constrain_GFED",
    "constrain_olson",
    "contrain_to_shape",
    "contrain_to_sow_shapefile",
    "constrain_natural_earth",
    "mask_data_with_geometry",
    "constrain_region",
    "contrain_coords",
    "constrain_BR_biomes",
]