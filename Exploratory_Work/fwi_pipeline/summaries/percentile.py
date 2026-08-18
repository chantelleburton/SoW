"""Generalises the existing apply_shapefile_inclusive -> ConstrainToYear ->
constrain_cube_to_months -> CountryPercentile -> TimePercentile pattern used
identically (and independently re-implemented) across Historical_FWI/*.py and
the xclim bias-correction script. Both baseline sources (ERA5Historical,
HadGEM3Historical) instantiate this with matching config so their output CSVs
stay directly comparable."""
from __future__ import annotations

import os
import sys

import iris
import iris.coord_categorisation as icc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.cubefuncs import apply_shapefile_inclusive
from utils.constrain_cubes_standard import sub_year_months

from .base import SummaryMetric


class PercentileSummary(SummaryMetric):
    def __init__(self, percentile: float, shp_file: str):
        self.percentile = percentile
        self.shp_file = shp_file

    def compute(self, cube, region_cfg: dict) -> pd.DataFrame:
        """region_cfg: {'shape_name': str, 'months': tuple[int, ...] (1-indexed),
        'start_year': int, 'end_year': int}"""
        shape_name = region_cfg["shape_name"]
        months = region_cfg["months"]
        start_year = region_cfg["start_year"]
        end_year = region_cfg["end_year"]

        cube = apply_shapefile_inclusive(self.shp_file, shape_name, cube)
        months_0idx = [m - 1 for m in months]
        cube = sub_year_months(cube, months_0idx)

        # temporal 95th per year, then spatial 95th (matches existing scripts' order)
        try:
            icc.add_year(cube, "time")
        except ValueError:
            pass
        yr_time_p = cube.aggregated_by("year", iris.analysis.PERCENTILE, percent=self.percentile)
        yr_country_p = yr_time_p.collapsed(["latitude", "longitude"], iris.analysis.PERCENTILE, percent=self.percentile)

        years = yr_country_p.coord("year").points
        values = np.ravel(yr_country_p.data)
        month_str = "/".join(f"{m:02d}" for m in months)
        year_month = [f"{int(y)}-{month_str}" for y in years]

        return pd.DataFrame({"Date": year_month, "FWI": values})
