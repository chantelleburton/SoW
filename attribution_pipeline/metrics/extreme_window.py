"""
ExtremeWindowMetric (FWI7X / DSR7X): extremes in fire-weather intensity defined
by the local maxima in a window-day (default 7) rolling mean.

Per grid cell: compute the `window`-day rolling mean over time.
Per year: take the maximum of that rolling mean across ALL cells in the
region (i.e. the single hottest cell/window in the event period that year).
"""

import iris.analysis

from attribution_pipeline.metrics.base import BaseMetric, _ensure_year_coord
from utils.cubefuncs import CountryMax, constrain_cube_to_months


class ExtremeWindowMetric(BaseMetric):
    def __init__(self, index: str, window: int = 7):
        super().__init__(index)
        self.window = window
        self.metric_id = f"{window}x"

    def compute(self, cube, months):
        cube = constrain_cube_to_months(cube, months)
        # 1) window-day rolling mean per cell (drops edge days that can't form a full window)
        rolled = cube.rolling_window("time", iris.analysis.MEAN, self.window)
        rolled = _ensure_year_coord(rolled)

        # 2) per year: max of the rolling mean over time, per cell
        yr_time_max = rolled.aggregated_by("year", iris.analysis.MAX)
        # 3) per year: max over all region cells -> single peak value per year
        yr_country_max = CountryMax(yr_time_max)

        years = list(yr_country_max.coord("year").points)
        values = list(yr_country_max.data.ravel())
        return years, values
