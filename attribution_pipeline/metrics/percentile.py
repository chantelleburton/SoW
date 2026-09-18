"""
PercentileMetric (FWI95 / DSR95, etc.): per-year percentile-over-time then
percentile-over-space, matching the existing FWI95 pipeline
(post-processing/Metric-HG3-A_Historical/FWI95-HG3-A_Historical.py).
"""

import iris.analysis
from utils.cubefuncs import CountryPercentile, constrain_cube_to_months

from attribution_pipeline.metrics.base import BaseMetric, _ensure_year_coord


class PercentileMetric(BaseMetric):
    metric_id = "p95"

    def __init__(self, index: str, percentile: float = 95):
        super().__init__(index)
        self.percentile = percentile
        self.metric_id = f"p{percentile:g}"

    def compute(self, cube, months):
        cube = constrain_cube_to_months(cube, months)
        cube = _ensure_year_coord(cube)

        # 1) percentile over time within each year
        yr_time_p = cube.aggregated_by("year", iris.analysis.PERCENTILE, percent=self.percentile)
        # 2) percentile over space (lat/lon) for each year
        yr_country_p = CountryPercentile(yr_time_p, self.percentile)

        years = list(yr_country_p.coord("year").points)
        values = list(yr_country_p.data.ravel())
        return years, values
