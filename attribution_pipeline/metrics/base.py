"""
BaseMetric: contract for interim metrics computed from daily FWI/DSR cubes.

A metric takes an iris cube already:
  - constrained to the event month(s) and a year range
  - masked to a region shapefile

and returns a per-year time series (one value per year) as the y-axis input for
bias correction / plotting. New metrics are added by subclassing BaseMetric and
implementing `compute`.
"""

from abc import ABC, abstractmethod

import iris.coord_categorisation as icc
from utils.cubefuncs import CountryMax, CountryMean, CountryPercentile


def _ensure_year_coord(cube):
    if not cube.coords("year"):
        icc.add_year(cube, "time")
    return cube


def spatial_reduce(cube, method: str, percentile: float = 95):
    """Collapse the remaining lat/lon dims to a scalar using one of
    'mean' | 'max' | 'p95' (or any percentile via `percentile`)."""
    if method == "mean":
        return CountryMean(cube)
    if method == "max":
        return CountryMax(cube)
    if method in ("p95", "percentile"):
        return CountryPercentile(cube, percentile)
    raise ValueError(f"Unknown spatial_reduction method: {method!r}. Expected 'mean', 'max', or 'p95'.")


class BaseMetric(ABC):
    #: short identifier used in output file naming, e.g. 'p95', '7x', 'cum7'
    metric_id: str = "base"

    #: extra days of antecedent context (before the event month(s) start) this
    #: metric needs, e.g. a 360-day cumulative window needs ~360 days of lead-in.
    #: run_metrics.py loads/masks a region cube spanning the full year range
    #: (not pre-constrained to event months) so metrics with context_days > 0
    #: can look back across the month/year boundary themselves.
    context_days: int = 0

    def __init__(self, index: str):
        """index: which sub-index this metric operates on, e.g. 'fwi' or 'dsr'."""
        self.index = index

    @abstractmethod
    def compute(self, cube, months):
        """Given a region-masked cube spanning the full year range (NOT yet
        constrained to event months) and the event month(s) (1-indexed tuple),
        return (years: list[int], values: list[float])."""
        raise NotImplementedError

    def output_stem(self) -> str:
        """Suffix used in output filenames, e.g. 'FWI_P95' or 'DSR_7X'."""
        return f"{self.index.upper()}_{self.metric_id.upper()}"
