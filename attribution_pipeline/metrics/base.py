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


def _ensure_year_coord(cube, months=None, season_wrap=False):
    """Add (if missing) a "year" aux coord to `cube`.

    When season_wrap is False (the default -- all pre-existing regions),
    this is identical to a plain icc.add_year: "year" == calendar year, no
    behaviour change.

    When season_wrap is True, `months` must be the region's chronological
    event-month list (e.g. [12, 1] for a Dec-Jan event). Timesteps whose
    month_number is BEFORE months[0] (the trailing, January-side months) are
    relabelled to calendar_year - 1, so the whole wrapping season is labelled
    by its START (December) year -- matching the config's event_year
    convention. Leading (December-side) months keep their calendar year.
    """
    if not cube.coords("year"):
        icc.add_year(cube, "time")
    if season_wrap:
        if not months:
            raise ValueError(
                "season_wrap=True requires `months` to determine the wrap threshold."
            )
        if not cube.coords("month_number"):
            icc.add_month_number(cube, "time")
        wrap_start = months[0]
        year_coord = cube.coord("year")
        adjusted = year_coord.points.copy()
        trailing = cube.coord("month_number").points < wrap_start
        adjusted[trailing] -= 1
        year_coord.points = adjusted
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
    raise ValueError(
        f"Unknown spatial_reduction method: {method!r}. Expected 'mean', 'max', or 'p95'."
    )


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
    def compute(self, cube, months, season_wrap=False):
        """Given a region-masked cube spanning the full year range (NOT yet
        constrained to event months), the event month(s) (chronological
        event-order tuple/list), and whether the event crosses the year
        boundary (region's "season_wrap" config flag), return
        (years: list[int], values: list[float]). `years` are labelled by the
        event's START year when season_wrap is True (see _ensure_year_coord)."""
        raise NotImplementedError

    def output_stem(self) -> str:
        """Suffix used in output filenames, e.g. 'FWI_P95' or 'DSR_7X'."""
        return f"{self.index.upper()}_{self.metric_id.upper()}"
