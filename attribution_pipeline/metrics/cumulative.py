"""
CumulativeMetric (cumulative DSR, etc.): accumulated value over the last
`window` days (default 360, tunable), with a tunable spatial reduction across
region cells: 'mean' | 'max' | 'p95'.

Because the accumulation window (default 360 days) is typically much longer
than the event month(s) themselves, this metric needs antecedent context that
can span back across a year (or calendar-month) boundary -- e.g. a Jan-Feb
event's 360-day accumulation reaches back into the previous December. It is
therefore given the *full* region-masked cube (not pre-constrained to event
months) and does its own explicit date-window slicing per year.

Per grid cell: `window`-day rolling SUM over time.
Per year: take the MAX of the rolling sum within the event month(s) (the peak
accumulated value reached during the event), then reduce over region cells via
`spatial_reduction`.
"""

from datetime import date, timedelta

import iris
import iris.analysis

from attribution_pipeline.metrics.base import BaseMetric, spatial_reduce


class CumulativeMetric(BaseMetric):
    def __init__(self, index: str, window: int = 360, spatial_reduction: str = "mean"):
        super().__init__(index)
        self.window = window
        self.context_days = window
        self.spatial_reduction = spatial_reduction
        self.metric_id = f"cum{window}_{spatial_reduction}"

    def compute(self, cube, months):
        time_coord = cube.coord("time")
        dates = time_coord.units.num2date(time_coord.points)
        years_present = sorted({d.year for d in dates})

        start_month, end_month = min(months), max(months)
        years, values = [], []

        for y in years_present:
            event_start = date(y, start_month, 1)
            event_end = date(y + 1, 1, 1) if end_month == 12 else date(y, end_month + 1, 1)
            window_start = event_start - timedelta(days=self.window)

            context_constraint = iris.Constraint(
                time=lambda cell, ws=window_start, ee=event_end: ws <= cell.point < ee
            )
            sub = cube.extract(context_constraint)
            if sub is None or sub.coord("time").shape[0] <= self.window:
                continue  # not enough antecedent context available (e.g. first year in record)

            rolled = sub.rolling_window("time", iris.analysis.SUM, self.window)

            event_constraint = iris.Constraint(
                time=lambda cell, es=event_start, ee=event_end: es <= cell.point < ee
            )
            rolled_event = rolled.extract(event_constraint)
            if rolled_event is None:
                continue

            peak = rolled_event.collapsed("time", iris.analysis.MAX)
            reduced = spatial_reduce(peak, self.spatial_reduction)

            years.append(y)
            values.append(float(reduced.data))

        return years, values
