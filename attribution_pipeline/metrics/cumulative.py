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

from datetime import timedelta

import cftime
import iris
import iris.analysis
import numpy as np

from attribution_pipeline.metrics.base import BaseMetric, spatial_reduce


class CumulativeMetric(BaseMetric):
    def __init__(
        self, index: str, window: int = 360, spatial_reduction: str = "mean"
    ):
        super().__init__(index)
        self.window = window
        self.context_days = window
        self.spatial_reduction = spatial_reduction
        self.metric_id = f"cum{window}_{spatial_reduction}"

    def compute(self, cube, months):
        time_coord = cube.coord("time")
        calendar = time_coord.units.calendar
        dates = time_coord.units.num2date(time_coord.points)
        years_present = sorted({d.year for d in dates})

        start_month, end_month = min(months), max(months)
        years, values = [], []

        for y in years_present:
            # Build window bounds using the cube's own calendar (e.g. 360_day
            # for HadGEM3, standard/proleptic_gregorian for ERA5) rather than
            # plain datetime.date, which is not comparable to cftime objects
            # and raises TypeError when the cube uses a non-standard calendar.
            event_start = cftime.datetime(y, start_month, 1, calendar=calendar)
            event_end = (
                cftime.datetime(y + 1, 1, 1, calendar=calendar)
                if end_month == 12
                else cftime.datetime(y, end_month + 1, 1, calendar=calendar)
            )
            window_start = event_start - timedelta(days=self.window)

            context_constraint = iris.Constraint(
                time=lambda cell, ws=window_start, ee=event_end: (
                    ws <= cell.point < ee
                )
            )
            sub = cube.extract(context_constraint)
            n_context = 0 if sub is None else sub.coord("time").shape[0]
            if n_context < 1:
                continue

            # Use the full accumulation window when enough antecedent context
            # is available (e.g. self.window=360 days); otherwise fall back
            # to a partial window spanning whatever days actually exist (e.g.
            # 240 instead of 360 for the first year of a record) rather than
            # dropping the year entirely.
            use_window = min(self.window, n_context)
            if use_window < self.window:
                print(
                    f"[cumulative] {y}: partial window -- only {n_context} antecedent "
                    f"timesteps available (need {self.window}), using window={use_window}"
                )

            rolled = sub.rolling_window("time", iris.analysis.SUM, use_window)

            # `rolling_window` sets each output timestep's point to the
            # *midpoint* of its window, not the window's end/as-of date, so
            # matching against the event month must be done positionally
            # against the pre-roll time points rather than `rolled`'s own
            # (midpoint) time coordinate: the i-th rolled entry sums
            # sub.points[i : i + use_window], so its true end/as-of date is
            # sub.points[i + use_window - 1].
            sub_time_points = sub.coord("time").points
            end_dates = time_coord.units.num2date(
                sub_time_points[use_window - 1 :]
            )
            mask = np.array([event_start <= d < event_end for d in end_dates])
            if not mask.any():
                continue

            time_dim = rolled.coord_dims("time")[0]
            index = [slice(None)] * rolled.ndim
            index[time_dim] = mask
            rolled_event = rolled[tuple(index)]

            peak = rolled_event.collapsed("time", iris.analysis.MAX)
            reduced = spatial_reduce(peak, self.spatial_reduction)

            years.append(y)
            values.append(float(reduced.data))

        return years, values
