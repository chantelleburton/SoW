"""
Metric-agnostic scalar extraction for a single member/target-year.

All metrics -- percentile ('p95' etc.), 7x, cum -- now go through the same
attribution_pipeline/metrics/ classes (via run_metrics.py's METRICS registry),
using their time-then-space order consistently. This used to differ for
percentile metrics (the legacy reduced_set_HG_bias_correction.py did
space-then-time via CountryPercentile/TimePercentile); that inconsistency has
been reconciled in favour of time-then-space, matching every other
baseline/metric-generating script in the repo (see
/memories/session/plan_bias_correction.md for the history). This changes the
numeric output of bias correction relative to the old reduced-set CSVs.

Metrics receive the FULL region-masked cube (not year/month pre-constrained),
since CumulativeMetric needs antecedent context across the year boundary and
PercentileMetric/ExtremeWindowMetric constrain internally.
"""

from attribution_pipeline.bias_correction.member_loader import validate_member_window
from attribution_pipeline.metrics.run_metrics import METRICS


def extract_scalar(cube, months, data_year: int, metric_name: str, index: str,
                    percentile: float = 95, **metric_kwargs) -> float:
    """cube: full-window, shapefile-masked member cube (from
    member_loader.load_member_cube). Returns a single scalar value for
    data_year/months."""
    # Validate first -- raises MissingMemberError/InvalidMemberDataError.
    validate_member_window(cube, data_year, months)

    kwargs = dict(metric_kwargs)
    if metric_name.startswith("p"):
        kwargs.setdefault("percentile", percentile)

    metric = METRICS[metric_name](index, **kwargs)
    years, values = metric.compute(cube, months)
    years = list(years)
    if data_year not in years:
        raise ValueError(f"No {metric_name} value for year {data_year}; available years: {years}")
    return float(values[years.index(data_year)])
