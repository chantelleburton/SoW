"""
ERA5 threshold lookup for probability-ratio statistics.

Reads the per-year CSV produced by attribution_pipeline/metrics/run_metrics.py
for the ERA5 dataset (Year,<METRIC_STEM> columns) and picks out the region's
event-year value. Using the same metrics framework as the ensemble side keeps
the threshold and the ensemble data consistent for any metric (FWI95, DSR95,
FWI7X, DSR7X, cumulative DSR, ...).
"""

import os

import pandas as pd

METRICS_OUT_DIR = "/data/scratch/bob.potts/sowf/attribution_pipeline/metrics"


def get_era5_threshold(country: str, event_year: int, metric_stem: str, member: str = "1") -> float:
    """metric_stem: BaseMetric.output_stem(), e.g. 'FWI_P95', 'DSR_7X', 'DSR_CUM360_MEAN'."""
    path = os.path.join(METRICS_OUT_DIR, f"era5_{metric_stem}_{country}_{member}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No ERA5 metric CSV found for {country}/{metric_stem}: {path}. "
            f"Run attribution_pipeline/metrics/run_metrics.py for dataset=era5 first."
        )
    df = pd.read_csv(path)
    row = df[df["Year"] == event_year]
    if row.empty:
        raise ValueError(f"No ERA5 {metric_stem} value for {country} event_year={event_year} in {path}")
    return float(row[metric_stem].values[0])
