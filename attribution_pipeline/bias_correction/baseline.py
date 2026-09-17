"""
Baseline series loader: reads the ERA5 / HadGEM3-historical baseline metric
CSVs already produced by attribution_pipeline/metrics/run_metrics.py, instead
of the separate legacy baseline-CSV generation step.

Replaces the reduced_set_HG_bias_correction.py step-0 pandas reads of
`ERA5_FWI_{start}-{end}_{country}_{percentile}%.csv` /
`HadGEM3_FWI_{start}-{end}_{country}_{member}_{percentile}%.csv`.
"""

import os

import numpy as np
import pandas as pd

from attribution_pipeline.bias_correction.regression import soft_log
from attribution_pipeline.pipeline_config import METRICS_OUT_DIR

# Legacy NaN placeholder used before the soft-log transform (log(0) is -inf).
_NAN_FILL = 1e-12


def load_baseline_series(dataset: str, country: str, metric_stem: str, member=None,
                          start: int = 1980, end: int = 2013):
    """Returns (years: np.ndarray[int], soft_log(values): np.ndarray[float]),
    filtered to [start, end] inclusive.

    dataset: 'era5' or 'hg3_historical'.
    member: baseline realisation (1-15) for 'hg3_historical'; ignored for 'era5'.
    """
    member_str = str(member) if member is not None else "1"
    path = os.path.join(METRICS_OUT_DIR, f"{dataset}_{metric_stem}_{country}_{member_str}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No baseline metric CSV for {dataset}/{country}/{metric_stem}: {path}. "
            f"Run attribution_pipeline/metrics/run_metrics.py for dataset={dataset} first."
        )

    df = pd.read_csv(path)
    df = df[(df["Year"] >= start) & (df["Year"] <= end)].sort_values("Year")

    values = df[metric_stem].replace(np.nan, _NAN_FILL).values
    years = df["Year"].values.astype(int)
    return years, soft_log(values)
