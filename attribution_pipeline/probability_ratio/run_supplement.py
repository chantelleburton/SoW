"""
Entrypoint for the 5-panel supplement figure
(attribution_pipeline/probability_ratio/plotting.py: plot_supplement /
generate_all_supplements).

Runs generate_all_supplements() across all configured regions, for a given
index + metric. historical_source ('xclim' default, or 'impacttb') selects
which bias_corrected_metrics/{historical_source}/ ensemble folder panels
(b)/(e) read, and nests output plots under exports/{historical_source}/,
alongside the risk_ratio/amplification summary CSVs and plots written by
run_probability_ratio.py. Output filenames are Supplement_{country}.png.

Usage (mirrors the CYLC_TASK_PARAM_* convention used elsewhere in the repo):
    CYLC_TASK_PARAM_index=fwi \
    CYLC_TASK_PARAM_metric_name=p95 \
    CYLC_TASK_PARAM_historical_source=xclim \
    python -m attribution_pipeline.probability_ratio.run_supplement
"""

import os

from attribution_pipeline.pipeline_config import EXPORTS
from attribution_pipeline.probability_ratio.plotting import (
    generate_all_supplements,
)

DEFAULT_OUTPUT_DIR = EXPORTS


if __name__ == "__main__":
    index = os.environ.get("CYLC_TASK_PARAM_index", "fwi")
    metric_name = os.environ.get("CYLC_TASK_PARAM_metric_name", "p95")
    historical_source = os.environ.get(
        "CYLC_TASK_PARAM_historical_source", "xclim"
    )
    percentile = float(os.environ.get("CYLC_TASK_PARAM_percentile", "95"))
    paired_only = (
        os.environ.get("CYLC_TASK_PARAM_paired_only", "true").lower()
        != "false"
    )

    out_dir = DEFAULT_OUTPUT_DIR
    generate_all_supplements(
        index,
        metric_name,
        out_dir,
        percentile=percentile,
        historical_source=historical_source,
        paired_only=paired_only,
    )
