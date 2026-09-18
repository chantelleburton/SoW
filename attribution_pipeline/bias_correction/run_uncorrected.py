"""
Entrypoint for attribution_pipeline/bias_correction/uncorrected.py, mirroring
the CYLC_TASK_PARAM_* convention used by run_bias_correction.py.

Usage:
    CYLC_TASK_PARAM_country=Iberia \
    CYLC_TASK_PARAM_runtype=hist \
    CYLC_TASK_PARAM_index=fwi \
    CYLC_TASK_PARAM_metric=p95 \
    python -m attribution_pipeline.bias_correction.run_uncorrected

Unlike run_bias_correction.py, there is no baseline_member/historical_source
parameter -- this stage extracts each attribution-ensemble member's raw scalar
directly, with no baseline regression involved. Output CSVs are written under
uncorrected_metrics/.
"""

import os

from attribution_pipeline.bias_correction.uncorrected import (
    run_uncorrected_extraction,
)

if __name__ == "__main__":
    country = os.environ.get("CYLC_TASK_PARAM_country", "Iberia")
    run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")
    index = os.environ.get("CYLC_TASK_PARAM_index", "fwi")
    metric_name = os.environ.get("CYLC_TASK_PARAM_metric", "p95")
    percentile = float(os.environ.get("CYLC_TASK_PARAM_percentile", "95"))

    metric_kwargs = {}
    if os.environ.get("CYLC_TASK_PARAM_window"):
        metric_kwargs["window"] = os.environ["CYLC_TASK_PARAM_window"]
    if os.environ.get("CYLC_TASK_PARAM_spatial_reduction"):
        metric_kwargs["spatial_reduction"] = os.environ[
            "CYLC_TASK_PARAM_spatial_reduction"
        ]

    run_uncorrected_extraction(
        country,
        run_type,
        index,
        metric_name,
        percentile=percentile,
        **metric_kwargs,
    )
