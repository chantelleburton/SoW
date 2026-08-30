"""
Entrypoint for attribution_pipeline/bias_correction, mirroring the
CYLC_TASK_PARAM_* convention used by run_fwi.py / run_metrics.py.

Usage:
    CYLC_TASK_PARAM_country=Iberia \
    CYLC_TASK_PARAM_member=1 \
    CYLC_TASK_PARAM_runtype=hist \
    CYLC_TASK_PARAM_index=fwi \
    CYLC_TASK_PARAM_metric=p95 \
    python -m attribution_pipeline.bias_correction.run_bias_correction
"""

import os

from attribution_pipeline.bias_correction.core import run_bias_correction

if __name__ == "__main__":
    country = os.environ.get("CYLC_TASK_PARAM_country", "Iberia")
    baseline_member = int(os.environ.get("CYLC_TASK_PARAM_member", "1"))
    run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")
    index = os.environ.get("CYLC_TASK_PARAM_index", "fwi")
    metric_name = os.environ.get("CYLC_TASK_PARAM_metric", "p95")
    percentile = float(os.environ.get("CYLC_TASK_PARAM_percentile", "95"))

    metric_kwargs = {}
    if os.environ.get("CYLC_TASK_PARAM_window"):
        metric_kwargs["window"] = os.environ["CYLC_TASK_PARAM_window"]
    if os.environ.get("CYLC_TASK_PARAM_spatial_reduction"):
        metric_kwargs["spatial_reduction"] = os.environ["CYLC_TASK_PARAM_spatial_reduction"]

    run_bias_correction(country, baseline_member, run_type, index, metric_name,
                         percentile=percentile, **metric_kwargs)
