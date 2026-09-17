"""
Entrypoint for the generalised probability-ratio framework.

Runs either the 'risk_ratio' statistic (probability ratio + bootstrap CI +
summary CSV + density-histogram plot) or the 'amplification' statistic
(per-member intensity amplification + summary CSV + box-whisker plot) across
all configured regions, for a given metric (matching the metric produced by
attribution_pipeline/metrics/).

historical_source ('xclim' default, or 'impacttb') selects which
bias_corrected_metrics/{historical_source}/ ensemble folder to read by
default, and nests both the summary CSV and the plot under
exports/{historical_source}/ -- mirroring bias_correction's own
{historical_source}-nested output layout. Pass CYLC_TASK_PARAM_ensemble_folder
explicitly to override the default folder.

Usage (mirrors the CYLC_TASK_PARAM_* convention used elsewhere in the repo):
    CYLC_TASK_PARAM_metric=FWI_P95 \
    CYLC_TASK_PARAM_statistic=risk_ratio \
    CYLC_TASK_PARAM_historical_source=xclim \
    python -m attribution_pipeline.probability_ratio.run_probability_ratio
"""

import os
import numpy as np
import pandas as pd

from attribution_pipeline.metrics.pipeline_config import REGION_CONFIGS
from attribution_pipeline.probability_ratio.core import compute_region_amplification, compute_region_risk_ratio
from attribution_pipeline.probability_ratio.plotting import plot_amplification, plot_risk_ratio_grid

BIAS_CORRECTED_BASE = "/data/scratch/bob.potts/sowf/attribution_pipeline/bias_corrected_metrics"
DEFAULT_OUTPUT_DIR = "/data/scratch/bob.potts/sowf/attribution_pipeline/exports"


def run_risk_ratio(metric_stem: str, ensemble_folder: str, countries, bootstrap_size: int, paired_only: bool,
                   historical_source: str = "xclim"):
    results = {}
    for country in countries:
        print(f"[probability_ratio] risk_ratio: {country} ({metric_stem})")
        results[country] = compute_region_risk_ratio(
            country, metric_stem, ensemble_folder, bootstrap_size=bootstrap_size, paired_only=paired_only
        )

    # Nest both the summary CSV and the plot under exports/{historical_source}/,
    # mirroring bias_correction's bias_corrected_metrics/{historical_source}/ layout.
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, historical_source)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for country, res in results.items():
        likelihood = (res["replicates"] >= 1).sum() / len(res["replicates"]) * 100
        rows.append({
            "Country": country,
            "Metric": metric_stem,
            "ERA5_Threshold": res["threshold"],
            "Hist_95th": res["hist_p95"],
            "HistNat_95th": res["histnat_p95"],
            "N_Hist_Members": res["n_hist_members"],
            "N_HistNat_Members": res["n_nat_members"],
            "RR_Median": res["median"],
            "RR_5th": res["ci_5"],
            "RR_25th": res["ci_25"],
            "RR_75th": res["ci_75"],
            "RR_95th": res["ci_95"],
            "Likelihood": likelihood,
        })
    summary_path = os.path.join(output_dir, f"{metric_stem}_Risk_Ratio_Summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[probability_ratio] Saved: {summary_path}")

    plot_path = os.path.join(output_dir, f"{metric_stem}_Risk_Ratio.png")
    plot_risk_ratio_grid(results, metric_stem, plot_path)
    print(f"[probability_ratio] Saved: {plot_path}")


def run_amplification(metric_stem: str, ensemble_folder: str, countries, paired_only: bool,
                      historical_source: str = "xclim"):
    results = {}
    for country in countries:
        print(f"[probability_ratio] amplification: {country} ({metric_stem})")
        results[country] = compute_region_amplification(country, metric_stem, ensemble_folder, paired_only=paired_only)

    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, historical_source)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for country, res in results.items():
        diffs = res["amplification"]
        if len(diffs) == 0:
            continue
        rows.append({
            "Country": country,
            "Metric": metric_stem,
            "N_Members": len(diffs),
            "Amp_Mean": np.mean(diffs),
            "Amp_Median": np.median(diffs),
            "Amp_5th": np.percentile(diffs, 5),
            "Amp_25th": np.percentile(diffs, 25),
            "Amp_75th": np.percentile(diffs, 75),
            "Amp_95th": np.percentile(diffs, 95),
        })
    summary_path = os.path.join(output_dir, f"{metric_stem}_Intensity_Amplification_Summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[probability_ratio] Saved: {summary_path}")

    plot_path = os.path.join(output_dir, f"{metric_stem}_Intensity_Amplification.png")
    plot_amplification(results, metric_stem, plot_path)
    print(f"[probability_ratio] Saved: {plot_path}")


if __name__ == "__main__":
    # metric_stem can be given directly (e.g. FWI_P95), or built from
    # index + metric (+ metric kwargs) exactly as BaseMetric.output_stem does --
    # the latter is what the flow.cylc passes, so 'cum' correctly expands to
    # e.g. DSR_CUM360_MEAN rather than a bare DSR_CUM.
    metric_stem = os.environ.get("CYLC_TASK_PARAM_metric")
    if not metric_stem:
        from attribution_pipeline.metrics.run_metrics import METRICS

        index = os.environ.get("CYLC_TASK_PARAM_index", "fwi")
        metric_name = os.environ.get("CYLC_TASK_PARAM_metric_name", "p95")
        metric_kwargs = {}
        if os.environ.get("CYLC_TASK_PARAM_window"):
            metric_kwargs["window"] = os.environ["CYLC_TASK_PARAM_window"]
        if os.environ.get("CYLC_TASK_PARAM_percentile"):
            metric_kwargs["percentile"] = os.environ["CYLC_TASK_PARAM_percentile"]
        if os.environ.get("CYLC_TASK_PARAM_spatial_reduction"):
            metric_kwargs["spatial_reduction"] = os.environ["CYLC_TASK_PARAM_spatial_reduction"]
        metric_stem = METRICS[metric_name](index, **metric_kwargs).output_stem()

    statistic = os.environ.get("CYLC_TASK_PARAM_statistic", "risk_ratio")
    historical_source = os.environ.get("CYLC_TASK_PARAM_historical_source", "xclim")
    ensemble_folder = os.environ.get(
        "CYLC_TASK_PARAM_ensemble_folder", os.path.join(BIAS_CORRECTED_BASE, historical_source)
    )
    bootstrap_size = int(os.environ.get("CYLC_TASK_PARAM_bootstrap_size", "10000"))
    paired_only = os.environ.get("CYLC_TASK_PARAM_paired_only", "true").lower() != "false"

    countries_env = os.environ.get("CYLC_TASK_PARAM_countries")
    countries = countries_env.split(",") if countries_env else list(REGION_CONFIGS)
    print(countries)
    if statistic == "risk_ratio":
        run_risk_ratio(metric_stem, ensemble_folder, countries, bootstrap_size, paired_only,
                       historical_source=historical_source)
    elif statistic == "amplification":
        run_amplification(metric_stem, ensemble_folder, countries, paired_only,
                          historical_source=historical_source)
    else:
        raise SystemExit(f"Unknown statistic {statistic!r}. Expected 'risk_ratio' or 'amplification'.")
