"""
Per-region orchestration for the probability-ratio framework: wires together
the region config, ERA5 threshold, ensemble loader (auto-paired), and a
statistic to produce one result dict per country.
"""

import numpy as np

from attribution_pipeline.metrics.pipeline_config import get_region
from attribution_pipeline.probability_ratio.ensemble import EnsembleLoader
from attribution_pipeline.probability_ratio.statistics import AmplificationStatistic, RiskRatioStatistic
from attribution_pipeline.probability_ratio.threshold import get_era5_threshold


def compute_region_risk_ratio(
    country: str,
    metric_stem: str,
    ensemble_folder: str,
    bootstrap_size: int = 10000,
    paired_only: bool = True,
) -> dict:
    region = get_region(country)
    threshold = get_era5_threshold(country, region["event_year"], metric_stem)

    loader = EnsembleLoader(
        ensemble_folder, metric_stem=metric_stem, percentile=region["percentile"],
        baseline_start=region["baseline_start"], baseline_end=region["baseline_end"],
    )
    paired = loader.derive_paired_members(country) if paired_only else None

    hist_data, hist_members = loader.load(country, "hist", paired)
    nat_data, nat_members = loader.load(country, "histnat", paired)
    if len(hist_data) == 0 or len(nat_data) == 0:
        raise RuntimeError(f"No ensemble data found for {country} in {ensemble_folder}")

    stat = RiskRatioStatistic(bootstrap_size)
    result = stat.compute(hist_data, nat_data, threshold)
    result.update(
        {
            "country": country,
            "metric_stem": metric_stem,
            "threshold": threshold,
            "hist_p95": np.nanpercentile(hist_data, region["percentile"]),
            "histnat_p95": np.nanpercentile(nat_data, region["percentile"]),
            "hist_data": hist_data,
            "nat_data": nat_data,
            "n_hist_members": len(hist_members),
            "n_nat_members": len(nat_members),
        }
    )
    return result


def compute_region_amplification(
    country: str,
    metric_stem: str,
    ensemble_folder: str,
    paired_only: bool = True,
) -> dict:
    region = get_region(country)
    loader = EnsembleLoader(
        ensemble_folder, metric_stem=metric_stem, percentile=region["percentile"],
        baseline_start=region["baseline_start"], baseline_end=region["baseline_end"],
    )
    paired = loader.derive_paired_members(country) if paired_only else None

    hist_means = loader.load_member_means(country, "hist", paired)
    nat_means = loader.load_member_means(country, "histnat", paired)

    diffs = AmplificationStatistic().compute(hist_means, nat_means)
    return {"country": country, "amplification": diffs}
