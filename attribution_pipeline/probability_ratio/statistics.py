"""
Statistics computed from a region's hist ('historicalExt', factual) vs
histnat ('historicalNatExt', counterfactual) ensemble data.

Two statistics, both as a common `compute(...)` shape so new statistics can be
added the same way:
  - RiskRatioStatistic: probability ratio + bootstrap confidence interval.
  - AmplificationStatistic: per-member intensity amplification (factual mean
    minus counterfactual mean), a sibling diagnostic (not threshold-based).
"""

import numpy as np

from utils.cubefuncs import RiskRatio, draw_bs_replicates


class RiskRatioStatistic:
    def __init__(self, bootstrap_size: int = 10000):
        self.bootstrap_size = bootstrap_size

    def compute(self, hist_data, nat_data, threshold: float) -> dict:
        replicates = draw_bs_replicates(hist_data, nat_data, threshold, RiskRatio, self.bootstrap_size)
        return {
            "median": np.median(replicates),
            "ci_5": np.percentile(replicates, 5),
            "ci_25": np.percentile(replicates, 25),
            "ci_75": np.percentile(replicates, 75),
            "ci_95": np.percentile(replicates, 95),
            "replicates": replicates,
        }


class AmplificationStatistic:
    def compute(self, hist_means: dict, nat_means: dict) -> np.ndarray:
        """hist_means/nat_means: {member: mean value}, as returned by
        EnsembleLoader.load_member_means(). Returns per-member (hist - nat)
        differences for members present in both."""
        common = sorted(set(hist_means) & set(nat_means))
        return np.array([hist_means[m] - nat_means[m] for m in common])
