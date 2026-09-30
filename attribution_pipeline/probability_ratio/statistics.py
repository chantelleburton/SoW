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


def RiskRatio(Alldata, Natdata, Threshold):
    """
    Calculate the Risk Ratio between ALL (anthropogenic) and NAT (natural) scenarios.

    Parameters
    ----------
    Alldata : array-like
        FWI values from ALL forcing scenario
    Natdata : array-like
        FWI values from NAT (natural-only) forcing scenario
    Threshold : float
        The threshold value (e.g., ERA5 2025 observed value)

    Returns
    -------
    float
        Risk Ratio (ALL exceedance count / NAT exceedance count)
    """
    ALL_count = np.count_nonzero(Alldata > Threshold)
    NAT_count = np.count_nonzero(Natdata > Threshold)

    if NAT_count == 0:
        return np.inf  # Handle division by zero

    return ALL_count / NAT_count


def draw_bs_replicates(ALL, NAT, threshold, func, size):
    """
    Create bootstrap replicates for uncertainty estimation.

    Uses a two-step resampling: first subsample 90% without replacement,
    then resample to original size with replacement.

    Parameters
    ----------
    ALL : array-like
        FWI values from ALL forcing scenario
    NAT : array-like
        FWI values from NAT forcing scenario
    threshold : float
        The threshold value for Risk Ratio calculation
    func : callable
        Function to compute statistic (e.g., RiskRatio)
    size : int
        Number of bootstrap replicates to generate

    Returns
    -------
    np.ndarray
        Array of bootstrap replicates
    """
    RR_replicates = np.empty(size)

    ALL_subsample_size = int(np.round(len(ALL) * 0.9))
    NAT_subsample_size = int(np.round(len(NAT) * 0.9))

    for i in range(size):
        # Step 1: Subsample 90% without replacement
        ALL_subsample = np.random.choice(ALL, size=ALL_subsample_size, replace=False)
        NAT_subsample = np.random.choice(NAT, size=NAT_subsample_size, replace=False)

        # Step 2: Resample to original size with replacement
        ALL_sample = np.random.choice(ALL_subsample, size=len(ALL), replace=True)
        NAT_sample = np.random.choice(NAT_subsample, size=len(NAT), replace=True)

        # Compute statistic
        RR_replicates[i] = func(ALL_sample, NAT_sample, threshold)

    return RR_replicates


class RiskRatioStatistic:
    def __init__(self, bootstrap_size: int = 10000):
        self.bootstrap_size = bootstrap_size

    def compute(self, hist_data, nat_data, threshold: float) -> dict:
        replicates = draw_bs_replicates(
            hist_data, nat_data, threshold, RiskRatio, self.bootstrap_size
        )
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
