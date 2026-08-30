"""
EnsembleLoader: loads the corrected, multi-year, per-baseline ensemble CSVs
(hist='historicalExt' / histnat='historicalNatExt') and auto-derives the set
of "paired" members -- those with complete (no-NaN) data in every loaded file
for BOTH run types -- so downstream statistics can restrict to a strictly
matched ensemble without needing to re-scan raw attribution-ensemble
directories (cf. reduced_set_risk_ratio.py's get_paired_members()).

File naming follows the existing corrected-CSV convention (see
Plotting/Explore_Risk_Ratio.py / post-processing/Bias_Correction/):
    {country}_baseline{N}_{run_type}{percentile}percent_LogTransform_Target_{year}_DataYear_{year}_BaselinePeriod_{start}_{end}.csv

NOTE: this pattern currently has no metric token (it predates the generalised
metrics framework and is FWI95-only). Once the bias-correction stage emits
metric-tagged CSVs, update `_glob` accordingly -- everything downstream
(paired-member derivation, flattening, per-member means) is metric-agnostic.
"""

import glob
import os

import numpy as np
import pandas as pd

RUN_TYPE_TOKEN = {"hist": "historicalExt", "histnat": "historicalNatExt"}


class EnsembleLoader:
    def __init__(
        self,
        folder: str,
        baseline_start: int,
        baseline_end: int,
        metric_stem: str = None,
        percentile: float = 95,
    ):
        self.folder = folder
        # attribution_pipeline/bias_correction writes filenames as
        # {country}_{metric_stem}_baseline{N}_..., e.g. 'FWI_P95'. Pass None
        # to match any metric token (useful for pre-metric-token legacy CSVs).
        self.metric_stem = metric_stem
        self.percentile = percentile
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end

    def _glob(self, country: str, run_type: str):
        metric_token = self.metric_stem if self.metric_stem else "*"
        pattern = os.path.join(
            self.folder,
            f"{country}_{metric_token}_baseline*_{run_type}{self.percentile:g}percent_LogTransform_"
            f"Target_*_DataYear_*_BaselinePeriod_{self.baseline_start}_{self.baseline_end}.csv",
        )
        return sorted(glob.glob(pattern))

    def _read_all(self, country: str, run_type: str):
        files = self._glob(country, run_type)
        return [pd.read_csv(f) for f in files], files

    def _all_member_columns(self, dfs):
        members = set()
        for df in dfs:
            members.update(c for c in df.columns if c != "Year")
        return members

    def complete_members(self, country: str, run_type: str) -> set:
        """Members present in every loaded file with no NaN values, for this run_type."""
        dfs, files = self._read_all(country, run_type)
        if not dfs:
            return set()
        all_members = self._all_member_columns(dfs)
        complete = set()
        for m in all_members:
            ok = True
            for df in dfs:
                if m not in df.columns or df[m].isna().any():
                    ok = False
                    break
            if ok:
                complete.add(m)
        return complete

    def derive_paired_members(self, country: str) -> set:
        """Members complete in BOTH hist and histnat -- the strictly-paired ensemble."""
        hist_complete = self.complete_members(country, "hist")
        nat_complete = self.complete_members(country, "histnat")
        return hist_complete & nat_complete

    def load(self, country: str, run_type: str, member_filter: set = None):
        """Return (flattened non-NaN values, members used) for a country/run_type."""
        dfs, files = self._read_all(country, run_type)
        if not dfs:
            return np.array([]), set()
        all_members = self._all_member_columns(dfs)
        cols = [m for m in all_members if member_filter is None or m in member_filter]

        parts = []
        for df in dfs:
            present = [c for c in cols if c in df.columns]
            if present:
                parts.append(df[present].values.flatten())
        if not parts:
            return np.array([]), set()
        values = np.concatenate(parts)
        values = values[~np.isnan(values)]
        return values, set(cols)

    def load_member_means(self, country: str, run_type: str, member_filter: set = None) -> dict:
        """Return {member: mean value across all baselines/years it appears in}."""
        dfs, files = self._read_all(country, run_type)
        if not dfs:
            return {}
        all_members = self._all_member_columns(dfs)
        cols = [m for m in all_members if member_filter is None or m in member_filter]

        sums = {m: 0.0 for m in cols}
        counts = {m: 0 for m in cols}
        for df in dfs:
            for m in cols:
                if m in df.columns:
                    vals = df[m].dropna().values
                    sums[m] += vals.sum()
                    counts[m] += len(vals)
        return {m: sums[m] / counts[m] for m in cols if counts[m] > 0}
