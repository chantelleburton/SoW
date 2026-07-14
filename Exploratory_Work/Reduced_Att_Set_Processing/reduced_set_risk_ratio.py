"""
TEMPORARY / EXPLORATORY reduced-set Risk Ratio script.

Companion to Exploratory_Work/reduced_set_HG_bias_correction.py. It computes Risk Ratios from the
bias-corrected CSVs produced for the REDUCED set of attribution ensemble members (the members with
complete data over 2020-2024), rather than the full 105 x 5 grid.

Design goal: "only use what's available".
  * The ensemble loader GLOBS every bias-corrected CSV that exists for a country/run_type, across
    whatever baselines and target years are present, and flattens every member column it finds.
  * It therefore requires no hard-coded list of baselines, years or members. If more members (extra
    columns) or more files (extra baselines/years) are generated later, they are picked up
    automatically on the next run with no code changes.
  * NaNs (members missing for a given file) are dropped so only genuinely available values are used.

Everything else (ERA5 threshold, bootstrap, plotting, summary export) mirrors
Plotting/Explore_Risk_Ratio.py, writing to separate "Reduced_Set" outputs so the full-ensemble
results are never overwritten.
"""

import os
import glob
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
from utils.cubefuncs import (
    RiskRatio,
    draw_bs_replicates,
    GetERA5ThresholdFromMonthly,
)

mpl.rcParams['font.family'] = 'Work Sans'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


############# Configuration #############
# Reduced-set bias-corrected CSVs (output of reduced_set_HG_bias_correction.py)
LOG_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Reduced_Set_Log_Transforms'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
PLOT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Plots'
EXPORT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Exports'
ERA5_FWI_DIR = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI'
BOOTSTRAP_SIZE = 10000

# Directories to scan for complete ensemble members
HIST_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalExt'
)
HISTNAT_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalNatExt'
)

# Expected 63 consecutive monthly date stamps for a complete member (Nov 2019 - Jan 2025)
_EXPECTED_DATES = {
    '20191101-20191201', '20191201-20200101',
    '20200101-20200201', '20200201-20200301', '20200301-20200401',
    '20200401-20200501', '20200501-20200601', '20200601-20200701',
    '20200701-20200801', '20200801-20200901', '20200901-20201001',
    '20201001-20201101', '20201101-20201201', '20201201-20210101',
    '20210101-20210201', '20210201-20210301', '20210301-20210401',
    '20210401-20210501', '20210501-20210601', '20210601-20210701',
    '20210701-20210801', '20210801-20210901', '20210901-20211001',
    '20211001-20211101', '20211101-20211201', '20211201-20220101',
    '20220101-20220201', '20220201-20220301', '20220301-20220401',
    '20220401-20220501', '20220501-20220601', '20220601-20220701',
    '20220701-20220801', '20220801-20220901', '20220901-20221001',
    '20221001-20221101', '20221101-20221201', '20221201-20230101',
    '20230101-20230201', '20230201-20230301', '20230301-20230401',
    '20230401-20230501', '20230501-20230601', '20230601-20230701',
    '20230701-20230801', '20230801-20230901', '20230901-20231001',
    '20231001-20231101', '20231101-20231201', '20231201-20240101',
    '20240101-20240201', '20240201-20240301', '20240301-20240401',
    '20240401-20240501', '20240501-20240601', '20240601-20240701',
    '20240701-20240801', '20240801-20240901', '20240901-20241001',
    '20241001-20241101', '20241101-20241201', '20241201-20250101',
    '20250101-20250201',
}
_N_EXPECTED = len(_EXPECTED_DATES)
_DATE_RE = re.compile(r'(\d{8}-\d{8})\.nc$')

# Set to True to restrict both hist and histnat to the 38 members present in both columns,
# giving a strictly paired ensemble. Set to False to use all available members independently.
PAIRED_ONLY = True

REGION_CONFIGS = {
    'Korea': {
        'Month': 3,
        'month_name': 'March',
        'percentile': 95,
        'shape_name': 'Southeast South Korea',
        'event_year': 2025,
    },
    'Iberia': {
        'Month': 8,
        'month_name': 'Aug',
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
        'event_year': 2025,
    },
    'Scotland': {
        'Month': (6, 7),
        'month_name': 'June-July',
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
        'event_year': 2025,
    },
    'Chile': {
        'Month': (1, 2),
        'month_name': 'January-February',
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
        'event_year': 2026,
    },
    'Canada': {
        'Month': (7, 8),
        'month_name': 'July-August',
        'percentile': 95,
        'shape_name': 'Midwestern Canadian Shield forests',
        'event_year': 2025,
    },
}

DISPLAY_NAMES = {
    'Northwest Iberia': 'NW Iberia',
    'Southeast South Korea': 'SE S. Korea',
    'Scottish Highlands': 'Scottish Highlands',
    'Chilean Temperate Forests and Matorral': 'Chile Forests & Matorral',
    'Midwestern Canadian Shield forests': 'Canadian Shield Forests',
}


############# Helper Functions #############


def load_reduced_ensemble_data(country, percentile, run_type, folder,
                               allowed_members=None):
    """
    Load every available reduced-set ensemble value for a country / run_type.

    Globs all bias-corrected CSVs matching the country, run_type and percentile, regardless of
    baseline member or target year, and flattens all member columns from each file. This means the
    function uses whatever data is present on disk - add more members (columns) or more baseline/year
    files later and they are included automatically.

    Parameters
    ----------
    allowed_members : set or None
        If provided, only columns whose name is in this set are included. Use this to restrict to
        the paired sub-ensemble (members present in both hist and histnat).

    Returns
    -------
    values : np.ndarray
        Flattened array of all available (non-NaN) bias-corrected FWI values.
    n_files : int
        Number of CSV files that were read.
    members : set
        Set of unique member column names used (across all files).
    """
    pattern = os.path.join(
        folder,
        f"{country}_baseline*_{run_type}{percentile}percent_LogTransform_"
        f"Target_*_DataYear_*_BaselinePeriod_*.csv",
    )
    files = sorted(glob.glob(pattern))

    all_data = []
    members = set()
    for filepath in files:
        df = pd.read_csv(filepath)
        col_names = [col for col in df.columns if col != 'Year']
        if allowed_members is not None:
            col_names = [col for col in col_names if col in allowed_members]
        members.update(col_names)
        if col_names:
            all_data.append(df[col_names].values.flatten())

    if all_data:
        values = np.concatenate(all_data)
        values = values[~np.isnan(values)]  # only use what is actually available
        return values, len(files), members

    return np.array([]), 0, members


def calculate_risk_ratio_with_ci(all_data, nat_data, threshold, bootstrap_size=10000):
    """Calculate Risk Ratio with confidence intervals via bootstrapping."""
    rr_replicates = draw_bs_replicates(
        all_data, nat_data, threshold, RiskRatio, bootstrap_size
    )

    return {
        'median': np.median(rr_replicates),
        'ci_interquartile': np.percentile(rr_replicates, [25, 75]),
        'ci_5': np.percentile(rr_replicates, 5),
        'ci_95': np.percentile(rr_replicates, 95),
        'replicates': rr_replicates,
    }


############# Main Analysis #############


def get_paired_members(hist_dir, histnat_dir):
    """
    Dynamically scan historicalExt and historicalNatExt directories and return
    the set of member IDs (e.g. 'r037i1p1') that are complete (63/63 monthly
    files with no gaps) in BOTH directories.
    """
    paired = set()
    # Check every subdirectory that exists in both
    try:
        hist_members = set(os.listdir(hist_dir))
    except FileNotFoundError:
        hist_members = set()
    try:
        histnat_members = set(os.listdir(histnat_dir))
    except FileNotFoundError:
        histnat_members = set()

    for member in hist_members & histnat_members:
        hist_path = os.path.join(hist_dir, member)
        histnat_path = os.path.join(histnat_dir, member)
        if _is_member_complete(hist_path) and _is_member_complete(histnat_path):
            paired.add(member)
    return paired


def _is_member_complete(member_path):
    """Check whether a member subdirectory contains all 63 expected monthly files."""
    if not os.path.isdir(member_path):
        return False
    stamps = set()
    for fname in os.listdir(member_path):
        m = _DATE_RE.search(fname)
        if m:
            stamps.add(m.group(1))
    return len(stamps & _EXPECTED_DATES) == _N_EXPECTED


def main():
    """Run the reduced-set Risk Ratio analysis for all configured regions."""

    os.makedirs(PLOT_FOLDER, exist_ok=True)
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    # Resolve paired member filter once up front
    if PAIRED_ONLY:
        paired_members = get_paired_members(HIST_DIR, HISTNAT_DIR)
        label_suffix = 'paired'
        print(f"PAIRED_ONLY mode: restricting to {len(paired_members)} members "
              f"complete in both HistExt and HistNatExt")
    else:
        paired_members = None
        label_suffix = 'reduced_set'

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    results = {}
    plot_idxs = [0, 1, 2, 3, 4]
    n_regions = len(REGION_CONFIGS)

    for idx, (country, config) in enumerate(REGION_CONFIGS.items()):
        print(f"\n{'='*50}")
        print(f"Processing {country} (reduced set)")
        print('='*50)

        # ERA5 threshold from monthly files (event month(s) of the event year)
        print("Loading ERA5 threshold...")
        threshold = GetERA5ThresholdFromMonthly(
            ERA5_FWI_DIR, SHP_FILE, config['shape_name'],
            config['Month'], config['event_year'], config['percentile']
        )
        print(f"ERA5 threshold: {threshold:.2f}")

        # Load whatever reduced-set ensemble data is available
        print("Loading reduced-set ensemble data from CSVs...")
        all_data, n_hist_files, hist_members = load_reduced_ensemble_data(
            country, config['percentile'], 'hist', LOG_FOLDER,
            allowed_members=paired_members,
        )
        nat_data, n_nat_files, nat_members = load_reduced_ensemble_data(
            country, config['percentile'], 'histnat', LOG_FOLDER,
            allowed_members=paired_members,
        )
        print(f"  hist:    {len(all_data)} values from {n_hist_files} files, "
              f"{len(hist_members)} unique members")
        print(f"  histnat: {len(nat_data)} values from {n_nat_files} files, "
              f"{len(nat_members)} unique members")

        if len(all_data) == 0 or len(nat_data) == 0:
            print(f"Skipping {country} due to missing data")
            continue

        # Empirical 95th percentiles from loaded hist/histnat ensembles
        hist_p95 = np.percentile(all_data, 95)
        histnat_p95 = np.percentile(nat_data, 95)

        # Risk Ratio with bootstrapped confidence intervals
        print("Calculating Risk Ratio...")
        rr_results = calculate_risk_ratio_with_ci(
            all_data, nat_data, threshold, BOOTSTRAP_SIZE
        )

        results[country] = {
            'threshold': threshold,
            'hist_p95': hist_p95,
            'histnat_p95': histnat_p95,
            'rr': rr_results,
            'all_data': all_data,
            'nat_data': nat_data,
            'n_hist_members': len(hist_members),
            'n_nat_members': len(nat_members),
        }

        print(f"hist 95th percentile: {hist_p95:.2f}")
        print(f"histnat 95th percentile: {histnat_p95:.2f}")
        print(f"Risk Ratio: {rr_results['median']:.2f} "
              f"[{rr_results['ci_5']:.2f} - {rr_results['ci_95']:.2f}] "
              f"(Interquartile Range: {rr_results['ci_interquartile'][0]:.2f} - "
              f"{rr_results['ci_interquartile'][1]:.2f})")

        # Export bootstrap replicates
        pd.DataFrame({'rr_replicates': rr_results['replicates']}).to_csv(
            f'{EXPORT_FOLDER}/{country}_Reduced_Set_{label_suffix}_Risk_Ratio_Bootstrap_Replicates.csv',
            index=False,
        )

        # Plot
        ax = axes[plot_idxs[idx]]
        sns.histplot(all_data, kde=True, color='#C7403D', label='Factual (Current Climate)',
                     alpha=0.5, ax=ax, stat='density')
        sns.histplot(nat_data, kde=True, color='#008787', label='Counterfactual (Natural Only Climate)',
                     alpha=0.5, ax=ax, stat='density')
        ax.axvline(x=threshold, color='black', linewidth=2.5,
                   label=f'ERA5 {config["month_name"]} {config["event_year"]}')

        display_name = DISPLAY_NAMES.get(config['shape_name'], config['shape_name'])
        mode_label = 'paired' if PAIRED_ONLY else 'reduced set'
        ax.set_title(f"{display_name}\nFWI {config['month_name']} ({mode_label})")
        ax.set_xlabel('Fire Weather Index')

        if idx % ncols == 0:
            ax.set_ylabel('Density')
        if idx == n_regions - 1:
            ax.legend()

    # Summary panel in the last subplot (bottom right)
    summary_ax = axes[-1]
    summary_ax.axis('off')
    mode_title = 'PAIRED' if PAIRED_ONLY else 'REDUCED SET'
    summary_lines = [f"SUMMARY OF RESULTS ({mode_title})", ""]
    for country, res in results.items():
        rr = res['rr']
        summary_lines.append(f"{country}: RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}]")
    summary_text = "\n".join(summary_lines)
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=12, wrap=True, family='monospace')

    plt.tight_layout()
    plot_file = f'{PLOT_FOLDER}/Reduced_Set_{label_suffix}_Risk_Ratio.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

    # Print summary
    mode_title = 'PAIRED' if PAIRED_ONLY else 'REDUCED SET'
    print("\n" + "="*60)
    print(f"SUMMARY OF RESULTS ({mode_title})")
    print("="*60)
    for country, res in results.items():
        rr = res['rr']
        print(
            f"{country}: hist_p95 = {res['hist_p95']:.2f}, "
            f"histnat_p95 = {res['histnat_p95']:.2f}, "
            f"RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}] "
            f"(hist members: {res['n_hist_members']}, histnat members: {res['n_nat_members']})"
        )

    # Export summary CSV
    summary_rows = []
    for country, res in results.items():
        rr = res['rr']
        likelihood = np.sum(rr['replicates'] >= 1) / len(rr['replicates']) * 100
        summary_rows.append({
            'Country': country,
            'ERA5_Threshold': res['threshold'],
            'Hist_95th': res['hist_p95'],
            'HistNat_95th': res['histnat_p95'],
            'N_Hist_Members': res['n_hist_members'],
            'N_HistNat_Members': res['n_nat_members'],
            'RR_Median': rr['median'],
            'RR_5th': rr['ci_5'],
            'RR_25th': rr['ci_interquartile'][0],
            'RR_50th': rr['median'],
            'RR_75th': rr['ci_interquartile'][1],
            'RR_95th': rr['ci_95'],
            'Likelihood': likelihood,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f'{EXPORT_FOLDER}/Reduced_Set_{label_suffix}_Risk_Ratio_Summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary exported to: {summary_path}")
    print(f"Plot saved to:       {plot_file}")

    return results


if __name__ == "__main__":
    results = main()
