"""
Altered version of Supplement2_5Panel.py
Identical to the original except subplot (e) uses the multiyear reduced-set
bias-corrected ensemble produced by reduced_set_risk_ratio.py, rather than the
fixed single-year Condensed_Log_Transforms data.

The reduced set loader globs every matching CSV in Reduced_Set_Log_Transforms
across all baselines and target years, so it picks up the full multi-year
ensemble automatically.  PAIRED_ONLY (default True) restricts both hist and
histnat to the members that are complete in both run types, matching the
behaviour of reduced_set_risk_ratio.py.
"""

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import warnings
import os
import sys

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
from utils.cubefuncs import GetERA5ThresholdFromMonthly

warnings.filterwarnings("ignore", module=r"^seaborn(\.|$)")
warnings.filterwarnings("ignore", module=r"^iris(\.|$)")

############# Configuration #############

FOLDER = '/data/scratch/bob.potts/sowf/'
OUTPUT_FOLDER = FOLDER + 'test_output/Plots'
BASELINE_FOLDER = FOLDER + 'test_output/Baseline/'
UNCORRECTED_ENSEMBLE_FOLDER = FOLDER + 'test_output/Uncorrected_Attribution_Ensembles/'
# Subplot (e) — reduced-set multi-year bias-corrected CSVs
REDUCED_SET_FOLDER = FOLDER + 'test_output/Reduced_Set_Log_Transforms'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
ERA5_FWI_DIR = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI'

BASELINE_START_YEAR = 1980
BASELINE_END_YEAR = 2013
UNCORRECTED_DATA_YEAR = 2024
DATA_YEARS = [2024]
N_BASELINES = 15
N_MEMBERS = 105
YEARS = np.arange(BASELINE_START_YEAR, BASELINE_END_YEAR + 1)

# Set to True to restrict both hist and histnat to the members present in both,
# giving a strictly paired ensemble (matches reduced_set_risk_ratio.py default).
PAIRED_ONLY = True

# Directories to scan for complete ensemble members (used when PAIRED_ONLY=True)
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

REGION_CONFIGS = {
    'Korea': {
        'month_name': 'March',
        'percentile': 95,
        'shape_name': 'Southeast South Korea',
        'Month': 3,
        'event_year': 2025,
    },
    'Iberia': {
        'month_name': 'Aug',
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
        'Month': 8,
        'event_year': 2025,
    },
    'Scotland': {
        'month_name': 'June-July',
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
        'Month': (6, 7),
        'event_year': 2025,
    },
    'Chile': {
        'month_name': 'January-February',
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
        'Month': (1, 2),
        'event_year': 2026,
    },
    'Canada': {
        'month_name': 'July-August',
        'percentile': 95,
        'shape_name': 'Midwestern Canadian Shield forests',
        'Month': (7, 8),
        'event_year': 2025,
    },
}

############# Helper Functions (reduced set) #############


def _is_member_complete(member_path):
    """Check whether a member subdirectory contains all expected monthly files."""
    if not os.path.isdir(member_path):
        return False
    stamps = set()
    for fname in os.listdir(member_path):
        m = _DATE_RE.search(fname)
        if m:
            stamps.add(m.group(1))
    return len(stamps & _EXPECTED_DATES) == _N_EXPECTED


def get_paired_members(hist_dir, histnat_dir):
    """
    Scan historicalExt and historicalNatExt directories and return the set of
    member IDs that are complete in both.
    """
    paired = set()
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


def load_reduced_ensemble_data(country, percentile, run_type, folder,
                               allowed_members=None):
    """
    Load every available reduced-set ensemble value for a country / run_type.

    Globs all bias-corrected CSVs matching country, run_type and percentile,
    regardless of baseline member or target year, and flattens all member
    columns from each file.

    Parameters
    ----------
    allowed_members : set or None
        If provided, only columns whose name is in this set are included.

    Returns
    -------
    values : np.ndarray
        Flattened array of all available (non-NaN) bias-corrected FWI values.
    n_files : int
        Number of CSV files read.
    members : set
        Set of unique member column names used.
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
        values = values[~np.isnan(values)]
        return values, len(files), members

    return np.array([]), 0, members


############# Helper Functions (unchanged from original) #############

def _baseline_years_str():
    """Return the baseline year range string used in filenames."""
    return f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"


def load_era5_baseline(country, percentile):
    """Load ERA5 baseline FWI from CSV (Date,FWI columns)."""
    filepath = f"{BASELINE_FOLDER}ERA5_FWI_{_baseline_years_str()}_{country}_{percentile}%.csv"
    df = pd.read_csv(filepath)
    return df['FWI'].values


def load_hadgem3_baseline(country, percentile, n_members=15):
    """Load all HadGEM3 baseline members from CSV (Date,FWI columns)."""
    all_data = []
    for member in range(1, n_members + 1):
        filepath = f"{BASELINE_FOLDER}HadGEM3_FWI_{_baseline_years_str()}_{country}_{member}_{percentile}%.csv"
        try:
            df = pd.read_csv(filepath)
            all_data.append(df['FWI'].values)
        except FileNotFoundError:
            print(f"Warning: Missing file for member {member}")
            continue
    return all_data


def load_uncorrected_ensemble(country, percentile, run_type):
    """Load uncorrected attribution ensemble from CSVs across all data years."""
    all_data = []
    for data_year in DATA_YEARS:
        filepath = f"{UNCORRECTED_ENSEMBLE_FOLDER}{country}_NoCorrection_{run_type}_{percentile}percent_DataYear_{data_year}.csv"
        try:
            df = pd.read_csv(filepath)
            col_names = [col for col in df.columns if col != 'Year']
            all_data.append(df[col_names].values.ravel())
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            continue

    if all_data:
        return np.concatenate(all_data)
    return np.array([])


def compute_bias_correction(era5_baseline, hadgem3_members):
    """
    Compute bias correction for all baseline members using pre-loaded data.

    Returns
    -------
    fwi_obs, fwi_sim_all, fwi_detrended_all, years
    """
    years = YEARS
    t = years - 2024
    X = sm.add_constant(t)

    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()
        fwi0, delta = results.params
        return fwi0, delta, np.std(fwi - delta * t)

    obs_arr = np.where(np.isnan(era5_baseline), 1e-12, era5_baseline)
    fwi_obs_log = np.log(np.exp(obs_arr) - 1)
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs_log)

    fwi_sim_all = []
    fwi_detrended_all = []

    for sim_arr in hadgem3_members:
        sim_clean = np.where(np.isnan(sim_arr), 1e-12, sim_arr)
        fwi_sim_log = np.log(np.exp(sim_clean) - 1)
        fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim_log)
        fwi_detrended_log = fwi0_obs + (fwi_sim_log - delta_sim * t - fwi0_sim)
        fwi_sim_inv = np.log(np.exp(fwi_sim_log) + 1)
        fwi_detrended_inv = np.log(np.exp(fwi_detrended_log) + 1)
        fwi_sim_all.append(fwi_sim_inv)
        fwi_detrended_all.append(fwi_detrended_inv)

    fwi_obs_inv = np.log(np.exp(fwi_obs_log) + 1)
    return fwi_obs_inv, fwi_sim_all, fwi_detrended_all, years


############# Plotting Functions #############

def plot_subplot_a(ax, hadgem3_arr, era5_arr, era5_2025, month_name, event_year):
    """Plot (a): Historical PDF uncorrected."""
    if len(hadgem3_arr) > 0:
        sns.histplot(np.ravel(hadgem3_arr), kde=True, color='#7A44FF', label='HadGEM3',
                     alpha=0.5, ax=ax, stat='density')
    if len(era5_arr) > 0:
        sns.histplot(era5_arr, kde=True, color='#E98400', label='ERA5',
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('')
    ax.set_title(f'a) {month_name} {BASELINE_START_YEAR}-{BASELINE_END_YEAR} (Uncorrected)')
    ax.legend(loc='best')


def plot_subplot_b(ax, fwi_detrended_all, era5_arr, era5_2025, month_name, event_year):
    """Plot (b): Historical PDF bias-corrected."""
    if len(fwi_detrended_all) > 0:
        fwi_detrended_ensemble = np.ravel(fwi_detrended_all)
        sns.histplot(fwi_detrended_ensemble, kde=True, color='#7A44FF', label='HadGEM3 (Corrected)',
                     alpha=0.5, ax=ax, stat='density')
    if len(era5_arr) > 0:
        sns.histplot(era5_arr, kde=True, color='#E98400', label='ERA5',
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('')
    ax.set_title(f'b) {month_name} {BASELINE_START_YEAR}-{BASELINE_END_YEAR} (Corrected)')
    ax.legend(loc='best')


def plot_subplot_c(ax, years, fwi_obs, fwi_sim_all, fwi_detrended_all, month_name):
    """Plot (c): Timeseries of bias correction."""
    if len(fwi_obs) > 0:
        ax.plot(years, fwi_obs, label='ERA5', color='blue', linewidth=1.5)

    if len(fwi_sim_all) > 0:
        fwi_sim_array = np.array(fwi_sim_all)
        fwi_sim_mean = np.mean(fwi_sim_array, axis=0)
        fwi_sim_std = np.std(fwi_sim_array, axis=0)
        ax.plot(years, fwi_sim_mean, label='HadGEM3 (mean)', color='red', linewidth=1.5)
        ax.fill_between(years, fwi_sim_mean - fwi_sim_std,
                        fwi_sim_mean + fwi_sim_std, color='red', alpha=0.2)

    if len(fwi_detrended_all) > 0:
        fwi_detrended_array = np.array(fwi_detrended_all)
        fwi_detrended_mean = np.mean(fwi_detrended_array, axis=0)
        fwi_detrended_std = np.std(fwi_detrended_array, axis=0)
        ax.plot(years, fwi_detrended_mean, label='Detrended & Shifted (mean)',
                color='purple', linewidth=1.5)
        ax.fill_between(years, fwi_detrended_mean - fwi_detrended_std,
                        fwi_detrended_mean + fwi_detrended_std, color='purple', alpha=0.2)

    ax.set_xlim(BASELINE_START_YEAR, BASELINE_END_YEAR)
    ax.set_xlabel('Year')
    ax.set_ylabel('FWI')
    ax.set_title(f'c) {month_name} Time Series of FWI and Detrended & Shifted FWI')
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)


def plot_subplot_d(ax, all_data, nat_data, era5_2025, month_name, event_year):
    """Plot (d): Uncorrected ALL vs NAT."""
    if len(all_data) > 0:
        sns.histplot(all_data, kde=True, color='#C7403D', label='Factual (Current Climate)',
                     alpha=0.5, ax=ax, stat='density')
    if len(nat_data) > 0:
        sns.histplot(nat_data, kde=True, color='#008787', label='Counterfactual (Natural Only Climate)',
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('Fire Weather Index')
    ax.set_title(f'd) {month_name} {event_year} (Uncorrected)')
    ax.legend(fontsize='small')


def plot_subplot_e(ax, all_data, nat_data, era5_2025, month_name, event_year, mode_label):
    """Plot (e): Reduced-set corrected ALL vs NAT."""
    if len(all_data) > 0:
        sns.histplot(all_data, kde=True, color='#C7403D', label='Factual (Current Climate)',
                     alpha=0.5, ax=ax, stat='density')
    if len(nat_data) > 0:
        sns.histplot(nat_data, kde=True, color='#008787', label='Counterfactual (Natural Only Climate)',
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('Fire Weather Index')
    ax.set_title(f'e) {month_name} {event_year} (Corrected, {mode_label})')
    ax.legend(fontsize='small')


############# Main Plotting Function #############

def create_supplement2_plot(country, config, paired_members, save=True):
    """
    Create the 5-panel Supplement 2 plot for a given country.
    Subplot (e) uses the multiyear reduced-set ensemble.
    """
    month_name = config['month_name']
    percentile = config['percentile']

    print(f"\n{'='*50}")
    print(f"Creating plot for {country}")
    print('='*50)

    # Load uncorrected baseline data for subplot (a)
    print("Loading ERA5 baseline...")
    try:
        era5_arr = load_era5_baseline(country, percentile)
        print(f"  Loaded {len(era5_arr)} ERA5 baseline values")
    except FileNotFoundError:
        print(f"  Warning: ERA5 baseline file not found")
        era5_arr = np.array([])

    print("Loading HadGEM3 baseline (all members)...")
    hadgem3_members = load_hadgem3_baseline(country, percentile, N_BASELINES)
    hadgem3_arr = np.concatenate(hadgem3_members) if hadgem3_members else np.array([])
    print(f"  Loaded {len(hadgem3_arr)} HadGEM3 baseline values")

    # ERA5 event-year threshold
    event_year = config['event_year']
    print(f"Computing ERA5 {event_year} threshold...")
    try:
        era5_2025 = GetERA5ThresholdFromMonthly(
            ERA5_FWI_DIR,
            SHP_FILE,
            config['shape_name'],
            config['Month'],
            event_year,
            config['percentile'])
        print(f"  Threshold Value: {era5_2025:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute ERA5 threshold: {e}")
        era5_2025 = None

    # Bias correction for subplots (b) and (c)
    print("Computing bias correction for all baseline members...")
    try:
        fwi_obs, fwi_sim_all, fwi_detrended_all, years = compute_bias_correction(
            era5_arr, hadgem3_members)
        print(f"  Processed {len(fwi_sim_all)} baseline members")
    except Exception as e:
        print(f"  Warning: Could not compute bias correction: {e}")
        import traceback
        traceback.print_exc()
        fwi_obs, fwi_sim_all, fwi_detrended_all, years = np.array([]), [], [], YEARS

    # Subplot (d): uncorrected ensemble
    print("Loading uncorrected attribution ensemble data...")
    all_uncorrected = load_uncorrected_ensemble(country, percentile, 'hist')
    nat_uncorrected = load_uncorrected_ensemble(country, percentile, 'histnat')
    print(f"  Loaded {len(all_uncorrected)} ALL uncorrected, {len(nat_uncorrected)} NAT uncorrected")

    # Subplot (e): reduced-set multiyear corrected ensemble
    mode_label = 'paired' if PAIRED_ONLY else 'reduced set'
    print(f"Loading reduced-set corrected ensemble ({mode_label})...")
    all_corrected, n_hist, hist_members = load_reduced_ensemble_data(
        country, percentile, 'hist', REDUCED_SET_FOLDER, allowed_members=paired_members)
    nat_corrected, n_nat, nat_members = load_reduced_ensemble_data(
        country, percentile, 'histnat', REDUCED_SET_FOLDER, allowed_members=paired_members)
    print(f"  hist:    {len(all_corrected)} values from {n_hist} files, "
          f"{len(hist_members)} unique members")
    print(f"  histnat: {len(nat_corrected)} values from {n_nat} files, "
          f"{len(nat_members)} unique members")

    # Build figure
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    plot_subplot_a(ax_a, hadgem3_arr, era5_arr, era5_2025, month_name, event_year)
    plot_subplot_b(ax_b, fwi_detrended_all, fwi_obs, era5_2025, month_name, event_year)
    plot_subplot_c(ax_c, years, fwi_obs, fwi_sim_all, fwi_detrended_all, month_name)
    plot_subplot_d(ax_d, all_uncorrected, nat_uncorrected, era5_2025, month_name, event_year)
    plot_subplot_e(ax_e, all_corrected, nat_corrected, era5_2025, month_name, event_year, mode_label)

    plt.suptitle(f'{country} {percentile}th percentile FWI', y=0.995, fontsize=14)
    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = os.path.join(
            OUTPUT_FOLDER, f'Supplement2_5Panel_ReducedSet_{country}.svg')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='svg')
        print(f"Saved: {output_file}")

    #plt.show()

    return fig


def main():
    """Generate plots for all configured regions."""
    # Resolve paired member filter once up front
    if PAIRED_ONLY:
        paired_members = get_paired_members(HIST_DIR, HISTNAT_DIR)
        print(f"PAIRED_ONLY mode: {len(paired_members)} members complete in both run types")
    else:
        paired_members = None

    for country, config in REGION_CONFIGS.items():
        try:
            create_supplement2_plot(country, config, paired_members)
        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue


if __name__ == "__main__":
    main()
