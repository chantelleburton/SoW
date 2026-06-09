"""
Plotting-only version of Supplement2.py
Loads pre-computed CSV files and creates 4-panel PDF/timeseries plots.
Generalised to all 15 baseline members.
"""

import numpy as np
import matplotlib.pyplot as plt
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
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
ERA5_FWI_DIR = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI'

BASELINE_START_YEAR = 1980
BASELINE_END_YEAR = 2013
DATA_YEAR = 2024
N_BASELINES = 15
N_MEMBERS = 105
YEARS = np.arange(BASELINE_START_YEAR, BASELINE_END_YEAR + 1)

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

############# Helper Functions #############

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
    """Load uncorrected attribution ensemble from CSV (Year + Ens columns)."""
    filepath = f"{UNCORRECTED_ENSEMBLE_FOLDER}{country}_NoCorrection_{run_type}_{percentile}percent_DataYear_{DATA_YEAR}.csv"
    try:
        df = pd.read_csv(filepath)
        col_names = [col for col in df.columns if col != 'Year']
        return df[col_names].values.ravel()
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return np.array([])


def compute_bias_correction(era5_baseline, hadgem3_members):
    """
    Compute bias correction for all baseline members using pre-loaded data.
    
    Args:
        era5_baseline: 1D array of ERA5 FWI baseline values
        hadgem3_members: list of 1D arrays, one per HadGEM3 baseline member
    
    Returns:
        fwi_obs: observed ERA5 FWI values (inverse log transformed)
        fwi_sim_all: list of raw HadGEM3 FWI values per member (inverse log transformed)
        fwi_detrended_all: list of detrended & shifted FWI values per member (inverse log transformed)
        years: array of years
    """
    years = YEARS
    t = years - 2024  # shift years to be relative to 2024
    X = sm.add_constant(t)  # add a constant term for intercept
    
    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()
        fwi0, delta = results.params
        return fwi0, delta, np.std(fwi - delta * t)
    
    # Replace NaNs without modifying original arrays
    obs_arr = np.where(np.isnan(era5_baseline), 1e-12, era5_baseline)
    
    # Log transform observations
    fwi_obs_log = np.log(np.exp(obs_arr) - 1)
    
    # Get regression parameters for observations
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs_log)
    
    fwi_sim_all = []
    fwi_detrended_all = []
    
    for sim_arr in hadgem3_members:
        sim_clean = np.where(np.isnan(sim_arr), 1e-12, sim_arr)
        
        # Log transform simulation
        fwi_sim_log = np.log(np.exp(sim_clean) - 1)
        
        # Get regression parameters for simulation
        fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim_log)
        
        # Detrend and shift to observations
        fwi_detrended_log = fwi0_obs + (fwi_sim_log - delta_sim * t - fwi0_sim)
        
        # Inverse log transform
        fwi_sim_inv = np.log(np.exp(fwi_sim_log) + 1)
        fwi_detrended_inv = np.log(np.exp(fwi_detrended_log) + 1)
        
        fwi_sim_all.append(fwi_sim_inv)
        fwi_detrended_all.append(fwi_detrended_inv)
    
    # Inverse log transform observations
    fwi_obs_inv = np.log(np.exp(fwi_obs_log) + 1)
    
    return fwi_obs_inv, fwi_sim_all, fwi_detrended_all, years


############# Plotting Functions #############

def plot_subplot_a(ax, hadgem3_arr, era5_arr, era5_2025, month_name, event_year):
    """Plot (a): Historical PDF uncorrected."""
    if len(hadgem3_arr) > 0:
        sns.histplot(np.ravel(hadgem3_arr), kde=True, color='yellow', label='HadGEM3', 
                     alpha=0.5, ax=ax, stat='density')
    if len(era5_arr) > 0:
        sns.histplot(era5_arr, kde=True, color='grey', label='ERA5', 
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('')
    ax.set_title(f'a) {month_name} {BASELINE_START_YEAR}-{BASELINE_END_YEAR} (Uncorrected)')
    ax.legend(loc='best')


def plot_subplot_b(ax, fwi_detrended_all, era5_arr, era5_2025, month_name, event_year):
    """Plot (b): Historical PDF bias-corrected."""
    if len(fwi_detrended_all) > 0:
        # Flatten all detrended members into one array
        fwi_detrended_ensemble = np.ravel(fwi_detrended_all)
        sns.histplot(fwi_detrended_ensemble, kde=True, color='yellow', label='HadGEM3 (Corrected)', 
                     alpha=0.5, ax=ax, stat='density')
    if len(era5_arr) > 0:
        sns.histplot(era5_arr, kde=True, color='grey', label='ERA5', 
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('')
    ax.set_title(f'b) {month_name} {BASELINE_START_YEAR}-{BASELINE_END_YEAR} (Corrected)')
    ax.legend(loc='best')


def plot_subplot_c(ax, years, fwi_obs, fwi_sim_all, fwi_detrended_all, month_name):
    """
    Plot (c): Timeseries of bias correction.
    
    Shows ERA5 observations, HadGEM3 raw (mean of members), 
    and HadGEM3 detrended & shifted (mean of members).
    """
    # Plot ERA5 observations
    if len(fwi_obs) > 0:
        ax.plot(years, fwi_obs, label='ERA5', color='blue', linewidth=1.5)
    
    # Plot HadGEM3 raw mean (with shading for spread)
    if len(fwi_sim_all) > 0:
        fwi_sim_array = np.array(fwi_sim_all)
        fwi_sim_mean = np.mean(fwi_sim_array, axis=0)
        fwi_sim_std = np.std(fwi_sim_array, axis=0)
        ax.plot(years, fwi_sim_mean, label='HadGEM3 (mean)', color='red', linewidth=1.5)
        ax.fill_between(years, fwi_sim_mean - fwi_sim_std, 
                        fwi_sim_mean + fwi_sim_std, color='red', alpha=0.2)
    
    # Plot HadGEM3 detrended & shifted mean (with shading for spread)
    if len(fwi_detrended_all) > 0:
        fwi_detrended_array = np.array(fwi_detrended_all)
        fwi_detrended_mean = np.mean(fwi_detrended_array, axis=0)
        fwi_detrended_std = np.std(fwi_detrended_array, axis=0)
        ax.plot(years, fwi_detrended_mean, label='Detrended & Shifted (mean)', 
                color='purple', linewidth=1.5)
        ax.fill_between(years, fwi_detrended_mean - fwi_detrended_std, 
                        fwi_detrended_mean + fwi_detrended_std, color='purple', alpha=0.2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('FWI')
    ax.set_title(f'c) {month_name} Time Series of FWI and Detrended & Shifted FWI')
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)


def plot_subplot_d(ax, all_data, nat_data, era5_2025, month_name, event_year):
    """Plot (d): Uncorrected ALL vs NAT."""
    if len(all_data) > 0:
        sns.histplot(all_data, kde=True, color='orange', label='ALL', 
                     alpha=0.5, ax=ax, stat='density')
    if len(nat_data) > 0:
        sns.histplot(nat_data, kde=True, color='blue', label='NAT', 
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} {event_year}')
    ax.set_xlabel('FWI')
    ax.set_title(f'd) {month_name} {event_year} Uncorrected')
    ax.legend()


############# Main Plotting Function #############

def create_supplement2_plot(country, config, save=True):
    """
    Create the 4-panel Supplement 2 plot for a given country.
    Computes bias correction on the fly for subplots (b) and (c).
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
    
    # Compute ERA5 event-year threshold from monthly files
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
    
    # Compute bias correction for subplots (b) and (c) using already-loaded data
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
    
    # Load uncorrected attribution ensemble data for subplot (d)
    print("Loading uncorrected attribution ensemble data...")
    all_uncorrected = load_uncorrected_ensemble(country, percentile, 'hist')
    nat_uncorrected = load_uncorrected_ensemble(country, percentile, 'histnat')
    print(f"  Loaded {len(all_uncorrected)} ALL uncorrected, {len(nat_uncorrected)} NAT uncorrected")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot (a): Historical PDF uncorrected
    plot_subplot_a(axes[0, 0], hadgem3_arr, era5_arr, era5_2025, month_name, event_year)
    
    # Subplot (b): Historical PDF bias-corrected
    plot_subplot_b(axes[0, 1], fwi_detrended_all, fwi_obs, era5_2025, month_name, event_year)
    
    # Subplot (c): Timeseries of bias correction
    plot_subplot_c(axes[1, 0], years, fwi_obs, fwi_sim_all, fwi_detrended_all, month_name)
    
    # Subplot (d): Uncorrected ALL vs NAT
    plot_subplot_d(axes[1, 1], all_uncorrected, nat_uncorrected, era5_2025, month_name, event_year)
    
    plt.suptitle(f'{country} {percentile}th percentile FWI', y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = f"{OUTPUT_FOLDER}{country}_NewSupplement2024.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    #plt.show()
    
    return fig


def main():
    """Generate plots for all configured regions."""
    for country, config in REGION_CONFIGS.items():
        try:
            create_supplement2_plot(country, config)
        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue


if __name__ == "__main__":
    main()