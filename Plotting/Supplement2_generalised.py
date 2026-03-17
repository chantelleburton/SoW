"""
Plotting-only version of Supplement2.py
Loads pre-computed .dat files and creates 4-panel PDF/timeseries plots.
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
import iris

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
from utils.cubefuncs import GetERA5Threshold
from utils.constrain_cubes_standard import contrain_to_sow_shapefile

warnings.filterwarnings("ignore", module=r"^seaborn(\.|$)")
warnings.filterwarnings("ignore", module=r"^iris(\.|$)")

############# Configuration #############

FOLDER = '/data/scratch/bob.potts/sowf/'
OUTPUT_FOLDER = FOLDER + 'test_output/Plots/'
BASELINE_FOLDER = FOLDER + 'test_output/Baseline/'
HISTORICAL_FOLDER = FOLDER + 'test_output/Historical_Ensembles/'
LOG_TRANSFORMS_FOLDER = FOLDER + 'test_output/Log_Transforms/'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'

N_BASELINES = 15
N_MEMBERS = 105
YEARS = np.arange(1960, 2014)

REGION_CONFIGS = {
    'Korea': {
        'month_name': 'March',
        'percentile': 95,
        'shape_name': 'Southeast South Korea',
        'Month': 3,
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Iberia': {
        'month_name': 'Aug',
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
        'Month': 8,
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Scotland': {
        'month_name': 'July',
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
        'Month': 7,
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Chile': {
        'month_name': 'January-February',
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
        'Month': (1, 2),
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-11-01-2026-02-28_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Canada': {
        'month_name': 'July-August',
        'percentile': 95,
        'shape_name': 'Midwestern Canadian Shield forests',
        'Month': (7, 8),
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    }
}

############# Helper Functions #############

def load_dat_file(filepath):
    """Load a .dat file and return as numpy array."""
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                data.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(data)


def load_era5_baseline(country, percentile):
    """Load ERA5 baseline data."""
    filepath = f"{BASELINE_FOLDER}ERA5_FWI_1960-2013_{country}_{percentile}%.dat"
    return load_dat_file(filepath)


def load_hadgem3_baseline(country, percentile, n_members=15):
    """Load all HadGEM3 baseline members."""
    all_data = []
    for member in range(1, n_members + 1):
        filepath = f"{BASELINE_FOLDER}HadGEM3_FWI_1960-2013_{country}_{member}_{percentile}%.dat"
        try:
            data = load_dat_file(filepath)
            all_data.append(data)
        except FileNotFoundError:
            print(f"Warning: Missing file for member {member}")
            continue
    return all_data


def load_historical_ensemble(country, percentile, run_type):
    """Load historical ensemble data (hist or histnat)."""
    filepath = f"{HISTORICAL_FOLDER}{country}_Uncorrected_{run_type}{percentile}%.dat"
    data = []
    try:
        with open(filepath) as f:
            for line in f.readlines():
                try:
                    data.append(float(line.rstrip(',\n')))
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    return np.array(data)


def load_bias_corrected_data(country, percentile, n_baselines, n_members, run_type):
    """
    Load pre-computed bias-corrected log-transformed data from .dat files.
    
    File naming scheme: {country}_baseline{baseline}_ens{member}_{run_type}{percentile}percent_LogTransform.dat
    """
    all_data = []
    files_found = 0
    
    for baseline in range(1, n_baselines + 1):
        for member in range(1, n_members + 1):
            filepath = f"{LOG_TRANSFORMS_FOLDER}{country}_baseline{baseline}_ens{member}_{run_type}{percentile}percent_LogTransform.dat"
            
            try:
                with open(filepath) as f:
                    for line in f:
                        numbers = line.strip().split(',')
                        all_data.extend([float(num) for num in numbers if num])
                files_found += 1
            except FileNotFoundError:
                continue
    
    print(f"  Found {files_found} files for {run_type}")
    return np.array(all_data)


def compute_bias_correction(country, percentile, n_baselines):
    """
    Compute bias correction for all baseline members.
    
    Returns:
        fwi_obs: observed ERA5 FWI values (inverse log transformed)
        fwi_sim_all: list of raw HadGEM3 FWI values per member (inverse log transformed)
        fwi_detrended_all: list of detrended & shifted FWI values per member (inverse log transformed)
        years: array of years
    """
    years = YEARS
    t = years - 2025  # shift years to be relative to 2025
    X = sm.add_constant(t)  # add a constant term for intercept
    
    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()
        fwi0, delta = results.params
        return fwi0, delta, np.std(fwi - delta * t)
    
    # Load ERA5 observations
    era5_filepath = f"{BASELINE_FOLDER}ERA5_FWI_1960-2013_{country}_{percentile}%.dat"
    df_obs = pd.read_csv(era5_filepath, header=None)
    df_obs[np.isnan(df_obs)] = 0.000000000001
    
    # Log transform observations
    df_obs_log = np.log(np.exp(df_obs) - 1)
    fwi_obs_log = df_obs_log.values[:, 0]
    
    # Get regression parameters for observations
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs_log)
    
    fwi_sim_all = []
    fwi_detrended_all = []
    
    for member in range(1, n_baselines + 1):
        hadgem3_filepath = f"{BASELINE_FOLDER}HadGEM3_FWI_1960-2013_{country}_{member}_{percentile}%.dat"
        
        try:
            df_sim = pd.read_csv(hadgem3_filepath, header=None)
            df_sim[np.isnan(df_sim)] = 0.000000000001
            
            # Log transform simulation
            df_sim_log = np.log(np.exp(df_sim) - 1)
            fwi_sim_log = df_sim_log.values[:, 0]
            
            # Get regression parameters for simulation
            fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim_log)
            
            # Detrend and shift to observations
            fwi_detrended_log = fwi0_obs + (fwi_sim_log - delta_sim * t - fwi0_sim)
            
            # Inverse log transform
            fwi_sim_inv = np.log(np.exp(fwi_sim_log) + 1)
            fwi_detrended_inv = np.log(np.exp(fwi_detrended_log) + 1)
            
            fwi_sim_all.append(fwi_sim_inv)
            fwi_detrended_all.append(fwi_detrended_inv)
            
        except FileNotFoundError:
            print(f"  Warning: Missing HadGEM3 baseline file for member {member}")
            continue
    
    # Inverse log transform observations
    fwi_obs_inv = np.log(np.exp(fwi_obs_log) + 1)
    
    return fwi_obs_inv, fwi_sim_all, fwi_detrended_all, years


############# Plotting Functions #############

def plot_subplot_a(ax, hadgem3_arr, era5_arr, era5_2025, month_name):
    """Plot (a): Historical PDF uncorrected."""
    if len(hadgem3_arr) > 0:
        sns.histplot(np.ravel(hadgem3_arr), kde=True, color='yellow', label='HadGEM3', 
                     alpha=0.5, ax=ax, stat='density')
    if len(era5_arr) > 0:
        sns.histplot(era5_arr, kde=True, color='grey', label='ERA5', 
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} 2025')
    ax.set_xlabel('')
    ax.set_title(f'a) {month_name} 1960-2013 (Uncorrected)')
    ax.legend(loc='best')


def plot_subplot_b(ax, fwi_detrended_all, era5_arr, era5_2025, month_name):
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
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} 2025')
    ax.set_xlabel('')
    ax.set_title(f'b) {month_name} 1960-2013 (Corrected)')
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


def plot_subplot_d(ax, all_data, nat_data, era5_2025, month_name):
    """Plot (d): Uncorrected 2025 ALL vs NAT."""
    if len(all_data) > 0:
        sns.histplot(all_data, kde=True, color='orange', label='ALL', 
                     alpha=0.5, ax=ax, stat='density')
    if len(nat_data) > 0:
        sns.histplot(nat_data, kde=True, color='blue', label='NAT', 
                     alpha=0.5, ax=ax, stat='density')
    if era5_2025 is not None:
        ax.axvline(x=era5_2025, color='black', linewidth=2.5, label=f'ERA5 {month_name} 2025')
    ax.set_xlabel('FWI')
    ax.set_title(f'd) {month_name} 2025 Uncorrected')
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
    
    # Compute ERA5 2025 threshold using GetERA5Threshold
    print("Computing ERA5 2025 threshold...")
    try:
        era5_2025 = GetERA5Threshold(
            config['era5_file'],
            SHP_FILE,
            config['shape_name'],
            config['Month'],
            config['percentile'])
        print(f"  Threshold Value: {era5_2025:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute ERA5 threshold: {e}")
        era5_2025 = None
    
    # Compute bias correction for subplots (b) and (c)
    print("Computing bias correction for all baseline members...")
    try:
        fwi_obs, fwi_sim_all, fwi_detrended_all, years = compute_bias_correction(
            country, percentile, N_BASELINES)
        print(f"  Processed {len(fwi_sim_all)} baseline members")
    except Exception as e:
        print(f"  Warning: Could not compute bias correction: {e}")
        import traceback
        traceback.print_exc()
        fwi_obs, fwi_sim_all, fwi_detrended_all, years = np.array([]), [], [], YEARS
    
    # Load uncorrected historical ensemble data for subplot (d)
    print("Loading uncorrected historical ensemble data...")
    all_uncorrected = load_historical_ensemble(country, percentile, 'hist')
    nat_uncorrected = load_historical_ensemble(country, percentile, 'histnat')
    print(f"  Loaded {len(all_uncorrected)} ALL uncorrected, {len(nat_uncorrected)} NAT uncorrected")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot (a): Historical PDF uncorrected
    plot_subplot_a(axes[0, 0], hadgem3_arr, era5_arr, era5_2025, month_name)
    
    # Subplot (b): Historical PDF bias-corrected
    plot_subplot_b(axes[0, 1], fwi_detrended_all, fwi_obs, era5_2025, month_name)
    
    # Subplot (c): Timeseries of bias correction
    plot_subplot_c(axes[1, 0], years, fwi_obs, fwi_sim_all, fwi_detrended_all, month_name)
    
    # Subplot (d): Uncorrected 2025 ALL vs NAT
    plot_subplot_d(axes[1, 1], all_uncorrected, nat_uncorrected, era5_2025, month_name)
    
    plt.suptitle(f'{country} {percentile}th percentile FWI', y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = f"{OUTPUT_FOLDER}{country}_Supplement2.png"
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
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()