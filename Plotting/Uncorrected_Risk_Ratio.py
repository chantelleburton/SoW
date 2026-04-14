"""
Explore Risk Ratios for State of Fires 2025-26 Attribution Study.

This script calculates and visualises Risk Ratios comparing ALL (anthropogenic)
and NAT (natural-only) climate scenarios for multiple regions using UNCORRECTED data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Work Sans'
import seaborn as sns
import warnings
import sys
import pandas as pd

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')

from utils.cubefuncs import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


############# Configuration #############

FOLDER = '/data/scratch/chantelle.burton/SoW2526/'
UNCORRECTED_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Uncorrected_Attribution_Ensembles/'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
PLOT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Plots'
EXPORT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Exports'
BOOTSTRAP_SIZE = 10000
N_MEMBERS = 105
N_BASELINES = 15
DATA_YEARS = 2024


REGION_CONFIGS = {
    'Iberia': {
        'Month': 8,
        'month_name': 'Aug',
        'event_year':2025,
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Korea': {
        'Month': 3,
        'month_name': 'March',
        'event_year':2025,
        'percentile': 95,
        'shape_name': 'Southeast South Korea',
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Scotland': {
        'Month': (6,7),
        'month_name': 'June-July',
        'event_year':2026,
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    },
    'Chile': {
        'Month': (1, 2),
        'month_name': 'January-February',
        'event_year':2026,
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-11-01-2026-02-28_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
            },
    'Canada': {
        'Month': (7, 8),
        'month_name': 'July-August',
        'percentile': 95,
        'event_year':2025,
        'shape_name': 'Midwestern Canadian Shield forests',
        'era5_file': FOLDER + 'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc'
    }
 }

DISPLAY_NAMES = {
    'Northwest Iberia': 'NW Iberia',
    'Southeast South Korea': 'SE S. Korea',
    'Scottish Highlands': 'Scottish Highlands',
    'Chilean Temperate Forests and Matorral': 'Chile Forests & Matorral',
    'Midwestern Canadian Shield forests': 'Canadian Shield Forests'
}


############# Helper Functions #############


# New loader for UNCORRECTED ALL and NAT ensemble data from new CSV outputs
def load_uncorrected_ensemble_data_csv(country, percentile, run_type, uncorrected_folder, data_year):
    """
    Load UNCORRECTED ALL or NAT ensemble data from new CSV files.
    Returns: flattened numpy array of all Ens/Real columns
    """
    filename = f"{country}_NoCorrection_{run_type}_{percentile}percent_DataYear_{data_year}.csv"
    filepath = os.path.join(uncorrected_folder, filename)
    print(f"  Loading from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        col_names = [col for col in df.columns if col != 'Year']
        return df[col_names].values.flatten()
    except FileNotFoundError:
        print(f"  Warning: File not found: {filepath}")
        return np.array([])


def calculate_risk_ratio_with_ci(all_data, nat_data, threshold, bootstrap_size=10000):
    """
    Calculate Risk Ratio with confidence intervals via bootstrapping.
    
    Parameters
    ----------
    all_data : np.ndarray
        ALL forcing scenario data
    nat_data : np.ndarray
        NAT forcing scenario data
    threshold : float
        Threshold for exceedance counting
    bootstrap_size : int
        Number of bootstrap samples
    
    Returns
    -------
    dict
        Dictionary with 'median', 'ci_5', 'ci_95' keys
    """
    rr_replicates = draw_bs_replicates(
        all_data, nat_data, threshold, RiskRatio, bootstrap_size
    )
    
    return {
        'median': np.median(rr_replicates),
        'ci_5': np.percentile(rr_replicates, 5),
        'ci_95': np.percentile(rr_replicates, 95),
        'replicates': rr_replicates
    }


############# Main Analysis #############

def main():
    """Run the Risk Ratio analysis for all configured regions using UNCORRECTED data."""
    
    n_regions = len(REGION_CONFIGS)
    # 2 rows: 3 on top, 2 on bottom, 3rd space in bottom row for summary
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    results = {}
    plot_idxs = [0, 1, 2, 3, 4]  # 5 regions, 5 axes for plots, 6th for summary
    
    for idx, (country, config) in enumerate(REGION_CONFIGS.items()):
        print(f"\n{'='*50}")
        print(f"Processing {country} (UNCORRECTED)")
        print('='*50)
        
        # Load ERA5 threshold
        print("Loading ERA5 threshold...")
        threshold = GetERA5Threshold(
            config['era5_file'], SHP_FILE, config['shape_name'], config['Month'], config['percentile']
        )
        print(f"ERA5 threshold: {threshold:.2f}")
        
        # Load UNCORRECTED ensemble data from new CSVs
        print("Loading UNCORRECTED ensemble data from CSVs...")
        all_data = load_uncorrected_ensemble_data_csv(
            country, config['percentile'], 'hist', UNCORRECTED_FOLDER, DATA_YEARS
        )

        nat_data = load_uncorrected_ensemble_data_csv(
            country, config['percentile'], 'histnat', UNCORRECTED_FOLDER, DATA_YEARS
        )
        print(f"Loaded {len(all_data)} ALL values, {len(nat_data)} NAT values")
        if len(all_data) == 0 or len(nat_data) == 0:
            print(f"Skipping {country} due to missing data")
            continue

        # Calculate Risk Ratio with bootstrapped confidence intervals
        print("Calculating Risk Ratio...")
        rr_results = calculate_risk_ratio_with_ci(
            all_data, nat_data, threshold, BOOTSTRAP_SIZE
        )
        
        results[country] = {
            'threshold': threshold,
            'rr': rr_results,
            'all_data': all_data,
            'nat_data': nat_data
        }
        
        print(f"Risk Ratio: {rr_results['median']:.2f} "
              f"[{rr_results['ci_5']:.2f} - {rr_results['ci_95']:.2f}]")
        
        # Save bootstrap replicates
        pd.DataFrame({'rr_replicates': rr_results['replicates']}).to_csv(
            f'{EXPORT_FOLDER}/{country}_Uncorrected_Risk_Ratio_Bootstrap_Replicates.csv', index=False
        )
        
        # Plot
        ax = axes[plot_idxs[idx]]
        sns.histplot(all_data, kde=True, color='#C7403D', label='Factual (Current Climate)', 
                    alpha=0.5, ax=ax, stat='density')
        sns.histplot(nat_data, kde=True, color='#008787', label='Counterfactual (Natural Only Climate)', 
                    alpha=0.5, ax=ax, stat='density')
        ax.axvline(x=threshold, color='black', linewidth=2.5, label=f'ERA5 {config["month_name"]} {config["event_year"]}')

        # Use shorter display name for plotting
        display_name = DISPLAY_NAMES.get(config['shape_name'], config['shape_name'])
        title = f"{display_name}\nFWI {config['month_name']} (Uncorrected)"
        ax.set_title(title)
        ax.set_xlabel('Fire Weather Index')
        
        if idx % ncols == 0:
            ax.set_ylabel('Density')
        if idx == n_regions - 1:
            ax.legend()
    
    # Add summary of risk ratios in the last subplot (bottom right)
    summary_ax = axes[-1]
    summary_ax.axis('off')
    summary_lines = ["SUMMARY OF RESULTS (UNCORRECTED)", ""]
    for country, res in results.items():
        rr = res['rr']
        summary_lines.append(f"{country}: RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}]")
    summary_text = "\n".join(summary_lines)
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, wrap=True, family='monospace')

    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/Uncorrected_Risk_Ratio.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS (UNCORRECTED)")
    print("="*60)
    for country, res in results.items():
        rr = res['rr']
        print(f"{country}: RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}]")

    return results


if __name__ == "__main__":
    results = main()