"""
Explore Risk Ratios for State of Fires 2025-26 Attribution Study.

This script calculates and visualises Risk Ratios comparing ALL (anthropogenic)
and NAT (natural-only) climate scenarios for all regions, producing a single
3x2 figure with a summary panel.

"""

import numpy as np
import iris
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Work Sans'
import seaborn as sns
import warnings
import sys
import pandas as pd
import os
from datetime import date
sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')

from utils.cubefuncs import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


############# Configuration #############
LOG_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Condensed_Log_Transforms/'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
PLOT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Plots'
EXPORT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Exports'
ERA5_FWI_DIR = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI'
BOOTSTRAP_SIZE = 10000
N_BASELINES = 15
BASELINE_START_YEAR = 1980
BASELINE_END_YEAR = 2013
DATA_YEARS = 2024

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


def get_era5_monthly_files(era5_dir, year, months):
    """
    Build file paths for ERA5 monthly FWI files.
    """
    if isinstance(months, int):
        months = (months,)

    files = []
    for m in months:
        start = date(year, m, 1)
        if m == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, m + 1, 1)
        fname = f"FWI_ERA5_global_day_{start:%Y%m%d}-{end:%Y%m%d}.nc"
        files.append(os.path.join(era5_dir, fname))
    return files


def GetERA5ThresholdFromMonthly(era5_dir, shp_file, shape_name, months, event_year, percentile):
    """
    Compute ERA5 threshold from monthly files.

    Loads the correct monthly file(s), applies shapefile mask,
    then computes spatial percentile -> temporal percentile.
    """
    files = get_era5_monthly_files(era5_dir, event_year, months)

    cubes = iris.cube.CubeList()
    reference_time_units = None
    for f in files:
        print(f"  Loading {f}")
        cube = iris.load_cube(f, 'Canadian Fire Weather Index')
        if len(files) > 1:
            # Remove scalar time-related coordinates that differ between months
            for coord in list(cube.coords()):
                if coord.long_name in ('month', 'month_number', 'season', 'season_year', 'year'):
                    cube.remove_coord(coord)
            # Standardise time units to the first cube's reference time
            if cube.coords('time'):
                time_coord = cube.coord('time')
                if reference_time_units is None:
                    reference_time_units = time_coord.units
                else:
                    time_coord.convert_units(reference_time_units)
                time_coord.attributes = {}
                time_coord.var_name = None
                time_coord.long_name = None
                time_coord.standard_name = 'time'
        cubes.append(cube)

    if len(cubes) > 1:
        era5_cube = cubes.concatenate_cube()
    else:
        era5_cube = cubes[0]

    # Apply shapefile mask
    era5_cube = apply_shapefile_inclusive(shp_file,shape_name, era5_cube)

    # Spatial percentile then temporal percentile
    era5_cube = CountryPercentile(era5_cube, percentile)
    era5_cube = TimePercentile(era5_cube, percentile)

    return float(np.array(era5_cube.data))


def load_ensemble_data_csv(country, percentile, run_type, folder, target_year, n_baselines, baseline_start, baseline_end):
    """
    Load ALL or NAT ensemble data from the new CSV format for all baselines.
    Returns: flattened numpy array of all values (all years, all Ens/Real columns, all baselines)
    """
    all_data = []
    for baseline in range(1, n_baselines + 1):
        filename = f"{country}_baseline{baseline}_{run_type}{percentile}percent_LogTransform_Target_{target_year}_DataYear_{target_year}_BaselinePeriod_{baseline_start}_{baseline_end}.csv"
        filepath = os.path.join(folder, filename)
        try:
            df = pd.read_csv(filepath)
            col_names = [col for col in df.columns if col != 'Year']
            all_data.append(df[col_names].values.flatten())
        except FileNotFoundError:
            print(f"Warning: Missing file {filepath}")
            continue
    if all_data:
        return np.concatenate(all_data)
    else:
        return np.array([])


def calculate_risk_ratio_with_ci(all_data, nat_data, threshold, bootstrap_size=10000):
    """
    Calculate Risk Ratio with confidence intervals via bootstrapping.
    """
    rr_replicates = draw_bs_replicates(
        all_data, nat_data, threshold, RiskRatio, bootstrap_size
    )

    return {
        'median': np.median(rr_replicates),
        'ci_interquartile': np.percentile(rr_replicates, [25, 75]),
        'ci_5': np.percentile(rr_replicates, 5),
        'ci_95': np.percentile(rr_replicates, 95),
        'replicates': rr_replicates
    }


############# Main Analysis #############


def main():
    """Run the Risk Ratio analysis for all configured regions."""

    n_regions = len(REGION_CONFIGS)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    results = {}
    plot_idxs = [0, 1, 2, 3, 4]

    for idx, (country, config) in enumerate(REGION_CONFIGS.items()):
        print(f"\n{'='*50}")
        print(f"Processing {country}")
        print('='*50)

        # Load ERA5 threshold from monthly files
        print("Loading ERA5 threshold...")
        threshold = GetERA5ThresholdFromMonthly(
            ERA5_FWI_DIR, SHP_FILE, config['shape_name'],
            config['Month'], config['event_year'], config['percentile']
        )
        print(f"ERA5 threshold: {threshold:.2f}")

        # Load ensemble data from CSVs
        print("Loading ensemble data from CSVs...")
        all_data = load_ensemble_data_csv(
            country, config['percentile'], 'hist', LOG_FOLDER, DATA_YEARS,
            N_BASELINES, BASELINE_START_YEAR, BASELINE_END_YEAR
        )
        nat_data = load_ensemble_data_csv(
            country, config['percentile'], 'histnat', LOG_FOLDER, DATA_YEARS,
            N_BASELINES, BASELINE_START_YEAR, BASELINE_END_YEAR
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
            'nat_data': nat_data,
        }

        print(f"Risk Ratio: {rr_results['median']:.2f} "
              f"[{rr_results['ci_5']:.2f} - {rr_results['ci_95']:.2f}] "
              f"(Interquartile Range: {rr_results['ci_interquartile'][0]:.2f} - {rr_results['ci_interquartile'][1]:.2f})")

        # Export bootstrap replicates
        pd.DataFrame({'rr_replicates': rr_results['replicates']}).to_csv(
            f'{EXPORT_FOLDER}/{country}_Corrected_Risk_Ratio_Bootstrap_Replicates.csv', index=False
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
        ax.set_title(f"{display_name}\nFWI {config['month_name']}")
        ax.set_xlabel('Fire Weather Index')

        if idx % ncols == 0:
            ax.set_ylabel('Density')
        if idx == n_regions - 1:
            ax.legend()

    # Summary panel in the last subplot (bottom right)
    summary_ax = axes[-1]
    summary_ax.axis('off')
    summary_lines = ["SUMMARY OF RESULTS", ""]
    for country, res in results.items():
        rr = res['rr']
        summary_lines.append(f"{country}: RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}]")
    summary_text = "\n".join(summary_lines)
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=12, wrap=True, family='monospace')

    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/Corrected_Risk_Ratio.png', dpi=300, bbox_inches='tight')

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for country, res in results.items():
        rr = res['rr']
        print(f"{country}: RR = {rr['median']:.2f} [{rr['ci_5']:.2f} - {rr['ci_95']:.2f}]")

    # Export summary CSV
    summary_rows = []
    for country, res in results.items():
        rr = res['rr']
        summary_rows.append({
            'Country': country,
            'ERA5_Threshold': res['threshold'],
            'RR_Median': rr['median'],
            'RR_5th': rr['ci_5'],
            'RR_25th': rr['ci_interquartile'][0],
            'RR_50th': rr['median'],
            'RR_75th': rr['ci_interquartile'][1],
            'RR_95th': rr['ci_95'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f'{EXPORT_FOLDER}/Corrected_Risk_Ratio_Summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary exported to: {summary_path}")

    return results


if __name__ == "__main__":
    results = main()


