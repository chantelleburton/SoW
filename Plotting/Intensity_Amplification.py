"""
Intensity Amplification of FWI 95th Percentile:
Factual (ALL) minus Counterfactual (NAT) comparison.

Produces vertical box-and-whisker plots (one per region) showing the
distribution of per-ensemble-member intensity differences across all
baselines. Box at IQR (25th-75th), whiskers at 5th/95th, mean line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Work Sans'
import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
from utils.branded_colours import SoW_gradient_teal, SoW_gradient_red, SoW_categorial

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

############# Configuration #############

LOG_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Condensed_Log_Transforms'
PLOT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Plots'

N_BASELINES = 15
BASELINE_START_YEAR = 1980
BASELINE_END_YEAR = 2013
TARGET_YEAR = 2024
DATA_YEARS = [2024]

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
    'Chilean Temperate Forests and Matorral': 'Chile Forests\n& Matorral',
    'Midwestern Canadian Shield forests': 'Canadian Shield\nForests',
}


############# Helper Functions #############


def load_member_means(country, percentile, run_type, folder, target_year,
                      data_years, n_baselines, baseline_start, baseline_end):
    """
    Load CSV data and compute per-ensemble-member mean across years.

    For each baseline and data_year, reads the CSV, drops the Year column,
    and computes the mean of each ensemble column across all year-rows.
    Returns a 1D array of member means (concatenated across all baselines/data_years).
    """
    all_means = []
    for data_year in data_years:
        for baseline in range(1, n_baselines + 1):
            filename = (
                f"{country}_baseline{baseline}_{run_type}{percentile}percent"
                f"_LogTransform_Target_{target_year}_DataYear_{data_year}"
                f"_BaselinePeriod_{baseline_start}_{baseline_end}.csv"
            )
            filepath = os.path.join(folder, filename)
            try:
                df = pd.read_csv(filepath)
                col_names = [col for col in df.columns if col != 'Year']
                member_means = df[col_names].mean(axis=0).values  # mean across years per member
                all_means.append(member_means)
            except FileNotFoundError:
                print(f"Warning: Missing file {filepath}")
                continue
    if all_means:
        return np.concatenate(all_means)
    else:
        return np.array([])


def compute_amplification(country, config):
    """
    Compute intensity amplification (factual - counterfactual) for a region.

    Returns a 1D array of per-member differences (factual mean - counterfactual mean).
    """
    all_means = load_member_means(
        country, config['percentile'], 'hist', LOG_FOLDER, TARGET_YEAR,
        DATA_YEARS, N_BASELINES, BASELINE_START_YEAR, BASELINE_END_YEAR
    )
    nat_means = load_member_means(
        country, config['percentile'], 'histnat', LOG_FOLDER, TARGET_YEAR,
        DATA_YEARS, N_BASELINES, BASELINE_START_YEAR, BASELINE_END_YEAR
    )

    if len(all_means) == 0 or len(nat_means) == 0:
        print(f"Skipping {country} due to missing data")
        return np.array([])

    if len(all_means) != len(nat_means):
        print(f"Warning: {country} has mismatched ALL ({len(all_means)}) "
              f"and NAT ({len(nat_means)}) array sizes. Using minimum length.")
        n = min(len(all_means), len(nat_means))
        all_means = all_means[:n]
        nat_means = nat_means[:n]

    return all_means - nat_means


############# Main #############

#
def main():
    """Compute and plot intensity amplification for all regions."""

    # Phase 1: Compute amplification per region
    amplification = {}
    for country, config in REGION_CONFIGS.items():
        print(f"Processing {country}...")
        amp = compute_amplification(country, config)
        if len(amp) > 0:
            amplification[country] = amp
            p5, p25, mean, p75, p95 = (
                np.percentile(amp, 5), np.percentile(amp, 25),
                np.mean(amp), np.percentile(amp, 75), np.percentile(amp, 95)
            )
            print(f"  {country}: mean={mean:.2f}, IQR=[{p25:.2f}, {p75:.2f}], "
                  f"5th-95th=[{p5:.2f}, {p95:.2f}], n={len(amp)}")

    if not amplification:
        print("No data loaded for any region. Exiting.")
        return

    # Phase 2: Plot
    regions = list(amplification.keys())
    n_regions = len(regions)

    fig, axes = plt.subplots(1, n_regions, figsize=(3 * n_regions, 7), sharey=True)
    if n_regions == 1:
        axes = [axes]

    box_colour = '#ee007f'   # '#008787' darkest teal
    mean_colour = "#0096a1"   # '#C7403D' darkest red

    for i, region in enumerate(regions):
        ax = axes[i]
        data = amplification[region]
        config = REGION_CONFIGS[region]
        display_name = DISPLAY_NAMES.get(config['shape_name'], config['shape_name'])

        # Custom box-and-whisker: box at IQR, whiskers at 5th/95th, mean line
        stats = {
            'med': np.mean(data),           # central line = mean
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'whislo': np.percentile(data, 5),
            'whishi': np.percentile(data, 95),
            'fliers': [],
        }

        bp = ax.bxp(
            [stats], positions=[0], widths=[0.5], patch_artist=True,
            showfliers=False, manage_ticks=False,
        )

        # Style the box
        for patch in bp['boxes']:
            patch.set_facecolor(box_colour)
            patch.set_alpha(0.4)
            patch.set_edgecolor(box_colour)
            patch.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color(box_colour)
            whisker.set_linewidth(1.5)
        for cap in bp['caps']:
            cap.set_color(box_colour)
            cap.set_linewidth(1.5)
        for median_line in bp['medians']:
            median_line.set_color(mean_colour)
            median_line.set_linewidth(2.5)
            median_line.set_label('Mean')

        # Reference line at zero
        ax.axhline(y=0, color='grey', linewidth=1, linestyle='--', alpha=0.7)

        # Labels
        ax.set_title(f"{display_name}\n{config['month_name']}", fontsize=11)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)

        if i == 0:
            ax.set_ylabel('Intensity Amplification\n(FWI 95th Percentile Difference)', fontsize=11)

    fig.suptitle(
        'Intensity Amplification: Factual − Counterfactual FWI 95th Percentile',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    outpath = os.path.join(PLOT_FOLDER, 'Intensity_Amplification_BoxWhisker.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {outpath}")

    plt.show()


if __name__ == "__main__":
    main()
