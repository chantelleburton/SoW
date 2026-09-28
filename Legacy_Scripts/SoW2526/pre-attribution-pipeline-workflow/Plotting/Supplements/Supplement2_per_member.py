"""
Per-member bias correction timeseries plots.
Generates 15 figures (one per HadGEM3 baseline member) for a given region,
each showing ERA5 observations, the individual member's raw FWI,
and its bias-corrected FWI.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import sys

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code/Plotting')
from Supplement2_generalised import (
    load_era5_baseline,
    load_hadgem3_baseline,
    REGION_CONFIGS,
    OUTPUT_FOLDER,
    N_BASELINES,
    BASELINE_START_YEAR,
    BASELINE_END_YEAR,
)



def compute_bias_correction_custom(era5_baseline, hadgem3_members, years):
    """
    Bias correction using a custom year range.
    Same logic as Supplement2_generalised.compute_bias_correction but
    accepts an explicit years array instead of using the global YEARS.
    """
    t = years - 2024
    X = sm.add_constant(t)

    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()
        fwi0, delta = results.params
        return fwi0, delta, np.std(fwi - delta * t)

    def get_trend_slope(values):
        """OLS trend slope of values against years."""
        model = sm.OLS(values, X)
        results = model.fit()
        return results.params[1]

    obs_arr = np.where(np.isnan(era5_baseline), 1e-12, era5_baseline)
    fwi_obs_log = np.log(np.exp(obs_arr) - 1)
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs_log)

    fwi_sim_all = []
    fwi_detrended_all = []
    deltas_sim = []
    deltas_corrected = []

    for sim_arr in hadgem3_members:
        sim_clean = np.where(np.isnan(sim_arr), 1e-12, sim_arr)
        fwi_sim_log = np.log(np.exp(sim_clean) - 1)
        fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim_log)
        fwi_detrended_log = fwi0_obs + delta_obs * t + (fwi_sim_log - delta_sim * t - fwi0_sim)

        fwi_sim_inv = np.log(np.exp(fwi_sim_log) + 1)
        fwi_detrended_inv = np.log(np.exp(fwi_detrended_log) + 1)

        fwi_sim_all.append(fwi_sim_inv)
        fwi_detrended_all.append(fwi_detrended_inv)
        deltas_sim.append(get_trend_slope(fwi_sim_inv))
        deltas_corrected.append(get_trend_slope(fwi_detrended_inv))

    fwi_obs_inv = np.log(np.exp(fwi_obs_log) + 1)
    delta_obs_final = get_trend_slope(fwi_obs_inv)

    return (fwi_obs_inv, fwi_sim_all, fwi_detrended_all, years,
            delta_obs_final, deltas_sim, deltas_corrected)


def plot_member_on_ax(ax, years, fwi_obs, fwi_sim, fwi_detrended, member_num):
    """Plot a single member's timeseries on the given axes."""
    ax.plot(years, fwi_obs, label='ERA5', color='blue', linewidth=1.0)
    ax.plot(years, fwi_sim, label=f'HadGEM3',
            color='red', linewidth=1.0)
    ax.plot(years, fwi_detrended,
            label=f'Corrected',
            color='purple', linewidth=1.0)
    ax.set_title(f'Member {member_num}', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def create_per_member_plots(country, config, start_year=None, save=True):
    """Generate 15 individual bias-correction timeseries plots for a region."""
    month_name = config['month_name']
    percentile = config['percentile']
    if start_year is None:
        start_year = BASELINE_START_YEAR

    print(f"\n{'='*50}")
    print(f"Creating per-member plots for {country} (from {start_year})")
    print('='*50)

    # Load full baseline data (always 1980-2013)
    print("Loading ERA5 baseline...")
    era5_arr = load_era5_baseline(country, percentile)
    print(f"  Loaded {len(era5_arr)} ERA5 baseline values")

    print("Loading HadGEM3 baseline (all members)...")
    hadgem3_members = load_hadgem3_baseline(country, percentile, N_BASELINES)
    print(f"  Loaded {len(hadgem3_members)} HadGEM3 members")

    # Trim to requested start year
    full_years = np.arange(BASELINE_START_YEAR, BASELINE_END_YEAR + 1)
    trim_idx = int(start_year - BASELINE_START_YEAR)
    years = full_years[trim_idx:]
    era5_trimmed = era5_arr[trim_idx:]
    hadgem3_trimmed = [m[trim_idx:] for m in hadgem3_members]
    print(f"  Using years {years[0]}-{years[-1]} ({len(years)} values)")

    # Compute bias correction with custom year range
    print("Computing bias correction...")
    fwi_obs, fwi_sim_all, fwi_detrended_all, years, \
        delta_obs, deltas_sim, deltas_corrected = \
        compute_bias_correction_custom(era5_trimmed, hadgem3_trimmed, years)
    print(f"  Processed {len(fwi_sim_all)} members")

    # Print trend summary
    print(f"\n--- Trend Summary (FWI/year, {start_year}-{BASELINE_END_YEAR}) ---")
    print(f"  ERA5 trend:          {delta_obs:.6f}")
    print(f"  {'Member':<10} {'Raw trend':<14} {'Corrected trend':<16} {'Change':<14}")
    for i in range(len(deltas_sim)):
        change = deltas_corrected[i] - deltas_sim[i]
        print(f"  {i+1:<10} {deltas_sim[i]:<14.6f} {deltas_corrected[i]:<16.6f} {change:<+14.6f}")

    changes = [deltas_corrected[i] - deltas_sim[i] for i in range(len(deltas_sim))]
    print(f"  {'':<10} {'':<14} {'':<16} {'':<14}")
    print(f"  {'Mean':<10} {np.mean(deltas_sim):<14.6f} {np.mean(deltas_corrected):<16.6f} {np.mean(changes):<+14.6f}")
    print(f"  {'Range':<10} {np.min(deltas_sim):.6f} to {np.max(deltas_sim):.6f}  "
          f"{np.min(deltas_corrected):.6f} to {np.max(deltas_corrected):.6f}  "
          f"{np.min(changes):+.6f} to {np.max(changes):+.6f}")

    # Plot all members in a 5x3 grid
    fig, axes = plt.subplots(5, 3, figsize=(18, 20), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        if i < len(fwi_sim_all):
            member_num = i + 1
            plot_member_on_ax(ax, years, fwi_obs, fwi_sim_all[i],
                              fwi_detrended_all[i], member_num)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Year', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('FWI', fontsize=9)

    # Single legend from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(f'{country} {month_name} {percentile}th percentile FWI '
                 f'({start_year}-{BASELINE_END_YEAR})',
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"{country}_AllMembers_BiasCorrectionFIXED_from{start_year}.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    plt.close(fig)


############# Configuration #############
# Set these before running:
COUNTRY = 'Korea'
START_YEAR = 1997
#########################################


def main():
    country = COUNTRY
    start_year = START_YEAR

    if country not in REGION_CONFIGS:
        available = ', '.join(REGION_CONFIGS.keys())
        print(f"Unknown region '{country}'. Available: {available}")
        sys.exit(1)

    if start_year < BASELINE_START_YEAR or start_year >= BASELINE_END_YEAR:
        print(f"start_year must be between {BASELINE_START_YEAR} and {BASELINE_END_YEAR - 1}")
        sys.exit(1)

    create_per_member_plots(country, REGION_CONFIGS[country], start_year=start_year)


if __name__ == "__main__":
    main()
