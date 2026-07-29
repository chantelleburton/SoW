"""
Compare log-bias-corrected FWI: reduced-set log transforms vs xclim-derived attribution ensembles.

This script only supports the current ensemble-key CSV layout, where the data columns are
named by the ensemble identifier key itself, for example ``r002i1p5``. The deprecated
Condensed log-transform outputs that used ``EnsX_RealY`` naming are intentionally not
supported here.

For each (country, baseline_member, runtype, year) where a matching pair of CSVs exists in
both output directories, this script:
    - Loads the corrected-FWI matrices from both sources
    - Aligns them on the common ensemble-key columns
    - Stacks all paired values and computes RMSE / correlation / mean bias
    - Plots a scatter + marginal distribution per country

Output
------
    comparison_stats.csv     -- one row per (country, member, runtype, year)
    scatter_<country>.png    -- one figure per country
    summary_table.png        -- RMSE / corr bar chart across countries
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Paths ----
ORIGINAL_DIR = '/data/scratch/bob.potts/sowf/test_output/Reduced_Set_Log_Transforms/'
XCLIM_DIR = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/Condensed_Log_Transforms/'
OUT_DIR   = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/validation/bias_correction_comparison/'
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES  = ['Iberia', 'Chile', 'Canada']#'Korea', 'Scotland'
PERCENTILE = 95
ENSEMBLE_KEY_RE = re.compile(r'^r\d{3}i1p\d+$')


# ---- File discovery ----

def parse_filename(fname):
    """Return (country, baseline_member, runtype, year) or None."""
    parts = fname.replace('.csv', '').split('_')
    try:
        country         = parts[0]
        baseline_member = int(parts[1].replace('baseline', ''))
        runtype         = parts[2].replace(f'{PERCENTILE}percent', '')
        target_idx      = parts.index('Target')
        year            = int(parts[target_idx + 1])
        return country, baseline_member, runtype, year
    except (ValueError, IndexError):
        return None


def index_files(directories):
    """Index CSVs from one or more directories. Later directories win on duplicate keys."""
    if isinstance(directories, str):
        directories = [directories]
    index = {}
    for directory in directories:
        pattern = os.path.join(directory, f'*{PERCENTILE}percent_LogTransform*.csv')
        for f in glob.glob(pattern):
            meta = parse_filename(os.path.basename(f))
            if meta:
                index[meta] = f
    return index


# ---- Loader ----

def load_ensemble_table(filepath):
    """Load a CSV and return the frame plus current-format ensemble-key columns.

    The deprecated ``EnsX_RealY`` layout is not supported.
    """
    df = pd.read_csv(filepath)
    ens_cols = [c for c in df.columns if ENSEMBLE_KEY_RE.match(c)]
    if not ens_cols:
        raise ValueError(
            f"No ensemble-key columns found in {filepath}. "
            "Deprecated EnsX_RealY layout is not supported.")
    return df, ens_cols


# ---- Statistics ----

def paired_stats(orig_vals, xc_vals):
    """Compute comparison stats over paired 1-D arrays (NaN-masked)."""
    if orig_vals.shape != xc_vals.shape:
        return None
    mask = ~(np.isnan(orig_vals) | np.isnan(xc_vals))
    o, x = orig_vals[mask], xc_vals[mask]
    if len(o) < 2:
        return None
    diff = x - o
    return {
        'n':          int(mask.sum()),
        'orig_mean':  float(np.mean(o)),
        'xclim_mean': float(np.mean(x)),
        'mean_bias':  float(np.mean(diff)),
        'rmse':       float(np.sqrt(np.mean(diff ** 2))),
        'corr':       float(np.corrcoef(o, x)[0, 1]),
    }


# ---- Main ----

def main():
    print('=' * 72)
    print('Comparing Original (ImpactTB) vs Xclim bias-corrected FWI')
    print('Matching on all common (country, member, runtype, year) pairs')
    print('=' * 72)

    orig_index = index_files(ORIGINAL_DIR)
    xc_index   = index_files(XCLIM_DIR)

    common = set(orig_index.keys()) & set(xc_index.keys())
    orig_years   = sorted({k[3] for k in orig_index})
    xc_years     = sorted({k[3] for k in xc_index})
    common_years = sorted({k[3] for k in common})
    print(f'\nOriginal files: {len(orig_index)}  (years: {orig_years})')
    print(f'Xclim files:    {len(xc_index)}  (years: {xc_years})')
    print(f'Matched pairs:  {len(common)}  (years: {common_years})\n')

    if not common:
        print('No matching pairs -- check both pipelines have been run.')
        print(f'  Original dir:  {ORIGINAL_DIR}')
        print(f'  Xclim dir:     {XCLIM_DIR}')
        return

    country_vals = {c: ([], []) for c in COUNTRIES}
    rows = []

    for (country, member, runtype, year) in sorted(common):
        orig_f = orig_index[(country, member, runtype, year)]
        xc_f   = xc_index[(country, member, runtype, year)]
        try:
            orig_df, orig_cols = load_ensemble_table(orig_f)
            xc_df, xc_cols = load_ensemble_table(xc_f)
        except Exception as e:
            print(f'  LOAD ERROR {country} m{member} {runtype} {year}: {e}')
            continue

        common_cols = sorted(set(orig_cols) & set(xc_cols))
        if not common_cols:
            print(f'  NO COMMON ENSEMBLE KEYS {country} m{member} {runtype} {year} -- skipping')
            print(f'    orig keys:  {len(orig_cols)}')
            print(f'    xclim keys: {len(xc_cols)}')
            continue

        o_flat = orig_df[common_cols].to_numpy().ravel()
        x_flat = xc_df[common_cols].to_numpy().ravel()

        country_vals[country][0].append(o_flat)
        country_vals[country][1].append(x_flat)

        s = paired_stats(o_flat, x_flat)
        if s:
            rows.append({'country': country, 'member': member, 'runtype': runtype, 'year': year, **s})

    if not rows:
        print('All pairs failed to load. Check file integrity.')
        return

    df = pd.DataFrame(rows)

    # ---- Print summary table ----
    years_present = sorted(df['year'].unique())
    print('─' * 80)
    print(f'{"Country":<12} {"RunType":<10} {"Year":>6} {"Pairs":>5} {"RMSE":>8} {"Corr":>8} {"Bias":>8}')
    print('─' * 80)
    for country in COUNTRIES:
        for rt in ['hist', 'histnat']:
            for yr in years_present:
                sub = df[(df['country'] == country) & (df['runtype'] == rt) & (df['year'] == yr)]
                if len(sub) == 0:
                    continue
                print(f'{country:<12} {rt:<10} {yr:>6} {len(sub):>5} '
                      f'{sub["rmse"].mean():>8.4f} '
                      f'{sub["corr"].mean():>8.4f} '
                      f'{sub["mean_bias"].mean():>8.4f}')
    print('─' * 80)
    print(f'{"OVERALL":<29} {len(df):>5} '
          f'{df["rmse"].mean():>8.4f} '
          f'{df["corr"].mean():>8.4f} '
          f'{df["mean_bias"].mean():>8.4f}')
    print('─' * 80)

    csv_path = os.path.join(OUT_DIR, 'comparison_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved stats to {csv_path}')

    # ---- Per-country scatter + distribution ----
    print('\nGenerating plots...')

    for country in COUNTRIES:
        o_lists, x_lists = country_vals[country]
        if not o_lists:
            print(f'  {country}: no data, skipping')
            continue

        o_all = np.concatenate(o_lists)
        x_all = np.concatenate(x_lists)
        mask  = ~(np.isnan(o_all) | np.isnan(x_all))
        o_all, x_all = o_all[mask], x_all[mask]

        s = paired_stats(o_all, x_all)

        fig, (ax_sc, ax_di) = plt.subplots(1, 2, figsize=(12, 5))
        years_str = ', '.join(str(y) for y in sorted(df[df['country'] == country]['year'].unique()))
        fig.suptitle(
            f'{country}  |  ImpactTB vs Xclim bias-corrected FWI (years: {years_str})',
            fontsize=12, fontweight='bold')

        # Scatter
        ax_sc.scatter(o_all, x_all, alpha=0.04, s=3, color='steelblue', rasterized=True)
        lim_lo = min(np.nanpercentile(o_all, 0.5), np.nanpercentile(x_all, 0.5))
        lim_hi = max(np.nanpercentile(o_all, 99.5), np.nanpercentile(x_all, 99.5))
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', lw=1, label='y = x')
        ax_sc.set_xlim(lim_lo, lim_hi)
        ax_sc.set_ylim(lim_lo, lim_hi)
        ax_sc.set_xlabel('ImpactTB (log-bias-corrected FWI)')
        ax_sc.set_ylabel('Xclim (log-bias-corrected FWI)')
        ax_sc.set_title(
            f'r = {s["corr"]:.4f}  |  RMSE = {s["rmse"]:.4f}  |  bias = {s["mean_bias"]:+.4f}')
        ax_sc.legend(fontsize=8)
        ax_sc.grid(alpha=0.3)

        # Overlapping histograms
        bins = np.linspace(lim_lo, lim_hi, 80)
        ax_di.hist(o_all, bins=bins, alpha=0.5, color='steelblue',
                   density=True, label='ImpactTB')
        ax_di.hist(x_all, bins=bins, alpha=0.5, color='coral',
                   density=True, label='Xclim')
        ax_di.set_xlabel('Log-bias-corrected FWI')
        ax_di.set_ylabel('Density')
        ax_di.set_title('All ensemble members x baseline years pooled')
        ax_di.legend()
        ax_di.grid(alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f'scatter_{country}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')

    # ---- Summary bar chart ----
    by_country = (df.groupby('country')[['rmse', 'corr']]
                    .mean()
                    .reindex(COUNTRIES)
                    .dropna())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    years_str = ', '.join(str(y) for y in sorted(df['year'].unique()))
    fig.suptitle(
        f'Summary: ImpactTB vs Xclim (years: {years_str}, all members + runtypes)',
        fontsize=12, fontweight='bold')

    by_country['rmse'].plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Mean RMSE (lower = more similar)')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', alpha=0.3)

    by_country['corr'].plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Mean Correlation (higher = more similar)')
    ax2.set_ylabel('Pearson r')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, 'summary_table.png')
    fig.savefig(summary_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {summary_path}')

    print(f'\nDone. All output in {OUT_DIR}')


if __name__ == '__main__':
    main()