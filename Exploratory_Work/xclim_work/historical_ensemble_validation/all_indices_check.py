"""
Compare HadGEM3-A historical FWI-family index percentile time series produced
by two different pipelines:

  - "xclim"    : new xclim-based pipeline metric outputs (attribution_pipeline's
                 run_metrics.py), e.g.
                 /data/scratch/bob.potts/sowf/attribution_pipeline/metrics/
                 hg3_historical_BUI_P95_Iberia_8.csv
  - "impactTB" : legacy ImpactTB-based pipeline outputs, e.g.
                 /data/scratch/bob.potts/sowf/test_output/Baseline/
                 HadGEM3_ISI_1980-2013_Chile_15_95%.csv

For every (index, country) combination present in both pipelines, this script:
  1. Loads every available ensemble member's yearly percentile time series.
  2. Plots a time series comparison: ensemble mean line + shaded spread
     (min-max across members) for each pipeline.
  3. Plots the per-year, per-member difference (xclim - impactTB) as a
     mean +/- std bar chart, highlighting the single largest-magnitude year.
  4. Writes a CSV summary of member-level and year-level differences, ranked
     by absolute difference, so the biggest discrepancies can be inspected.
"""
import os
import re
import glob
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Work Sans'

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

############# Configuration #############

# xclim metric CSVs are written by attribution_pipeline/metrics/run_metrics.py
# for dataset='hg3_historical', filenames: hg3_historical_{INDEX}_{METRICID}_{country}_{member}.csv
# (only the percentile metric ('p95' -> METRICID 'P95') is comparable here --
# the 7x/cum metrics have no ImpactTB-side equivalent to diff against).
XCLIM_DIR = '/data/scratch/bob.potts/sowf/attribution_pipeline/metrics'
IMPACTTB_DIR = '/data/scratch/bob.potts/sowf/test_output/Baseline/'
PLOT_DIR = '/data/scratch/bob.potts/sowf/attribution_pipeline/validation/XCLIM_vs_ImpactTB'
SUMMARY_CSV = os.path.join(PLOT_DIR, 'xclim_vs_impacttb_differences.csv')

INDEX_NAMES = {
    'FFMC': 'Fine Fuel Moisture Content',
    'FWI':  'Canadian Fire Weather Index',
    'ISI':  'Initial Spread Index',
    'BUI':  'Build Up Index',
    'DMC':  'Duff Moisture Content',
    'DC':   'Drought Code',
}

COLOUR_XCLIM = '#008787'     # teal
COLOUR_IMPACTTB = '#E27226'  # orange
COLOUR_DIFF = '#862976'      # hotpink/purple, from SoW_gradient_hues

# xclim filenames look like: hg3_historical_BUI_P95_Iberia_8.csv
# impactTB filenames look like: HadGEM3_ISI_1980-2013_Chile_15_95%.csv
XCLIM_PATTERN = re.compile(
    r'hg3_historical_(?P<index>[A-Z]+)_P(?P<pct>\d+)_(?P<country>[A-Za-z]+)_(?P<member>\d+)\.csv$'
)
IMPACTTB_PATTERN = re.compile(
    r'HadGEM3_(?P<index>[A-Z]+)_(?P<start>\d+)-(?P<end>\d+)_(?P<country>[A-Za-z]+)_(?P<member>\d+)_(?P<pct>\d+)%\.csv$'
)


def _scan_dir(directory, pattern):
    """Return {(index, country, member): filepath} for all files matching pattern."""
    found = {}
    for fpath in sorted(glob.glob(os.path.join(directory, '*.csv'))):
        m = pattern.search(os.path.basename(fpath))
        if not m:
            continue
        key = (m.group('index'), m.group('country'), int(m.group('member')))
        found[key] = fpath
    return found


def _load_series(fpath, missing_report):
    """Load a Date,<Value> csv and return a Series indexed by year (int).

    Non-numeric values (e.g. '--' placeholders for missing data) are coerced
    to NaN rather than raising, and every such occurrence is appended to
    missing_report so it can be surfaced to the user at the end of the run.
    """
    df = pd.read_csv(fpath)
    date_col, value_col = df.columns[0], df.columns[1]
    years = df[date_col].astype(str).str.split('-').str[0].astype(int)
    raw_values = df[value_col]
    values = pd.to_numeric(raw_values, errors='coerce')

    bad_mask = values.isna() & raw_values.notna()
    for year, raw in zip(years[bad_mask], raw_values[bad_mask]):
        missing_report.append({'file': fpath, 'year': int(year), 'raw_value': raw})

    series = pd.Series(values.values, index=years, name=value_col)
    return series[~series.index.duplicated(keep='first')].sort_index()


def _spread_plot(ax, years, mean, lo, hi, colour, label):
    ax.plot(years, mean, color=colour, linewidth=2, label=f'{label} (mean)')
    ax.fill_between(years, lo, hi, color=colour, alpha=0.2, label=f'{label} (member spread)')


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    xclim_files = _scan_dir(XCLIM_DIR, XCLIM_PATTERN)
    impacttb_files = _scan_dir(IMPACTTB_DIR, IMPACTTB_PATTERN)

    xclim_keys = {(idx, country) for idx, country, member in xclim_files}
    impacttb_keys = {(idx, country) for idx, country, member in impacttb_files}
    common_keys = sorted(xclim_keys & impacttb_keys)

    if not common_keys:
        raise FileNotFoundError(
            "No overlapping (index, country) combinations found between "
            f"{XCLIM_DIR} and {IMPACTTB_DIR}"
        )
    print(f"Found {len(common_keys)} (index, country) combinations present in both pipelines: {common_keys}")

    diff_rows = []
    missing_report = []

    for index_code, country in common_keys:
        members = sorted({m for (idx, c, m) in xclim_files if idx == index_code and c == country}
                          & {m for (idx, c, m) in impacttb_files if idx == index_code and c == country})
        if not members:
            continue

        try:
            xclim_series = {m: _load_series(xclim_files[(index_code, country, m)], missing_report) for m in members}
            impacttb_series = {m: _load_series(impacttb_files[(index_code, country, m)], missing_report) for m in members}

            xclim_df = pd.DataFrame(xclim_series)   # rows=year, cols=member
            impacttb_df = pd.DataFrame(impacttb_series)

            common_years = sorted(set(xclim_df.index) & set(impacttb_df.index))
            xclim_df = xclim_df.loc[common_years]
            impacttb_df = impacttb_df.loc[common_years]
            diff_df = xclim_df - impacttb_df  # rows=year, cols=member (NaN where either side is missing)

            # --- record per-member, per-year differences for the summary CSV ---
            for year in common_years:
                for m in members:
                    diff_rows.append({
                        'index': index_code,
                        'country': country,
                        'member': m,
                        'year': year,
                        'xclim_value': xclim_df.loc[year, m],
                        'impacttb_value': impacttb_df.loc[year, m],
                        'difference': diff_df.loc[year, m],
                    })

            # --- Figure: time series comparison + difference panel ---
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1]})

            _spread_plot(ax0, common_years, xclim_df.mean(axis=1), xclim_df.min(axis=1), xclim_df.max(axis=1),
                         COLOUR_XCLIM, 'xclim')
            _spread_plot(ax0, common_years, impacttb_df.mean(axis=1), impacttb_df.min(axis=1), impacttb_df.max(axis=1),
                         COLOUR_IMPACTTB, 'impactTB')
            index_name = INDEX_NAMES.get(index_code, index_code)
            ax0.set_title(f'{index_name} ({index_code}) — {country}: xclim vs impactTB (95th percentile, all members)')
            ax0.set_ylabel(index_name)
            ax0.legend(loc='upper left', fontsize=8, ncol=2)

            diff_mean = diff_df.mean(axis=1)
            diff_std = diff_df.std(axis=1)
            ax1.bar(common_years, diff_mean, yerr=diff_std, color=COLOUR_DIFF, alpha=0.8, capsize=3)
            ax1.axhline(0, color='black', linewidth=0.8)

            if diff_mean.notna().any():
                biggest_year = diff_mean.abs().idxmax()
                ax1.annotate(
                    f'largest mean diff: {diff_mean[biggest_year]:+.2f} ({biggest_year})',
                    xy=(biggest_year, diff_mean[biggest_year]),
                    xytext=(0, 15 if diff_mean[biggest_year] >= 0 else -25),
                    textcoords='offset points', ha='center', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                )
            else:
                print(f"  Warning: {index_code}/{country} has no overlapping non-NaN years — skipping diff annotation")
            ax1.set_ylabel('xclim − impactTB\n(mean ± std across members)')
            ax1.set_xlabel('Year')

            fig.tight_layout()
            out_path = os.path.join(PLOT_DIR, f'{index_code}_{country}_xclim_vs_impactTB_modified.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"  Skipping {index_code}/{country}: plotting failed with {e!r}")
            continue

    # --- Report any missing/non-numeric ('--' etc.) values found while loading ---
    if missing_report:
        missing_df = pd.DataFrame(missing_report)
        missing_csv = os.path.join(PLOT_DIR, 'missing_values_report_modified.csv')
        missing_df.to_csv(missing_csv, index=False)
        print(f"\n=== WARNING: {len(missing_df)} non-numeric/missing values ('--' etc.) found and treated as NaN ===")
        print(f"Full details saved to {missing_csv}")
        print(missing_df.groupby('file').size().rename('n_missing').to_string())
    else:
        print("\nNo missing/non-numeric values found in any input file.")

    if not diff_rows:
        print("\nNo overlapping (index, country, member, year) rows were successfully compared — nothing to summarise.")
        return

    # --- Summary CSV + console report of biggest differences ---
    summary = pd.DataFrame(diff_rows)
    summary['abs_difference'] = summary['difference'].abs()
    summary = summary.sort_values('abs_difference', ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved full difference summary to {SUMMARY_CSV}")

    print("\n=== Average absolute difference per index (across all countries/members/years) ===")
    print(summary.groupby('index')['abs_difference'].mean().sort_values(ascending=False).round(3))

    print("\n=== Average absolute difference per (index, country) ===")
    print(summary.groupby(['index', 'country'])['abs_difference'].mean().sort_values(ascending=False).round(3))

    print("\n=== Top 20 biggest single differences ===")
    print(summary.head(20)[['index', 'country', 'member', 'year', 'xclim_value', 'impacttb_value', 'difference']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
