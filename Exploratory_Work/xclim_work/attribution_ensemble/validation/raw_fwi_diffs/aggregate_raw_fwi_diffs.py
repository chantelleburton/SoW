"""
Aggregate the per-member raw-FWI diff outputs written by compute_raw_fwi_diffs.py.

Produces, across every member/experiment/region found in OUT_DIR:
  * raw_fwi_diffs_summary_stats.csv     -- per experiment/country/member: n, n_days,
                                            expected_n_days, pct_complete, mean_bias,
                                            rmse, corr, max_abs_diff, date_of_max_diff
  * raw_fwi_diffs_incomplete_members.csv -- rows from the above where ImpactTB only
                                            covered a subset of the expected period
                                            (pct_complete < 100); excluded from all
                                            ensemble-mean stats/plots below
  * raw_fwi_diffs_summary_overall.csv   -- per experiment/country: ensemble means of the
                                            above, computed over complete members only
  * ensemble_daily_diff_<experiment>.png -- WHEN: per-region daily diff, ensemble mean +/- spread
  * seasonal_diff_<experiment>.png       -- WHEN: per-region mean diff by calendar month
  * spatial_diff_map_<country>_<experiment>.png -- WHERE: ensemble-mean spatial diff map

Run after all compute_raw_fwi_diffs<run_type, member> tasks complete.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

OUT_DIR = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/validation/raw_fwi_diffs'
REGION_ORDER = ['Iberia', 'Chile', 'Canada']

DAILY_RE = re.compile(r'daily_diff_(?P<country>[^_]+)_(?P<exp>[^_]+)_(?P<member>[^_]+)\.csv$')
MAP_RE = re.compile(r'meanmap_(?P<country>[^_]+)_(?P<exp>[^_]+)_(?P<member>[^_]+)\.csv$')


def _regions_present(sub):
    present = list(sub['country'].unique())
    ordered = [r for r in REGION_ORDER if r in present]
    return ordered + [r for r in present if r not in REGION_ORDER]


def collect_daily():
    """Read every per-member daily-diff CSV into one long dataframe + a summary table."""
    records = []
    frames = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'daily_diff_*_*_*.csv'))):
        m = DAILY_RE.search(os.path.basename(path))
        if not m:
            continue
        country, exp, member = m.group('country'), m.group('exp'), m.group('member')
        df = pd.read_csv(path, parse_dates=False)
        if df.empty:
            continue

        # HadGEM3-A driver data uses a 360_day calendar (e.g. Feb 30 exists), which
        # pandas' Gregorian date parser can't represent. Keep 'Date' as plain
        # 'YYYY-MM-DD' strings throughout (never parse to Timestamp) so every row
        # across every member/file is handled uniformly -- mixing Timestamp and
        # str within a concatenated column breaks matplotlib's date axis later.
        valid = df.dropna(subset=['mean_impacttb', 'mean_xclim'])
        corr = valid['mean_impacttb'].corr(valid['mean_xclim']) if len(valid) > 1 else np.nan
        max_idx = df['max_abs_diff'].idxmax() if df['max_abs_diff'].notna().any() else None
        date_of_max_diff = str(df.loc[max_idx, 'Date']) if max_idx is not None else None

        # Completeness (added by compute_raw_fwi_diffs.py): ImpactTB sometimes only
        # delivers a subset of the full period (e.g. 1 month out of ~62). Older
        # daily_diff CSVs written before this tracking existed won't have these
        # columns -- treat those as complete (can't tell otherwise) rather than
        # crashing or silently mis-flagging them as partial.
        if 'pct_complete' in df.columns:
            pct_complete = float(df['pct_complete'].iloc[0])
            n_days = int(df['n_days'].iloc[0])
            expected_n_days = int(df['expected_n_days'].iloc[0])
        else:
            pct_complete, n_days, expected_n_days = 100.0, int(len(df)), int(len(df))

        records.append({
            'experiment': exp, 'country': country, 'member': member,
            'n': int(len(df)),
            'n_days': n_days,
            'expected_n_days': expected_n_days,
            'pct_complete': pct_complete,
            'mean_bias': df['mean_diff'].mean(),
            'rmse': float(np.sqrt(np.mean(df['mean_diff'].dropna() ** 2))) if df['mean_diff'].notna().any() else np.nan,
            'corr': corr,
            'max_abs_diff': df['max_abs_diff'].max(),
            'date_of_max_diff': date_of_max_diff,
        })

        df['country'] = country
        df['experiment'] = exp
        df['member'] = member
        frames.append(df)

    if not records:
        raise SystemExit(f"No daily-diff CSVs found in {OUT_DIR}. Run compute_raw_fwi_diffs.py first.")

    summary = pd.DataFrame(records).sort_values(['experiment', 'country', 'member'])
    alldata = pd.concat(frames, ignore_index=True)
    return summary, alldata


def collect_maps(keep_set=None):
    """Average the time-mean spatial diff maps across all members, per country/experiment.

    keep_set, if given, is a set of (experiment, country, member) tuples to include
    -- used to exclude members with incomplete ImpactTB coverage from the ensemble
    spatial-mean map, consistent with write_summaries()'s exclusion from the stats.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'meanmap_*_*_*.csv'))):
        m = MAP_RE.search(os.path.basename(path))
        if not m:
            continue
        country, exp, member = m.group('country'), m.group('exp'), m.group('member')
        if keep_set is not None and (exp, country, member) not in keep_set:
            continue
        df = pd.read_csv(path)
        df['country'] = country
        df['experiment'] = exp
        df['member'] = member
        frames.append(df)

    if not frames:
        print(f"No meanmap CSVs found in {OUT_DIR}. Skipping spatial diff maps.")
        return None

    alldata = pd.concat(frames, ignore_index=True)
    ensemble_mean = (alldata.groupby(['experiment', 'country', 'lat', 'lon'])
                     .agg(mean_diff=('mean_diff', 'mean'), std_diff=('std_diff', 'mean'),
                          n_members=('member', 'nunique'))
                     .reset_index())
    return ensemble_mean


def write_summaries(summary):
    # Split off any member/country/experiment whose ImpactTB source didn't cover the
    # full expected period (e.g. only 1 of ~62 months delivered). These are reported
    # separately rather than silently folded into the ensemble means, since a partial
    # sample (e.g. 30 days) is not comparable to a full ~1860-day comparison and can
    # bias mean_bias/rmse/max_abs_diff if included with equal weight.
    full = summary[summary['pct_complete'] >= 100.0 - 1e-6].copy()
    partial = summary[summary['pct_complete'] < 100.0 - 1e-6].copy()

    summary.to_csv(os.path.join(OUT_DIR, 'raw_fwi_diffs_summary_stats.csv'), index=False)
    partial.sort_values(['experiment', 'country', 'pct_complete']).to_csv(
        os.path.join(OUT_DIR, 'raw_fwi_diffs_incomplete_members.csv'), index=False)

    if not partial.empty:
        print(f"{len(partial)} member/country/experiment rows are incomplete "
              f"(partial ImpactTB coverage) -- excluded from ensemble means, "
              f"see raw_fwi_diffs_incomplete_members.csv")
        print(partial[['experiment', 'country', 'member', 'n_days', 'expected_n_days', 'pct_complete']]
              .sort_values('pct_complete').to_string(index=False))

    overall = (full.groupby(['experiment', 'country'])
               .agg(n_members=('member', 'nunique'),
                    mean_bias=('mean_bias', 'mean'),
                    mean_rmse=('rmse', 'mean'),
                    mean_corr=('corr', 'mean'),
                    mean_max_abs_diff=('max_abs_diff', 'mean'))
               .reset_index())
    overall.to_csv(os.path.join(OUT_DIR, 'raw_fwi_diffs_summary_overall.csv'), index=False)
    print("Overall raw-FWI diff agreement (ensemble means, complete members only):")
    print(overall.round(3).to_string(index=False))
    return overall, full


def plot_ensemble_daily(alldata, experiment):
    """WHEN: per-region daily diff, ensemble mean +/- min/max spread across members."""
    sub = alldata[alldata['experiment'] == experiment]
    regions = _regions_present(sub)
    if not regions:
        return
    ncols = 2
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, region in zip(axes, regions):
        g = sub[sub['country'] == region]
        n_members = g['member'].nunique()
        # Date is a plain 'YYYY-MM-DD' string (360_day-calendar dates like Feb 30
        # can't be represented as a Timestamp), so group/sort on the string directly
        # and plot against a categorical integer index rather than a date axis.
        grp = g.groupby('Date')['mean_diff']
        mean, lo, hi = grp.mean().sort_index(), grp.min().sort_index(), grp.max().sort_index()
        x = range(len(mean))
        ax.fill_between(x, lo.values, hi.values, color='#d62728', alpha=0.15)
        ax.plot(x, mean.values, '-', color='#d62728', lw=1.2, label='Xclim - ImpactTB')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        step = max(1, len(mean) // 6)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(mean.index[::step], rotation=45, ha='right', fontsize=7)
        ax.set_title(f"{region}  (n={n_members} members)")
        ax.set_ylabel('Mean diff (FWI)')
        ax.grid(alpha=0.3)

    for ax in axes[len(regions):]:
        ax.set_visible(False)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Daily raw FWI diff: mean +/- spread across members — {experiment}', y=1.0)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'ensemble_daily_diff_{experiment}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_seasonal(alldata, experiment):
    """WHEN: mean diff by calendar month, pooled across members and years."""
    sub = alldata[alldata['experiment'] == experiment].copy()
    regions = _regions_present(sub)
    if not regions:
        return
    # 'Date' is a plain 'YYYY-MM-DD' string (360_day calendar dates like Feb 30
    # can't be represented as a Timestamp) -- extract the month directly.
    sub['month'] = sub['Date'].str.split('-').str[1].astype(int)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for region, colour in zip(regions, ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']):
        g = sub[sub['country'] == region]
        monthly = g.groupby('month')['mean_diff'].agg(['mean', 'std'])
        ax.plot(monthly.index, monthly['mean'], '-o', ms=4, color=colour, label=region)
        ax.fill_between(monthly.index, monthly['mean'] - monthly['std'],
                         monthly['mean'] + monthly['std'], color=colour, alpha=0.12)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean diff (FWI, Xclim - ImpactTB)')
    ax.set_title(f'Seasonal pattern of raw FWI diff (all members/years pooled) — {experiment}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'seasonal_diff_{experiment}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_spatial_maps(ensemble_mean):
    """WHERE: ensemble-mean spatial diff map per country/experiment."""
    if ensemble_mean is None:
        return
    for (experiment, country), g in ensemble_mean.groupby(['experiment', 'country']):
        lats = np.sort(g['lat'].unique())
        lons = np.sort(g['lon'].unique())
        grid = g.pivot(index='lat', columns='lon', values='mean_diff').reindex(index=lats, columns=lons)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        vmax = np.nanmax(np.abs(grid.values)) if np.isfinite(grid.values).any() else 1.0
        im = ax.pcolormesh(grid.columns.values, grid.index.values, grid.values,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
        fig.colorbar(im, ax=ax, label='Mean diff (FWI, Xclim - ImpactTB)')
        n_members = int(g['n_members'].max())
        ax.set_title(f'{country} ({experiment})  ensemble-mean spatial diff  (n={n_members} members)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f'spatial_diff_map_{country}_{experiment}.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_png}")


if __name__ == '__main__':
    summary, alldata = collect_daily()
    overall, full = write_summaries(summary)

    # Restrict plots/spatial aggregation to fully-compared members only, matching
    # the exclusion already applied to the ensemble-mean stats in write_summaries().
    keep_set = set(zip(full['experiment'], full['country'], full['member']))
    alldata_full = alldata.merge(
        full[['experiment', 'country', 'member']], on=['experiment', 'country', 'member'], how='inner')

    ensemble_mean_maps = collect_maps(keep_set=keep_set)

    for experiment in sorted(alldata_full['experiment'].unique()):
        plot_ensemble_daily(alldata_full, experiment)
        plot_seasonal(alldata_full, experiment)

    plot_spatial_maps(ensemble_mean_maps)
    print("Aggregation complete.")
