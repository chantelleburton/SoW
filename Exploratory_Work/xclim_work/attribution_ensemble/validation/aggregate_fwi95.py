"""
Aggregate the per-member FWI95 validation CSVs written by compare_fwi95_timeseries.py.

Produces, across every member/experiment found in OUT_DIR:
  * FWI95_summary_stats.csv    - per experiment/country/member: n, mean_bias, mean_abs_diff, corr
  * FWI95_summary_overall.csv  - per experiment/country: ensemble means of the above
  * FWI95_ensemble_timeseries_<experiment>.png - per-region subplots, ensemble mean +/- spread
  * FWI95_scatter_<experiment>.png              - per-region Xclim-vs-ImpactTB with 1:1 line

Run after all compare_fwi95<run_type, member> tasks complete.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# ---- Config (mirrors compare_fwi95_timeseries.py) ----
OUT_DIR = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/validation'
PERCENTILE = 95
REGION_ORDER = ['Korea', 'Iberia', 'Scotland', 'Chile', 'Canada']

# Country/experiment/member tokens contain no underscores, so a simple split is safe,
# but use a regex for robustness.
FNAME_RE = re.compile(r'FWI95_(?P<country>[^_]+)_(?P<exp>[^_]+)_(?P<member>[^_]+)\.csv$')


def collect():
    """Read every per-member CSV into a per-member stats table and a long dataframe."""
    records = []
    frames = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'FWI95_*_*_*.csv'))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        country, exp, member = m.group('country'), m.group('exp'), m.group('member')
        df = pd.read_csv(path, index_col=0)
        df.index.name = 'Date'
        if not {'ImpactTB', 'Xclim'}.issubset(df.columns):
            continue

        diff = df['Xclim'] - df['ImpactTB']
        valid = df[['ImpactTB', 'Xclim']].dropna()
        corr = valid['ImpactTB'].corr(valid['Xclim']) if len(valid) > 1 else np.nan
        records.append({
            'experiment': exp, 'country': country, 'member': member,
            'n': int(len(valid)),
            'mean_bias': diff.mean(),
            'mean_abs_diff': diff.abs().mean(),
            'corr': corr,
        })

        long = df.reset_index()[['Date', 'ImpactTB', 'Xclim']].copy()
        long['country'] = country
        long['experiment'] = exp
        long['member'] = member
        frames.append(long)

    if not records:
        raise SystemExit(f"No per-member FWI95 CSVs found in {OUT_DIR}. Nothing to aggregate.")

    summary = pd.DataFrame(records).sort_values(['experiment', 'country', 'member'])
    alldata = pd.concat(frames, ignore_index=True)
    alldata['Date'] = pd.to_datetime(alldata['Date'] + '-15', format='%Y-%m-%d')
    return summary, alldata


def write_summaries(summary):
    summary.to_csv(os.path.join(OUT_DIR, 'FWI95_summary_stats.csv'), index=False)
    overall = (summary.groupby(['experiment', 'country'])
               .agg(n_members=('member', 'nunique'),
                    mean_bias=('mean_bias', 'mean'),
                    mean_abs_diff=('mean_abs_diff', 'mean'),
                    mean_corr=('corr', 'mean'))
               .reset_index())
    overall.to_csv(os.path.join(OUT_DIR, 'FWI95_summary_overall.csv'), index=False)
    print("Overall agreement (ensemble means):")
    print(overall.round(3).to_string(index=False))
    return overall


def _regions_present(sub):
    present = list(sub['country'].unique())
    ordered = [r for r in REGION_ORDER if r in present]
    return ordered + [r for r in present if r not in REGION_ORDER]


def plot_ensemble_timeseries(alldata, experiment):
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
        for col, colour, label in [('ImpactTB', '#1f77b4', 'ImpactTB'),
                                    ('Xclim', '#d62728', 'Xclim')]:
            grp = g.groupby('Date')[col]
            mean = grp.mean()
            lo, hi = grp.min(), grp.max()
            ax.fill_between(mean.index, lo.values, hi.values, color=colour, alpha=0.15)
            ax.plot(mean.index, mean.values, '-', color=colour, lw=1.5, label=label)
        ax.set_title(f"{region}  (n={n_members} members)")
        ax.set_ylabel(f'FWI {PERCENTILE}th')
        ax.grid(alpha=0.3)

    for ax in axes[len(regions):]:
        ax.set_visible(False)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Ensemble monthly FWI{PERCENTILE}: mean +/- spread — {experiment}', y=1.0)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'FWI95_ensemble_timeseries_{experiment}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_scatter(alldata, experiment):
    sub = alldata[alldata['experiment'] == experiment].dropna(subset=['ImpactTB', 'Xclim'])
    regions = _regions_present(sub)
    if not regions:
        return
    ncols = 2
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, region in zip(axes, regions):
        g = sub[sub['country'] == region]
        ax.scatter(g['ImpactTB'], g['Xclim'], s=6, alpha=0.3, color='#555555')
        lim = [0, max(g['ImpactTB'].max(), g['Xclim'].max()) * 1.05]
        ax.plot(lim, lim, 'k--', lw=1, label='1:1')
        r = g['ImpactTB'].corr(g['Xclim'])
        bias = (g['Xclim'] - g['ImpactTB']).mean()
        ax.set_title(f"{region}  (r={r:.3f}, bias={bias:+.2f})")
        ax.set_xlabel('ImpactTB FWI95')
        ax.set_ylabel('Xclim FWI95')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.3)

    for ax in axes[len(regions):]:
        ax.set_visible(False)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Xclim vs ImpactTB FWI{PERCENTILE} (all members/months) — {experiment}', y=1.0)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'FWI95_scatter_{experiment}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == '__main__':
    summary, alldata = collect()
    write_summaries(summary)
    for experiment in sorted(alldata['experiment'].unique()):
        plot_ensemble_timeseries(alldata, experiment)
        plot_scatter(alldata, experiment)
    print("Aggregation complete.")
