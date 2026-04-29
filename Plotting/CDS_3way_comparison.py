import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

baseline_dir = '/data/scratch/bob.potts/sowf/test_output/AllMonths_Baseline'
export_dir = '/data/scratch/bob.potts/sowf/test_output/Exports'
plot_dir = '/data/scratch/bob.potts/sowf/test_output/Plots/CDS_Comparison'

REGION_CONFIGS = {
    'Iberia':   {'percentile': 95, 'shape_name': 'Northwest Iberia',
                 'key_months': [8], 'key_months_label': 'August'},
    'Korea':    {'percentile': 95, 'shape_name': 'Southeast South Korea',
                 'key_months': [3], 'key_months_label': 'March'},
    'Scotland': {'percentile': 95, 'shape_name': 'Scottish Highlands',
                 'key_months': [6, 7], 'key_months_label': 'June-July'},
    'Chile':    {'percentile': 95, 'shape_name': 'Chilean Temperate Forests and Matorral',
                 'key_months': [1, 2], 'key_months_label': 'January-February'},
    'Canada':   {'percentile': 95, 'shape_name': 'Midwestern Canadian Shield forests',
                 'key_months': [7, 8], 'key_months_label': 'July-August'},
}


def compute_pairwise_stats(df, pairs, prefix=''):
    """Print pairwise comparison stats."""
    for col_a, col_b, label in pairs:
        overlap = df.dropna(subset=[col_a, col_b])
        if len(overlap) == 0:
            print(f"  {prefix}{label}: no overlapping months")
            continue
        diff = overlap[col_a] - overlap[col_b]
        corr = overlap[col_a].corr(overlap[col_b])
        print(f"\n  {prefix}--- {label} ({len(overlap)} overlapping months) ---")
        print(f"  {prefix}  Correlation:     {corr:.4f}")
        print(f"  {prefix}  Mean diff:       {diff.mean():.4f}")
        print(f"  {prefix}  Mean abs diff:   {diff.abs().mean():.4f}")
        print(f"  {prefix}  RMSE:            {np.sqrt((diff**2).mean()):.4f}")
        print(f"  {prefix}  Max abs diff:    {diff.abs().max():.4f}")
        print(f"  {prefix}  Bias (median):   {diff.median():.4f}")


def print_source_summary(df, prefix=''):
    """Print per-source summary stats."""
    print(f"\n  {prefix}--- Per-source summary ---")
    for col, label in [('CDS_FWI_95', 'CDS'), ('XClim_FWI_95', 'XClim'),
                        ('ERA5_FWI_95', 'ImpactTB')]:
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"  {prefix}  {label:8s}: n={len(valid):4d}, "
                  f"mean={valid.mean():.2f}, median={valid.median():.2f}, "
                  f"min={valid.min():.2f}, max={valid.max():.2f}")


def plot_timeseries(df, region_name, pct, suffix, title_extra=''):
    """Plot 3-way time series + difference panel."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    ax1 = axes[0]
    ax1.plot(df['DateParsed'], df['ERA5_FWI_95'], '-',
             label='ImpactTB 95%', color='blue', linewidth=0.8, alpha=0.8)
    ax1.plot(df['DateParsed'], df['CDS_FWI_95'], '-',
             label='CDS 95%', color='red', linewidth=0.8, alpha=0.8)
    ax1.plot(df['DateParsed'], df['XClim_FWI_95'], '-',
             label='XClim 95%', color='green', linewidth=0.8, alpha=0.8)
    ax1.set_ylabel(f'FWI {pct}th Percentile')
    ax1.set_title(f'{region_name} Monthly FWI {pct}th Percentile: '
                  f'CDS vs ImpactTB vs XClim{title_extra}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df['DateParsed'], df['CDS_minus_ERA5'], '-',
             label='CDS - ImpactTB', color='red', linewidth=0.8, alpha=0.7)
    ax2.plot(df['DateParsed'], df['XClim_minus_ERA5'], '-',
             label='XClim - ImpactTB', color='green', linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Difference vs ImpactTB')
    ax2.set_title('Differences (where ImpactTB baseline available)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plot_dir, f'FWI_3way_{suffix}_{region_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Time series plot saved: {os.path.basename(out)}")


def plot_scatter(df, region_name, suffix):
    """Plot pairwise scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    scatter_pairs = [
        ('ERA5_FWI_95', 'CDS_FWI_95', 'ImpactTB', 'CDS', 'red'),
        ('ERA5_FWI_95', 'XClim_FWI_95', 'ImpactTB', 'XClim', 'green'),
        ('CDS_FWI_95', 'XClim_FWI_95', 'CDS', 'XClim', 'purple'),
    ]
    for ax, (xcol, ycol, xlabel, ylabel, color) in zip(axes, scatter_pairs):
        overlap = df.dropna(subset=[xcol, ycol])
        if len(overlap) == 0:
            ax.text(0.5, 0.5, 'No overlap', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{xlabel} vs {ylabel}')
            continue
        ax.scatter(overlap[xcol], overlap[ycol], alpha=0.3, s=10, color=color)
        lims = [min(overlap[xcol].min(), overlap[ycol].min()),
                max(overlap[xcol].max(), overlap[ycol].max())]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='1:1')
        corr = overlap[xcol].corr(overlap[ycol])
        ax.set_xlabel(f'{xlabel} FWI 95th')
        ax.set_ylabel(f'{ylabel} FWI 95th')
        ax.set_title(f'{xlabel} vs {ylabel} (r={corr:.3f}, n={len(overlap)})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{region_name} — Pairwise FWI Scatter ({suffix})', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(plot_dir, f'FWI_3way_scatter_{suffix}_{region_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scatter plot saved: {os.path.basename(out)}")


PAIRS = [
    ('CDS_FWI_95', 'ERA5_FWI_95', 'CDS vs ImpactTB'),
    ('XClim_FWI_95', 'ERA5_FWI_95', 'XClim vs ImpactTB'),
    ('CDS_FWI_95', 'XClim_FWI_95', 'CDS vs XClim'),
]

for region_name, cfg in REGION_CONFIGS.items():
    pct = cfg['percentile']
    key_months = cfg['key_months']
    key_months_label = cfg['key_months_label']

    print(f"\n{'='*60}")
    print(f"Plotting: {region_name}")
    print(f"{'='*60}")

    # --- Load the 3 CSVs ---
    cds_path = os.path.join(baseline_dir, f'CDS_FWI_AllMonths_{region_name}.csv')
    xclim_path = os.path.join(baseline_dir, f'XClim_FWI_AllMonths_{region_name}.csv')
    era5_path = os.path.join(baseline_dir,
                             f'ERA5_FWI_1980-2013_{region_name}_allmonths_{pct}%.csv')

    cds_df = None
    if os.path.exists(cds_path):
        cds_df = pd.read_csv(cds_path).rename(columns={'FWI': 'CDS_FWI_95'})
        print(f"  CDS: {len(cds_df)} months "
              f"({cds_df['Date'].iloc[0]} to {cds_df['Date'].iloc[-1]})")
    else:
        print(f"  CDS CSV not found: {cds_path}")

    xclim_df = None
    if os.path.exists(xclim_path):
        xclim_df = pd.read_csv(xclim_path).rename(columns={'FWI': 'XClim_FWI_95'})
        print(f"  XClim: {len(xclim_df)} months "
              f"({xclim_df['Date'].iloc[0]} to {xclim_df['Date'].iloc[-1]})")
    else:
        print(f"  XClim CSV not found: {xclim_path}")

    era5_df = None
    if os.path.exists(era5_path):
        era5_df = pd.read_csv(era5_path).rename(columns={'FWI': 'ERA5_FWI_95'})
        print(f"  ImpactTB: {len(era5_df)} months "
              f"({era5_df['Date'].iloc[0]} to {era5_df['Date'].iloc[-1]})")
    else:
        print(f"  ImpactTB CSV not found: {era5_path}")

    # Need at least 2 sources to make a comparison
    sources = [df for df in [cds_df, xclim_df, era5_df] if df is not None]
    if len(sources) < 2:
        print(f"  Skipping {region_name}: fewer than 2 data sources available")
        continue

    # --- 3-way merge (outer join) ---
    comparison = sources[0]
    for df in sources[1:]:
        comparison = comparison.merge(df, on='Date', how='outer')

    for col in ['CDS_FWI_95', 'XClim_FWI_95', 'ERA5_FWI_95']:
        if col not in comparison.columns:
            comparison[col] = np.nan

    comparison['DateParsed'] = pd.to_datetime(comparison['Date'], format='%Y-%m')
    comparison = comparison.sort_values('DateParsed').reset_index(drop=True)

    comparison['CDS_minus_ERA5'] = comparison['CDS_FWI_95'] - comparison['ERA5_FWI_95']
    comparison['XClim_minus_ERA5'] = comparison['XClim_FWI_95'] - comparison['ERA5_FWI_95']
    comparison['CDS_minus_XClim'] = comparison['CDS_FWI_95'] - comparison['XClim_FWI_95']

    # Save combined CSV
    combined_csv_path = os.path.join(export_dir, f'FWI_3way_AllMonths_{region_name}.csv')
    comparison.drop(columns=['DateParsed']).to_csv(combined_csv_path, index=False)
    print(f"  Saved 3-way CSV: {combined_csv_path}")

    # ===== ALL MONTHS =====
    print(f"\n  ===== All Months =====")
    compute_pairwise_stats(comparison, PAIRS)
    print_source_summary(comparison)
    plot_timeseries(comparison, region_name, pct, 'allmonths')
    plot_scatter(comparison, region_name, 'allmonths')

    # ===== KEY MONTHS ONLY =====
    key_mask = comparison['DateParsed'].dt.month.isin(key_months)
    key_df = comparison[key_mask].copy()
    n_key = len(key_df)
    print(f"\n  ===== Key Months: {key_months_label} ({n_key} entries) =====")

    if n_key > 0:
        compute_pairwise_stats(key_df, PAIRS)
        print_source_summary(key_df)
        plot_timeseries(key_df, region_name, pct, f'keymonths',
                        title_extra=f' — {key_months_label} only')
        plot_scatter(key_df, region_name, f'keymonths')

        # Save key-months CSV
        key_csv_path = os.path.join(export_dir,
                                    f'FWI_3way_KeyMonths_{region_name}.csv')
        key_df.drop(columns=['DateParsed']).to_csv(key_csv_path, index=False)
        print(f"  Saved key-months CSV: {key_csv_path}")
    else:
        print(f"  No data for key months {key_months_label}")

print("\nDone.")
