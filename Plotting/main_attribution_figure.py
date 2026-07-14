import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('ticks')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3

# --- Region configurations ---
DATA_DIR = '/data/scratch/bob.potts/sowf/test_output/Exports'
PLOT_DIR = '/data/scratch/bob.potts/sowf/test_output/Plots'

regions = {
    'Canada': {
        'file': f'{DATA_DIR}/Canada_Attribution_FP.ods',
        'title': 'FWI$_{95}$ - Midwestern Canadian Shield Forests',
        'has_synthesis': True,
        'reanalysis_style': 'box',       # normal box
        'future_red_line': 5,
        'fut_clip_upper': {'3p0': 5},
    },
    'Chile': {
        'file': f'{DATA_DIR}/Chile_Attribution_FP.ods',
        'title': 'FWI$_{95}$ - Chilean Temperature Forests and Matorral',
        'has_synthesis': True,
        'reanalysis_style': 'arrow_up',
        'cap_ylim': 2000,
        'future_red_line': 2,
    },
    'Iberia': {
        'file': f'{DATA_DIR}/Iberia_Attribution_FP.ods',
        'title': 'FWI$_{95}$ - Northwestern Iberia',
        'has_synthesis': False,
        'reanalysis_style': 'star',       # just a star at top
        'future_red_line': 10,
    },
}

# Colours
brown  = '#724B49'
teal   = '#0096A1'
indigo = '#7A44FF'


def load_data(filepath):
    """Load attribution data from ODS file."""
    df = pd.read_excel(filepath, engine='odf', header=None)
    hadgem3    = df.iloc[1, 1:6].astype(float).tolist()
    reanalysis = df.iloc[2, 1:6].astype(float).tolist()
    models     = df.iloc[3, 1:6].astype(float).tolist()
    synthesis  = df.iloc[4, 1:6].astype(float).tolist()
    fut_1p5 = df.iloc[8, 1:6].astype(float).tolist()
    fut_2p0 = df.iloc[9, 1:6].astype(float).tolist()
    fut_3p0 = df.iloc[10, 1:6].astype(float).tolist()
    return {
        'reanalysis': reanalysis,
        'models': models,
        'synthesis': synthesis,
        'hadgem3_median': hadgem3[2],
        'fut_1p5': fut_1p5,
        'fut_2p0': fut_2p0,
        'fut_3p0': fut_3p0,
    }


def draw_box(ax, pos, data, color, width=0.6, clip_upper=None):
    """Draw a box-and-whisker from [p5, p25, median, p75, p95]."""
    p5, p25, med, p75, p95 = data
    p95_top = min(p95, clip_upper) if clip_upper is not None else p95
    box = mpatches.FancyBboxPatch(
        (pos - width/2, p25), width, p75 - p25,
        boxstyle="square,pad=0", facecolor=color, alpha=0.7, linewidth=1.2
    )
    ax.add_patch(box)
    ax.plot([pos - width/2, pos + width/2], [med, med], color='black', linewidth=2, zorder=5)
    ax.plot([pos, pos], [p5, p25], color=color, linewidth=1.2, zorder=4)
    ax.plot([pos, pos], [p75, p95_top], color=color, linewidth=1.2, zorder=4)
    cap_w = width * 0.4
    ax.plot([pos - cap_w/2, pos + cap_w/2], [p5, p5], color=color, linewidth=1.2, zorder=4)
    ax.plot([pos - cap_w/2, pos + cap_w/2], [p95_top, p95_top], color=color, linewidth=1.2, zorder=4)


def draw_box_arrow_up(ax, pos, data, color, width=0.6):
    """Draw a box clipped to visible area, without top whisker."""
    p5, p25, med, p75, p95 = data
    ylim = ax.get_ylim()
    # Clip box top to visible area
    box_top = min(p75, ylim[1])
    box = mpatches.FancyBboxPatch(
        (pos - width/2, p25), width, box_top - p25,
        boxstyle="square,pad=0", facecolor=color, alpha=0.7, linewidth=1.2
    )
    ax.add_patch(box)
    # Only draw median if it's in visible range
    if med <= ylim[1]:
        ax.plot([pos - width/2, pos + width/2], [med, med], color='black', linewidth=2, zorder=5)
    # Bottom whisker
    ax.plot([pos, pos], [p5, p25], color=color, linewidth=1.2, zorder=4)
    cap_w = width * 0.4
    ax.plot([pos - cap_w/2, pos + cap_w/2], [p5, p5], color=color, linewidth=1.2, zorder=4)


def add_triangle_cap(ax, pos, color, width=0.6):
    """Add triangle ^ cap at top of visible plot area. Call after set_ylim."""
    ylim = ax.get_ylim()
    box_top = ylim[1]
    tri_height = ylim[1] * 1.15
    triangle = plt.Polygon(
        [[pos - width/2, box_top], [pos + width/2, box_top], [pos, tri_height]],
        closed=True, facecolor=color, edgecolor=color, alpha=0.7, zorder=4,
        clip_on=False
    )
    ax.add_patch(triangle)


def plot_region(data, cfg):
    """Create the attribution figure for one region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5))
    fig.suptitle(cfg['title'], fontsize=14, y=1)
    fig.subplots_adjust(top=0.750)

    # ---- Left plot: Present Day ----
    ax1.set_yscale('log')

    # Auto-scale y-axis based on relevant data
    all_vals = []
    if cfg['reanalysis_style'] == 'box':
        all_vals.extend(data['reanalysis'])
    elif cfg['reanalysis_style'] == 'arrow_up':
        # Use p25-p75 for scaling, cap_ylim sets upper limit
        all_vals.extend(data['reanalysis'][1:4])  # p25, median, p75
    all_vals.extend(data['models'])
    if cfg['has_synthesis']:
        all_vals.extend(data['synthesis'])
    all_vals = [v for v in all_vals if v is not None and not np.isnan(v) and v > 0]
    ymin = min(all_vals) * 0.5
    ymax = max(all_vals) * 1.5
    if 'cap_ylim' in cfg:
        ymax = cfg['cap_ylim']
    ax1.set_ylim(ymin, ymax)

    # Determine positions based on whether synthesis is present
    if cfg['has_synthesis']:
        pos_re, pos_mo, pos_sy = 1, 1.7, 2.4
        xlim = (0.5, 2.9)
        xticks = [pos_re, pos_mo, pos_sy]
        xlabels = ['Reanalysis', 'Models', 'Synthesis']
    else:
        pos_re, pos_mo = 1, 1.7
        xlim = (0.5, 2.2)
        xticks = [pos_re, pos_mo]
        xlabels = ['Reanalysis', 'Models']

    # Draw reanalysis based on style
    if cfg['reanalysis_style'] == 'box':
        draw_box(ax1, pos_re, data['reanalysis'], brown)
    elif cfg['reanalysis_style'] == 'arrow_up':
        draw_box_arrow_up(ax1, pos_re, data['reanalysis'], brown)
        add_triangle_cap(ax1, pos_re, brown)
    elif cfg['reanalysis_style'] == 'star':
        # Star marker just inside the top of the plot
        ax1.plot(pos_re, ymax * 0.9, '*', color=brown, markersize=15, zorder=6)

    # Models box
    draw_box(ax1, pos_mo, data['models'], teal)

    # Synthesis box (if present)
    if cfg['has_synthesis']:
        draw_box(ax1, pos_sy, data['synthesis'], indigo)

    # HadGEM3 median dot on Models box
    ax1.plot(pos_mo, data['hadgem3_median'], 'ko', markersize=7, zorder=6)

    # Reference lines
    # Reference lines and ticks - adapt to data range
    ax1.axhline(y=1, color='grey', linestyle='-', linewidth=0.8, alpha=0.7)
    # Build major ticks based on what's in range
    major_ticks = [1]
    major_labels = ['No Change']
    for val in [10, 100, 1000]:
        if val <= ymax:
            ax1.axhline(y=val, color='grey', linestyle='--', linewidth=0.6, alpha=0.7)
            major_ticks.append(val)
            major_labels.append(str(val))

    ax1.set_yticks(major_ticks)
    ax1.set_yticklabels(major_labels)
    # Minor ticks within visible range
    minor_vals = [v for v in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                              2, 3, 4, 5, 6, 7, 8, 9,
                              20, 30, 40, 50, 60, 70, 80, 90,
                              200, 300, 400, 500, 600, 700, 800, 900]
                  if ymin <= v <= ymax]
    ax1.yaxis.set_minor_locator(ticker.FixedLocator(minor_vals))
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.tick_params(axis='y', which='minor', length=3, direction='out')
    ax1.set_xlim(*xlim)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel('Probability Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('a) Past to Present', loc='left', fontsize=12)
    ax1.grid(axis='y', which='major', alpha=0.5)

    # ---- Right plot: Future Projections ----
    ax2.set_yscale('log')

    clips = cfg.get('fut_clip_upper', {})
    draw_box(ax2, 1, data['fut_1p5'], teal, clip_upper=clips.get('1p5'))
    draw_box(ax2, 1.7, data['fut_2p0'], teal, clip_upper=clips.get('2p0'))
    draw_box(ax2, 2.4, data['fut_3p0'], teal, clip_upper=clips.get('3p0'))

    red_line = cfg['future_red_line']
    ax2.axhline(y=1, color='grey', linestyle='-', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=2, color='grey', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.axhline(y=red_line, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

    # Auto-scale right plot y-axis based on data and red line
    fut_vals = data['fut_1p5'] + data['fut_2p0'] + data['fut_3p0']
    fut_vals = [v for v in fut_vals if v is not None and not np.isnan(v) and v > 0]
    fut_ymax = max(max(fut_vals), red_line) * 1.3
    fut_ymin = min(fut_vals) * 0.7

    # Align No Change (y=1) between plots:
    # On log scale, fractional position of y=1 is: -log(ymin) / (log(ymax) - log(ymin))
    # Match right plot ymin so y=1 is at same fractional position as left plot
    left_frac = -np.log(ymin) / (np.log(ymax) - np.log(ymin))
    # Solve for fut_ymin: left_frac = -log(fut_ymin) / (log(fut_ymax) - log(fut_ymin))
    # => left_frac * log(fut_ymax) - left_frac * log(fut_ymin) = -log(fut_ymin)
    # => left_frac * log(fut_ymax) = log(fut_ymin) * (left_frac - 1)
    # => log(fut_ymin) = left_frac * log(fut_ymax) / (left_frac - 1)
    aligned_fut_ymin = np.exp(left_frac * np.log(fut_ymax) / (left_frac - 1))
    ax2.set_ylim(aligned_fut_ymin, fut_ymax)

    # Build ticks for right plot
    right_ticks = [1]
    right_labels = ['No Change']
    for val in [2, 5, red_line]:
        if val not in right_ticks and val <= fut_ymax:
            right_ticks.append(val)
            right_labels.append(str(val))
    ax2.set_yticks(sorted(right_ticks))
    ax2.set_yticklabels([right_labels[right_ticks.index(t)] for t in sorted(right_ticks)])

    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.set_xlim(0.5, 2.9)
    ax2.set_xticks([1, 1.7, 2.4])
    ax2.set_xticklabels(['+1.5°C', '+2.0°C', '+3.0°C'])
    ax2.set_title('b) Present to Future', loc='left', fontsize=12)
    ax2.grid(axis='y', which='major', alpha=0.3)

    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    return fig


# --- Main ---
import os
os.makedirs(PLOT_DIR, exist_ok=True)

for region_name, cfg in regions.items():
    print(f"Plotting {region_name}...")
    data = load_data(cfg['file'])
    fig = plot_region(data, cfg)
    save_path = os.path.join(PLOT_DIR, f'{region_name}_Attribution_FP_Reduced_HG3_Values.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {save_path}")
    #plt.show()
