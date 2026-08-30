"""
Plotting for the probability-ratio framework: density-histogram grid + summary
panel for risk ratio (per Plotting/Explore_Risk_Ratio.py), and box-and-whisker
for intensity amplification (per Plotting/Intensity_Amplification.py).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams["font.family"] = "Work Sans"

from attribution_pipeline.metrics.pipeline_config import get_region


def plot_risk_ratio_grid(results: dict, metric_stem: str, out_path: str):
    """results: {country: result_dict} as returned by core.compute_region_risk_ratio."""
    countries = list(results.keys())
    n = len(countries)
    ncols = 3
    nrows = -(-n // ncols) if n > 1 else 1
    nrows = max(nrows, 2)  # keep room for the summary panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, country in enumerate(countries):
        region = get_region(country)
        res = results[country]
        ax = axes[idx]
        sns.histplot(res["hist_data"], kde=True, color="#C7403D", label="Factual (Current Climate)",
                     alpha=0.5, ax=ax, stat="density")
        sns.histplot(res["nat_data"], kde=True, color="#008787", label="Counterfactual (Natural Only Climate)",
                     alpha=0.5, ax=ax, stat="density")
        ax.axvline(x=res["threshold"], color="black", linewidth=2.5,
                   label=f'ERA5 {region["month_name"]} {region["event_year"]}')
        ax.set_title(f'{region["display_name"]}\n{metric_stem} {region["month_name"]}')
        ax.set_xlabel(metric_stem)
        if idx % ncols == 0:
            ax.set_ylabel("Density")
        if idx == n - 1:
            ax.legend()

    summary_ax = axes[-1] if n < len(axes) else axes[n]
    summary_ax.axis("off")
    lines = ["SUMMARY OF RESULTS", ""]
    for country, res in results.items():
        lines.append(f"{country}: RR = {res['median']:.2f} [{res['ci_5']:.2f} - {res['ci_95']:.2f}]")
    summary_ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=12,
                     wrap=True, family="monospace")

    for extra_ax in axes[len(countries) + 1:]:
        extra_ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_amplification(amplification_results: dict, metric_stem: str, out_path: str):
    """amplification_results: {country: result_dict} from core.compute_region_amplification."""
    countries = [c for c in amplification_results if len(amplification_results[c]["amplification"]) > 0]
    n = len(countries)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 7), sharey=True)
    axes = np.atleast_1d(axes)

    box_colour = "#ee007f"
    mean_colour = "#0096a1"

    for i, country in enumerate(countries):
        region = get_region(country)
        data = amplification_results[country]["amplification"]
        ax = axes[i]
        stats = {
            "med": np.mean(data),
            "q1": np.percentile(data, 25),
            "q3": np.percentile(data, 75),
            "whislo": np.percentile(data, 5),
            "whishi": np.percentile(data, 95),
            "fliers": [],
        }
        bp = ax.bxp([stats], positions=[0], widths=[0.5], patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(box_colour)
            patch.set_alpha(0.4)
            patch.set_edgecolor(box_colour)
            patch.set_linewidth(1.5)
        for whisker in bp["whiskers"]:
            whisker.set_color(box_colour)
            whisker.set_linewidth(1.5)
        for cap in bp["caps"]:
            cap.set_color(box_colour)
            cap.set_linewidth(1.5)
        for median_line in bp["medians"]:
            median_line.set_color(mean_colour)
            median_line.set_linewidth(2.5)

        ax.axhline(y=0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(f'{region["display_name"]}\n{region["month_name"]}', fontsize=11)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        if i == 0:
            ax.set_ylabel(f"Intensity Amplification\n({metric_stem} Difference)", fontsize=11)

    fig.suptitle(f"Intensity Amplification: Factual - Counterfactual {metric_stem}", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
