"""
Plotting for the probability-ratio framework: density-histogram grid + summary
panel for risk ratio (per Plotting/Explore_Risk_Ratio.py), box-and-whisker
for intensity amplification (per Plotting/Intensity_Amplification.py), and the
5-panel supplement figure (per Plotting/Supplements/Supplement2_5Panel_ReducedSet.py).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

mpl.rcParams["font.family"] = "Work Sans"

from attribution_pipeline.bias_correction.baseline import load_baseline_series
from attribution_pipeline.bias_correction.member_loader import paired_members
from attribution_pipeline.bias_correction.regression import (
    bias_correct,
    find_regression_parameters,
    inverse_soft_log,
)
from attribution_pipeline.metrics.run_metrics import METRICS
from attribution_pipeline.pipeline_config import (
    BIAS_CORRECTED_METRICS,
    REGION_CONFIGS,
    UNCORRECTED_METRICS,
    get_region,
)
from attribution_pipeline.probability_ratio.ensemble import EnsembleLoader
from attribution_pipeline.probability_ratio.threshold import get_era5_threshold

# Reference target year for the illustrative single-year regression shown in
# panels (b)/(c)
SUPPLEMENT_TARGET_YEAR = 2024

BIAS_CORRECTED_FOLDER = BIAS_CORRECTED_METRICS
UNCORRECTED_FOLDER = UNCORRECTED_METRICS


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
        sns.histplot(
            res["hist_data"],
            kde=True,
            color="#C7403D",
            label="Factual (Current Climate)",
            alpha=0.5,
            ax=ax,
            stat="density",
        )
        sns.histplot(
            res["nat_data"],
            kde=True,
            color="#008787",
            label="Counterfactual (Natural Only Climate)",
            alpha=0.5,
            ax=ax,
            stat="density",
        )
        ax.axvline(
            x=res["threshold"],
            color="black",
            linewidth=2.5,
            label=f"ERA5 {region['month_name']} {region['event_year']}",
        )
        ax.set_title(
            f"{region['display_name']}\n{metric_stem} {region['month_name']}"
        )
        ax.set_xlabel(metric_stem)
        if idx % ncols == 0:
            ax.set_ylabel("Density")
        if idx == n - 1:
            ax.legend()

    summary_ax = axes[-1] if n < len(axes) else axes[n]
    summary_ax.axis("off")
    lines = ["SUMMARY OF RESULTS", ""]
    for country, res in results.items():
        lines.append(
            f"{country}: RR = {res['median']:.2f} [{res['ci_5']:.2f} - {res['ci_95']:.2f}]"
        )
    summary_ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
        family="monospace",
    )

    for extra_ax in axes[len(countries) + 1 :]:
        extra_ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_amplification(
    amplification_results: dict, metric_stem: str, out_path: str
):
    """amplification_results: {country: result_dict} from core.compute_region_amplification."""
    countries = [
        c
        for c in amplification_results
        if len(amplification_results[c]["amplification"]) > 0
    ]
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
        bp = ax.bxp(
            [stats],
            positions=[0],
            widths=[0.5],
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
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
        ax.set_title(
            f"{region['display_name']}\n{region['month_name']}", fontsize=11
        )
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        if i == 0:
            ax.set_ylabel(
                f"Intensity Amplification\n({metric_stem} Difference)",
                fontsize=11,
            )

    fig.suptitle(
        f"Intensity Amplification: Factual - Counterfactual {metric_stem}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- 5-panel supplement (per Plotting/Supplements/Supplement2_5Panel_ReducedSet.py) ----

N_BASELINE_MEMBERS = 15  # HadGEM3-historical baseline realisations (1..15).


def _load_all_baseline_members(
    country,
    metric_stem,
    historical_source,
    baseline_start,
    baseline_end,
    n_baseline_members=N_BASELINE_MEMBERS,
):
    """Loads ERA5 (once) and every available HadGEM3-historical baseline member's
    soft-logged series. Returns (years, era5_log, {member: hg3_log})."""
    hg3_dataset = (
        "hg3_historical_impacttb"
        if historical_source == "impacttb"
        else "hg3_historical_xclim"
    )
    era5_years, era5_log = load_baseline_series(
        "era5", country, metric_stem, start=baseline_start, end=baseline_end
    )

    hg3_logs = {}
    for member in range(1, n_baseline_members + 1):
        try:
            hg3_years, hg3_log = load_baseline_series(
                hg3_dataset,
                country,
                metric_stem,
                member=member,
                start=baseline_start,
                end=baseline_end,
            )
        except FileNotFoundError as e:
            print(f"[plot_supplement] Warning: {e}")
            continue
        if not np.array_equal(era5_years, hg3_years):
            print(
                f"[plot_supplement] Warning: baseline member {member} years differ from ERA5; skipping"
            )
            continue
        hg3_logs[member] = hg3_log

    return era5_years, era5_log, hg3_logs


def _bias_correct_baseline_members(
    years, era5_log, hg3_logs, target_year=SUPPLEMENT_TARGET_YEAR
):
    """Fits obs/sim regressions per HadGEM3-historical member against `target_year`
    and returns (era5_raw, hg3_raw_per_member, detrended_raw_per_member)."""
    t = years - target_year
    fwi0_obs, _delta_obs, _ = find_regression_parameters(era5_log, t)

    hg3_raw = []
    detrended_raw = []
    for hg3_log in hg3_logs.values():
        fwi0_sim, delta_sim, _ = find_regression_parameters(hg3_log, t)
        detrended_log = bias_correct(hg3_log, t, fwi0_obs, delta_sim, fwi0_sim)
        hg3_raw.append(inverse_soft_log(hg3_log))
        detrended_raw.append(inverse_soft_log(detrended_log))

    era5_raw = inverse_soft_log(era5_log)
    return era5_raw, hg3_raw, detrended_raw


def _plot_pdf_pair(
    ax,
    sim_arr,
    obs_arr,
    threshold,
    title,
    sim_label,
    obs_label,
    threshold_label,
):
    if len(sim_arr) > 0:
        sns.histplot(
            np.ravel(sim_arr),
            kde=True,
            color="#7A44FF",
            label=sim_label,
            alpha=0.5,
            ax=ax,
            stat="density",
        )
    if len(obs_arr) > 0:
        sns.histplot(
            obs_arr,
            kde=True,
            color="#E98400",
            label=obs_label,
            alpha=0.5,
            ax=ax,
            stat="density",
        )
    if threshold is not None:
        ax.axvline(
            x=threshold, color="black", linewidth=2.5, label=threshold_label
        )
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(loc="best")


def _plot_timeseries(ax, years, era5_raw, hg3_raw, detrended_raw, title):
    ax.plot(years, era5_raw, label="ERA5", color="blue", linewidth=1.5)

    if hg3_raw:
        hg3_arr = np.array(hg3_raw)
        hg3_mean, hg3_std = hg3_arr.mean(axis=0), hg3_arr.std(axis=0)
        ax.plot(
            years, hg3_mean, label="HadGEM3 (mean)", color="red", linewidth=1.5
        )
        ax.fill_between(
            years,
            hg3_mean - hg3_std,
            hg3_mean + hg3_std,
            color="red",
            alpha=0.2,
        )

    if detrended_raw:
        det_arr = np.array(detrended_raw)
        det_mean, det_std = det_arr.mean(axis=0), det_arr.std(axis=0)
        ax.plot(
            years,
            det_mean,
            label="Detrended & Shifted (mean)",
            color="purple",
            linewidth=1.5,
        )
        ax.fill_between(
            years,
            det_mean - det_std,
            det_mean + det_std,
            color="purple",
            alpha=0.2,
        )

    ax.set_xlim(years.min(), years.max())
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)


def _plot_factual_counterfactual(
    ax, hist_data, nat_data, threshold, title, threshold_label, xlabel=""
):
    if len(hist_data) > 0:
        sns.histplot(
            hist_data,
            kde=True,
            color="#C7403D",
            label="Factual (Current Climate)",
            alpha=0.5,
            ax=ax,
            stat="density",
        )
    if len(nat_data) > 0:
        sns.histplot(
            nat_data,
            kde=True,
            color="#008787",
            label="Counterfactual (Natural Only Climate)",
            alpha=0.5,
            ax=ax,
            stat="density",
        )
    if threshold is not None:
        ax.axvline(
            x=threshold, color="black", linewidth=2.5, label=threshold_label
        )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize="small")


def plot_supplement(
    country: str,
    index: str,
    metric_name: str,
    out_path: str,
    percentile: float = 95,
    historical_source: str = "xclim",
    paired_only: bool = True,
    n_baseline_members: int = N_BASELINE_MEMBERS,
):
    """5-panel supplement figure for one region:
    a) uncorrected baseline PDF (ERA5 vs HadGEM3-historical, all baseline members)
    b) bias-corrected baseline PDF (ERA5 vs detrended & shifted HadGEM3-historical)
    c) baseline timeseries (ERA5, HadGEM3 mean+/-std, detrended mean+/-std)
    d) uncorrected attribution ensemble (factual vs counterfactual, pooled over
       all of the region's bias_correction_years)
    e) bias-corrected attribution ensemble (factual vs counterfactual, pooled
       over all of the region's bias_correction_years)
    """
    region = get_region(country)
    month_name = region["month_name"]
    event_year = region["event_year"]
    baseline_start = region["baseline_start"]
    baseline_end = region["baseline_end"]

    metric_stem = METRICS[metric_name](
        index, percentile=percentile
    ).output_stem()
    print(
        f"[plot_supplement] country={country} index={index} metric={metric_stem} "
        f"historical_source={historical_source}"
    )

    try:
        era5_threshold = get_era5_threshold(country, event_year, metric_stem)
    except (FileNotFoundError, ValueError) as e:
        print(
            f"[plot_supplement] Warning: could not compute ERA5 threshold: {e}"
        )
        era5_threshold = None
    threshold_label = f"ERA5 {month_name} {event_year}"

    # Panels (a)-(c): baseline regression across all HadGEM3-historical members.
    years, era5_log, hg3_logs = _load_all_baseline_members(
        country,
        metric_stem,
        historical_source,
        baseline_start,
        baseline_end,
        n_baseline_members,
    )
    era5_raw, hg3_raw, detrended_raw = _bias_correct_baseline_members(
        years, era5_log, hg3_logs
    )
    hg3_arr = np.concatenate(hg3_raw) if hg3_raw else np.array([])
    detrended_arr = (
        np.concatenate(detrended_raw) if detrended_raw else np.array([])
    )

    # Panels (d)/(e): attribution ensemble, pooled over the region's bias_correction_years.
    paired = paired_members(index) if paired_only else None

    uncorrected_loader = EnsembleLoader(
        UNCORRECTED_FOLDER,
        baseline_start,
        baseline_end,
        metric_stem=metric_stem,
        percentile=percentile,
        mode="uncorrected",
    )
    hist_uncorrected, _ = uncorrected_loader.load(
        country, "hist", member_filter=paired
    )
    nat_uncorrected, _ = uncorrected_loader.load(
        country, "histnat", member_filter=paired
    )

    corrected_folder = os.path.join(BIAS_CORRECTED_FOLDER, historical_source)
    corrected_loader = EnsembleLoader(
        corrected_folder,
        baseline_start,
        baseline_end,
        metric_stem=metric_stem,
        percentile=percentile,
        mode="corrected",
    )
    hist_corrected, _ = corrected_loader.load(
        country, "hist", member_filter=paired
    )
    nat_corrected, _ = corrected_loader.load(
        country, "histnat", member_filter=paired
    )

    # Build figure.
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    _plot_pdf_pair(
        ax_a,
        hg3_arr,
        era5_raw,
        era5_threshold,
        f"a) {month_name} {baseline_start}-{baseline_end} (Uncorrected)",
        "HadGEM3",
        "ERA5",
        threshold_label,
    )
    _plot_pdf_pair(
        ax_b,
        detrended_arr,
        era5_raw,
        era5_threshold,
        f"b) {month_name} {baseline_start}-{baseline_end} (Corrected)",
        "HadGEM3 (Corrected)",
        "ERA5",
        threshold_label,
    )
    _plot_timeseries(
        ax_c,
        years,
        era5_raw,
        hg3_raw,
        detrended_raw,
        f"c) {month_name} Time Series of {metric_stem} and Detrended & Shifted {metric_stem}",
    )
    _plot_factual_counterfactual(
        ax_d,
        hist_uncorrected,
        nat_uncorrected,
        era5_threshold,
        f"d) {month_name} {event_year} (Uncorrected)",
        threshold_label,
        xlabel=metric_stem,
    )
    _plot_factual_counterfactual(
        ax_e,
        hist_corrected,
        nat_corrected,
        era5_threshold,
        f"e) {month_name} {event_year} (Corrected)",
        threshold_label,
        xlabel=metric_stem,
    )

    plt.suptitle(
        f"{region['display_name']} {percentile:g}th percentile {metric_stem}",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_supplement] Saved: {out_path}")
    return out_path


def generate_all_supplements(
    index: str,
    metric_name: str,
    out_dir: str,
    percentile: float = 95,
    historical_source: str = "xclim",
    paired_only: bool = True,
):
    """Runs plot_supplement() for every region in REGION_CONFIGS (dynamic region
    count -- no hardcoded country list). Outputs are nested under
    out_dir/{historical_source}/, mirroring bias_correction's
    bias_corrected_metrics/{historical_source}/ layout."""
    source_dir = os.path.join(out_dir, historical_source)
    written = []
    for country in REGION_CONFIGS:
        out_path = os.path.join(source_dir, f"Supplement_{country}.png")
        try:
            written.append(
                plot_supplement(
                    country,
                    index,
                    metric_name,
                    out_path,
                    percentile=percentile,
                    historical_source=historical_source,
                    paired_only=paired_only,
                )
            )
        except Exception as e:  # noqa: BLE001 -- intentionally skip any failing
            # country so one bad region doesn't abort the whole batch run.
            print(f"[plot_supplement] Error processing {country}: {e}")
            continue
    return written
