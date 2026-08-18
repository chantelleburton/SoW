"""Config loading for the xclim-based FWI pipeline.

A single YAML file (`configs/pipeline.yaml`) holds settings shared across all
dataset variants (shapefile path, region definitions, scratch_root, summary
defaults) plus one block per variant under `datasets:`. `load_config(path,
dataset)` picks out the requested dataset's block and merges in the shared
fields, so paths/regions/shapefile are defined exactly once.

Three families exist (see PIPELINE_PLAN / session notes):
    - baseline    : ERA5HistoricalSource + HadGEM3HistoricalSource (1980-2013,
                    matched pair feeding bias correction together)
    - attribution : HadGEM3AttributionSource (525-member event ensemble)
    - present     : ERA5PresentSource (2024-2026 event period)

All three share the same summary-config shape so stage2 stays generic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import yaml


@dataclass
class SummaryConfig:
    method: str = "percentile"
    percentile: float = 95.0
    regions: dict = field(default_factory=dict)
    output_dir: str = ""
    shapefile: str = ""


@dataclass
class RunConfig:
    """Generic config shared by all dataset families.

    family-specific fields (e.g. `members`, `run_types`, `wind_stat`) are kept
    in `extra` rather than exploded into subclasses, since stage1/stage2 only
    need a handful of common fields (`dataset`, `source`, `metrics`,
    `interim_dir`, `summary`) plus whatever the chosen DataSource needs from
    `extra`.
    """
    dataset: str = ""
    family: str = ""
    source: str = ""
    metrics: list = field(default_factory=lambda: ["fwi"])
    interim_dir: str = ""
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-create output locations so users never have to make these by hand.
        if self.interim_dir:
            os.makedirs(self.interim_dir, exist_ok=True)
        if self.summary.output_dir:
            os.makedirs(self.summary.output_dir, exist_ok=True)

    def interim_filename(self, metric_name: str, **task_params) -> str:
        """Build the interim .nc path for a given metric + task parameters.

        Naming mirrors the conventions already used by the working scripts:
            era5_{metric}_{run_label}_{start}-{end}.nc
            hadgem3a_{metric}_{run_type}_{member}.nc
        Callers pass whatever task_params their family needs (member,
        run_type, start_year/end_year, run_label, ...).
        """
        if self.family == "attribution":
            run_type = task_params["run_type"]
            member = task_params["member"]
            return os.path.join(self.interim_dir, f"hadgem3a_{metric_name}_{run_type}_{member}.nc")
        if self.source == "hadgem3_historical":
            member = task_params["member"]
            start = self.extra["start_year"]
            end = self.extra["end_year"]
            return os.path.join(self.interim_dir, f"hadgem3a_historical_{metric_name}_r1i1p{member}_{start}-{end}.nc")
        # ERA5 present / historical
        run_label = task_params.get("run_label") or f"{self.extra.get('rh_stat', 'mean').capitalize()}_RH_" \
            f"{self.extra.get('wind_stat', 'mean').capitalize()}_Wind"
        start = task_params.get("start_year", self.extra.get("start_year"))
        end = task_params.get("end_year", self.extra.get("end_year"))
        return os.path.join(self.interim_dir, f"era5_{metric_name}_{run_label}_{start}-{end}.nc")

    def summary_filename(self, region: str, **task_params) -> str:
        pct = int(self.summary.percentile)
        start = self.extra.get("start_year")
        end = self.extra.get("end_year")
        if self.family == "baseline" and self.source == "hadgem3_historical":
            member = task_params["member"]
            return os.path.join(
                self.summary.output_dir,
                f"HadGEM3_FWI_{start}-{end}_{region}_{member}_{pct}%.csv",
            )
        if self.family == "baseline" and self.source == "era5_historical":
            return os.path.join(self.summary.output_dir, f"ERA5_FWI_{start}-{end}_{region}_{pct}%.csv")
        # present / attribution: keep dataset name in the filename to avoid clashes
        suffix = "_".join(str(v) for v in task_params.values())
        return os.path.join(self.summary.output_dir, f"{self.dataset}_FWI_{region}_{suffix}_{pct}%.csv")


def _derive_attribution_window_fields(extra: dict) -> dict:
    """hadgem3_attribution only specifies `window_start`/`window_end` (e.g.
    '2019-11-01'/'2024-12-30'). Everything else derived from that window
    (YYYYMM month bounds used for file selection, plus start_year/end_year
    used for filenames/region summaries) is computed here so it's defined in
    exactly one place instead of three.

    window_start is typically Nov/Dec lead-in data used only for DC/DMC
    spin-up (not a complete calendar year), so the first *complete* year is
    the following one unless window_start is 1 January. window_end is
    typically the final month of its year (e.g. Dec 30 in HadGEM3's 360-day
    calendar), so that year counts as complete unless window_end's month
    isn't December.
    """
    window_start = extra.get("window_start")
    window_end = extra.get("window_end")
    if window_start is None or window_end is None:
        return extra
    start_date = date.fromisoformat(str(window_start))
    end_date = date.fromisoformat(str(window_end))
    extra["window_start_month"] = start_date.year * 100 + start_date.month
    extra["window_end_month"] = end_date.year * 100 + end_date.month
    extra["start_year"] = start_date.year if start_date.month == 1 else start_date.year + 1
    extra["end_year"] = end_date.year if end_date.month == 12 else end_date.year - 1
    return extra


def load_config(path: str, dataset: str) -> RunConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    datasets_raw = raw.get("datasets", {})
    if dataset not in datasets_raw:
        raise KeyError(f"Unknown dataset {dataset!r}; known datasets: {sorted(datasets_raw)}")
    ds_raw: dict[str, Any] = dict(datasets_raw[dataset])

    scratch_root = raw.get("scratch_root", "")
    interim_dir = os.path.join(scratch_root, ds_raw.pop("interim_subdir"))
    output_dir = os.path.join(scratch_root, ds_raw.pop("output_subdir"))

    summary_raw = {**raw.get("summary_defaults", {}), **ds_raw.pop("summary", {})}
    summary = SummaryConfig(
        method=summary_raw.get("method", "percentile"),
        percentile=summary_raw.get("percentile", 95.0),
        regions=raw.get("regions", {}),
        output_dir=output_dir,
        shapefile=raw.get("shapefile", ""),
    )

    known_fields = {"family", "source", "metrics"}
    extra = {k: v for k, v in ds_raw.items() if k not in known_fields}
    extra = _derive_attribution_window_fields(extra)

    return RunConfig(
        dataset=dataset,
        family=ds_raw.get("family", ""),
        source=ds_raw.get("source", ""),
        metrics=ds_raw.get("metrics", ["fwi"]),
        interim_dir=interim_dir,
        summary=summary,
        extra=extra,
    )


def parse_cylc_task_params() -> dict:
    """Collect all CYLC_TASK_PARAM_* environment variables into a plain dict.

    Cylc auto-sets one CYLC_TASK_PARAM_<name> per parameter used directly in
    a task's name (e.g. hg3_attr_member, hg3_hist_member). Those raw,
    dataset-specific names are re-exported under clean aliases (member,
    run_type) directly in flow.cylc's task scripts, so the raw hg3_* ones are
    dropped here to avoid duplicate/conflicting values downstream.
    """
    prefix = "CYLC_TASK_PARAM_"
    return {
        k[len(prefix):]: v
        for k, v in os.environ.items()
        if k.startswith(prefix) and not k[len(prefix):].startswith("hg3_")
    }
