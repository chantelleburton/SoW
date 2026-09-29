"""
Used by attribution_pipeline/index_calculation/ (raw FWI/DSR generation),
attribution_pipeline/metrics/ (interim metric CSVs),
attribution_pipeline/bias_correction/ (baseline regression + read window), and
attribution_pipeline/probability_ratio/ (risk ratio / amplification / supplement figure).

Loads its values from config.json (sibling to this file) so paths and region
definitions can be edited without touching code. Override the JSON file
location via ATTRIBUTION_PIPELINE_CONFIG; override the output root via
ATTRIBUTION_PIPELINE_ROOT (as before).
"""

import json
import os

_CONFIG_PATH = os.environ.get(
    "ATTRIBUTION_PIPELINE_CONFIG",
    os.path.join(os.path.dirname(__file__), "config.json"),
)
with open(_CONFIG_PATH) as _f:
    _CONFIG = json.load(_f)

# --- Pipeline output root ---------------------------------------------------
# Every attribution_pipeline-generated file (raw FWI/DSR, interim metric CSVs,
# bias-corrected/uncorrected ensemble CSVs, risk-ratio/amplification/supplement
# exports) lives under this single root. Override via env var to relocate
PIPELINE_ROOT = os.environ.get(
    "ATTRIBUTION_PIPELINE_ROOT",
    _CONFIG["pipeline_root_default"],
)

# index_calculation/ raw FWI/DSR output (one subfolder per data source).
# Each entry may be an absolute path  or a path relative to
# PIPELINE_ROOT (the default). Falls back to the pre-config.json convention if
# config.json doesn't have a raw_fwi_output section (older config.json files).
_RAW_FWI = _CONFIG.get("raw_fwi_output", {})
RAW_FWI_ERA5 = os.path.join(
    PIPELINE_ROOT, _RAW_FWI.get("era5", "raw_fwi/era5")
)
RAW_FWI_HG3_HISTORICAL = os.path.join(
    PIPELINE_ROOT, _RAW_FWI.get("hg3_historical", "raw_fwi/hg3_historical")
)
RAW_FWI_HG3_ATTRIBUTION = os.path.join(
    PIPELINE_ROOT, _RAW_FWI.get("hg3_attribution", "raw_fwi/hg3_attribution")
)

# metrics/ interim per-year metric CSVs (read by bias_correction/ and
# probability_ratio/ alike
METRICS_OUT_DIR = os.path.join(PIPELINE_ROOT, "metrics")

# bias_correction/ ensemble CSV output.
BIAS_CORRECTED_METRICS = os.path.join(PIPELINE_ROOT, "bias_corrected_metrics")
UNCORRECTED_METRICS = os.path.join(PIPELINE_ROOT, "uncorrected_metrics")

# probability_ratio/ final exports (summary CSVs + plots), nested by
# {historical_source}
EXPORTS = os.path.join(PIPELINE_ROOT, "exports")

# --- External, read-only data sources ---------------------------------------
_EXTERNAL = _CONFIG["external_data_sources"]
ERA5_OBS_BASEPATH = _EXTERNAL["era5_obs_basepath"]
IMPACTTB_HISTORICAL_FWI_DIR = _EXTERNAL["impacttb_historical_fwi_dir"]
SHAPEFILE = _EXTERNAL["shapefile"]

# HadGEM3-A Attribution raw-data read window: a dataset-availability constraint
# on the whole ensemble (hurs single-month files only start at 201911, and all
# four variables have continuous single-month files 201911..202502) -- global,
# not region-specific, since it gates which raw files the loader reads before
# any region/country is selected.
_WINDOW = _CONFIG["hadgem3_attribution_window"]
WINDOW_START_MONTH = _WINDOW["start_month"]
WINDOW_END_MONTH = _WINDOW["end_month"]
WINDOW_START = _WINDOW["start"]
WINDOW_END = _WINDOW["end"]

# Region/country definitions. All regions always live here; which ones
# actually get run is controlled by the Rose-suite COUNTRY task parameter
# in metrics/rose-suite.conf and bias_correction/rose-suite.conf, not here.
REGION_CONFIGS = _CONFIG["regions"]


def get_region(country: str) -> dict:
    if country not in REGION_CONFIGS:
        raise ValueError(
            f"Unknown Country: {country}. Expected one of: {sorted(REGION_CONFIGS)}"
        )
    return REGION_CONFIGS[country]


def month_label(months) -> str:
    """e.g. (8,) -> 'Aug'; (6, 7) -> 'Jun-Jul'."""
    import calendar

    return "-".join(calendar.month_abbr[m] for m in months)

