"""
Central pipeline configuration -- the single source of truth for per-region
event/fire-season definitions, the shared observed/simulated baseline period,
and the HadGEM3-A Attribution ensemble's raw-data read window.

Mirrors (and replaces the duplicated) REGION_CONFIGS blocks previously
scattered across Plotting/Explore_Risk_Ratio.py, Plotting/Uncorrected_Risk_Ratio.py,
Exploratory_Work/Reduced_Att_Set_Processing/reduced_set_risk_ratio.py,
Plotting/Supplements/*.py, and post-processing/Metric-HG3-A_Historical/FWI95-HG3-A_Historical.py.

Used by attribution_pipeline/metrics/ (interim metric CSVs),
attribution_pipeline/bias_correction/ (baseline regression + read window), and
attribution_pipeline/probability_ratio/ (risk ratio / amplification).
"""

# HadGEM3-A Attribution raw-data read window: a dataset-availability constraint
# on the whole ensemble (hurs single-month files only start at 201911, and all
# four variables have continuous single-month files 201911..202502) -- global,
# not region-specific, since it gates which raw files the loader reads before
# any region/country is selected.
WINDOW_START_MONTH = 201911
WINDOW_END_MONTH = 202412
WINDOW_START = "2019-11-01"
WINDOW_END = "2024-12-30"

REGION_CONFIGS = {
    "Korea": {
        "months": (3,),
        "month_name": "March",
        "shape_name": "Southeast South Korea",
        "display_name": "SE S. Korea",
        "event_year": 2025,
        "percentile": 95,
        "baseline_start": 1980,
        "baseline_end": 2013,
        "bias_correction_years": (2020, 2021, 2022, 2023, 2024),
    },
    "Iberia": {
        "months": (8,),
        "month_name": "Aug",
        "shape_name": "Northwest Iberia",
        "display_name": "NW Iberia",
        "event_year": 2025,
        "percentile": 95,
        "baseline_start": 1980,
        "baseline_end": 2013,
        "bias_correction_years": (2020, 2021, 2022, 2023, 2024),
    },
    "Scotland": {
        "months": (6, 7),
        "month_name": "June-July",
        "shape_name": "Scottish Highlands",
        "display_name": "Scottish Highlands",
        "event_year": 2025,
        "percentile": 95,
        "baseline_start": 1980,
        "baseline_end": 2013,
        "bias_correction_years": (2020, 2021, 2022, 2023, 2024),
    },
    "Chile": {
        "months": (1, 2),
        "month_name": "January-February",
        "shape_name": "Chilean Temperate Forests and Matorral",
        "display_name": "Chile Forests & Matorral",
        "event_year": 2026,
        "percentile": 95,
        "baseline_start": 1980,
        "baseline_end": 2013,
        # Chile's spin-up requirement excludes 2020.
        "bias_correction_years": (2021, 2022, 2023, 2024),
    },
    "Canada": {
        "months": (7, 8),
        "month_name": "July-August",
        "shape_name": "Midwestern Canadian Shield forests",
        "display_name": "Canadian Shield Forests",
        "event_year": 2025,
        "percentile": 95,
        "baseline_start": 1980,
        "baseline_end": 2013,
        "bias_correction_years": (2020, 2021, 2022, 2023, 2024),
    },
}

SHAPEFILE = "/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp"


def get_region(country: str) -> dict:
    if country not in REGION_CONFIGS:
        raise ValueError(f"Unknown Country: {country}. Expected one of: {sorted(REGION_CONFIGS)}")
    return REGION_CONFIGS[country]


def month_label(months) -> str:
    """e.g. (8,) -> 'Aug'; (6, 7) -> 'Jun-Jul'."""
    import calendar
    return "-".join(calendar.month_abbr[m] for m in months)
