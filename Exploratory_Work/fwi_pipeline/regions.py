"""Helper for building the per-region dict passed to SummaryMetric.compute().

Region definitions (shape_name, months) are user-defined parameters and live
in each dataset's YAML config under `summary.regions`, not here."""
from __future__ import annotations


def get_region(name: str, region_def: dict, start_year: int, end_year: int) -> dict:
    cfg = dict(region_def)
    cfg["start_year"] = start_year
    cfg["end_year"] = end_year
    return cfg
