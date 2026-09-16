"""
Per-(country, run_type, index, metric) uncorrected-scalar extraction: loads
each paired attribution-ensemble member once, then for every configured
target year writes each member's RAW (non-bias-corrected) scalar to a CSV.

Sibling to core.py's run_bias_correction(), but skips the soft_log/
bias_correct/inverse_soft_log steps entirely -- this is the "uncorrected"
counterpart used for supplement-figure panels that compare factual/
counterfactual ensembles before bias correction is applied.
"""

import os

import pandas as pd

from attribution_pipeline.bias_correction.member_loader import (
    InvalidMemberDataError,
    MissingMemberError,
    load_member_cube,
    paired_members,
)
from attribution_pipeline.bias_correction.metric_extract import extract_scalar
from attribution_pipeline.metrics.pipeline_config import get_region
from attribution_pipeline.metrics.run_metrics import METRICS

OUTPUT_DIR = "/data/scratch/bob.potts/sowf/attribution_pipeline/uncorrected_metrics"


def run_uncorrected_extraction(country: str, run_type: str, index: str, metric_name: str,
                                percentile: float = 95, **metric_kwargs):
    region = get_region(country)
    shape_name = region["shape_name"]
    months = region["months"]
    data_years = region["bias_correction_years"]

    metric_stem = METRICS[metric_name](index, percentile=percentile, **metric_kwargs).output_stem()
    print(f"[uncorrected] country={country} run_type={run_type} metric={metric_stem}")

    members = sorted(paired_members(index))
    print(f"[uncorrected] {len(members)} paired members available for index={index}")

    member_cubes = {}
    load_missing = []
    for member in members:
        try:
            member_cubes[member] = load_member_cube(index, run_type, member, shape_name)
        except MissingMemberError as e:
            load_missing.append((member, str(e)))
    print(f"[uncorrected] Loaded {len(member_cubes)}/{len(members)} member cubes "
          f"({len(load_missing)} missing on disk)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    written = []
    for data_year in data_years:
        col_names = list(member_cubes.keys())
        row = {}
        successful = []
        missing = list(load_missing)
        errors = []

        for member in col_names:
            cube = member_cubes[member]
            try:
                row[member] = extract_scalar(cube, months, data_year, metric_name, index,
                                              percentile=percentile, **metric_kwargs)
                successful.append(member)
            except MissingMemberError as e:
                missing.append((member, str(e)))
            except (InvalidMemberDataError, ValueError) as e:
                errors.append((member, str(e)))

        df_out = pd.DataFrame([row], columns=col_names)
        df_out.insert(0, "Year", data_year)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{country}_{metric_stem}_{run_type}{percentile:g}percent_Uncorrected_DataYear_{data_year}.csv",
        )
        df_out.to_csv(out_path, index=False)
        written.append(out_path)

        total = len(members)
        print(f"[uncorrected] DATA_YEAR={data_year}: {len(successful)}/{total} successful, "
              f"{len(missing)}/{total} missing, {len(errors)}/{total} errors -> {out_path}")

    return written
