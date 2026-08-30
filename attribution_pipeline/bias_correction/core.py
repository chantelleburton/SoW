"""
Per-(country, baseline_member, run_type, index, metric) orchestration:
fits the obs/sim baseline regression, loads each paired attribution-ensemble
member once, then for every configured target year bias-corrects each
member's scalar and writes a CSV -- generalizing
Exploratory_Work/Reduced_Att_Set_Processing/reduced_set_HG_bias_correction.py.
"""

import os

import numpy as np
import pandas as pd

from attribution_pipeline.bias_correction.baseline import load_baseline_series
from attribution_pipeline.bias_correction.member_loader import (
    InvalidMemberDataError,
    MissingMemberError,
    load_member_cube,
    paired_members,
)
from attribution_pipeline.bias_correction.metric_extract import extract_scalar
from attribution_pipeline.bias_correction.regression import (
    bias_correct,
    find_regression_parameters,
    inverse_soft_log,
    soft_log,
)
from attribution_pipeline.metrics.pipeline_config import get_region
from attribution_pipeline.metrics.run_metrics import METRICS

OUTPUT_DIR = "/data/scratch/bob.potts/sowf/attribution_pipeline/bias_corrected_metrics"
DEFAULT_DATA_YEARS = (2020, 2021, 2022, 2023, 2024)


def run_bias_correction(country: str, baseline_member: int, run_type: str, index: str,
                         metric_name: str, percentile: float = 95, **metric_kwargs):
    region = get_region(country)
    shape_name = region["shape_name"]
    months = region["months"]
    data_years = region.get("bias_correction_years", DEFAULT_DATA_YEARS)
    baseline_start_year = region["baseline_start"]
    baseline_end_year = region["baseline_end"]

    metric_stem = METRICS[metric_name](index, percentile=percentile, **metric_kwargs).output_stem()
    print(f"[bias_correction] country={country} baseline_member={baseline_member} run_type={run_type} "
          f"metric={metric_stem}")

    era5_years, era5_vals = load_baseline_series(
        "era5", country, metric_stem, start=baseline_start_year, end=baseline_end_year
    )
    hg3_years, hg3_vals = load_baseline_series(
        "hg3_historical", country, metric_stem, member=baseline_member,
        start=baseline_start_year, end=baseline_end_year,
    )
    if not np.array_equal(era5_years, hg3_years):
        common = sorted(set(era5_years) & set(hg3_years))
        era5_vals = era5_vals[np.isin(era5_years, common)]
        hg3_vals = hg3_vals[np.isin(hg3_years, common)]
        era5_years = np.array(common)
        print(f"[bias_correction] WARNING: ERA5/HadGEM3-historical baseline years differ; "
              f"using intersection of {len(common)} years")

    baseline_years = era5_years

    members = sorted(paired_members(index))
    print(f"[bias_correction] {len(members)} paired members available for index={index}")

    member_cubes = {}
    load_missing = []
    for member in members:
        try:
            member_cubes[member] = load_member_cube(index, run_type, member, shape_name)
        except MissingMemberError as e:
            load_missing.append((member, str(e)))
    print(f"[bias_correction] Loaded {len(member_cubes)}/{len(members)} member cubes "
          f"({len(load_missing)} missing on disk)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    written = []
    for data_year in data_years:
        t = baseline_years - data_year
        fwi0_obs, delta_obs, std_obs = find_regression_parameters(era5_vals, t)
        fwi0_sim, delta_sim, std_sim = find_regression_parameters(hg3_vals, t)

        col_names = list(member_cubes.keys())
        data_matrix = np.full((len(baseline_years), len(col_names)), np.nan)
        successful = []
        missing = list(load_missing)
        errors = []

        for col_idx, member in enumerate(col_names):
            cube = member_cubes[member]
            try:
                scalar = extract_scalar(cube, months, data_year, metric_name, index,
                                         percentile=percentile, **metric_kwargs)
                scalar_log = soft_log(scalar)
                corrected_log = bias_correct(scalar_log, t, fwi0_obs, delta_sim, fwi0_sim)
                corrected = inverse_soft_log(corrected_log)
                data_matrix[:, col_idx] = corrected
                successful.append(member)
            except MissingMemberError as e:
                missing.append((member, str(e)))
            except (InvalidMemberDataError, ValueError) as e:
                errors.append((member, str(e)))

        df_out = pd.DataFrame(data_matrix, columns=col_names)
        df_out.insert(0, "Year", baseline_years)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{country}_{metric_stem}_baseline{baseline_member}_{run_type}{percentile:g}percent_"
            f"LogTransform_Target_{data_year}_DataYear_{data_year}_BaselinePeriod_"
            f"{baseline_start_year}_{baseline_end_year}.csv",
        )
        df_out.to_csv(out_path, index=False)
        written.append(out_path)

        total = len(members)
        print(f"[bias_correction] DATA_YEAR={data_year}: {len(successful)}/{total} successful, "
              f"{len(missing)}/{total} missing, {len(errors)}/{total} errors -> {out_path}")

    return written
