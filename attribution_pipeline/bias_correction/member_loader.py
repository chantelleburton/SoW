"""
Loads attribution-ensemble member FWI/DSR data directly from our own
FWI-creation output (attribution_pipeline/index_calculation/loaders/hadgem3_attribution.py),
instead of scanning raw monthly netCDFs. Replaces
Exploratory_Work/Reduced_Att_Set_Processing/find_matching_members.py's 63-file
completeness scan with a simple "does the file exist for both hist and
histnat" check, plus read-in validation (calendar-aware day-count + sanity
checks) since we lose the old per-month-file completeness guarantee.
"""

import glob
import os

import iris
import numpy as np

from attribution_pipeline.pipeline_config import RAW_FWI_HG3_ATTRIBUTION, SHAPEFILE
from utils.cubefuncs import ConstrainToYear, apply_shapefile_inclusive, constrain_cube_to_months

RAW_FWI_DIR = RAW_FWI_HG3_ATTRIBUTION

# bias_correction's CYLC_TASK_PARAM_runtype convention ('hist'/'histnat') ->
# hadgem3_attribution.py's CYLC_TASK_PARAM_run_type convention.
RUN_TYPE_MAP = {"hist": "historicalExt", "histnat": "historicalNatExt"}


class MissingMemberError(Exception):
    """Member file/year not found."""


class InvalidMemberDataError(Exception):
    """Member file exists but the data looks wrong (day-count, all-NaN, etc)."""


def _run_type_token(run_type: str) -> str:
    return RUN_TYPE_MAP.get(run_type, run_type)


def _member_file(index: str, run_type: str, member: str) -> str:
    token = _run_type_token(run_type)
    return os.path.join(RAW_FWI_DIR, f"hadgem3a_{index}_{token}_{member}.nc")


def list_available_members(index: str, run_type: str) -> set:
    token = _run_type_token(run_type)
    prefix = f"hadgem3a_{index}_{token}_"
    pattern = os.path.join(RAW_FWI_DIR, f"{prefix}*.nc")
    members = set()
    for f in glob.glob(pattern):
        stem = os.path.basename(f)[: -len(".nc")]
        members.add(stem[len(prefix):])
    return members


def paired_members(index: str) -> set:
    """Members with an output file for BOTH hist and histnat -- the strictly
    paired ensemble (mirrors reduced_set_risk_ratio.py's get_paired_members,
    but checked against our own FWI-creation output rather than raw dirs)."""
    return list_available_members(index, "hist") & list_available_members(index, "histnat")


def load_member_cube(index: str, run_type: str, member: str, shape_name: str):
    """Load and shapefile-mask a member's full-window cube (2019-11..2024-12).
    Does NOT constrain to a specific year/month -- callers needing antecedent
    context (e.g. CumulativeMetric) need the full window; use
    validate_member_window() for a year/month-constrained + validated view."""
    path = _member_file(index, run_type, member)
    if not os.path.exists(path):
        raise MissingMemberError(f"No FWI-creation output for member {member}: {path}")

    cube = iris.load_cube(path, iris.NameConstraint(var_name=index))
    for coord_name in ("year", "season_year"):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)
    cube = apply_shapefile_inclusive(SHAPEFILE, shape_name, cube)
    return cube


def _expected_days_in_month(year: int, month: int, calendar: str) -> int:
    if calendar == "360_day":
        return 30
    import calendar as cal
    return cal.monthrange(year, month)[1]


def validate_member_window(cube, data_year: int, months):
    """Constrain `cube` to data_year/months and validate it. Raises
    MissingMemberError if the year isn't present, InvalidMemberDataError if
    the day-count (calendar-aware) or data looks wrong. Returns the
    constrained cube on success."""
    try:
        yr_cube = ConstrainToYear(cube, data_year)
    except ValueError as e:
        raise MissingMemberError(str(e))
    yr_cube = constrain_cube_to_months(yr_cube, months)

    calendar = yr_cube.coord("time").units.calendar
    expected = sum(_expected_days_in_month(data_year, m, calendar) for m in months)
    actual = yr_cube.coord("time").shape[0]
    if actual != expected:
        raise InvalidMemberDataError(
            f"Expected {expected} days for {data_year}/{months} (calendar={calendar}), got {actual}"
        )
    if not np.any(np.isfinite(yr_cube.data)):
        raise InvalidMemberDataError(f"All-NaN/non-finite data for {data_year}/{months}")

    return yr_cube
