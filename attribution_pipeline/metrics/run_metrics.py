"""
Entrypoint for the interim metrics framework: turns daily FWI/DSR NetCDF files
(as produced by attribution_pipeline/index_calculation/run_fwi.py) into per-year, event-month CSVs
consumed by bias correction.

Adding a new metric: subclass attribution_pipeline.metrics.base.BaseMetric and
register it in METRICS below -- no other code needs to change.

Adding a new dataset's file layout: add a resolver function and register it in
DATASET_RESOLVERS below.

Usage (mirrors the CYLC_TASK_PARAM_* convention used elsewhere in the repo):
    CYLC_TASK_PARAM_dataset=hg3_historical_xclim \
    CYLC_TASK_PARAM_country=Iberia \
    CYLC_TASK_PARAM_index=fwi \
    CYLC_TASK_PARAM_metric=p95 \
    CYLC_TASK_PARAM_member=1 \
    python -m attribution_pipeline.metrics.run_metrics
"""

import glob
import os

import iris
import numpy as np

from attribution_pipeline.metrics.base import BaseMetric
from attribution_pipeline.metrics.cumulative import CumulativeMetric
from attribution_pipeline.metrics.extreme_window import ExtremeWindowMetric
from attribution_pipeline.metrics.percentile import PercentileMetric
from attribution_pipeline.pipeline_config import (
    METRICS_OUT_DIR,
    RAW_FWI_ERA5,
    RAW_FWI_HG3_ATTRIBUTION,
    RAW_FWI_HG3_HISTORICAL,
    IMPACTTB_HISTORICAL_FWI_DIR,
    SHAPEFILE,
    get_region,
)
from utils.cubefuncs import apply_shapefile_inclusive

# --- Metric registry -------------------------------------------------------
# metric name (as passed via CYLC_TASK_PARAM_metric) -> factory(index, **kwargs)
METRICS = {
    "p95": lambda index, **kw: PercentileMetric(index, percentile=float(kw.get("percentile", 95))),
    "7x": lambda index, **kw: ExtremeWindowMetric(index, window=int(kw.get("window", 7))),
    "cum": lambda index, **kw: CumulativeMetric(
        index, window=int(kw.get("window", 360)), spatial_reduction=kw.get("spatial_reduction", "mean")
    ),
}


# --- Cube invariant checks ---------------------------------------------------
# Catches structural corruption (duplicate/ambiguous coordinates, wrong dims,
# non-monotonic/duplicate time points, leaked auxiliary coordinates)
_ALLOWED_COORDS = {"time", "latitude", "longitude", "year", "season_year"}


def _validate_cube(cube: iris.cube.Cube, dataset: str, stage: str) -> None:
    context = f"{dataset} ({stage})"
    print(context + f": cube shape={cube.shape}, coords={sorted(c.name() for c in cube.coords())}")
    time_coords = cube.coords("time")
    if len(time_coords) != 1:
        raise RuntimeError(
            f"[{context}] Expected exactly 1 'time' coordinate, found {len(time_coords)}."
        )

    dim_coord_names = {c.name() for c in cube.coords(dim_coords=True)}
    if cube.ndim != 3 or dim_coord_names != {"time", "latitude", "longitude"}:
        raise RuntimeError(
            f"[{context}] Expected a 3-D (time, latitude, longitude) cube, got "
            f"ndim={cube.ndim}, dim coords={sorted(dim_coord_names)}, shape={cube.shape}."
        )

    time_coord = cube.coord("time")
    points = time_coord.points
    if len(points) > 1 and not (points[1:] > points[:-1]).all():
        raise RuntimeError(
            f"[{context}] 'time' coordinate is not strictly monotonically increasing "
            f"(duplicate timestamps or wrong concatenation axis)."
        )

    leaked = {c.name() for c in cube.coords()} - _ALLOWED_COORDS
    if leaked:
        raise RuntimeError(
            f"[{context}] Unexpected leftover coordinate(s) {sorted(leaked)} -- likely "
            f"leaked metadata from the raw input files (see era5.py attrs-contamination fix)."
        )

    # Soft check: warn (don't fail) on an implausible day count for the
    # cube's own calendar/date range -- legitimate small data gaps shouldn't
    # hard-fail the whole run.
    if len(points) > 1:
        calendar = time_coord.units.calendar
        first_date = time_coord.units.num2date(points[0])
        last_date = time_coord.units.num2date(points[-1])
        expected_days = (last_date - first_date).days + 1
        actual_days = len(points)
        if actual_days > expected_days or actual_days < 0.5 * expected_days:
            print(f"[{context}] WARNING: implausible day count -- {actual_days} timesteps "
                  f"spanning {first_date} to {last_date} ({calendar} calendar, "
                  f"~{expected_days} days expected).")


# --- Dataset file resolvers --------------------------------------------------
# Each resolver returns a single concatenated iris cube spanning the full
# available period for (index, member/run_type). Extend this dict to plug in
# new datasets; the metric/CSV-writing code below is dataset-agnostic.

def _concatenate_yearly_cubes(cubes: iris.cube.CubeList) -> iris.cube.Cube:
    """Concatenate per-year cubes loaded from separate NetCDF files.

    Each file gets its own auto-derived global attrs (e.g. a unique 'history'
    timestamp) and Iris re-derives its own time-coordinate metadata per load,
    so even with a shared numeric time `units` on disk, residual per-cube
    metadata (attributes dict, var_name, differing units objects) can still
    fail Iris's concatenate_cube() equality check. Normalise both before
    concatenating -- same pattern as utils/cubefuncs.py's historical-percentile
    helper and the validation scripts' _strip_aux_time_coords().
    """
    if len(cubes) == 1:
        return cubes[0]

    reference_units = None
    for cube in cubes:
        time_coord = cube.coord("time")
        if reference_units is None:
            reference_units = time_coord.units
        else:
            time_coord.convert_units(reference_units)
        time_coord.attributes = {}
        time_coord.var_name = None
        time_coord.long_name = None
        time_coord.standard_name = "time"

    iris.util.equalise_attributes(cubes)
    return cubes.concatenate_cube()


def _resolve_hg3_historical_xclim(index: str, member: str, **kw) -> iris.cube.Cube:
    """HadGEM3HistoricalLoader.write() (attribution_pipeline/index_calculation/loaders/hadgem3_historical.py)
    writes one file per calendar year (segmented ~10-year blocks, run
    independently with overwintering disabled -- see that module's docstring):
    hadgem3a_{index}_historical_{member}_{year}.nc.
    """
    pattern = os.path.join(RAW_FWI_HG3_HISTORICAL, f"hadgem3a_{index}_historical_r1i1p{member}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HadGEM3 historical {index} files found: {pattern}")

    cubes = iris.cube.CubeList(iris.load_cube(f, iris.NameConstraint(var_name=index)) for f in files)
    for coord_name in ("year", "season_year"):
        for cube in cubes:
            if cube.coords(coord_name):
                cube.remove_coord(coord_name)
    return _concatenate_yearly_cubes(cubes)


# NetCDF variable-name search set for impact-toolbox FWI files -- the file's
# actual variable is named 'canadian_fire_weather_index' (with a
# variable_id="fwi" attribute), unlike xclim's output where var_name=='fwi'
# directly, so iris.NameConstraint(var_name=index) won't match it.
_IMPACTTB_FWI_NAMES = {
    "fwi", "Fire Weather Index", "fire_weather_index",
    "Canadian Fire Weather Index", "canadian_fire_weather_index",
}


def _load_impacttb_fwi_cube(fpath: str) -> iris.cube.Cube:
    """Load the FWI cube from an impact-toolbox file that may contain several
    FWI sub-indices, matching by var_name/name()/long_name/standard_name."""
    cubes = iris.load(fpath)
    for c in cubes:
        names = {c.var_name, c.name(), getattr(c, "long_name", None), c.standard_name}
        if names & _IMPACTTB_FWI_NAMES:
            return c
    raise ValueError(f"No FWI cube found in {fpath}. Available: {[c.name() for c in cubes]}")


def _resolve_hg3_historical_impacttb(index: str, member: str, **kw) -> iris.cube.Cube:
    """Impact-toolbox HadGEM3-A historical FWI: monthly 'gwl' files (FWI only,
    no DSR), member r1i1p1..15, 1980-2013, 360_day calendar (confirmed via
    ncdump), time units 'days since 1960-01-01' (differs from xclim's
    reference date, handled internally below via unit-alignment before
    concatenation). Ported from the reference implementation in
    Exploratory_Work/xclim_work/historical_ensemble_validation/compute_raw_fwi_diffs.py
    (load_impacttb/_strip_aux_time_coords/_load_fwi_cube) to keep
    attribution_pipeline self-contained.
    """
    if index != "fwi":
        raise ValueError(
            f"hg3_historical_impacttb only has FWI data (no DSR); got index={index!r}."
        )
    pattern = os.path.join(
        IMPACTTB_HISTORICAL_FWI_DIR,
        f"FWI_HadGEM3-A-N216_r1i1p{member}_historical_gwl*_global_day_"
        f"initialise-from=previous-and-save-input-data=True.nc",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No impact-toolbox HadGEM3 historical FWI files found: {pattern}")

    cubes = iris.cube.CubeList()
    reference_units = None
    for f in files:
        cube = _load_impacttb_fwi_cube(f)
        for coord_name in ("month", "month_number", "season", "season_year", "year", "height"):
            if cube.coords(coord_name):
                cube.remove_coord(coord_name)
        time_coord = cube.coord("time")
        if reference_units is None:
            reference_units = time_coord.units
        else:
            time_coord.convert_units(reference_units)
        time_coord.attributes = {}
        time_coord.var_name = None
        time_coord.long_name = None
        time_coord.standard_name = "time"
        cubes.append(cube)

    iris.util.equalise_attributes(cubes)
    cube = cubes.concatenate_cube()

    # Drop any duplicate boundary timesteps shared between contiguous monthly files.
    _, idx = np.unique(cube.coord("time").points, return_index=True)
    if len(idx) != cube.coord("time").shape[0]:
        cube = cube[np.sort(idx)]
    return cube


def _resolve_era5(index: str, member: str = None, run_label: str = None, **kw) -> iris.cube.Cube:
    """ERA5Loader.write() (attribution_pipeline/index_calculation/loaders/era5.py) writes one file
    per calendar year: era5_{index}_{run_label}_{year}.nc, where run_label
    encodes the wind/RH statistic combo (e.g. 'Mean_RH_Mean_Wind'). ERA5 has no
    'member' concept -- pass an optional `run_label` kwarg (via
    CYLC_TASK_PARAM_run_label) to select a specific wind/RH combo when more
    than one has been generated; otherwise all matching files are used.
    """
    label_glob = run_label if run_label else "*"
    pattern = os.path.join(RAW_FWI_ERA5, f"era5_{index}_{label_glob}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ERA5 {index} files found: {pattern}")

    cubes = iris.cube.CubeList(iris.load_cube(f, iris.NameConstraint(var_name=index)) for f in files)
    for coord_name in ("year", "season_year"):
        for cube in cubes:
            if cube.coords(coord_name):
                cube.remove_coord(coord_name)
    return _concatenate_yearly_cubes(cubes)


def _resolve_hg3_attribution(index: str, member: str, run_type: str = "historicalExt", **kw) -> iris.cube.Cube:
    """HadGEM3AttributionLoader.write() (attribution_pipeline/index_calculation/loaders/hadgem3_attribution.py)
    writes one file per (run_type, member) spanning the full 2019-2024 window:
    hadgem3a_{index}_{run_type}_{member}.nc. `run_type` is 'historicalExt'
    (factual) or 'historicalNatExt' (counterfactual); pass via
    CYLC_TASK_PARAM_run_type.
    """
    pattern = os.path.join(RAW_FWI_HG3_ATTRIBUTION, f"hadgem3a_{index}_{run_type}_{member}.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HadGEM3 attribution {index} file found: {pattern}")
    assert len(files) == 1, f"Expected exactly one file for run_type={run_type} member={member}, found {len(files)}: {files}"
    cube = iris.load_cube(files[0], iris.NameConstraint(var_name=index))
    for coord_name in ("year", "season_year"):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)
    return cube


DATASET_RESOLVERS = {
    "hg3_historical_xclim": _resolve_hg3_historical_xclim,
    "hg3_historical_impacttb": _resolve_hg3_historical_impacttb,
    "era5": _resolve_era5,
    "hg3_attribution": _resolve_hg3_attribution,
}


def run(dataset: str, country: str, index: str, metric_name: str, member: str = "1", **metric_kwargs):
    region = get_region(country)
    months = region["months"]
    shape_name = region["shape_name"]

    if dataset not in DATASET_RESOLVERS:
        raise ValueError(f"Unknown/unregistered dataset {dataset!r}. Valid options: {sorted(DATASET_RESOLVERS)}")
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}. Valid options: {sorted(METRICS)}")

    metric: BaseMetric = METRICS[metric_name](index, **metric_kwargs)

    print(f"[metrics] dataset={dataset} country={country} index={index} metric={metric.output_stem()} member={member}")

    cube = DATASET_RESOLVERS[dataset](index, member=member, **metric_kwargs)
    print(cube)
    _validate_cube(cube, dataset, "post-resolve")
    print(cube)
    cube = apply_shapefile_inclusive(SHAPEFILE, shape_name, cube)
    _validate_cube(cube, dataset, "post-mask")

    years, values = metric.compute(cube, months)
    if not years:
        raise RuntimeError(f"No results computed for {dataset}/{country}/{metric.output_stem()}")

    out_dir = METRICS_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    # hg3_attribution has two run_types (historicalExt/historicalNatExt) per
    # member -- fold it into the filename so they don't overwrite each other.
    run_type = metric_kwargs.get("run_type")
    stem = f"{dataset}_{metric.output_stem()}_{country}_{member}"
    if run_type:
        stem += f"_{run_type}"
    out_path = os.path.join(out_dir, f"{stem}.csv")
    with open(out_path, "w") as f:
        f.write(f"Year,{metric.output_stem()}\n")
        for y, v in zip(years, values):
            f.write(f"{int(y)},{v:.6f}\n")
    print(f"[metrics] Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    dataset = os.environ.get("CYLC_TASK_PARAM_dataset", "hg3_historical_xclim")
    country = os.environ.get("CYLC_TASK_PARAM_country", "Iberia")
    index = os.environ.get("CYLC_TASK_PARAM_index", "fwi")
    metric_name = os.environ.get("CYLC_TASK_PARAM_metric", "p95")
    member = os.environ.get("CYLC_TASK_PARAM_member", "1")

    metric_kwargs = {}
    if os.environ.get("CYLC_TASK_PARAM_window"):
        metric_kwargs["window"] = os.environ["CYLC_TASK_PARAM_window"]
    if os.environ.get("CYLC_TASK_PARAM_percentile"):
        metric_kwargs["percentile"] = os.environ["CYLC_TASK_PARAM_percentile"]
    if os.environ.get("CYLC_TASK_PARAM_spatial_reduction"):
        metric_kwargs["spatial_reduction"] = os.environ["CYLC_TASK_PARAM_spatial_reduction"]
    # dataset-resolver-specific kwargs, forwarded through **metric_kwargs
    if os.environ.get("CYLC_TASK_PARAM_run_label"):
        metric_kwargs["run_label"] = os.environ["CYLC_TASK_PARAM_run_label"]
    if os.environ.get("CYLC_TASK_PARAM_run_type"):
        metric_kwargs["run_type"] = os.environ["CYLC_TASK_PARAM_run_type"]

    run(dataset, country, index, metric_name, member=member, **metric_kwargs)
