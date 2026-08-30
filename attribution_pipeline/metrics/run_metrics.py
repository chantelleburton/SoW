"""
Entrypoint for the interim metrics framework: turns daily FWI/DSR NetCDF files
(as produced by attribution_pipeline/run_fwi.py) into per-year, event-month CSVs
consumed by bias correction.

Adding a new metric: subclass attribution_pipeline.metrics.base.BaseMetric and
register it in METRICS below -- no other code needs to change.

Adding a new dataset's file layout: add a resolver function and register it in
DATASET_RESOLVERS below.

Usage (mirrors the CYLC_TASK_PARAM_* convention used elsewhere in the repo):
    CYLC_TASK_PARAM_dataset=hg3_historical \
    CYLC_TASK_PARAM_country=Iberia \
    CYLC_TASK_PARAM_index=fwi \
    CYLC_TASK_PARAM_metric=p95 \
    CYLC_TASK_PARAM_member=1 \
    python -m attribution_pipeline.metrics.run_metrics
"""

import glob
import os

import iris

from attribution_pipeline.metrics.base import BaseMetric
from attribution_pipeline.metrics.cumulative import CumulativeMetric
from attribution_pipeline.metrics.extreme_window import ExtremeWindowMetric
from attribution_pipeline.metrics.percentile import PercentileMetric
from attribution_pipeline.metrics.pipeline_config import SHAPEFILE, get_region
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


# --- Dataset file resolvers --------------------------------------------------
# Each resolver returns a single concatenated iris cube spanning the full
# available period for (index, member/run_type). Extend this dict to plug in
# new datasets; the metric/CSV-writing code below is dataset-agnostic.

def _resolve_hg3_historical(index: str, member: str, **kw) -> iris.cube.Cube:
    folder = "/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_historical/"
    pattern = os.path.join(folder, f"hadgem3a_{index}_historical_r1i1p{member}_*_modified.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HadGEM3 historical {index} file found: {pattern}")
    assert len(files) == 1, f"Expected exactly one file for member {member}, found {len(files)}: {files}"
    cube = iris.load_cube(files[0], iris.NameConstraint(var_name=index))
    for coord_name in ("year", "season_year"):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)
    return cube


def _resolve_era5(index: str, member: str = None, run_label: str = None, **kw) -> iris.cube.Cube:
    """ERA5Loader.write() (attribution_pipeline/loaders/era5.py) writes one file
    per calendar year: era5_{index}_{run_label}_{year}.nc, where run_label
    encodes the wind/RH statistic combo (e.g. 'Mean_RH_Mean_Wind'). ERA5 has no
    'member' concept -- pass an optional `run_label` kwarg (via
    CYLC_TASK_PARAM_run_label) to select a specific wind/RH combo when more
    than one has been generated; otherwise all matching files are used.
    """
    folder = "/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/era5/"
    label_glob = run_label if run_label else "*"
    pattern = os.path.join(folder, f"era5_{index}_{label_glob}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ERA5 {index} files found: {pattern}")

    cubes = iris.cube.CubeList(iris.load_cube(f, iris.NameConstraint(var_name=index)) for f in files)
    for coord_name in ("year", "season_year"):
        for cube in cubes:
            if cube.coords(coord_name):
                cube.remove_coord(coord_name)
    cube = cubes.concatenate_cube() if len(cubes) > 1 else cubes[0]
    return cube


def _resolve_hg3_attribution(index: str, member: str, run_type: str = "historicalExt", **kw) -> iris.cube.Cube:
    """HadGEM3AttributionLoader.write() (attribution_pipeline/loaders/hadgem3_attribution.py)
    writes one file per (run_type, member) spanning the full 2019-2024 window:
    hadgem3a_{index}_{run_type}_{member}.nc. `run_type` is 'historicalExt'
    (factual) or 'historicalNatExt' (counterfactual); pass via
    CYLC_TASK_PARAM_run_type.
    """
    folder = "/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_attribution/"
    pattern = os.path.join(folder, f"hadgem3a_{index}_{run_type}_{member}.nc")
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
    "hg3_historical": _resolve_hg3_historical,
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
    cube = apply_shapefile_inclusive(SHAPEFILE, shape_name, cube)

    years, values = metric.compute(cube, months)
    if not years:
        raise RuntimeError(f"No results computed for {dataset}/{country}/{metric.output_stem()}")

    out_dir = "/data/scratch/bob.potts/sowf/attribution_pipeline/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_{metric.output_stem()}_{country}_{member}.csv")
    with open(out_path, "w") as f:
        f.write(f"Year,{metric.output_stem()}\n")
        for y, v in zip(years, values):
            f.write(f"{int(y)},{v:.6f}\n")
    print(f"[metrics] Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    dataset = os.environ.get("CYLC_TASK_PARAM_dataset", "hg3_historical")
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
