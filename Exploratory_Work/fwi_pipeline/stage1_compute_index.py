"""Stage 1 CLI: source + metric -> interim .nc. Dataset-agnostic; driven
purely by $CONFIG_PATH and cylc CYLC_TASK_PARAM_* env vars. Same script runs
for all four dataset variants (era5_present, era5_historical,
hadgem3_historical, hadgem3_attribution)."""
from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


warnings.filterwarnings("ignore", category=UserWarning, message=".*chunking.*")
warnings.filterwarnings("ignore", category=FutureWarning,message="*")

import xarray as xr
from dask.distributed import Client, LocalCluster, wait

from config import load_config, parse_cylc_task_params
from metrics.registry import METRIC_REGISTRY
from sources.registry import SOURCE_REGISTRY


def write_netcdf(da: xr.DataArray, out_path: str, metric_name: str) -> None:
    n_times = da.sizes["time"]
    chunk_by_dim = {"time": 365, "latitude": 90, "longitude": 90}
    chunksizes = tuple(min(chunk_by_dim.get(dim, da.sizes[dim]), da.sizes[dim]) for dim in da.dims)
    enc = {metric_name: {"chunksizes": chunksizes}}
    ds = xr.Dataset({metric_name: da})
    ds.to_netcdf(out_path, encoding=enc)
    print(f"Saved {metric_name} to {out_path} ({n_times} timesteps)")


def main() -> None:
    start_time = time.time()
    cfg = load_config(os.environ["CONFIG_PATH"], os.environ["DATASET"])
    task_params = parse_cylc_task_params()
    print(f"dataset={cfg.dataset}, source={cfg.source}, task_params={task_params}")

    source_cls = SOURCE_REGISTRY[cfg.source]
    source = source_cls(cfg)

    cluster = LocalCluster(**source.dask_cluster_kwargs())
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    variables = source.load_variables(**task_params)
    variables_persisted = client.persist(list(variables.values()))
    wait(variables_persisted)
    variables = dict(zip(variables.keys(), variables_persisted))

    for metric_name in cfg.metrics:
        metric = METRIC_REGISTRY[metric_name]()
        indices = metric.compute(variables)
        for idx_name, da in indices.items():
            out_path = cfg.interim_filename(idx_name, **task_params)
            write_netcdf(da, out_path, idx_name)

    try:
        client.close(timeout=30)
        cluster.close(timeout=30)
    except Exception as e:
        print(f"Warning: cluster shutdown raised {e!r} (ignoring - all output already written)")
    print("--- %.2f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
