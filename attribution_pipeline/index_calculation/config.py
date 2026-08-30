"""
Shared configuration objects for the unified FWI creation framework.

Each loader owns its own dataset-specific paths/env-var parsing (kept there so the
"currently wrong but close" HadGEM3-A Historical loader can keep being tuned in
isolation). This module only holds the small pieces of config that are genuinely
shared across all three datasets: dask cluster sizing and the sub-indices to write.
"""

from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    n_workers: int = 3
    threads_per_worker: int = 1  # numpy inner loop is GIL-bound
    memory_per_worker_gb: int = 30


@dataclass
class DatasetConfig:
    """Generic per-run settings. Loaders may hold additional dataset-specific
    config (paths, var maps, env-var driven parameters) alongside this."""

    name: str
    out_dir: str
    spatial_chunk: int = 30
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    output_indices: list = field(default_factory=lambda: ["fwi"])
    # kwargs forwarded to xclim.indices.cffwis_indices, minus tas/pr/sfcWind/hurs/lat
    cffwis_kwargs: dict = field(default_factory=lambda: {"initial_start_up": True})
