"""
Shared configuration objects for the unified FWI creation framework.

Each loader owns its own dataset-specific paths/env-var parsing. This module only holds the small pieces of config that are genuinely
shared across all three datasets: dask cluster sizing and the sub-indices to write.
"""

from dataclasses import dataclass, field

#: All indices producible by xclim.indices.cffwis_indices, plus the derived
#: dsr. Used only to validate DATASET_INDICES entries below (catches typos). 
#validation step, keep entire list with specification below in DATASET_INDICES.
ALL_INDICES = ["fwi", "dsr", "dc", "dmc", "ffmc", "isi", "bui"]

#this is only used for what files to export.
DATASET_INDICES = {
    "era5": [
        "fwi",
        "dsr",
    ],
    "hg3_historical": [
        "fwi",
        "dsr",
        "dc",
        "dmc",
        "ffmc",
        "isi",
        "bui",
    ],
    "hg3_attribution": [
        "fwi",
        "dsr",
    ],
}


@dataclass
class ClusterConfig:
    n_workers: int = 3
    threads_per_worker: int = 1  # numpy inner loop is GIL-bound
    memory_per_worker_gb: int = 30


@dataclass
class DatasetConfig:
    """Generic per-run settings. Loaders may hold additional dataset-specific
    config (paths, var maps, env-var driven parameters) alongside this.

    `output_indices` defaults to DATASET_INDICES[name] (see above) if not
    passed explicitly -- loaders normally don't need to set it at all.
    """

    name: str
    out_dir: str
    spatial_chunk: int = 30
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    output_indices: list = field(default_factory=list)
    # kwargs forwarded to xclim.indices.cffwis_indices, minus tas/pr/sfcWind/hurs/lat
    cffwis_kwargs: dict = field(default_factory=lambda: {"initial_start_up": True})

    def __post_init__(self):
        if not self.output_indices:
            self.output_indices = list(DATASET_INDICES.get(self.name, ["fwi", "dsr"]))
        unknown = sorted(set(self.output_indices) - set(ALL_INDICES))
        if unknown:
            raise ValueError(f"Unknown output_indices {unknown}; valid options: {ALL_INDICES}")
