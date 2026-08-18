"""Abstract base classes for data sources.

DataSource is the generic interface every source implements. BaselineSource is
a marker/shared-interface subclass for the two sources that form the matched
ERA5-historical / HadGEM3-historical baseline pair (see session plan notes):
both must produce the same downstream summary-CSV schema so bias correction
can treat them symmetrically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr


class DataSource(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def load_variables(self, **task_params) -> dict[str, xr.DataArray]:
        """Return aligned, unit-converted, forward-filled tas/pr/sfcWind/hurs,
        ready to be passed straight into a Metric."""

    def dask_cluster_kwargs(self) -> dict:
        """Default per-source dask sizing; sources may override."""
        return {"n_workers": 3, "threads_per_worker": 1, "memory_limit": "40GB"}


class BaselineSource(DataSource):
    """Marker base for the ERA5-historical / HadGEM3-historical matched pair."""
