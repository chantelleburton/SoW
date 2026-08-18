"""ERA5 historical (baseline pair, ERA5 side). Shares loading mechanics with
ERA5PresentSource via ERA5LoaderMixin, but is a BaselineSource: its output
must match HadGEM3HistoricalSource's summary-CSV schema exactly (Date, FWI)
since bias correction consumes both together."""
from __future__ import annotations

from .base import BaselineSource
from .era5_loader import ERA5LoaderMixin

SPATIAL_CHUNK = 90


class ERA5HistoricalSource(ERA5LoaderMixin, BaselineSource):
    def load_variables(self, **task_params):
        start_year = int(self.cfg.extra["start_year"])
        end_year = int(self.cfg.extra["end_year"])
        chunks = {"latitude": SPATIAL_CHUNK, "longitude": SPATIAL_CHUNK}
        return ERA5LoaderMixin.load_variables(self, start_year, end_year, chunks)

    def dask_cluster_kwargs(self) -> dict:
        return {"n_workers": 3, "threads_per_worker": 1, "memory_limit": "40GB"}
