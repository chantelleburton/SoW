"""ERA5 present-day (event period) source. Own class/config/task, not part of
the baseline pair — shares loading mechanics with ERA5HistoricalSource via
ERA5LoaderMixin, differs in base class + date range/role."""
from __future__ import annotations

from .base import DataSource
from .era5_loader import ERA5LoaderMixin

SPATIAL_CHUNK = 90


class ERA5PresentSource(ERA5LoaderMixin, DataSource):
    def load_variables(self, **task_params):
        start_year = int(self.cfg.extra["start_year"])
        end_year = int(self.cfg.extra["end_year"])
        chunks = {"latitude": SPATIAL_CHUNK, "longitude": SPATIAL_CHUNK}
        return ERA5LoaderMixin.load_variables(self, start_year, end_year, chunks)

    def dask_cluster_kwargs(self) -> dict:
        return {"n_workers": 3, "threads_per_worker": 1, "memory_limit": "40GB"}
