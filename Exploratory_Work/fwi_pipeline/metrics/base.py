from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, variables: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
        """Return one or more named DataArrays, e.g. {'fwi': ..., 'dc': ...}."""
