"""FWI metric — thin wrapper around xclim.indices.cffwis_indices, used
identically by all four DataSource implementations."""
from __future__ import annotations

import xarray as xr
import xclim as xc

from .base import Metric

INDEX_META = {
    "dc": ("Drought Code", "1"),
    "dmc": ("Duff Moisture Code", "1"),
    "ffmc": ("Fine Fuel Moisture Code", "1"),
    "isi": ("Initial Spread Index", "1"),
    "bui": ("Build-Up Index", "1"),
    "fwi": ("Fire Weather Index", "FWI"),
}


class FWIMetric(Metric):
    name = "fwi"

    def __init__(self, output_indices: list[str] | None = None):
        self.output_indices = output_indices or ["fwi"]

    def compute(self, variables: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
        dc, dmc, ffmc, isi, bui, fwi = xc.indices.cffwis_indices(
            tas=variables["tasmax"],
            pr=variables["pr"],
            sfcWind=variables["sfcWind"],
            hurs=variables["hurs"],
            lat=variables["tasmax"].latitude,
            initial_start_up=True,
        )
        computed = {"dc": dc, "dmc": dmc, "ffmc": ffmc, "isi": isi, "bui": bui, "fwi": fwi}

        out = {}
        for idx_name in self.output_indices:
            da = computed[idx_name]
            long_name, units = INDEX_META[idx_name]
            da = da.rename(idx_name)
            da.attrs = {"long_name": long_name, "units": units}
            da.encoding = {}
            da = da.drop_vars([c for c in ("height",) if c in da.coords])
            da = da.transpose("time", "latitude", "longitude")
            out[idx_name] = da
        return out
