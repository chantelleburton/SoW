"""HadGEM3-A 525-member attribution ensemble source. Direct port of
Exploratory_Work/xclim_work/attribution_ensemble/explore_hadgem_attribution_xclim_FWI.py,
kept fully separate from HadGEM3HistoricalSource (different tree, member
scheme r{NNN}i1p{R}, experiment tokens, date window)."""
from __future__ import annotations

import glob
import os

import xarray as xr

from .base import DataSource
from .hadgem3_common import fix_time_dim, regrid_to_tracer

VAR_CONFIG = {
    "tasmax": {"dir": "tasmax/day", "nc_var": "tasmax", "units": "degC"},
    "pr": {"dir": "pr/day", "nc_var": "pr", "units": "mm/day"},
    "sfcWind": {"dir": "sfcWind/day", "nc_var": "sfcWind", "units": "m s-1"},
    "hurs": {"dir": "hurs/day", "nc_var": "hurs", "units": "%"},
}
SPATIAL_CHUNK = 30


def _token_overlaps_window(month_token: str, window_start: int, window_end: int) -> bool:
    if "-" in month_token:
        start_str, end_str = month_token.split("-", 1)
        if not (start_str.isdigit() and end_str.isdigit()):
            return False
        start, end = int(start_str), int(end_str)
    elif month_token.isdigit():
        start = end = int(month_token)
    else:
        return False
    return start <= window_end and end >= window_start


class HadGEM3AttributionSource(DataSource):
    def _load_variable(self, var_name: str, run_type: str, member: str, chunks: dict) -> xr.DataArray:
        cfg = VAR_CONFIG[var_name]
        tld = self.cfg.extra["tld"]
        window_start_month = int(self.cfg.extra["window_start_month"])
        window_end_month = int(self.cfg.extra["window_end_month"])

        var_dir = os.path.join(tld, run_type, cfg["dir"])
        pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_{run_type}_{member}_*.nc")
        files = sorted(glob.glob(pattern))
        assert files, f"No files found for {var_name}: {pattern}"

        windowed = [
            f for f in files
            if _token_overlaps_window(
                os.path.basename(f).rsplit("_", 1)[-1].replace(".nc", ""),
                window_start_month, window_end_month,
            )
        ]
        assert windowed, f"No files for {var_name} in window {window_start_month}..{window_end_month}: {pattern}"

        da = xr.open_mfdataset(
            windowed, chunks=chunks, combine="nested", concat_dim="time", preprocess=fix_time_dim,
        )[cfg["nc_var"]]
        da = da.sortby("time").drop_duplicates("time")

        if var_name == "tasmax":
            da = da - 273.15
        elif var_name == "pr":
            da = da * 86400
        elif var_name == "hurs":
            da = da.clip(min=0, max=100)

        da.attrs["units"] = cfg["units"]
        return da

    def load_variables(self, **task_params) -> dict[str, xr.DataArray]:
        run_type = task_params["run_type"]
        member = task_params["member"]
        chunks = {"latitude": SPATIAL_CHUNK, "longitude": SPATIAL_CHUNK, "time": -1}

        tas = self._load_variable("tasmax", run_type, member, chunks)
        pr = self._load_variable("pr", run_type, member, chunks)
        ws = self._load_variable("sfcWind", run_type, member, chunks)
        hurs = self._load_variable("hurs", run_type, member, chunks)

        ws = regrid_to_tracer(ws, tas)
        ws = ws.chunk(chunks)
        ws.attrs["units"] = VAR_CONFIG["sfcWind"]["units"]

        window_start = self.cfg.extra["window_start"]
        window_end = self.cfg.extra["window_end"]
        tas = tas.sel(time=slice(window_start, window_end))
        pr = pr.sel(time=slice(window_start, window_end))
        ws = ws.sel(time=slice(window_start, window_end))
        hurs = hurs.sel(time=slice(window_start, window_end))

        tas, pr, ws, hurs = xr.align(tas, pr, ws, hurs, join="inner")
        if tas.time.size == 0:
            raise ValueError("No overlapping dates after alignment.")

        tas = tas.ffill(dim="time")
        pr = pr.ffill(dim="time")
        ws = ws.ffill(dim="time")
        hurs = hurs.ffill(dim="time")

        compute_chunks = {"time": -1, "latitude": SPATIAL_CHUNK, "longitude": SPATIAL_CHUNK}
        return {
            "tasmax": tas.chunk(compute_chunks),
            "pr": pr.chunk(compute_chunks),
            "sfcWind": ws.chunk(compute_chunks),
            "hurs": hurs.chunk(compute_chunks),
        }

    def dask_cluster_kwargs(self) -> dict:
        return {"n_workers": 3, "threads_per_worker": 1, "memory_limit": "5GB"}
