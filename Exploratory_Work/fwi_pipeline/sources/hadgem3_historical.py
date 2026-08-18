"""HadGEM3-A 15-member historical source (baseline pair, HadGEM3 side).

Independent module from hadgem3_attribution.py: different directory layout
(decadal files, not near-monthly), different member scheme (r1i1p{1..15},
single realisation, not r{NNN}i1p{R}), single 'historical' experiment token.
Confirmed via ncdump that sfcWind IS staggered here too (lat=325 vs tasmax
lat=324), so regrid_to_tracer (shared helper) is required.

File pattern (confirmed via ls on
/data/users/opatt/HadGEM3-A-N216/historical/{var}/day/):
    {var}_day_HadGEM3-A-N216_historical_r1i1p{N}_{decade_start}-{decade_end}.nc
"""
from __future__ import annotations

import glob
import os

import xarray as xr

from .base import BaselineSource
from .hadgem3_common import fix_time_dim, regrid_to_tracer

VAR_CONFIG = {
    "tasmax": {"dir": "tasmax/day", "nc_var": "tasmax", "units": "degC"},
    "pr": {"dir": "pr/day", "nc_var": "pr", "units": "mm/day"},
    "sfcWind": {"dir": "sfcWind/day", "nc_var": "sfcWind", "units": "m s-1"},
    "hurs": {"dir": "hurs/day", "nc_var": "hurs", "units": "%"},
}
SPATIAL_CHUNK = 30


class HadGEM3HistoricalSource(BaselineSource):
    def _load_variable(self, var_name: str, member: str, chunks: dict) -> xr.DataArray:
        cfg = VAR_CONFIG[var_name]
        tld = self.cfg.extra["tld"]
        var_dir = os.path.join(tld, "historical", cfg["dir"])
        pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_historical_{member}_*.nc")
        files = sorted(glob.glob(pattern))
        assert files, f"No files found for {var_name}: {pattern}"

        # Keep only decade files overlapping the configured baseline window.
        start_year = int(self.cfg.extra["start_year"])
        end_year = int(self.cfg.extra["end_year"])
        windowed = []
        for f in files:
            token = os.path.basename(f).rsplit("_", 1)[-1].replace(".nc", "")
            dec_start, dec_end = token.split("-")
            dec_start_year, dec_end_year = int(dec_start[:4]), int(dec_end[:4])
            if dec_start_year <= end_year and dec_end_year >= start_year:
                windowed.append(f)
        assert windowed, f"No files overlapping {start_year}-{end_year} for {var_name}: {pattern}"

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
        member = f"r1i1p{task_params['member']}"
        start_year = int(self.cfg.extra["start_year"])
        end_year = int(self.cfg.extra["end_year"])
        chunks = {"latitude": SPATIAL_CHUNK, "longitude": SPATIAL_CHUNK, "time": -1}

        tas = self._load_variable("tasmax", member, chunks)
        pr = self._load_variable("pr", member, chunks)
        ws = self._load_variable("sfcWind", member, chunks)
        hurs = self._load_variable("hurs", member, chunks)

        # sfcWind is on the staggered grid here too (confirmed: lat=325 vs
        # tasmax lat=324) — regrid onto the tracer grid before aligning.
        ws = regrid_to_tracer(ws, tas)
        ws = ws.chunk(chunks)
        ws.attrs["units"] = VAR_CONFIG["sfcWind"]["units"]

        window_start = f"{start_year}-01-01"
        window_end = f"{end_year}-12-30"
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
