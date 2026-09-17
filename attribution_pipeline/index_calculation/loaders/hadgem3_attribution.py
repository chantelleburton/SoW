"""
HadGEM3-A Attribution loader: reads the hadgem3-a attribution input variables
and produces FWI for the period 2019-2024.

Ported from
index_calculation/fwi/FWI-HadGEM3-A_Attribution/hadgem3-a_attribution_fwi_calculation.py
"""

import glob
import os

import xarray as xr

from attribution_pipeline.index_calculation.config import ClusterConfig, DatasetConfig
from attribution_pipeline.index_calculation.loaders.base import BaseLoader
from attribution_pipeline.index_calculation.loaders._grid_utils import regrid_to_tracer, fix_anonymous_time_dim
from attribution_pipeline.pipeline_config import (
    RAW_FWI_HG3_ATTRIBUTION,
    WINDOW_START_MONTH,
    WINDOW_END_MONTH,
    WINDOW_START,
    WINDOW_END,
)

VAR_CONFIG = {
    "tasmax": {"dir": "tasmax/day", "nc_var": "tasmax", "units": "degC"},
    "pr": {"dir": "pr/day", "nc_var": "pr", "units": "mm/day"},
    "sfcWind": {"dir": "sfcWind/day", "nc_var": "sfcWind", "units": "m s-1"},
    "hurs": {"dir": "hurs/day", "nc_var": "hurs", "units": "%"},
}

# Fixed time units reference (rather than xarray's per-file default, which
# would pick each file's own start date) -- matches the reference used by the
# era5/hg3_historical loaders so time units are consistent across all three
# datasets rather than varying per attribution member/run_type.
TIME_UNITS = "days since 1900-01-01"



def _token_overlaps_window(month_token):
    """True if a filename month token ('YYYYMM' or 'YYYYMM-YYYYMM') overlaps the window."""
    if "-" in month_token:
        start_str, end_str = month_token.split("-", 1)
        if not (start_str.isdigit() and end_str.isdigit()):
            return False
        start, end = int(start_str), int(end_str)
    elif month_token.isdigit():
        start = end = int(month_token)
    else:
        return False
    return start <= WINDOW_END_MONTH and end >= WINDOW_START_MONTH


class HadGEM3AttributionLoader(BaseLoader):
    name = "hg3_attribution"
    tld = "/data/users/opatt/HadGEM3-A-N216"

    def __init__(self, run_type=None, member=None, out_dir=None):
        self.run_type = (run_type or os.environ.get("CYLC_TASK_PARAM_run_type", "historicalExt")).strip()
        self.member = (member or os.environ.get("CYLC_TASK_PARAM_member", "r001i1p1")).strip()

        cfg = DatasetConfig(
            name=self.name,
            out_dir=out_dir or "/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_attribution",
            spatial_chunk=30,
            cluster=ClusterConfig(n_workers=3, memory_per_worker_gb=5),
            cffwis_kwargs={"initial_start_up": True},
        )
        self.out_dir = cfg.out_dir
        self.spatial_chunk = cfg.spatial_chunk
        self.cluster = cfg.cluster
        self.output_indices = cfg.output_indices
        self.cffwis_kwargs = cfg.cffwis_kwargs

        print(f"[{self.name}] run_type={self.run_type}, member={self.member}")

    def _load_variable(self, var_name, cfg, chunks):
        var_dir = os.path.join(self.tld, self.run_type, cfg["dir"])
        pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_{self.run_type}_{self.member}_*.nc")
        files = sorted(glob.glob(pattern))
        assert len(files) > 0, f"No files found for {var_name}: {pattern}"

        files = [f for f in files if _token_overlaps_window(os.path.basename(f).rsplit("_", 1)[-1].replace(".nc", ""))]
        assert len(files) > 0, f"No files for {var_name} in window {WINDOW_START_MONTH}..{WINDOW_END_MONTH}: {pattern}"
        print(f"[{self.name}]  {var_name}: {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}")

        da = xr.open_mfdataset(
            files, chunks=chunks, combine="nested", concat_dim="time", preprocess=fix_anonymous_time_dim
        )[cfg["nc_var"]]
        da = da.sortby("time").drop_duplicates("time")

        if var_name == "tasmax":
            da = da - 273.15
        elif var_name == "pr":
            da = da * 86400  # kg m-2 s-1 -> mm/day
        elif var_name == "hurs":
            da = da.clip(min=0, max=100)

        da.attrs["units"] = cfg["units"]
        return da

    def load(self, chunks):
        chunks = {**chunks, "time": -1}

        print(f"[{self.name}] Loading variables...")
        tas = self._load_variable("tasmax", VAR_CONFIG["tasmax"], chunks)
        pr = self._load_variable("pr", VAR_CONFIG["pr"], chunks)
        ws = self._load_variable("sfcWind", VAR_CONFIG["sfcWind"], chunks)
        hurs = self._load_variable("hurs", VAR_CONFIG["hurs"], chunks)

        # sfcWind is on a staggered (velocity) grid; regrid onto the tracer grid.
        print(f"[{self.name}] Regridding sfcWind onto tracer grid...")
        ws = regrid_to_tracer(ws, tas)
        ws = ws.chunk({"latitude": self.spatial_chunk, "longitude": self.spatial_chunk, "time": -1})
        ws.attrs["units"] = VAR_CONFIG["sfcWind"]["units"]

        print(f"[{self.name}] Clipping all variables to {WINDOW_START} .. {WINDOW_END}...")
        tas = tas.sel(time=slice(WINDOW_START, WINDOW_END))
        pr = pr.sel(time=slice(WINDOW_START, WINDOW_END))
        ws = ws.sel(time=slice(WINDOW_START, WINDOW_END))
        hurs = hurs.sel(time=slice(WINDOW_START, WINDOW_END))

        return {"tas": tas, "pr": pr, "sfcWind": ws, "hurs": hurs}

    def write(self, index_map):
        chunk_by_dim = {"time": 365, "latitude": self.spatial_chunk, "longitude": self.spatial_chunk}
        for idx_name, (da, long_name, units) in index_map.items():
            da = da.rename(idx_name)
            da.attrs = {"long_name": long_name, "units": units}
            da.encoding = {}
            da = da.drop_vars([c for c in ("height",) if c in da.coords])
            da = da.transpose("time", "latitude", "longitude")

            out_path = os.path.join(self.out_dir, f"hadgem3a_{idx_name}_{self.run_type}_{self.member}.nc")
            chunksizes = tuple(min(chunk_by_dim.get(dim, da.sizes[dim]), da.sizes[dim]) for dim in da.dims)
            enc = {idx_name: {"chunksizes": chunksizes}, "time": {"units": TIME_UNITS}}
            ds = xr.Dataset({idx_name: da})
            # Source files carry a 'bounds' attr on time (time_bnds) referencing a
            # bounds variable that is never written out here -- leaving it in
            # place produces the same "missing CF-netCDF boundary variable"
            # warning/ambiguity as the ERA5 valid_time issue. Clear it.
            if "time" in ds.coords:
                ds["time"].attrs.pop("bounds", None)
            ds.to_netcdf(out_path, encoding=enc)
            print(f"[{self.name}] Saved {idx_name} to {out_path}")
