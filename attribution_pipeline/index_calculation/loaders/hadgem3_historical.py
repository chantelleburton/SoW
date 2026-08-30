"""
HadGEM3-A Historical loader: reads hadgem3-a historical input variables and
produces baseline FWI (typically 1980-2013).

Ported from
index_calculation/fwi/FWI-HadGEM3-A_Historical/hadgem3-a_historical_fwi_calculation.py

NOTE: this loader is known to be "wrong but very close" (per the pipeline plan) —
it is kept isolated from the shared core deliberately so it can keep being tuned
without touching the ERA5 / HG3-Attribution loaders or FWICalculator.
"""

import glob
import os
import re

import xarray as xr

from attribution_pipeline.index_calculation.config import ClusterConfig, DatasetConfig
from attribution_pipeline.index_calculation.loaders.base import BaseLoader
from attribution_pipeline.index_calculation.loaders._grid_utils import regrid_to_tracer, fix_anonymous_time_dim

VAR_CONFIG = {
    "tasmax": {"dir": "tasmax/day", "nc_var": "tasmax", "units": "degC"},
    "pr": {"dir": "pr/day", "nc_var": "pr", "units": "mm/day"},
    "sfcWind": {"dir": "sfcWind/day", "nc_var": "sfcWind", "units": "m s-1"},
    "hurs": {"dir": "hurs/day", "nc_var": "hurs", "units": "%"},
}

# Historical files run 1960-01-01 .. 2013-12-30 (360-day calendar) in 6 decadal
# chunks per member. We always load from the dataset start so the FWI moisture
# codes are fully spun up by START_YEAR, but only write out START_YEAR..END_YEAR.
DATA_START_YEAR = 1970
START_YEAR = 1980
END_YEAR = 2013


def _decade_file_overlaps_range(fpath, start_year, end_year):
    m = re.search(r"_(\d{8})-(\d{8})\.nc$", os.path.basename(fpath))
    if not m:
        return False
    file_start_year = int(m.group(1)[:4])
    file_end_year = int(m.group(2)[:4])
    return file_start_year <= end_year and file_end_year >= start_year


class HadGEM3HistoricalLoader(BaseLoader):
    name = "hg3_historical"
    tld = "/data/users/opatt/HadGEM3-A-N216/historical"

    def __init__(self, member=None, out_dir=None):
        member_num = int(member if member is not None else os.environ.get("CYLC_TASK_PARAM_member", "1"))
        self.member = f"r1i1p{member_num}"

        cfg = DatasetConfig(
            name=self.name,
            out_dir=out_dir or "/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_historical",
            spatial_chunk=30,
            cluster=ClusterConfig(n_workers=3, memory_per_worker_gb=30),
            output_indices=["fwi", "dc", "dmc", "ffmc", "isi", "bui", "dsr"],
            cffwis_kwargs={"initial_start_up": True, "season_method": "WF93", "overwintering": True},
        )
        self.out_dir = cfg.out_dir
        self.spatial_chunk = cfg.spatial_chunk
        self.cluster = cfg.cluster
        self.output_indices = cfg.output_indices
        self.cffwis_kwargs = cfg.cffwis_kwargs

        print(f"[{self.name}] member={self.member}")

    def _load_variable(self, var_name, cfg, chunks):
        var_dir = os.path.join(self.tld, cfg["dir"])
        pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_historical_{self.member}_*.nc")
        files = sorted(glob.glob(pattern))
        assert len(files) > 0, f"No files found for {var_name}: {pattern}"

        files = [f for f in files if _decade_file_overlaps_range(f, DATA_START_YEAR, END_YEAR)]
        assert len(files) > 0, f"No files for {var_name} in range {DATA_START_YEAR}..{END_YEAR}: {pattern}"
        print(f"[{self.name}]  {var_name}: {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}")

        da = xr.open_mfdataset(
            files, chunks=chunks, combine="nested", concat_dim="time", preprocess=fix_anonymous_time_dim
        )[cfg["nc_var"]]
        da = da.sortby("time").drop_duplicates("time")

        if var_name == "tasmax":
            da = da - 273.15
        elif var_name == "pr":
            da = da * 86400  # kg m-2 s-1 -> mm/day

        da.attrs["units"] = cfg["units"]
        # normalise dataset-native lat/lon dims to latitude/longitude
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "latitude"
        if "lon" in da.dims:
            rename["lon"] = "longitude"
        if rename:
            da = da.rename(rename)
        return da

    def load(self, chunks):
        # source files use lat/lon; translate the requested chunk spec for open_mfdataset
        native_chunks = {"lat": chunks["latitude"], "lon": chunks["longitude"], "time": -1}

        print(f"[{self.name}] Loading variables...")
        tas = self._load_variable("tasmax", VAR_CONFIG["tasmax"], native_chunks)
        pr = self._load_variable("pr", VAR_CONFIG["pr"], native_chunks)
        ws = self._load_variable("sfcWind", VAR_CONFIG["sfcWind"], native_chunks)
        hurs = self._load_variable("hurs", VAR_CONFIG["hurs"], native_chunks)
        hurs = hurs.clip(min=0, max=100)

        print(f"[{self.name}] Regridding sfcWind onto tracer grid...")
        ws = regrid_to_tracer(ws, tas)
        ws = ws.chunk({"latitude": self.spatial_chunk, "longitude": self.spatial_chunk, "time": -1})
        ws.attrs["units"] = VAR_CONFIG["sfcWind"]["units"]

        return {"tas": tas, "pr": pr, "sfcWind": ws, "hurs": hurs}

    def trim_output(self, index_map):
        output_start = f"{START_YEAR}-01-01"
        output_end = f"{END_YEAR}-12-30"
        print(f"[{self.name}] Trimming output to {output_start} .. {output_end}...")
        return {
            idx_name: (da.sel(time=slice(output_start, output_end)), long_name, units)
            for idx_name, (da, long_name, units) in index_map.items()
        }

    def write(self, index_map):
        chunk_by_dim = {"time": 360, "latitude": self.spatial_chunk, "longitude": self.spatial_chunk}
        for idx_name, (da, long_name, units) in index_map.items():
            da = da.rename(idx_name)
            da.attrs = {"long_name": long_name, "units": units}
            da.encoding = {}
            da = da.drop_vars([c for c in ("height",) if c in da.coords])
            da = da.transpose("time", "latitude", "longitude")

            out_path = os.path.join(
                self.out_dir, f"hadgem3a_{idx_name}_historical_{self.member}_{START_YEAR}-{END_YEAR}_modified.nc"
            )
            chunksizes = tuple(min(chunk_by_dim.get(dim, da.sizes[dim]), da.sizes[dim]) for dim in da.dims)
            enc = {idx_name: {"chunksizes": chunksizes}}
            ds = xr.Dataset({idx_name: da})
            ds.to_netcdf(out_path, encoding=enc)
            print(f"[{self.name}] Saved {idx_name} to {out_path}")
