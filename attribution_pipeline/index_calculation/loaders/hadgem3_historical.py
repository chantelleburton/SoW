"""
HadGEM3-A Historical loader: reads hadgem3-a historical input variables and
produces baseline FWI (typically 1980-2013).

Ported from
index_calculation/fwi/FWI-HadGEM3-A_Historical/hadgem3-a_historical_fwi_calculation.py

NOTE: this loader is known to be "wrong but very close" (per the pipeline plan) —
it is kept isolated from the shared core deliberately so it can keep being tuned
without touching the ERA5 / HG3-Attribution loaders or FWICalculator.

Runs in independent ~10-year BLOCKS (mirroring ERA5Loader's segmented
start_year/end_year pattern) rather than one 1970-2013 monolithic call, to
reduce memory footprint. Each block loads SPIN_UP_YEARS of lead-in before its
nominal start, discards that lead-in year(s) before writing (same idea as
ERA5's spin-up-year discard)."""

import glob
import os
import re

import xarray as xr

from attribution_pipeline.index_calculation.config import (
    ClusterConfig,
    DatasetConfig,
)
from attribution_pipeline.index_calculation.loaders._grid_utils import (
    fix_anonymous_time_dim,
    regrid_to_tracer,
)
from attribution_pipeline.index_calculation.loaders.base import BaseLoader
from attribution_pipeline.pipeline_config import RAW_FWI_HG3_HISTORICAL

VAR_CONFIG = {
    "tasmax": {"dir": "tasmax/day", "nc_var": "tasmax", "units": "degC"},
    "pr": {"dir": "pr/day", "nc_var": "pr", "units": "mm/day"},
    "sfcWind": {"dir": "sfcWind/day", "nc_var": "sfcWind", "units": "m s-1"},
    "hurs": {"dir": "hurs/day", "nc_var": "hurs", "units": "%"},
}

# Historical files run 1960-01-01 .. 2013-12-30 (360-day calendar) in 6 decadal
# chunks per member. Overall output window is END_YEAR-bounded; each block
# loads SPIN_UP_YEARS of lead-in before its own block_start_year for moisture
# code spin-up, then discards that lead-in before writing.
END_YEAR = 2013
BLOCK_LENGTH = 10
SPIN_UP_YEARS = 1


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

    def __init__(self, member=None, block_start_year=None, out_dir=None):
        member_num = int(
            member
            if member is not None
            else os.environ.get("CYLC_TASK_PARAM_member", "1")
        )
        self.member = f"r1i1p{member_num}"

        self.block_start_year = int(
            block_start_year
            if block_start_year is not None
            else os.environ.get("CYLC_TASK_PARAM_block_start_year", "1970")
        )
        self.block_end_year = min(
            self.block_start_year + BLOCK_LENGTH - 1, END_YEAR
        )
        # lead-in years loaded for spin-up but discarded before writing
        self.data_start_year = self.block_start_year - SPIN_UP_YEARS

        cfg = DatasetConfig(
            name=self.name,
            out_dir=out_dir or RAW_FWI_HG3_HISTORICAL,
            spatial_chunk=30,
            cluster=ClusterConfig(n_workers=3, memory_per_worker_gb=30),
            cffwis_kwargs={"initial_start_up": True},
        )
        self.out_dir = cfg.out_dir
        self.spatial_chunk = cfg.spatial_chunk
        self.cluster = cfg.cluster
        self.output_indices = cfg.output_indices
        self.cffwis_kwargs = cfg.cffwis_kwargs

        print(
            f"[{self.name}] member={self.member}, block={self.block_start_year}-{self.block_end_year} "
            f"(loading from {self.data_start_year} for spin-up)"
        )

    @property
    def output_years(self):
        return range(self.block_start_year, self.block_end_year + 1)

    def _load_variable(self, var_name, cfg, chunks):
        var_dir = os.path.join(self.tld, cfg["dir"])
        pattern = os.path.join(
            var_dir,
            f"{var_name}_day_HadGEM3-A-N216_historical_{self.member}_*.nc",
        )
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No files found for {var_name}: {pattern}")

        files = [
            f
            for f in files
            if _decade_file_overlaps_range(
                f, self.data_start_year, self.block_end_year
            )
        ]
        if len(files) == 0:
            raise FileNotFoundError(
                f"No files for {var_name} in range {self.data_start_year}..{self.block_end_year}: {pattern}"
            )
        print(
            f"[{self.name}]  {var_name}: {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}"
        )

        da = xr.open_mfdataset(
            files,
            chunks=chunks,
            combine="nested",
            concat_dim="time",
            preprocess=fix_anonymous_time_dim,
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
        native_chunks = {
            "lat": chunks["latitude"],
            "lon": chunks["longitude"],
            "time": -1,
        }

        print(f"[{self.name}] Loading variables...")
        tas = self._load_variable(
            "tasmax", VAR_CONFIG["tasmax"], native_chunks
        )
        pr = self._load_variable("pr", VAR_CONFIG["pr"], native_chunks)
        ws = self._load_variable(
            "sfcWind", VAR_CONFIG["sfcWind"], native_chunks
        )
        hurs = self._load_variable("hurs", VAR_CONFIG["hurs"], native_chunks)
        hurs = hurs.clip(min=0, max=100)

        print(f"[{self.name}] Regridding sfcWind onto tracer grid...")
        ws = regrid_to_tracer(ws, tas)
        ws = ws.chunk(
            {
                "latitude": self.spatial_chunk,
                "longitude": self.spatial_chunk,
                "time": -1,
            }
        )
        ws.attrs["units"] = VAR_CONFIG["sfcWind"]["units"]

        return {"tas": tas, "pr": pr, "sfcWind": ws, "hurs": hurs}

    def trim_output(self, index_map):
        output_start = f"{self.block_start_year}-01-01"
        output_end = f"{self.block_end_year}-12-30"
        print(
            f"[{self.name}] Discarding spin-up ({self.data_start_year}-{self.block_start_year - 1}); "
            f"trimming output to {output_start} .. {output_end}..."
        )
        return {
            idx_name: (
                da.sel(time=slice(output_start, output_end)),
                long_name,
                units,
            )
            for idx_name, (da, long_name, units) in index_map.items()
        }

    def write(self, index_map):
        for idx_name, (da, long_name, units) in index_map.items():
            da = da.rename(idx_name)
            da.attrs = {"long_name": long_name, "units": units}
            da.encoding = {}
            da = da.drop_vars([c for c in ("height",) if c in da.coords])
            da = da.transpose("time", "latitude", "longitude")

            for y in self.output_years:
                da_year = da.sel(time=slice(f"{y}-01-01", f"{y}-12-30"))
                n_times_year = da_year.sizes["time"]
                if n_times_year == 0:
                    print(
                        f"[{self.name}]  Skipping {idx_name} {y}: no data in range"
                    )
                    continue
                out_path = os.path.join(
                    self.out_dir,
                    f"hadgem3a_{idx_name}_historical_{self.member}_{y}.nc",
                )
                chunksizes = (
                    n_times_year,
                    self.spatial_chunk,
                    self.spatial_chunk,
                )
                enc = {
                    idx_name: {"chunksizes": chunksizes},
                    "time": {"units": self.TIME_UNITS},
                }
                ds = xr.Dataset({idx_name: da_year})
                # Source files carry 'bounds' attrs on time/lat/lon (time_bnds,
                # lat_bnds, lon_bnds) referencing bounds variables that are never
                # written out here -- leaving them in place produces the same
                # "missing CF-netCDF boundary variable" warning/ambiguity as the
                # ERA5 valid_time issue. Clear them so no dangling reference survives.
                for coord_name in ("time", "latitude", "longitude"):
                    if coord_name in ds.coords:
                        ds[coord_name].attrs.pop("bounds", None)
                ds.to_netcdf(out_path, encoding=enc)
                print(
                    f"[{self.name}] Saved {idx_name} {y} ({n_times_year} days) to {out_path}"
                )
