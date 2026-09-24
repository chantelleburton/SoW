"""
ERA5 loader: reads OBS-ERA5 daily variables and produces FWI for the period up
to the present day.

Ported from index_calculation/fwi/FWI-ERA5/era5_fwi_calculation.py — see that
file's module docstring for background on the DC spin-up / 10-year block /
wind-time-reconstruction quirks this loader replicates.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from attribution_pipeline.index_calculation.config import (
    ClusterConfig,
    DatasetConfig,
)
from attribution_pipeline.index_calculation.loaders.base import BaseLoader
from attribution_pipeline.pipeline_config import (
    ERA5_OBS_BASEPATH,
    RAW_FWI_ERA5,
)

# Fixed time units reference for all yearly output files -- see write() below.
TIME_UNITS = "days since 1900-01-01"

WIND_OPTIONS = {
    # Derived from daily-mean u/v components (hypot(u, v)) rather than a
    # precomputed daily-mean wind-speed file. NOTE: hypot(mean(u), mean(v)) is
    # not exactly the same as mean(hypot(u, v)) -- vector-averaging u/v first
    # underestimates true scalar wind speed when direction varies within the
    # day -- but this uses the only daily-resolution u/v data available
    # on-disk (no hourly u/v archive here).
    "mean": {
        "u_subdir": ("10m_u_component_of_wind", "daily_mean"),
        "v_subdir": ("10m_v_component_of_wind", "daily_mean"),
        "u_pattern": "era5_daily_mean_10m_u_component_of_wind_{year}*.nc",
        "v_pattern": "era5_daily_mean_10m_v_component_of_wind_{year}*.nc",
        "u_var": "u10",
        "v_var": "v10",
        "label": "Mean_Wind",
    },
    "max": {
        "subdir": "10m_max_wind_speed",
        "pattern": "era5_daily_max_10m_wind_speed_{year}-*.nc",
        "var": "wind_speed_max",
        "label": "Max_Wind",
    },
}
RH_OPTIONS = {
    "mean": {
        "subdirs": ("relative_humidity", "mean"),
        "pattern": "era5_daily_mean_relative_humidity_{year}*.nc",
        "label": "Mean_RH",
    },
    "minimum": {
        "subdirs": ("relative_humidity", "minimum"),
        "pattern": "era5_daily_minimum_relative_humidity_{year}*.nc",
        "label": "Minimum_RH",
    },
}


def _fix_valid_time(da):
    # ERA5's GRIB source data carries a leftover 'valid_time' coordinate
    # alongside the real 'time' dim -- CF standard_name='time' on both makes
    # Iris's cube.coord('time') ambiguous ("found 2 coordinates") once written
    # out and reloaded. Rename/drop it here so it never propagates downstream.
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "valid_time" in da.coords:
        da = da.drop_vars("valid_time")
    return da


class ERA5Loader(BaseLoader):
    name = "era5"
    basepath = ERA5_OBS_BASEPATH

    def __init__(
        self,
        start_year=None,
        wind_stat=None,
        rh_stat=None,
        max_end_year=None,
        out_dir=None,
    ):
        self.start_year = start_year or int(
            os.environ.get("CYLC_TASK_PARAM_start_year", 2025)
        )
        self.wind_stat = (
            (wind_stat or os.environ.get("CYLC_TASK_PARAM_wind_stat", "mean"))
            .strip()
            .lower()
        )
        self.rh_stat = (
            (rh_stat or os.environ.get("CYLC_TASK_PARAM_rh_stat", "mean"))
            .strip()
            .lower()
        )
        max_end_year = max_end_year or int(
            os.environ.get("MAX_END_YEAR", 2026)
        )
        self.end_year = min(self.start_year + 10, max_end_year)

        if self.wind_stat not in WIND_OPTIONS:
            raise ValueError(
                f"Unsupported wind_stat='{self.wind_stat}'. Valid options: {sorted(WIND_OPTIONS)}"
            )
        if self.rh_stat not in RH_OPTIONS:
            raise ValueError(
                f"Unsupported rh_stat='{self.rh_stat}'. Valid options: {sorted(RH_OPTIONS)}"
            )

        self.wind_cfg = WIND_OPTIONS[self.wind_stat]
        self.rh_cfg = RH_OPTIONS[self.rh_stat]
        self.run_label = f"{self.rh_cfg['label']}_{self.wind_cfg['label']}"

        cfg = DatasetConfig(
            name=self.name,
            out_dir=out_dir or RAW_FWI_ERA5,
            spatial_chunk=90,
            cluster=ClusterConfig(n_workers=3, memory_per_worker_gb=40),
            cffwis_kwargs={"initial_start_up": True},
        )
        self.out_dir = cfg.out_dir
        self.spatial_chunk = cfg.spatial_chunk
        self.cluster = cfg.cluster
        self.output_indices = cfg.output_indices
        self.cffwis_kwargs = cfg.cffwis_kwargs

        print(
            f"[{self.name}] start_year={self.start_year}, end_year={self.end_year}, "
            f"wind_stat={self.wind_stat}, rh_stat={self.rh_stat}, run_label={self.run_label}"
        )

    @property
    def years(self):
        return range(self.start_year, self.end_year + 1)

    def load(self, chunks):
        years = self.years

        tas_files = []
        for y in years:
            tas_files += sorted(
                glob.glob(
                    os.path.join(
                        self.basepath,
                        "2m_temperature",
                        "daily_maximum",
                        f"era5_daily_maximum_2m_temperature_{y}*.nc",
                    )
                )
            )
        if len(tas_files) == 0:
            raise FileNotFoundError(
                f"No temperature files found in {self.basepath}/2m_temperature/daily_maximum/"
            )
        tas = xr.open_mfdataset(tas_files, chunks=chunks)["t2m"] - 273.15
        tas = _fix_valid_time(tas)
        tas.attrs["units"] = "degC"

        pr_files = []
        for y in years:
            pr_files += sorted(
                glob.glob(
                    os.path.join(
                        self.basepath,
                        "total_precipitation",
                        "daily_sum",
                        f"era5_daily_sum_total_precipitation_{y}*.nc",
                    )
                )
            )
        if len(pr_files) == 0:
            raise FileNotFoundError(
                f"No precipitation files found in {self.basepath}/total_precipitation/daily_sum/"
            )
        pr = xr.open_mfdataset(pr_files, chunks=chunks)["tp"] * 1000  # m to mm
        pr = _fix_valid_time(pr)
        pr.attrs["units"] = "mm/day"

        if self.wind_stat == "mean":
            # Wind (mean) -- derived from daily-mean u/v component files
            # (hypot(u, v)); these have well-formed CF time encoding like
            # tas/pr/hurs, so no decode_times=False workaround needed here.
            u_files = []
            v_files = []
            for y in years:
                u_files += sorted(
                    glob.glob(
                        os.path.join(
                            self.basepath,
                            *self.wind_cfg["u_subdir"],
                            self.wind_cfg["u_pattern"].format(year=y),
                        )
                    )
                )
                v_files += sorted(
                    glob.glob(
                        os.path.join(
                            self.basepath,
                            *self.wind_cfg["v_subdir"],
                            self.wind_cfg["v_pattern"].format(year=y),
                        )
                    )
                )
            if len(u_files) == 0:
                raise FileNotFoundError(
                    f"No u-wind files found in {self.basepath}/{os.path.join(*self.wind_cfg['u_subdir'])}/"
                )
            if len(v_files) == 0:
                raise FileNotFoundError(
                    f"No v-wind files found in {self.basepath}/{os.path.join(*self.wind_cfg['v_subdir'])}/"
                )

            u = xr.open_mfdataset(u_files, chunks=chunks)[
                self.wind_cfg["u_var"]
            ]
            v = xr.open_mfdataset(v_files, chunks=chunks)[
                self.wind_cfg["v_var"]
            ]
            u = _fix_valid_time(u)
            v = _fix_valid_time(v)

            ws = np.hypot(u, v)
            ws = ws.chunk(chunks)
            ws.attrs["units"] = "m s-1"
        else:
            # Wind (max) -- time encoding in these files is broken (produced by
            # a separate pipeline), so we load with decode_times=False and
            # reconstruct time from the YYYY-MM in each filename.
            wind_files = []
            for y in years:
                wind_files += sorted(
                    glob.glob(
                        os.path.join(
                            self.basepath,
                            self.wind_cfg["subdir"],
                            self.wind_cfg["pattern"].format(year=y),
                        )
                    )
                )
            if len(wind_files) == 0:
                raise FileNotFoundError(
                    f"No wind files found in {self.basepath}/{self.wind_cfg['subdir']}/"
                )
            ws_parts = []
            for fpath in wind_files:
                ds_wind = xr.open_dataset(
                    fpath, decode_times=False, chunks=chunks
                )
                da = ds_wind[self.wind_cfg["var"]]
                m = re.search(r"(\d{4})-(\d{2})\.nc$", os.path.basename(fpath))
                if not m:
                    raise FileNotFoundError(
                        f"Cannot parse year-month from wind filename: {fpath}"
                    )
                yyyy, mm = int(m.group(1)), int(m.group(2))
                n_days = da.sizes["time"]
                new_time = pd.date_range(
                    f"{yyyy}-{mm:02d}-01", periods=n_days, freq="D"
                )
                da = da.assign_coords(time=new_time)
                da = _fix_valid_time(da)
                ws_parts.append(da)
            ws = xr.concat(ws_parts, dim="time")
            ws = ws.chunk(chunks)
            ws.attrs["units"] = "m s-1"

        hurs_files = []
        for y in years:
            hurs_files += sorted(
                glob.glob(
                    os.path.join(
                        self.basepath,
                        *self.rh_cfg["subdirs"],
                        self.rh_cfg["pattern"].format(year=y),
                    )
                )
            )
        if len(hurs_files) == 0:
            raise FileNotFoundError(
                f"No humidity files found in {self.basepath}/{'/'.join(self.rh_cfg['subdirs'])}/"
            )
        hurs = xr.open_mfdataset(hurs_files, chunks=chunks)["hurs"]
        hurs = _fix_valid_time(hurs)

        # normalise time-of-day so alignment works across sources
        tas = tas.assign_coords(time=tas.indexes["time"].normalize())
        pr = pr.assign_coords(time=pr.indexes["time"].normalize())
        ws = ws.assign_coords(time=ws.indexes["time"].normalize())
        hurs = hurs.assign_coords(time=hurs.indexes["time"].normalize())

        return {"tas": tas, "pr": pr, "sfcWind": ws, "hurs": hurs}

    def trim_output(self, index_map):
        # The block's first year (start_year) exists only to spin up the moisture
        # codes (esp. DC, ~52 day lag) and must be discarded before writing.
        output_years = [y for y in self.years if y > self.start_year]
        print(
            f"[{self.name}] Discarding spin-up year {self.start_year}; writing yearly files for {output_years}"
        )
        self._output_years = output_years
        return index_map

    def write(self, index_map):
        for idx_name, (da, long_name, units) in index_map.items():
            # Reset (not just update) attrs: xclim's cffwis_indices computation
            # leaks the original 'tas' input's raw GRIB attrs (GRIB_paramId,
            # GRIB_cfVarName='t2m', coordinates='day_of_month number surface',
            # etc.) through via xarray's keep_attrs propagation. Patching just
            # long_name/units with .update() leaves that stale metadata in
            # place, which later confuses Iris's CF coordinate parsing on load
            # (e.g. a leftover 'bounds' reference to a nonexistent variable).
            da = da.copy()
            da.attrs = {"long_name": long_name, "units": units}
            # Also strip any leaked non-dimension coordinates (e.g. 'number',
            # 'surface', 'day_of_month') that came along for the ride from the
            # raw GRIB-derived inputs and aren't meaningful for the computed index.
            extra_coords = [
                c
                for c in da.coords
                if c not in ("time", "latitude", "longitude")
            ]
            if extra_coords:
                da = da.drop_vars(extra_coords)
            # xclim's cffwis_indices output dim order follows its inputs, which
            # (unlike the HadGEM3 loaders) isn't guaranteed to be time-first --
            # force it here so downstream shapefile masking (which broadcasts a
            # 2-D (lat, lon) mask against the cube's trailing dims) lines up
            # correctly instead of a shape mismatch against a stray leading dim.
            da = da.transpose("time", "latitude", "longitude")
            for y in self._output_years:
                da_year = da.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
                n_times_year = da_year.sizes["time"]
                if n_times_year == 0:
                    print(
                        f"[{self.name}]  Skipping {idx_name} {y}: no data in range"
                    )
                    continue
                out_path = os.path.join(
                    self.out_dir, f"era5_{idx_name}_{self.run_label}_{y}.nc"
                )
                # Fixed time units reference (rather than xarray's per-file default,
                # which would pick each year's own start date) so Iris can
                # concatenate cubes loaded from different yearly files -- otherwise
                # concatenate_cube() sees differing time-coordinate metadata and errors.
                enc = {
                    idx_name: {
                        "chunksizes": (
                            n_times_year,
                            self.spatial_chunk,
                            self.spatial_chunk,
                        )
                    },
                    "time": {"units": TIME_UNITS},
                }
                ds = xr.Dataset({idx_name: da_year})
                ds["time"].attrs = {}
                ds.to_netcdf(out_path, encoding=enc)
                print(
                    f"[{self.name}] Saved {idx_name} {y} ({n_times_year} days) to {out_path}"
                )
