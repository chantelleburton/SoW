"""Shared ERA5 variable-loading mechanics.

Extracted from Exploratory_Work/xclim_work/explore_xclim_FWI.py. Both
ERA5PresentSource and ERA5HistoricalSource mix this in — the loading logic is
identical, only the start_year/end_year (and therefore which files get
globbed) differ, driven entirely by config.
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd
import xarray as xr

WIND_OPTIONS = {
    "mean": {
        "subdir": "10m_mean_wind_speed",
        "pattern": "era5_daily_mean_10m_wind_speed_{year}-*.nc",
        "var": "wind_speed_mean",
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


class ERA5LoaderMixin:
    """Requires self.cfg.extra to define: basepath, wind_stat, rh_stat.
    Requires the concrete class to pass start_year/end_year into load_variables."""

    def _run_label(self, wind_stat: str, rh_stat: str) -> str:
        return f"{RH_OPTIONS[rh_stat]['label']}_{WIND_OPTIONS[wind_stat]['label']}"

    def load_variables(self, start_year: int, end_year: int, chunks: dict) -> dict[str, xr.DataArray]:
        basepath = self.cfg.extra["basepath"]
        wind_stat = self.cfg.extra.get("wind_stat", "mean")
        rh_stat = self.cfg.extra.get("rh_stat", "mean")
        if wind_stat not in WIND_OPTIONS:
            raise ValueError(f"Unsupported wind_stat='{wind_stat}'")
        if rh_stat not in RH_OPTIONS:
            raise ValueError(f"Unsupported rh_stat='{rh_stat}'")
        wind_cfg = WIND_OPTIONS[wind_stat]
        rh_cfg = RH_OPTIONS[rh_stat]
        years = range(start_year, end_year + 1)

        tas_files = []
        for y in years:
            tas_files += sorted(glob.glob(os.path.join(
                basepath, "2m_temperature", "daily_maximum", f"era5_daily_maximum_2m_temperature_{y}*.nc")))
        assert tas_files, f"No temperature files found for {start_year}-{end_year}"
        tas = xr.open_mfdataset(tas_files, chunks=chunks)["t2m"] - 273.15
        if "valid_time" in tas.dims:
            tas = tas.rename({"valid_time": "time"})
        tas.attrs["units"] = "degC"

        pr_files = []
        for y in years:
            pr_files += sorted(glob.glob(os.path.join(
                basepath, "total_precipitation", "daily_sum", f"era5_daily_sum_total_precipitation_{y}*.nc")))
        assert pr_files, f"No precipitation files found for {start_year}-{end_year}"
        pr = xr.open_mfdataset(pr_files, chunks=chunks)["tp"] * 1000
        if "valid_time" in pr.dims:
            pr = pr.rename({"valid_time": "time"})
        pr.attrs["units"] = "mm/day"

        wind_files = []
        for y in years:
            wind_files += sorted(glob.glob(os.path.join(basepath, wind_cfg["subdir"], wind_cfg["pattern"].format(year=y))))
        assert wind_files, f"No wind files found for {start_year}-{end_year}"
        ws_parts = []
        for fpath in wind_files:
            ds_wind = xr.open_dataset(fpath, decode_times=False, chunks=chunks)
            da = ds_wind[wind_cfg["var"]]
            m = re.search(r"(\d{4})-(\d{2})\.nc$", os.path.basename(fpath))
            assert m, f"Cannot parse year-month from wind filename: {fpath}"
            yyyy, mm = int(m.group(1)), int(m.group(2))
            new_time = pd.date_range(f"{yyyy}-{mm:02d}-01", periods=da.sizes["time"], freq="D")
            ws_parts.append(da.assign_coords(time=new_time))
        ws = xr.concat(ws_parts, dim="time").chunk(chunks)
        ws.attrs["units"] = "m s-1"

        hurs_files = []
        for y in years:
            hurs_files += sorted(glob.glob(os.path.join(basepath, *rh_cfg["subdirs"], rh_cfg["pattern"].format(year=y))))
        assert hurs_files, f"No humidity files found for {start_year}-{end_year}"
        hurs = xr.open_mfdataset(hurs_files, chunks=chunks)["hurs"]
        if "valid_time" in hurs.dims:
            hurs = hurs.rename({"valid_time": "time"})

        tas = tas.assign_coords(time=tas.indexes["time"].normalize())
        pr = pr.assign_coords(time=pr.indexes["time"].normalize())
        ws = ws.assign_coords(time=ws.indexes["time"].normalize())
        hurs = hurs.assign_coords(time=hurs.indexes["time"].normalize())
        tas, pr, ws, hurs = xr.align(tas, pr, ws, hurs, join="inner")
        if tas.time.size == 0:
            raise ValueError("No overlapping dates after alignment.")

        tas = tas.ffill(dim="time")
        pr = pr.ffill(dim="time")
        ws = ws.ffill(dim="time")
        hurs = hurs.ffill(dim="time")
        hurs = hurs.clip(min=0, max=100)

        # xclim treats time as a core dimension, so it must be a single chunk
        # before being handed to cffwis_indices (spatial chunking is preserved).
        compute_chunks = {**chunks, "time": -1}
        tas = tas.chunk(compute_chunks)
        pr = pr.chunk(compute_chunks)
        ws = ws.chunk(compute_chunks)
        hurs = hurs.chunk(compute_chunks)

        return {"tasmax": tas, "pr": pr, "sfcWind": ws, "hurs": hurs}
