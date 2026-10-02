#!/usr/bin/env python
"""
One-off fix for existing per-year FWI/DSR NetCDF files written before the
TIME_UNITS fix in attribution_pipeline/index_calculation/loaders/era5.py and
hadgem3_historical.py: each yearly file previously got its own auto-derived
time `units` reference (e.g. "days since 2019-01-01"), which made Iris's
concatenate_cube() fail with "Dimension coordinates metadata differ: time
!= time" when metrics tried to join multiple years into one cube.

This re-encodes files IN PLACE with a fixed time units reference. It does
NOT recompute FWI -- the underlying dates/values are unchanged, only the
NetCDF time encoding is normalised so Iris treats the time coordinate
metadata as identical across files.

Usage:
    python spice_scripts/fix_time_encoding.py \
        /data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/era5 \
        /data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_historical
"""

import sys
import os
import glob
import tempfile

import xarray as xr

TIME_UNITS = "days since 1900-01-01"


def fix_file(path):
    with xr.open_dataset(path) as ds:
        ds = ds.load()
        enc = {}
        for var in ds.data_vars:
            old_enc = ds[var].encoding
            var_enc = {}
            if "chunksizes" in old_enc:
                var_enc["chunksizes"] = old_enc["chunksizes"]
            enc[var] = var_enc
        enc["time"] = {"units": TIME_UNITS}

        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".nc")
        os.close(fd)
        try:
            ds.to_netcdf(tmp_path, encoding=enc)
        except Exception:
            os.remove(tmp_path)
            raise
    os.replace(tmp_path, path)
    print(f"Fixed: {path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: fix_time_encoding.py <dir> [<dir> ...]")

    for d in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(d, "*.nc")))
        print(f"{d}: {len(files)} files")
        for f in files:
            fix_file(f)


if __name__ == "__main__":
    main()
