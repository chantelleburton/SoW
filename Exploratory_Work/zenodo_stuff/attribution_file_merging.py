"""
Merge per-member Zenodo attribution NetCDF files into one file per region/run-type.

Reads HadGEM3-A_FWI_2020-2024_*_{Region}_{RunType}.nc files from the Zenodo
interim directory and stacks them along a 'member' dimension that always holds
all 525 canonical members (r001i1p1 .. r105i1p5). Missing members and missing
days are filled with NaN. Days are matched by calendar date.

Outputs:
    {zenodo_output_folder}/HadGEM3-A_FWI_2020-2024_{Region}_{RunType}.nc
    {zenodo_output_folder}/HadGEM3-A_FWI_2020-2024_member_completeness.csv
"""

import csv
import datetime
import glob
import os
import re
from collections import defaultdict

import cftime
import dask
import dask.array as da
import iris
import iris.coords
import iris.cube
import numpy as np

zenodo_input_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Interim/'
zenodo_output_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Export/'
completeness_csv = os.path.join(zenodo_output_folder, 'HadGEM3-A_FWI_2020-2024_member_completeness.csv')

START_YEAR, END_YEAR = 2020, 2024
MEMBER_IDS = [f"r{r:03d}i1p{p}" for r in range(1, 106) for p in range(1, 6)]


def _canonical_dates(calendar, hour, minute):
    d = cftime.datetime(START_YEAR, 1, 1, hour, minute, calendar=calendar)
    dates = []
    while d.year <= END_YEAR:
        dates.append(d)
        d = d + datetime.timedelta(days=1)
    return dates


def _time_lat_lon(cube):
    order = [cube.coord_dims(name)[0] for name in ('time', 'latitude', 'longitude')]
    if order != [0, 1, 2]:
        cube.transpose(order)
    return cube


def _member_dates(cube):
    t = cube.coord('time')
    return [(d.year, d.month, d.day) for d in t.units.num2date(t.points)]


def _member_slab(fpath, date_index, template):
    """Return a (time, lat, lon) float32 array for one member, NaN where missing."""
    nt = len(date_index)
    ny, nx = template.shape[1:]
    slab = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    if fpath is None:
        return slab

    cube = _time_lat_lon(iris.load_cube(fpath))
    for name in ('latitude', 'longitude'):
        if not np.allclose(cube.coord(name).points, template.coord(name).points):
            raise ValueError(f"{fpath}: {name} grid does not match template")
    if cube.coord('time').units.calendar != template.coord('time').units.calendar:
        raise ValueError(f"{fpath}: calendar does not match template")

    data = np.ma.filled(cube.data.astype(np.float32), np.nan)
    for i, ymd in enumerate(_member_dates(cube)):
        if ymd in date_index:
            slab[date_index[ymd]] = data[i]
    return slab


# ---- Discover per-member files ----
pattern = os.path.join(zenodo_input_folder, 'HadGEM3-A_FWI_2020-2024_*.nc')
all_files = sorted(glob.glob(pattern))

fname_re = re.compile(
    r'HadGEM3-A_FWI_2020-2024_(r\d+i1p\d+)_([A-Za-z]+)_(Factual|Counterfactual)\.nc$'
)

# (region, run_type) -> {member_id: fpath}
groups = defaultdict(dict)
for fpath in all_files:
    m = fname_re.match(os.path.basename(fpath))
    if m:
        member_id, region, run_type = m.group(1), m.group(2), m.group(3)
        groups[(region, run_type)][member_id] = fpath

print(f"Found {len(all_files)} per-member files in {len(groups)} region/run-type groups")
os.makedirs(zenodo_output_folder, exist_ok=True)

completeness_rows = []

# Load one member at a time while saving, so memory holds a single member slab
dask.config.set(scheduler='synchronous')

for (region, run_type), files in sorted(groups.items()):
    print(f"\nMerging {len(files)}/{len(MEMBER_IDS)} members for {region} / {run_type}")

    template = _time_lat_lon(iris.load_cube(files[sorted(files)[0]]))
    first_date = template.coord('time').units.num2date(template.coord('time').points[0])
    time_units = template.coord('time').units
    dates = _canonical_dates(time_units.calendar, first_date.hour, first_date.minute)
    date_index = {(d.year, d.month, d.day): i for i, d in enumerate(dates)}
    nt = len(dates)
    ny, nx = template.shape[1:]

    member_arrays = []
    for member_id in MEMBER_IDS:
        fpath = files.get(member_id)
        if fpath is None:
            n_present = 0
        else:
            present = set(_member_dates(iris.load_cube(fpath))) & date_index.keys()
            n_present = len(present)
        completeness_rows.append({
            'region': region,
            'run_type': run_type,
            'member_id': member_id,
            'file_present': fpath is not None,
            'n_days_present': n_present,
            'n_days_expected': nt,
            'n_days_nan_filled': nt - n_present,
        })
        member_arrays.append(da.from_delayed(
            dask.delayed(_member_slab)(fpath, date_index, template),
            shape=(nt, ny, nx), dtype=np.float32,
        ))

    merged = iris.cube.Cube(
        da.stack(member_arrays),
        standard_name=template.standard_name,
        long_name=template.long_name,
        var_name=template.var_name,
        units=template.units,
        attributes=template.attributes,
        dim_coords_and_dims=[
            (iris.coords.DimCoord(np.arange(1, len(MEMBER_IDS) + 1, dtype=np.int32),
                                  long_name='member', var_name='member', units='1'), 0),
            (iris.coords.DimCoord(time_units.date2num(dates), standard_name='time', units=time_units), 1),
            (template.coord('latitude').copy(), 2),
            (template.coord('longitude').copy(), 3),
        ],
        aux_coords_and_dims=[
            (iris.coords.AuxCoord(np.array(MEMBER_IDS), long_name='member_id', var_name='member_id'), 0),
        ],
    )

    output_file = os.path.join(zenodo_output_folder, f"HadGEM3-A_FWI_2020-2024_{region}_{run_type}.nc")
    iris.save(merged, output_file)
    print(f"  Saved: {output_file} (shape {merged.shape})")

with open(completeness_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(completeness_rows[0]))
    writer.writeheader()
    writer.writerows(completeness_rows)
print(f"\nSaved completeness diagnostics: {completeness_csv}")

print("\nDone.")
