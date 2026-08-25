"""
Compute per-timestep, per-grid-cell raw FWI diffs (Xclim - ImpactTB) for one
HadGEM3-A_Historical ensemble member, for the 3 primary focal regions
(Iberia, Chile, Canada).

Compares:
  - ImpactTB monthly "gwl" historical files, e.g.
    FWI_HadGEM3-A-N216_r1i1p1_historical_gwl19800101-19800201_global_day_
    initialise-from=previous-and-save-input-data=True.nc
  - xclim single-file-per-member historical output, e.g.
    hadgem3a_fwi_historical_r1i1p6_1980-2013.nc

Unlike a script that collapses to a single monthly region-95th-percentile
value, this script keeps every daily timestep and every grid cell within each
region so that both the WHEN (temporal) and WHERE (spatial) origin of any
divergence between the two pipelines can be examined.

Outputs (per member/experiment/region), written to OUT_DIR:
    diffcube_<country>_<experiment>_<member>.nc   -- full diff cube (time, lat, lon)
    daily_diff_<country>_<experiment>_<member>.csv -- per-day region stats
    meanmap_<country>_<experiment>_<member>.csv    -- time-mean/std spatial diff map
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import iris
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')

from utils.cubefuncs import apply_shapefile_inclusive  # noqa: E402

# ---- Config ----
# ImpactTB historical source: monthly ("gwl") files directly under this folder, e.g.
# FWI_HadGEM3-A-N216_r1i1p1_historical_gwl19800101-19800201_global_day_initialise-from=previous-and-save-input-data=True.nc
IMPACTTB_ROOT = '/data/users/bob.potts/sowf_data/historicalFWI/HadGEM'
# xclim historical source: one file per member covering the full period, e.g.
# hadgem3a_fwi_historical_r1i1p6_1980-2013.nc
XCLIM_DIR = '/data/scratch/bob.potts/sowf/fwi-calculation-pipeline/HadGEM3-A_Historical'
SHP_FILE = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
OUT_DIR = '/data/scratch/bob.potts/sowf/fwi-calculation-pipeline/HadGEM3-A_Historical/validation/raw_fwi_diffs'

MEMBER     = os.environ.get("CYLC_TASK_PARAM_member", "r1i1p1").strip()
EXPERIMENT = os.environ.get("CYLC_TASK_PARAM_run_type", "historical").strip()
START_YEAR = 1980
END_YEAR = 2013

# The 3 primary focal regions only (Korea/Scotland deferred).
REGION_SHAPES = {
    'Iberia': 'Northwest Iberia',
    'Chile':  'Chilean Temperate Forests and Matorral',
    'Canada': 'Midwestern Canadian Shield forests',
}

region_months = {
    'Iberia': (8,),
    'Chile':  (1, 2),
    'Canada': (7, 8),
}
def _load_fwi_cube(fpath):
    """Load the FWI cube from a file that may contain several FWI sub-indices."""
    fwi_names = {'fwi', 'Fire Weather Index', 'fire_weather_index',
                 'Canadian Fire Weather Index'}
    cubes = iris.load(fpath)
    for c in cubes:
        names = {c.var_name, c.name(), getattr(c, 'long_name', None), c.standard_name}
        if names & fwi_names:
            return c
    raise ValueError(
        f"No FWI cube found in {fpath}. Available: {[c.name() for c in cubes]}")


def _strip_aux_time_coords(cube):
    """Drop scalar month/season/year coords that block concatenation across monthly files."""
    for name in ('month', 'month_number', 'season', 'season_year', 'year'):
        if cube.coords(name):
            cube.remove_coord(name)
    t = cube.coord('time')
    t.attributes = {}
    t.var_name = None
    t.long_name = None
    t.standard_name = 'time'
    return cube


def load_impacttb(member, experiment, start_year=START_YEAR, end_year=END_YEAR):
    """Concatenate the monthly ImpactTB 'gwl' files for one member/experiment into a single cube."""
    pattern = os.path.join(
        IMPACTTB_ROOT,
        f'FWI_HadGEM3-A-N216_{member}_{experiment}_gwl*_global_day_'
        f'initialise-from=previous-and-save-input-data=True.nc')
    files = sorted(glob.glob(pattern))
    print(files)
    assert files, f"No ImpactTB files: {pattern}"

    # Only keep monthly chunks that overlap the requested year range.
    def _overlaps(fpath):
        import re
        m = re.search(r'_gwl(\d{8})-(\d{8})_', os.path.basename(fpath))
        if not m:
            return False
        return int(m.group(1)[:4]) <= end_year and int(m.group(2)[:4]) >= start_year

    files = [f for f in files if _overlaps(f)]
    assert files, f"No ImpactTB files in range {start_year}-{end_year}: {pattern}"

    cubes = iris.cube.CubeList()
    ref_units = None
    for f in files:
        c = _load_fwi_cube(f)
        c = _strip_aux_time_coords(c)
        if ref_units is None:
            ref_units = c.coord('time').units
        else:
            c.coord('time').convert_units(ref_units)
        cubes.append(c)
    iris.util.equalise_attributes(cubes)
    cube = cubes.concatenate_cube()

    # Drop any duplicate boundary timesteps between contiguous monthly files.
    _, idx = np.unique(cube.coord('time').points, return_index=True)
    if len(idx) != cube.coord('time').shape[0]:
        cube = cube[np.sort(idx)]
    return cube


def load_xclim(member, experiment, start_year=START_YEAR, end_year=END_YEAR):
    fpath = os.path.join(XCLIM_DIR, f'hadgem3a_fwi_{experiment}_{member}_{start_year}-{end_year}_modified.nc')
    assert os.path.exists(fpath), f"Missing {fpath}"
    return _load_fwi_cube(fpath)


def _months_present(cube):
    """Sorted list of unique calendar months (1-12) present in a cube's time coordinate."""
    t = cube.coord('time')
    dts = t.units.num2date(t.points)
    return sorted({d.month for d in dts})


def intersect_time(cube_a, cube_b):
    """Constrain two cubes to their common (year, month, day) timestamps.

    Returns (cube_a_aligned, cube_b_aligned), both sorted ascending in time, or
    (None, None) if there is no overlap.
    """
    ta, tb = cube_a.coord('time'), cube_b.coord('time')
    da = ta.units.num2date(ta.points)
    db = tb.units.num2date(tb.points)
    set_a = {(d.year, d.month, d.day) for d in da}
    set_b = {(d.year, d.month, d.day) for d in db}
    common = set_a & set_b
    if not common:
        return None, None

    con = iris.Constraint(
        time=lambda cell, common=common: (cell.point.year, cell.point.month, cell.point.day) in common)
    a2 = cube_a.extract(con)
    b2 = cube_b.extract(con)
    if a2 is None or b2 is None:
        return None, None
    if a2.coord('time').shape != b2.coord('time').shape:
        return None, None
    return a2, b2


def grids_match(cube_a, cube_b, tol=1e-6):
    lat_a, lat_b = cube_a.coord('latitude').points, cube_b.coord('latitude').points
    lon_a, lon_b = cube_a.coord('longitude').points, cube_b.coord('longitude').points
    return (lat_a.shape == lat_b.shape and lon_a.shape == lon_b.shape
            and np.allclose(lat_a, lat_b, atol=tol) and np.allclose(lon_a, lon_b, atol=tol))


def compute_diff_cube(xc_masked, tb_masked):
    """Grid-cell diff (Xclim - ImpactTB), built from the xclim cube's metadata."""
    diff_data = xc_masked.data - tb_masked.data
    diff_cube = xc_masked.copy(data=diff_data)
    diff_cube.rename('fwi_diff_xclim_minus_impacttb')
    diff_cube.units = '1'
    diff_cube.attributes = {}
    return diff_cube


def daily_region_stats(diff_cube, tb_masked, xc_masked):
    """Per-timestep region stats: area-weighted mean diff, max abs diff, n valid cells."""
    coords = ('longitude', 'latitude')
    for c in coords:
        if not diff_cube.coord(c).has_bounds():
            diff_cube.coord(c).guess_bounds()
    weights = iris.analysis.cartography.area_weights(diff_cube)

    mean_diff = diff_cube.collapsed(coords, iris.analysis.MEAN, weights=weights)
    abs_cube = diff_cube.copy(data=np.abs(diff_cube.data))
    max_abs = abs_cube.collapsed(coords, iris.analysis.MAX)
    tb_mean = tb_masked.collapsed(coords, iris.analysis.MEAN, weights=weights)
    xc_mean = xc_masked.collapsed(coords, iris.analysis.MEAN, weights=weights)

    time_dim = diff_cube.coord_dims('time')[0]
    other_axes = tuple(i for i in range(diff_cube.ndim) if i != time_dim)
    n_cells = np.ma.count(diff_cube.data, axis=other_axes)

    t = diff_cube.coord('time')
    dts = t.units.num2date(t.points)
    dates = [d.strftime('%Y-%m-%d') for d in dts]

    return pd.DataFrame({
        'Date': dates,
        'mean_diff': np.ma.filled(mean_diff.data, np.nan),
        'max_abs_diff': np.ma.filled(max_abs.data, np.nan),
        'mean_impacttb': np.ma.filled(tb_mean.data, np.nan),
        'mean_xclim': np.ma.filled(xc_mean.data, np.nan),
        'n_cells': n_cells,
    })


def time_mean_map(diff_cube):
    """Time-collapsed mean/std diff map, flattened to a (lat, lon, mean, std) table."""
    mean_map = diff_cube.collapsed('time', iris.analysis.MEAN)
    std_map = diff_cube.collapsed('time', iris.analysis.STD_DEV)

    lat = mean_map.coord('latitude').points
    lon = mean_map.coord('longitude').points
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')

    df = pd.DataFrame({
        'lat': lat2d.ravel(),
        'lon': lon2d.ravel(),
        'mean_diff': np.ma.filled(np.asarray(mean_map.data), np.nan).ravel(),
        'std_diff': np.ma.filled(np.asarray(std_map.data), np.nan).ravel(),
    })
    return df.dropna(subset=['mean_diff'])


def process_region(country, shape_name, tb_cube, xc_cube, expected_n_days):
    tb_masked = apply_shapefile_inclusive(SHP_FILE, shape_name, tb_cube.copy())
    xc_masked = apply_shapefile_inclusive(SHP_FILE, shape_name, xc_cube.copy())

    tb_aligned, xc_aligned = intersect_time(tb_masked, xc_masked)
    if tb_aligned is None:
        print(f"  {country}: no overlapping timestamps -- skipping")
        return

    if not grids_match(tb_aligned, xc_aligned):
        print(f"  {country}: lat/lon grids do not match -- skipping")
        return

    diff_cube = compute_diff_cube(xc_aligned, tb_aligned)

    diffcube_path = os.path.join(OUT_DIR, f'diffcube_{country}_{EXPERIMENT}_{MEMBER}.nc')
    iris.save(diff_cube, diffcube_path)

    daily_df = daily_region_stats(diff_cube, tb_aligned, xc_aligned)

    # Completeness: ImpactTB sometimes only delivers a subset of the full period
    # (e.g. 1 month out of ~62), in which case intersect_time() silently limits the
    # comparison to whatever overlap exists. Record how much of the expected xclim
    # period this member/region actually covers, so downstream aggregation can
    # separate fully-compared members from partial ones rather than treating a
    # 30-day comparison the same as a full ~1860-day one.
    n = len(daily_df)
    daily_df['n_days'] = n
    daily_df['expected_n_days'] = expected_n_days
    daily_df['pct_complete'] = 100.0 * n / expected_n_days if expected_n_days else np.nan

    daily_path = os.path.join(OUT_DIR, f'daily_diff_{country}_{EXPERIMENT}_{MEMBER}_modified.csv')
    daily_df.to_csv(daily_path, index=False)

    map_df = time_mean_map(diff_cube)
    map_path = os.path.join(OUT_DIR, f'meanmap_{country}_{EXPERIMENT}_{MEMBER}_modified.csv')
    map_df.to_csv(map_path, index=False)

    print(f"  {country}: n_days={n}/{expected_n_days} ({daily_df['pct_complete'].iloc[0]:.1f}% complete), "
          f"mean|diff|={daily_df['mean_diff'].abs().mean():.3f}, "
          f"max|diff|={daily_df['max_abs_diff'].max():.3f}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Computing raw FWI diffs: member={MEMBER}, experiment={EXPERIMENT}")

    xc_path = os.path.join(XCLIM_DIR, f'hadgem3a_fwi_{EXPERIMENT}_{MEMBER}_{START_YEAR}-{END_YEAR}.nc')
    tb_pattern = os.path.join(
        IMPACTTB_ROOT,
        f'FWI_HadGEM3-A-N216_{MEMBER}_{EXPERIMENT}_gwl*_global_day_'
        f'initialise-from=previous-and-save-input-data=True.nc')
    has_xc = os.path.exists(xc_path)
    has_tb = bool(glob.glob(tb_pattern))
    if not (has_xc and has_tb):
        print(f"Skipping {MEMBER}/{EXPERIMENT}: missing inputs (xclim={has_xc}, impacttb={has_tb}).")
        sys.exit(0)

    tb_cube = load_impacttb(MEMBER, EXPERIMENT)
    print(tb_cube)
    xc_cube = load_xclim(MEMBER, EXPERIMENT)

    # ImpactTB was only ever computed for specific fire-season months per region
    # (e.g. Aug for Iberia, Jan/Feb for Chile, Jul/Aug for Canada), so its 'gwl'
    # files on disk only cover those months. Clip xclim down to the same months
    # up front, so the comparison is a direct, like-for-like match rather than
    # relying on intersect_time() to silently discard the other months later.
    impacttb_months = _months_present(tb_cube)
    print(f"  Months present in ImpactTB data: {impacttb_months}")
    xc_cube = xc_cube.extract(
        iris.Constraint(time=lambda cell, months=impacttb_months: cell.point.month in months))
    if xc_cube is None:
        print(f"Skipping {MEMBER}/{EXPERIMENT}: no xclim data left after clipping to ImpactTB months {impacttb_months}.")
        sys.exit(0)

    print(f"  ImpactTB: {tb_cube.summary(shorten=True)}")
    print(f"  Xclim:    {xc_cube.summary(shorten=True)}")

    # Full xclim record length for this member, restricted to the ImpactTB months
    # -- the expected number of days a complete comparison would cover, since
    # xclim's period is always the full WINDOW_START..WINDOW_END range for those
    # months (ImpactTB is the source that sometimes only delivers a partial period).
    expected_n_days = xc_cube.coord('time').shape[0]

    for country, shape_name in REGION_SHAPES.items():
        print(f"=== {country} ({shape_name}) ===")
        try:
            process_region(country, shape_name, tb_cube, xc_cube, expected_n_days)
        except Exception as e:
            print(f"  ERROR processing {country}: {e}")
            continue

    print("Done.")
