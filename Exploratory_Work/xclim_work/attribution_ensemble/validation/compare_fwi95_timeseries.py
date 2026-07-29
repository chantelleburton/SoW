"""
Validate the xclim-produced FWI against the known-good ImpactTB (impactstoolbox) FWI.

For the single common ensemble member (r002i1p5), for each focal region defined in the
workspace shapefile, compute a timeseries of each month's regional FWI 95th percentile from
both versions and plot one subplot per region.

Region cut uses the *inclusive* shapefile masker from utils (apply_shapefile_inclusive).
The per-month statistic follows the workspace convention (spatial-then-temporal percentile):
    CountryPercentile (over lat/lon)  ->  TimePercentile (over time within the month).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: cylc/spice batch jobs have no display
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")
import iris
import iris.coord_categorisation as icc

sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')
from utils.cubefuncs import apply_shapefile_inclusive, CountryPercentile, TimePercentile

# ---- Config ----
# Original impactstoolbox source tree: <root>/<experiment>/<member>/FWI_...nc
IMPACTTB_ROOT = '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/Fire-Weather/FWI/HadGEM3-A-N216'
XCLIM_DIR    = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/Xclim'
SHP_FILE     = '/data/users/bob.potts/StateOfFires_2025-26/code/Shapefiles/SoW2526_Focal_MASTER_20260218.shp'
OUT_DIR      = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/validation'

MEMBER     = os.environ.get("CYLC_TASK_PARAM_member", "r002i1p5").strip()
EXPERIMENT = os.environ.get("CYLC_TASK_PARAM_run_type", "historicalExt").strip()  # historicalExt | historicalNatExt
PERCENTILE = 95

REGION_SHAPES = {
    'Korea':    'Southeast South Korea',
    'Iberia':   'Northwest Iberia',
    'Scotland': 'Scottish Highlands',
    'Chile':    'Chilean Temperate Forests and Matorral',
    'Canada':   'Midwestern Canadian Shield forests',
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


def load_impacttb(member, experiment):
    """Concatenate the monthly ImpactTB files for one member/experiment into a single cube."""
    pattern = os.path.join(
        IMPACTTB_ROOT, experiment, member,
        f'FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{experiment}_{member}_global_day_*.nc')
    files = sorted(glob.glob(pattern))
    assert files, f"No ImpactTB files: {pattern}"

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


def load_xclim(member, experiment):
    fpath = os.path.join(XCLIM_DIR, f'hadgem3a_fwi_{experiment}_{member}.nc')
    assert os.path.exists(fpath), f"Missing {fpath}"
    return _load_fwi_cube(fpath)


def monthly_region_95th(cube, shp_file, shape_name, percentile):
    """Inclusive-mask to region, then per year-month compute spatial-then-temporal percentile.

    Returns a pandas Series indexed by 'YYYY-MM'.
    """
    masked = apply_shapefile_inclusive(shp_file, shape_name, cube.copy())

    if not masked.coords('year'):
        icc.add_year(masked, 'time', name='year')
    if not masked.coords('month_number'):
        icc.add_month_number(masked, 'time', name='month_number')

    years = masked.coord('year').points
    months = masked.coord('month_number').points

    out = {}
    for (yr, mo) in sorted(set(zip(years.tolist(), months.tolist()))):
        month_cube = masked.extract(
            iris.Constraint(time=lambda c, yr=yr, mo=mo: c.point.year == yr and c.point.month == mo))
        if month_cube is None:
            continue
        spatial = CountryPercentile(month_cube, percentile)   # collapse lat/lon -> per-day series
        scalar = TimePercentile(spatial, percentile)          # collapse time  -> scalar
        out[f'{yr}-{mo:02d}'] = float(np.array(scalar.data))
    return pd.Series(out).sort_index()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading member {MEMBER} / {EXPERIMENT}...")

    # Skip gracefully (exit 0) when either source is absent for this member/experiment,
    # so cylc tasks for members that don't exist in both datasets are no-ops, not failures.
    xc_path = os.path.join(XCLIM_DIR, f'hadgem3a_fwi_{EXPERIMENT}_{MEMBER}.nc')
    tb_pattern = os.path.join(
        IMPACTTB_ROOT, EXPERIMENT, MEMBER,
        f'FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{EXPERIMENT}_{MEMBER}_global_day_*.nc')
    has_xc = os.path.exists(xc_path)
    has_tb = bool(glob.glob(tb_pattern))
    if not (has_xc and has_tb):
        print(f"Skipping {MEMBER}/{EXPERIMENT}: missing inputs "
              f"(xclim={has_xc}, impacttb={has_tb}).")
        sys.exit(0)

    tb_cube = load_impacttb(MEMBER, EXPERIMENT)
    xc_cube = load_xclim(MEMBER, EXPERIMENT)
    print(f"  ImpactTB: {tb_cube.summary(shorten=True)}")
    print(f"  Xclim:    {xc_cube.summary(shorten=True)}")

    results = {}
    for country, shape_name in REGION_SHAPES.items():
        print(f"=== {country} ({shape_name}) ===")
        tb_series = monthly_region_95th(tb_cube, SHP_FILE, shape_name, PERCENTILE)
        xc_series = monthly_region_95th(xc_cube, SHP_FILE, shape_name, PERCENTILE)
        df = pd.DataFrame({'ImpactTB': tb_series, 'Xclim': xc_series})
        df['diff'] = df['Xclim'] - df['ImpactTB']
        results[country] = df
        df.to_csv(os.path.join(OUT_DIR, f'FWI95_{country}_{EXPERIMENT}_{MEMBER}.csv'))
        corr = df[['ImpactTB', 'Xclim']].corr().iloc[0, 1]
        print(f"  n={len(df)}, mean|diff|={df['diff'].abs().mean():.3f}, corr={corr:.4f}")

    # ---- Plot: one subplot per region ----
    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (country, df) in zip(axes, results.items()):
        x = pd.to_datetime(df.index + '-15', format='%Y-%m-%d')
        ax.plot(x, df['ImpactTB'], '-o', ms=3, label='ImpactTB (known good)', color='#1f77b4')
        ax.plot(x, df['Xclim'],    '-o', ms=3, label='Xclim',                color='#d62728')
        corr = df[['ImpactTB', 'Xclim']].corr().iloc[0, 1]
        ax.set_title(f"{country}  (r={corr:.3f})")
        ax.set_ylabel(f'FWI {PERCENTILE}th')
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Monthly FWI{PERCENTILE} — {EXPERIMENT} / {MEMBER}', y=1.0)
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f'FWI95_timeseries_{EXPERIMENT}_{MEMBER}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")
