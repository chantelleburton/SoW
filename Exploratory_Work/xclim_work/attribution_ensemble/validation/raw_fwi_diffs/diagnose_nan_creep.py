"""
Diagnose why the number of valid (non-NaN) grid cells within a focal region
shrinks over time in the raw FWI diff outputs (the n_cells column of
daily_diff_<country>_<experiment>_<member>.csv, produced by compute_raw_fwi_diffs.py).

n_cells on the *diff* cube is the count of cells unmasked in BOTH the xclim and
ImpactTB source cubes (masked-array subtraction ORs the two masks). The region
shapefile mask itself is static in time, so a declining count means NaN/fill
values are creeping into one or both underlying *source* cubes. This script:

  1. Counts valid cells per day PER SOURCE separately (not just the diff), to
     identify which source (xclim, ImpactTB, or both) is responsible.
  2. Builds a per-cell "first NaN day" map for the culprit source(s), to see
     whether the NaNs appear as a spatial creep (edge/regrid artefact), a
     scattered pattern (data corruption), or a single step-change (bad file).
  3. If xclim is implicated: cross-checks the 4 raw driver variables
     (tasmax, pr, sfcWind, hurs) over the same region/time to find which
     variable is the source.
  4. If ImpactTB is implicated: checks each individual monthly source file for
     the region bounding box to find which delivered file introduces new NaNs.

Outputs (per member/experiment/region), written to OUT_DIR:
    nan_counts_<country>_<experiment>_<member>.csv   -- daily n_valid per source
    nan_timeseries_<experiment>_<member>.png          -- 3-region n_valid plot
    first_nan_map_<source>_<country>_<experiment>_<member>.png
    driver_nan_check_<country>_<experiment>_<member>.csv   (xclim culprit only)
    monthly_file_nan_check_<country>_<experiment>_<member>.csv (ImpactTB culprit only)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../validation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # .../raw_fwi_diffs
sys.path.insert(0, '/data/users/bob.potts/StateOfFires_2025-26/code')

from compare_fwi95_timeseries import load_impacttb, load_xclim, IMPACTTB_ROOT, XCLIM_DIR, SHP_FILE  # noqa: E402
from compute_raw_fwi_diffs import intersect_time  # noqa: E402
from utils.cubefuncs import apply_shapefile_inclusive  # noqa: E402

# ---- Config ----
OUT_DIR = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/validation/raw_fwi_diffs/nan_diagnosis'

MEMBER     = os.environ.get("CYLC_TASK_PARAM_member", "r015i1p2").strip()
EXPERIMENT = os.environ.get("CYLC_TASK_PARAM_run_type", "historicalExt").strip()

# Iberia included as a control (not reported as declining).
REGION_SHAPES = {
    'Iberia': 'Northwest Iberia',
    'Chile':  'Chilean Temperate Forests and Matorral',
    'Canada': 'Midwestern Canadian Shield forests',
}

# Fraction-of-max threshold below which a source is flagged as "declining" for a region.
DECLINE_THRESHOLD = 0.98


def n_valid_per_day(cube):
    """Count unmasked (valid) cells at each timestep, robust to dim ordering."""
    time_dim = cube.coord_dims('time')[0]
    other_axes = tuple(i for i in range(cube.ndim) if i != time_dim)
    return np.ma.count(cube.data, axis=other_axes)


def region_bounds(shp_file, shape_name):
    gdf = gpd.read_file(shp_file)
    geom = gdf[gdf['name'] == shape_name]['geometry'].values[0]
    return geom.bounds  # (minx, miny, maxx, maxy), longitude in -180..180


def first_nan_map(cube):
    """Per-cell day-index of first time a cell (valid at t=0) becomes masked.

    Returns (first_nan_idx, ref_valid, lat_points, lon_points):
        first_nan_idx -- int array (lat, lon), -1 if never newly-masked
        ref_valid      -- bool array (lat, lon), True if valid at the first timestep
    """
    time_dim = cube.coord_dims('time')[0]
    mask = np.ma.getmaskarray(cube.data)
    if time_dim != 0:
        mask = np.moveaxis(mask, time_dim, 0)

    ref_valid = ~mask[0]
    first_nan_idx = np.full(mask.shape[1:], -1, dtype=int)
    still_open = ref_valid.copy()
    for t in range(1, mask.shape[0]):
        newly = mask[t] & still_open
        first_nan_idx[newly] = t
        still_open &= ~newly

    return first_nan_idx, ref_valid


def classify_decline(n_valid):
    """Return ('step'|'gradual'|'none', n_lost, onset_index)."""
    static_max = int(np.max(n_valid))
    n_lost = static_max - int(np.min(n_valid))
    if n_lost <= 0:
        return 'none', 0, None
    onset_index = int(np.argmax(n_valid < static_max))
    # If >=70% of the total loss happens within a single day-to-day transition,
    # call it a step change; otherwise it's a gradual creep.
    day_to_day_loss = -np.diff(n_valid.astype(int))
    day_to_day_loss = np.clip(day_to_day_loss, 0, None)
    biggest_single_drop = int(day_to_day_loss.max()) if day_to_day_loss.size else 0
    kind = 'step' if biggest_single_drop >= 0.7 * n_lost else 'gradual'
    return kind, n_lost, onset_index


def plot_first_nan_map(first_nan_idx, ref_valid, cube, title, out_path):
    lat = cube.coord('latitude').points
    lon = cube.coord('longitude').points
    plotted = np.where(ref_valid, first_nan_idx, np.nan)
    plotted = np.where((first_nan_idx == -1) & ref_valid, np.nan, plotted)  # never-NaN cells left blank too

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(plotted), cmap='viridis', shading='nearest')
    fig.colorbar(im, ax=ax, label='Day index of first NaN (since period start)')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {out_path}")


# ---- Step 6: xclim driver-variable cross-check ----

def diagnose_xclim_drivers(country, bounds, dates_index):
    """Count valid cells per day for each of the 4 xclim driver variables.

    sfcWind is regridded onto the tracer grid (same as the FWI pipeline itself
    does before calling cffwis_indices) rather than left on its native
    staggered grid, so its valid-cell count is directly comparable to
    tasmax/pr/hurs and any regrid-introduced NaNs are actually visible here.
    """
    # .../attribution_ensemble/validation/raw_fwi_diffs -> up 3 levels to attribution_ensemble
    attribution_ensemble_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, attribution_ensemble_dir)
    import explore_hadgem_attribution_xclim_FWI as xfwi  # noqa: E402

    minx, miny, maxx, maxy = bounds
    lon0, lon1 = minx % 360, maxx % 360

    records = {}
    tas = None  # regrid target for sfcWind (must be loaded first; VAR_CONFIG order guarantees this)
    for var_name, cfg in xfwi.VAR_CONFIG.items():
        try:
            da = xfwi.load_variable(var_name, cfg, xfwi.tld, EXPERIMENT, MEMBER,
                                     chunks={'latitude': -1, 'longitude': -1, 'time': -1})
        except AssertionError as e:
            print(f"    driver {var_name}: {e}")
            continue

        if var_name == 'tasmax':
            tas = da
        if var_name == 'sfcWind' and tas is not None:
            da = xfwi.regrid_to_tracer(da, tas)

        lat_mask = (da.latitude >= miny) & (da.latitude <= maxy)
        lon_mask = (da.longitude >= lon0) & (da.longitude <= lon1)
        da_region = da.where(lat_mask & lon_mask, drop=True).load()

        n_valid = (~np.isnan(da_region)).sum(dim=[d for d in da_region.dims if d != 'time'])
        s = n_valid.to_pandas()
        # HadGEM3-A driver data uses a 360_day calendar (cftime.Datetime360Day, incl.
        # e.g. Feb 30), which pd.to_datetime cannot parse -- keep dates as plain
        # 'YYYY-MM-DD' strings (via strftime) instead of converting to Timestamps.
        s.index = [d.strftime('%Y-%m-%d') for d in s.index.values]
        s.index.name = 'Date'
        records[var_name] = s

        if var_name == 'hurs':
            n_ge_100 = int((da_region.values >= 100).sum())
            n_total = int(da_region.size)
            print(f"    hurs >= 100% over {country} region/period: {n_ge_100}/{n_total} "
                  f"cell-days ({100.0 * n_ge_100 / n_total:.2f}%)")

    if not records:
        return None
    df = pd.DataFrame(records)
    df.index.name = 'Date'
    return df


def trace_poisoned_cell(xc_aligned, first_nan_idx, ref_valid):
    """Deep-dive the earliest-poisoned grid cell.

    Prints/returns the FWI value, the intermediate DC/DMC/FFMC sub-indices
    (recomputed at this single cell via xclim's own cffwis_indices, so we can
    see which sub-index is the first to go NaN), and all 4 driver values
    (sfcWind regridded to the tracer grid) for a window of days around the
    onset of NaN. This distinguishes two hypotheses:
      - an input driver goes NaN at/just before onset (regrid/data gap), vs
      - FWI alone goes NaN while all 4 drivers stay finite (the DC/DMC/FFMC
        recursion has been permanently poisoned by an internal numerical
        edge case, unrelated to any missing input) -- and if so, which of
        DC/DMC/FFMC poisons first.
    """
    candidates = np.where(ref_valid & (first_nan_idx >= 0))
    if len(candidates[0]) == 0:
        print("    No poisoned cell found to trace.")
        return None

    order = np.argsort(first_nan_idx[candidates])
    i, j = candidates[0][order[0]], candidates[1][order[0]]
    onset_t = int(first_nan_idx[i, j])

    lat_pts = xc_aligned.coord('latitude').points
    lon_pts = xc_aligned.coord('longitude').points
    cell_lat, cell_lon = lat_pts[i], lon_pts[j]
    print(f"    Tracing earliest-poisoned cell: lat={cell_lat:.3f}, lon={cell_lon:.3f}, "
          f"onset day index={onset_t}")

    time_dim = xc_aligned.coord_dims('time')[0]
    fwi_data = xc_aligned.data
    if time_dim != 0:
        fwi_data = np.moveaxis(fwi_data, time_dim, 0)
    fwi_series = np.ma.filled(fwi_data[:, i, j].astype(float), np.nan)

    t = xc_aligned.coord('time')
    dts = t.units.num2date(t.points)
    dates = [d.strftime('%Y-%m-%d') for d in dts]

    lo, hi = max(0, onset_t - 5), min(len(dates), onset_t + 6)
    trace_df = pd.DataFrame({'Date': dates[lo:hi], 'fwi': fwi_series[lo:hi]})

    attribution_ensemble_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, attribution_ensemble_dir)
    import explore_hadgem_attribution_xclim_FWI as xfwi  # noqa: E402
    import xarray as xr
    import xclim as xc

    lon_0_360 = cell_lon % 360
    tas = xfwi.load_variable('tasmax', xfwi.VAR_CONFIG['tasmax'], xfwi.tld, EXPERIMENT, MEMBER,
                              chunks={'latitude': -1, 'longitude': -1, 'time': -1})

    pt_das = {}
    for var_name, cfg in xfwi.VAR_CONFIG.items():
        da = tas if var_name == 'tasmax' else xfwi.load_variable(
            var_name, cfg, xfwi.tld, EXPERIMENT, MEMBER,
            chunks={'latitude': -1, 'longitude': -1, 'time': -1})
        if var_name == 'sfcWind':
            da = xfwi.regrid_to_tracer(da, tas)
        da_pt = da.sel(latitude=cell_lat, longitude=lon_0_360, method='nearest').load()
        pt_das[var_name] = da_pt
        pt_dates = [d.strftime('%Y-%m-%d') for d in da_pt['time'].values]
        pt_series = pd.Series(da_pt.values, index=pt_dates)
        trace_df[var_name] = trace_df['Date'].map(pt_series)

    # Recompute the FWI sub-indices (DC/DMC/FFMC) at this single cell using
    # xclim's own cffwis_indices, so we can see which sub-index poisons first.
    tas_pt, pr_pt, ws_pt, hurs_pt = xr.align(
        pt_das['tasmax'], pt_das['pr'], pt_das['sfcWind'], pt_das['hurs'], join='inner')
    for da_, units in ((tas_pt, 'degC'), (pr_pt, 'mm/day'), (ws_pt, 'm s-1'), (hurs_pt, '%')):
        da_.attrs['units'] = units
    try:
        lat_da = xr.full_like(tas_pt.isel(time=0), cell_lat, dtype=float)
        # Strip inherited tasmax attrs (e.g. units=degC) and replace with proper
        # latitude units -- cffwis_indices requires a units attr on lat, just not
        # a temperature one.
        lat_da.attrs = {'units': 'degrees_north'}
        dc, dmc, ffmc, isi, bui, fwi_recomp = xc.indices.cffwis_indices(
            tas=tas_pt, pr=pr_pt, sfcWind=ws_pt, hurs=hurs_pt,
            lat=lat_da,
            initial_start_up=True)
        recomp_dates = [d.strftime('%Y-%m-%d') for d in tas_pt['time'].values]
        for name, arr in (('dc', dc), ('dmc', dmc), ('ffmc', ffmc)):
            s = pd.Series(np.asarray(arr.values), index=recomp_dates)
            trace_df[name] = trace_df['Date'].map(s)
    except Exception as e:
        print(f"    (sub-index recompute failed: {e})")

    n_hurs_ge_100 = int((pt_das['hurs'].values >= 100).sum())
    n_hurs_total = int(pt_das['hurs'].values.size)
    print(f"    hurs >= 100% at this cell: {n_hurs_ge_100}/{n_hurs_total} days "
          f"({100.0 * n_hurs_ge_100 / n_hurs_total:.1f}%)")

    print(trace_df.to_string(index=False))
    return trace_df


# ---- Step 7: ImpactTB monthly-file cross-check ----

def diagnose_impacttb_monthly_files(country, shape_name, bounds):
    """Count valid cells within the region bounding box, per delivered monthly file."""
    from utils.constrain_cubes_standard import contrain_coords

    pattern = os.path.join(
        IMPACTTB_ROOT, EXPERIMENT, MEMBER,
        f'FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{EXPERIMENT}_{MEMBER}_global_day_*.nc')
    files = sorted(glob.glob(pattern))

    minx, miny, maxx, maxy = bounds
    rows = []
    for f in files:
        try:
            cube = iris.load_cube(f, 'canadian_fire_weather_index')
        except Exception:
            cubes = iris.load(f)
            cube = cubes[0]
        cube = contrain_coords(cube, (minx, maxx, miny, maxy))
        n_total = cube.data.size
        n_valid = np.ma.count(cube.data)
        t = cube.coord('time')
        dts = t.units.num2date(t.points)
        rows.append({
            'file': os.path.basename(f),
            'start_date': min(dts).strftime('%Y-%m-%d'),
            'end_date': max(dts).strftime('%Y-%m-%d'),
            'n_total_cellsteps': n_total,
            'n_valid_cellsteps': n_valid,
            'pct_valid': 100.0 * n_valid / n_total if n_total else np.nan,
        })
    return pd.DataFrame(rows)


def process_region(country, shape_name, tb_cube, xc_cube):
    print(f"=== {country} ({shape_name}) ===")
    tb_masked = apply_shapefile_inclusive(SHP_FILE, shape_name, tb_cube.copy())
    xc_masked = apply_shapefile_inclusive(SHP_FILE, shape_name, xc_cube.copy())

    tb_aligned, xc_aligned = intersect_time(tb_masked, xc_masked)
    if tb_aligned is None:
        print("  no overlapping timestamps -- skipping")
        return None

    n_valid_tb = n_valid_per_day(tb_aligned)
    n_valid_xc = n_valid_per_day(xc_aligned)

    t = tb_aligned.coord('time')
    dts = t.units.num2date(t.points)
    dates = [d.strftime('%Y-%m-%d') for d in dts]

    df = pd.DataFrame({
        'Date': dates,
        'n_valid_impacttb': n_valid_tb,
        'n_valid_xclim': n_valid_xc,
    })
    df.to_csv(os.path.join(OUT_DIR, f'nan_counts_{country}_{EXPERIMENT}_{MEMBER}.csv'), index=False)

    kind_tb, lost_tb, onset_tb = classify_decline(n_valid_tb)
    kind_xc, lost_xc, onset_xc = classify_decline(n_valid_xc)
    print(f"  ImpactTB: max={n_valid_tb.max()}, min={n_valid_tb.min()}, "
          f"lost={lost_tb} ({kind_tb}), onset={dates[onset_tb] if onset_tb is not None else '-'}")
    print(f"  Xclim:    max={n_valid_xc.max()}, min={n_valid_xc.min()}, "
          f"lost={lost_xc} ({kind_xc}), onset={dates[onset_xc] if onset_xc is not None else '-'}")

    bounds = region_bounds(SHP_FILE, shape_name)

    if lost_tb > 0:
        idx, ref_valid = first_nan_map(tb_aligned)
        plot_first_nan_map(
            idx, ref_valid, tb_aligned,
            f'{country} ImpactTB: first-NaN day ({EXPERIMENT}/{MEMBER})',
            os.path.join(OUT_DIR, f'first_nan_map_impacttb_{country}_{EXPERIMENT}_{MEMBER}.png'))
        print("  ImpactTB declining -- checking individual monthly files...")
        monthly_df = diagnose_impacttb_monthly_files(country, shape_name, bounds)
        monthly_df.to_csv(
            os.path.join(OUT_DIR, f'monthly_file_nan_check_{country}_{EXPERIMENT}_{MEMBER}.csv'), index=False)
        print(monthly_df[['file', 'start_date', 'end_date', 'pct_valid']].to_string(index=False))

    if lost_xc > 0:
        idx, ref_valid = first_nan_map(xc_aligned)
        plot_first_nan_map(
            idx, ref_valid, xc_aligned,
            f'{country} Xclim: first-NaN day ({EXPERIMENT}/{MEMBER})',
            os.path.join(OUT_DIR, f'first_nan_map_xclim_{country}_{EXPERIMENT}_{MEMBER}.png'))
        print("  Tracing earliest-poisoned cell...")
        trace_df = trace_poisoned_cell(xc_aligned, idx, ref_valid)
        if trace_df is not None:
            trace_df.to_csv(
                os.path.join(OUT_DIR, f'poisoned_cell_trace_{country}_{EXPERIMENT}_{MEMBER}.csv'), index=False)
        print("  Xclim declining -- checking raw driver variables (sfcWind regridded)...")
        driver_df = diagnose_xclim_drivers(country, bounds, df['Date'])
        if driver_df is not None:
            driver_df.to_csv(
                os.path.join(OUT_DIR, f'driver_nan_check_{country}_{EXPERIMENT}_{MEMBER}.csv'))
            print(driver_df.describe().loc[['min', 'max']].to_string())

    return df


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Diagnosing NaN creep: member={MEMBER}, experiment={EXPERIMENT}")

    xc_path = os.path.join(XCLIM_DIR, f'hadgem3a_fwi_{EXPERIMENT}_{MEMBER}.nc')
    tb_pattern = os.path.join(
        IMPACTTB_ROOT, EXPERIMENT, MEMBER,
        f'FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{EXPERIMENT}_{MEMBER}_global_day_*.nc')
    if not (os.path.exists(xc_path) and glob.glob(tb_pattern)):
        print(f"Skipping {MEMBER}/{EXPERIMENT}: missing source inputs.")
        sys.exit(0)

    tb_cube = load_impacttb(MEMBER, EXPERIMENT)
    xc_cube = load_xclim(MEMBER, EXPERIMENT)

    results = {}
    for country, shape_name in REGION_SHAPES.items():
        df = process_region(country, shape_name, tb_cube, xc_cube)
        if df is not None:
            results[country] = df

    # ---- Combined 3-region n_valid timeseries plot ----
    if results:
        ncols = 2
        nrows = int(np.ceil(len(results) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False)
        axes = np.atleast_1d(axes).ravel()
        for ax, (country, df) in zip(axes, results.items()):
            # 360-day-calendar dates (e.g. Feb 30) can't be parsed by pd.to_datetime,
            # so plot against a plain integer index and label a sparse subset of ticks.
            x = range(len(df))
            ax.plot(x, df['n_valid_impacttb'], '-', color='#1f77b4', lw=1.2, label='ImpactTB')
            ax.plot(x, df['n_valid_xclim'], '-', color='#d62728', lw=1.2, label='Xclim')
            step = max(1, len(df) // 6)
            ax.set_xticks(list(x)[::step])
            ax.set_xticklabels(df['Date'].iloc[::step], rotation=45, ha='right', fontsize=7)
            ax.set_title(country)
            ax.set_ylabel('n valid cells')
            ax.grid(alpha=0.3)
        for ax in axes[len(results):]:
            ax.set_visible(False)
        axes[0].legend(loc='lower left', fontsize=8)
        fig.suptitle(f'Valid-cell count over time by source — {EXPERIMENT}/{MEMBER}', y=1.0)
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f'nan_timeseries_{EXPERIMENT}_{MEMBER}.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_png}")

    print("Done.")
