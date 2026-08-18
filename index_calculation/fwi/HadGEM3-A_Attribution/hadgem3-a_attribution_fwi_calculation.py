import xarray as xr
import time
import os
import logging
logging.getLogger("distributed").setLevel(logging.WARNING)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import xclim as xc
import glob
import warnings
import time
import numpy as np
from dask.distributed import Client, LocalCluster, wait
warnings.filterwarnings("ignore", category=UserWarning, message=".*chunking.*")
warnings.filterwarnings("ignore", category=FutureWarning)
#swap back to opatt which has redownloaded wind data and pray. 
# ─── Configuration ───────────────────────────────────────────────────────────
tld = '/data/users/opatt/HadGEM3-A-N216'
out_dir = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim/Raw_FWI_Files'

start_time = time.time()
run_type = os.environ.get("CYLC_TASK_PARAM_run_type", "historicalExt").strip()
if run_type == "historicalExt":
    print("Using fallback run_type: historicalExt")

member = os.environ.get("CYLC_TASK_PARAM_member", "r001i1p1").strip()
if member == "r001i1p1":
    print("Using fallback member: r001i1p1")

# Variable config: directory name, netCDF variable name, unit conversion
VAR_CONFIG = {
    'tasmax':  {'dir': 'tasmax/day',  'nc_var': 'tasmax',  'units': 'degC'},
    'pr':      {'dir': 'pr/day',      'nc_var': 'pr',      'units': 'mm/day'},
    'sfcWind': {'dir': 'sfcWind/day', 'nc_var': 'sfcWind', 'units': 'm s-1'},
    'hurs':    {'dir': 'hurs/day',    'nc_var': 'hurs',    'units': '%'},
}

OUTPUT_INDICES = ['fwi']

SPATIAL_CHUNK = 30  # lat/lon chunk size — larger chunks = smaller task graph
MEMORY_PER_WORKER = 5  # GB

# Common reliable read-in window. hurs single-month files only start at 201911,
# and all four variables have continuous single-month files 201911..202502.
WINDOW_START_MONTH = 201911
WINDOW_END_MONTH = 202412
WINDOW_START = '2019-11-01'
WINDOW_END = '2024-12-30'


def _fix_time_dim(ds):
    """Fix files where time is a coordinate on an anonymous dim (e.g. dim0) rather than a dimension itself."""
    if 'time' in ds.coords and 'time' not in ds.indexes: #relates to  sfcWind_day_HadGEM3-A-N216_historicalExt_r001i1p1_202210.nc
        anon_dim = ds['time'].dims[0]
        ds = ds.swap_dims({anon_dim: 'time'})
        ds = ds.drop_duplicates('time')  # bad file has 50 steps with duplicate days
    return ds


def _token_overlaps_window(month_token):
    """Return True if a filename month token overlaps the read-in window.

    Handles both single-month tokens ('YYYYMM', e.g. '202305') and multi-month
    range tokens ('YYYYMM-YYYYMM', e.g. '202306-202311'). YYYYMM ints are
    monotonic so integer comparison is a valid ordering.
    """
    if '-' in month_token:
        start_str, end_str = month_token.split('-', 1)
        if not (start_str.isdigit() and end_str.isdigit()):
            return False
        start, end = int(start_str), int(end_str)
    elif month_token.isdigit():
        start = end = int(month_token)
    else:
        return False
    # Interval-overlap test against [WINDOW_START_MONTH, WINDOW_END_MONTH].
    return start <= WINDOW_END_MONTH and end >= WINDOW_START_MONTH


def load_variable(var_name, cfg, tld, experiment, member, chunks):
    """Load all files for a given variable/experiment/member and apply unit conversion."""
    var_dir = os.path.join(tld, experiment, cfg['dir'])
    pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_{experiment}_{member}_*.nc")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No files found for {var_name}: {pattern}"

    # Keep every file whose month token overlaps the read-in window. This includes
    # both single-month files ('..._202305.nc') and multi-month range files
    # ('..._202306-202311.nc'), which are the only source for some months (e.g.
    # sfcWind Jun-Nov 2023 for r002i1p5). Overlapping single/range files are
    # harmless: the sortby + drop_duplicates('time') below removes any duplicate days.
    windowed = [
        f for f in files
        if _token_overlaps_window(os.path.basename(f).rsplit('_', 1)[-1].replace('.nc', ''))
    ]
    files = windowed
    assert len(files) > 0, f"No files for {var_name} in window {WINDOW_START_MONTH}..{WINDOW_END_MONTH}: {pattern}"
    print(f"  {var_name}: {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}")

    da = xr.open_mfdataset(files, chunks=chunks, combine='nested', concat_dim='time', preprocess=_fix_time_dim)[cfg['nc_var']]

    # Sort by time and drop any cross-file duplicate timestamps (_fix_time_dim only de-dupes within a file).
    da = da.sortby('time').drop_duplicates('time')


    # Unit conversions
    if var_name == 'tasmax':
        da = da - 273.15
    elif var_name == 'pr':
        da = da * 86400  # kg m-2 s-1 → mm/day
    elif var_name == 'hurs':
        # Occasional supersaturated/rounding values slightly above 100% cause
        # xclim's FFMC calculation (kl term: ((100-h)/100)**1.7) to hit a
        # negative base under a fractional exponent -> NaN, which then
        # poisons the FFMC recursion permanently for that cell. 
        da = da.clip(min=0, max=100)

    da.attrs['units'] = cfg['units']
    return da


def regrid_to_tracer(da, target):
    """Regrid a variable on the staggered wind grid onto the tracer grid via linear
    interpolation. Longitude is padded periodically so the wrap-around column is not NaN."""
    if da.latitude.equals(target.latitude) and da.longitude.equals(target.longitude):
        return da
    lon = da.longitude
    left  = da.isel(longitude=[-1]).assign_coords(longitude=[lon.values[-1] - 360.0])
    right = da.isel(longitude=[0]).assign_coords(longitude=[lon.values[0] + 360.0])
    da_ext = xr.concat([left, da, right], dim='longitude')
    return da_ext.interp(latitude=target.latitude, longitude=target.longitude, method='linear')


if __name__ == '__main__':
    print(f"Task parameters: run_type={run_type}, member={member}")

    # --- Dask cluster setup ---
    cluster = LocalCluster(
        n_workers=3,
        threads_per_worker=1,
        memory_limit=f'{MEMORY_PER_WORKER}GB',
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    chunks = {'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK, 'time': -1}

    # --- Load all input variables ---
    print("Loading variables...")
    tas = load_variable('tasmax', VAR_CONFIG['tasmax'], tld, run_type, member, chunks)
    pr = load_variable('pr', VAR_CONFIG['pr'], tld, run_type, member, chunks)
    ws = load_variable('sfcWind', VAR_CONFIG['sfcWind'], tld, run_type, member, chunks)
    hurs = load_variable('hurs', VAR_CONFIG['hurs'], tld, run_type, member, chunks)

    # sfcWind is on a staggered (velocity) grid; regrid it onto the tracer grid so it
    # aligns with tasmax/pr/hurs. Without this, inner-join alignment collapses lat/lon.
    print("Regridding sfcWind onto tracer grid...")
    print(f"  sfcWind grid before: lat={ws.latitude.size}, lon={ws.longitude.size}")
    ws = regrid_to_tracer(ws, tas)
    ws = ws.chunk({'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK, 'time': -1})
    ws.attrs['units'] = VAR_CONFIG['sfcWind']['units']
    print(f"  sfcWind grid after:  lat={ws.latitude.size}, lon={ws.longitude.size}")

    # --- Clip to common reliable window (2019-11 .. 2024-12) ---
    print(f"Clipping all variables to {WINDOW_START} .. {WINDOW_END}...")
    tas  = tas.sel(time=slice(WINDOW_START, WINDOW_END))
    pr   = pr.sel(time=slice(WINDOW_START, WINDOW_END))
    ws   = ws.sel(time=slice(WINDOW_START, WINDOW_END))
    hurs = hurs.sel(time=slice(WINDOW_START, WINDOW_END))

    # --- Align all variables on their common time axis ---
    tas, pr, ws, hurs = xr.align(tas, pr, ws, hurs, join='inner')
    print(f"Aligned time dimension: {tas.time.size} steps")

    if tas.time.size == 0:
        raise ValueError("No overlapping dates after alignment.")

    # --- Forward-fill NaN values in inputs ---
    for name, da in [('tas', tas), ('pr', pr), ('ws', ws), ('hurs', hurs)]:
        n_nan = da.isnull().sum().values
        if n_nan > 0:
            print(f"  {name}: {int(n_nan)} NaN values detected, applying forward-fill")
    tas = tas.ffill(dim='time')
    pr = pr.ffill(dim='time')
    ws = ws.ffill(dim='time')
    hurs = hurs.ffill(dim='time')

    # xclim treats time as a core dimension, so it must be a single chunk.
    compute_chunks = {'time': -1, 'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK}
    tas = tas.chunk(compute_chunks)
    pr = pr.chunk(compute_chunks)
    ws = ws.chunk(compute_chunks)
    hurs = hurs.chunk(compute_chunks)
    print(f"tas shape: {tas.shape}, pr shape: {pr.shape}, ws shape: {ws.shape}, hurs shape: {hurs.shape}")

    # Persist to avoid bloating the FWI task graph with unit-conversion ops
    tas, pr, ws, hurs = client.persist([tas, pr, ws, hurs])
    wait([tas, pr, ws, hurs])

    # --- Compute FWI ---
    print(f"Computing FWI for {run_type} / {member}...")
    dc, dmc, ffmc, isi, bui, fwi = xc.indices.cffwis_indices(
        tas=tas,
        pr=pr,
        sfcWind=ws,
        hurs=hurs,
        lat=tas.latitude,
        initial_start_up=True
    )
    print(f"FWI dtype: {fwi.dtype}, shape: {fwi.shape}, chunks: {getattr(fwi, 'chunks', None)}")

    # --- Save output ---
    index_map = {
        'dc':   (dc,   'Drought Code',          '1'),
        'dmc':  (dmc,  'Duff Moisture Code',     '1'),
        'ffmc': (ffmc, 'Fine Fuel Moisture Code', '1'),
        'isi':  (isi,  'Initial Spread Index',   '1'),
        'bui':  (bui,  'Build-Up Index',         '1'),
        'fwi':  (fwi,  'Fire Weather Index',     'FWI'),
    }
    os.makedirs(out_dir, exist_ok=True)
    # Map desired chunk size per named dim, clamped to the dimension's actual size.
    chunk_by_dim = {'time': 365, 'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK}
    for idx_name in OUTPUT_INDICES:
        da, long_name, units = index_map[idx_name]

        # xclim copies tasmax metadata onto the output — wipe it and set clean attrs.
        da = da.rename(idx_name)
        da.attrs = {'long_name': long_name, 'units': units}
        da.encoding = {}

        # Drop inherited scalar coords (e.g. height=10m) that don't belong on FWI.
        da = da.drop_vars([c for c in ('height',) if c in da.coords])

        # Conventional dim order.
        da = da.transpose('time', 'latitude', 'longitude')

        out_path = os.path.join(out_dir, f'hadgem3a_{idx_name}_{run_type}_{member}.nc')
        chunksizes = tuple(
            min(chunk_by_dim.get(dim, da.sizes[dim]), da.sizes[dim]) for dim in da.dims
        )
        enc = {idx_name: {'chunksizes': chunksizes}}
        ds = xr.Dataset({idx_name: da})
        print(ds)
        ds.to_netcdf(out_path, encoding=enc)
        print(f"Saved {idx_name} to {out_path}")

    print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
    try:
        client.close(timeout=30)
        cluster.close(timeout=30)
    except Exception as e:
        print(f"Warning: cluster shutdown raised {e!r} (ignoring - all output already written)")
    print('Finished')