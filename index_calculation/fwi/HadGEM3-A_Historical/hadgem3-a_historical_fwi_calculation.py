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
import re
import warnings
import time
import numpy as np
from dask.distributed import Client, LocalCluster, wait
warnings.filterwarnings("ignore", category=UserWarning, message=".*chunking.*")
warnings.filterwarnings("ignore", category=FutureWarning)
# ─── Configuration ───────────────────────────────────────────────────────────
tld = '/data/users/opatt/HadGEM3-A-N216/historical'
out_dir = '/data/scratch/bob.potts/sowf/fwi-calculation-pipeline/HadGEM3-A_Historical'

start_time = time.time()

member_num = int(os.environ.get("CYLC_TASK_PARAM_member", "1"))
member = f"r1i1p{member_num}"

# Variable config: directory name, netCDF variable name, unit conversion
VAR_CONFIG = {
    'tasmax':  {'dir': 'tasmax/day',  'nc_var': 'tasmax',  'units': 'degC'},
    'pr':      {'dir': 'pr/day',      'nc_var': 'pr',      'units': 'mm/day'},
    'sfcWind': {'dir': 'sfcWind/day', 'nc_var': 'sfcWind', 'units': 'm s-1'},
    'hurs':    {'dir': 'hurs/day',    'nc_var': 'hurs',    'units': '%'},
}

OUTPUT_INDICES = ['fwi']

SPATIAL_CHUNK = 30  # lat/lon chunk size — larger chunks = smaller task graph
MEMORY_PER_WORKER = 30  # GB

# Historical files run 1960-01-01 .. 2013-12-30 (360-day calendar) in 6 decadal
# chunks per member. We always load from the dataset start so the FWI moisture
# codes are fully spun up by START_YEAR, but only write out START_YEAR..END_YEAR
# (mirrors Legacy_Scripts/Historical_FWI.py's START_YEAR/END_YEAR baseline).
DATA_START_YEAR = 1978
START_YEAR = 1980
END_YEAR = 2013


def _fix_time_dim(ds):
    """Fix files where time is a coordinate on an anonymous dim (e.g. dim0) rather than a dimension itself."""
    if 'time' in ds.coords and 'time' not in ds.indexes: #relates to  sfcWind_day_HadGEM3-A-N216_historicalExt_r001i1p1_202210.nc
        anon_dim = ds['time'].dims[0]
        ds = ds.swap_dims({anon_dim: 'time'})
        ds = ds.drop_duplicates('time')  # bad file has 50 steps with duplicate days
    return ds


def _decade_file_overlaps_range(fpath, start_year, end_year):
    """Return True if a decadal filename's YYYYMMDD-YYYYMMDD token overlaps
    [start_year, end_year]. E.g. '..._19800101-19891230.nc' overlaps [1960, 2013]."""
    m = re.search(r'_(\d{8})-(\d{8})\.nc$', os.path.basename(fpath))
    if not m:
        return False
    file_start_year = int(m.group(1)[:4])
    file_end_year = int(m.group(2)[:4])
    return file_start_year <= end_year and file_end_year >= start_year


def load_variable(var_name, cfg, tld, member, chunks):
    """Load all decadal files for a given variable/member and apply unit conversion."""
    var_dir = os.path.join(tld, cfg['dir'])
    pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_historical_{member}_*.nc")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No files found for {var_name}: {pattern}"

    # Only load decadal chunks that overlap [DATA_START_YEAR, END_YEAR] — this
    # always includes the full spin-up lead-in from the dataset start, while
    # skipping any decades entirely after END_YEAR.
    files = [f for f in files if _decade_file_overlaps_range(f, DATA_START_YEAR, END_YEAR)]
    assert len(files) > 0, f"No files for {var_name} in range {DATA_START_YEAR}..{END_YEAR}: {pattern}"
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
    if da.lat.equals(target.lat) and da.lon.equals(target.lon):
        return da
    lon = da.lon
    left  = da.isel(lon=[-1]).assign_coords(lon=[lon.values[-1] - 360.0])
    right = da.isel(lon=[0]).assign_coords(lon=[lon.values[0] + 360.0])
    da_ext = xr.concat([left, da, right], dim='lon')
    return da_ext.interp(lat=target.lat, lon=target.lon, method='linear')


if __name__ == '__main__':
    print(f"Task parameters: member={member}")

    # --- Dask cluster setup ---
    cluster = LocalCluster(
        n_workers=3,
        threads_per_worker=1,
        memory_limit=f'{MEMORY_PER_WORKER}GB',
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    chunks = {'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK, 'time': -1}

    # --- Load all input variables ---
    print("Loading variables...")
    tas = load_variable('tasmax', VAR_CONFIG['tasmax'], tld, member, chunks)
    pr = load_variable('pr', VAR_CONFIG['pr'], tld, member, chunks)
    ws = load_variable('sfcWind', VAR_CONFIG['sfcWind'], tld, member, chunks)
    hurs = load_variable('hurs', VAR_CONFIG['hurs'], tld, member, chunks)

    # sfcWind is on a staggered (velocity) grid; regrid it onto the tracer grid so it
    # aligns with tasmax/pr/hurs. Without this, inner-join alignment collapses lat/lon.
    print("Regridding sfcWind onto tracer grid...")
    print(f"  sfcWind grid before: lat={ws.lat.size}, lon={ws.lon.size}")
    ws = regrid_to_tracer(ws, tas)
    ws = ws.chunk({'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK, 'time': -1})
    ws.attrs['units'] = VAR_CONFIG['sfcWind']['units']
    print(f"  sfcWind grid after:  lat={ws.lat.size}, lon={ws.lon.size}")

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
    compute_chunks = {'time': -1, 'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK}
    tas = tas.chunk(compute_chunks)
    pr = pr.chunk(compute_chunks)
    ws = ws.chunk(compute_chunks)
    hurs = hurs.chunk(compute_chunks)
    print(f"tas shape: {tas.shape}, pr shape: {pr.shape}, ws shape: {ws.shape}, hurs shape: {hurs.shape}")

    # Persist to avoid bloating the FWI task graph with unit-conversion ops
    tas, pr, ws, hurs = client.persist([tas, pr, ws, hurs])
    wait([tas, pr, ws, hurs])

    # --- Compute FWI ---
    print(f"Computing FWI for historical / {member}...")
    dc, dmc, ffmc, isi, bui, fwi = xc.indices.cffwis_indices(
        tas=tas,
        pr=pr,
        sfcWind=ws,
        hurs=hurs,
        lat=tas.lat,
        initial_start_up=True
    )
    print(f"FWI dtype: {fwi.dtype}, shape: {fwi.shape}, chunks: {getattr(fwi, 'chunks', None)}")

    # --- Trim to the output window (discards the pre-START_YEAR spin-up lead-in) ---
    output_start = f'{START_YEAR}-01-01'
    output_end = f'{END_YEAR}-12-30'
    print(f"Trimming output to {output_start} .. {output_end}...")
    dc, dmc, ffmc, isi, bui, fwi = (
        da.sel(time=slice(output_start, output_end)) for da in (dc, dmc, ffmc, isi, bui, fwi)
    )

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
    chunk_by_dim = {'time': 360, 'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK}
    for idx_name in OUTPUT_INDICES:
        da, long_name, units = index_map[idx_name]

        # xclim copies tasmax metadata onto the output — wipe it and set clean attrs.
        da = da.rename(idx_name)
        da.attrs = {'long_name': long_name, 'units': units}
        da.encoding = {}

        # Drop inherited scalar coords (e.g. height=10m) that don't belong on FWI.
        da = da.drop_vars([c for c in ('height',) if c in da.coords])

        # Conventional dim order.
        da = da.transpose('time', 'lat', 'lon')

        out_path = os.path.join(out_dir, f'hadgem3a_{idx_name}_historical_{member}_{START_YEAR}-{END_YEAR}.nc')
        chunksizes = tuple(
            min(chunk_by_dim.get(dim, da.sizes[dim]), da.sizes[dim]) for dim in da.dims
        )
        enc = {idx_name: {'chunksizes': chunksizes}}
        ds = xr.Dataset({idx_name: da})
        print(ds)
        ds.to_netcdf(out_path, encoding=enc)
        print(f"Saved {idx_name} to {out_path}")

    print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
    client.close()
    cluster.close()
    print('Finished')