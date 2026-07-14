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

# ─── Configuration ───────────────────────────────────────────────────────────
tld = '/data/users/opatt/HadGEM3-A-N216'
out_dir = '/data/scratch/bob.potts/sowf/Attribution_Ensemble_xclim'

start_time = time.time()
run_type = os.environ.get("CYLC_TASK_PARAM_run_type", "historicalExt").strip()
member = os.environ.get("CYLC_TASK_PARAM_member", "r001i1p1").strip()

# Variable config: directory name, netCDF variable name, unit conversion
VAR_CONFIG = {
    'tasmax':  {'dir': 'tasmax/day',  'nc_var': 'tasmax',  'units': 'degC'},
    'pr':      {'dir': 'pr/day',      'nc_var': 'pr',      'units': 'mm/day'},
    'sfcWind': {'dir': 'sfcWind/day', 'nc_var': 'sfcWind', 'units': 'm s-1'},
    'hurs':    {'dir': 'hurs/day',    'nc_var': 'hurs',    'units': '%'},
}

OUTPUT_INDICES = ['fwi']

SPATIAL_CHUNK = 30  # lat/lon chunk size — larger chunks = smaller task graph
MEMORY_PER_WORKER = 10  # GB


def load_variable(var_name, cfg, tld, experiment, member, chunks):
    """Load all files for a given variable/experiment/member and apply unit conversion."""
    var_dir = os.path.join(tld, experiment, cfg['dir'])
    pattern = os.path.join(var_dir, f"{var_name}_day_HadGEM3-A-N216_{experiment}_{member}_*.nc")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No files found for {var_name}: {pattern}"
    print(f"  {var_name}: {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}")

    da = xr.open_mfdataset(files, chunks=chunks)[cfg['nc_var']]

    # Unit conversions
    if var_name == 'tasmax':
        da = da - 273.15
    elif var_name == 'pr':
        da = da * 86400  # kg m-2 s-1 → mm/day

    da.attrs['units'] = cfg['units']
    return da


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

    chunks = {'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK}

    # --- Load all input variables ---
    print("Loading variables...")
    tas = load_variable('tasmax', VAR_CONFIG['tasmax'], tld, run_type, member, chunks)
    pr = load_variable('pr', VAR_CONFIG['pr'], tld, run_type, member, chunks)
    ws = load_variable('sfcWind', VAR_CONFIG['sfcWind'], tld, run_type, member, chunks)
    hurs = load_variable('hurs', VAR_CONFIG['hurs'], tld, run_type, member, chunks)

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
    n_times = tas.sizes['time']
    for idx_name in OUTPUT_INDICES:
        da, long_name, units = index_map[idx_name]
        da.attrs.update({'long_name': long_name, 'units': units})
        out_path = os.path.join(out_dir, f'hadgem3a_{idx_name}_{run_type}_{member}.nc')
        enc = {idx_name: {'chunksizes': (min(n_times, 365), SPATIAL_CHUNK, SPATIAL_CHUNK)}}
        ds = xr.Dataset({idx_name: da})
        print(ds)
        ds.to_netcdf(out_path, encoding=enc)
        print(f"Saved {idx_name} to {out_path}")

    print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
    client.close()
    cluster.close()
    print('Finished')