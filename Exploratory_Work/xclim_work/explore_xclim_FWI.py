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
import pandas as pd
from dask.distributed import Client, LocalCluster, wait
warnings.filterwarnings("ignore", category=UserWarning, message=".*chunking.*")
warnings.filterwarnings("ignore", category=FutureWarning)
#Due to its depth, the DC is the slowest-changing moisture code with a time lag of 52 d (Van Wagner, 1987).

start_time = time.time()
start_year = int(os.environ.get("CYLC_TASK_PARAM_start_year", 2025))
MAX_END_YEAR = int(os.environ.get("MAX_END_YEAR", 2025))
wind_stat = os.environ.get("CYLC_TASK_PARAM_wind_stat", "mean").strip().lower()
rh_stat = os.environ.get("CYLC_TASK_PARAM_rh_stat", "mean").strip().lower()
end_year = min(start_year + 10, MAX_END_YEAR)

basepath = '/data/users/appldata/Data/OBS-ERA5/daily'
out_dir = '/data/scratch/bob.potts/sowf/test_output/XClim_FWI'

WIND_OPTIONS = {
    'mean': {
        'subdir': '10m_mean_wind_speed',
        'pattern': 'era5_daily_mean_10m_wind_speed_{year}-*.nc',
        'var': 'wind_speed_mean',
        'label': 'Mean_Wind',
    },
    'max': {
        'subdir': '10m_max_wind_speed',
        'pattern': 'era5_daily_max_10m_wind_speed_{year}-*.nc',
        'var': 'wind_speed_max',
        'label': 'Max_Wind',
    },
}
RH_OPTIONS = {
    'mean': {
        'subdirs': ('relative_humidity', 'mean'),
        'pattern': 'era5_daily_mean_relative_humidity_{year}*.nc',
        'label': 'Mean_RH',
    },
    'minimum': {
        'subdirs': ('relative_humidity', 'minimum'),
        'pattern': 'era5_daily_minimum_relative_humidity_{year}*.nc',
        'label': 'Minimum_RH',
    },
}

if wind_stat not in WIND_OPTIONS:
    raise ValueError(f"Unsupported wind_stat='{wind_stat}'. Valid options: {sorted(WIND_OPTIONS)}")
if rh_stat not in RH_OPTIONS:
    raise ValueError(f"Unsupported rh_stat='{rh_stat}'. Valid options: {sorted(RH_OPTIONS)}")

wind_cfg = WIND_OPTIONS[wind_stat]
rh_cfg = RH_OPTIONS[rh_stat]
RUN_LABEL = f"{rh_cfg['label']}_{wind_cfg['label']}"

# Sub-indices to write. Remove any you don't need.
OUTPUT_INDICES = ['fwi'] #'dc', 'dmc', 'ffmc', 'isi', 'bui', 

SPATIAL_CHUNK = 90  # lat/lon chunk size — larger chunks = smaller task graph
MEMORY_PER_WORKER = 40  # GB — set based on your cluster's worker memory
if __name__ == '__main__':
    print(
        f"Task parameters: start_year={start_year}, end_year={end_year}, "
        f"wind_stat={wind_stat}, rh_stat={rh_stat}, run_label={RUN_LABEL}"
    )
    # --- Dask cluster setup ---
    cluster = LocalCluster(
        n_workers=3,
        threads_per_worker=1,  # numpy inner loop is GIL-bound
        memory_limit=f'{MEMORY_PER_WORKER}GB',
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    start_time = time.time()
    # --- Merge all input files for each variable over the full period ---
    years = range(start_year, end_year + 1)
    chunks = {'latitude': SPATIAL_CHUNK, 'longitude': SPATIAL_CHUNK}

    # --- Lazy loading with spatial chunking ---
    # xclim's cffwis_indices requires time as a single chunk (sequential dependency)
    # but parallelises independently over spatial chunks via dask.

    # Temperature
    tas_files = []
    for y in years:
        tas_files += sorted(glob.glob(os.path.join(basepath, '2m_temperature', 'daily_maximum', f'era5_daily_maximum_2m_temperature_{y}*.nc')))
    assert len(tas_files) > 0, f"No temperature files found in {basepath}/2m_temperature/daily_maximum/"
    tas = xr.open_mfdataset(tas_files, chunks=chunks)['t2m'] - 273.15
    if 'valid_time' in tas.dims:
        tas = tas.rename({'valid_time': 'time'})
    tas.attrs['units'] = 'degC'
    print(f'Loaded Temp Data (lazy): {tas.dims}, chunks={tas.chunks}')

    # Precipitation
    pr_files = []
    for y in years:
        pr_files += sorted(glob.glob(os.path.join(basepath, 'total_precipitation', 'daily_sum', f'era5_daily_sum_total_precipitation_{y}*.nc')))
    assert len(pr_files) > 0, f"No precipitation files found in {basepath}/total_precipitation/daily_sum/"
    pr = xr.open_mfdataset(pr_files, chunks=chunks)['tp'] * 1000  # m to mm
    if 'valid_time' in pr.dims:
        pr = pr.rename({'valid_time': 'time'})
    pr.attrs['units'] = 'mm/day'
    print(f'Loaded Precip Data (lazy): {pr.dims}, chunks={pr.chunks}')

    # Wind — time encoding in these files is broken (produced by a separate
    # pipeline), so we load with decode_times=False and reconstruct time from
    # the YYYY-MM in each filename.
    wind_files = []
    for y in years:
        wind_files += sorted(glob.glob(os.path.join(basepath, wind_cfg['subdir'], wind_cfg['pattern'].format(year=y))))
    assert len(wind_files) > 0, f"No wind files found in {basepath}/{wind_cfg['subdir']}/"

    ws_parts = []
    for fpath in wind_files:
        ds_wind = xr.open_dataset(fpath, decode_times=False, chunks=chunks)
        da = ds_wind[wind_cfg['var']]
        # Extract YYYY-MM from filename
        m = re.search(r'(\d{4})-(\d{2})\.nc$', os.path.basename(fpath))
        assert m, f"Cannot parse year-month from wind filename: {fpath}"
        yyyy, mm = int(m.group(1)), int(m.group(2))
        n_days = da.sizes['time']
        new_time = pd.date_range(f"{yyyy}-{mm:02d}-01", periods=n_days, freq='D')
        da = da.assign_coords(time=new_time)
        ws_parts.append(da)
    ws = xr.concat(ws_parts, dim='time')
    ws = ws.chunk(chunks)
    ws.attrs['units'] = 'm s-1'
    print(f'Loaded Wind Data (lazy): {ws.dims}, chunks={ws.chunks}')

    # Humidity
    hurs_files = []
    for y in years:
        hurs_files += sorted(glob.glob(os.path.join(basepath, *rh_cfg['subdirs'], rh_cfg['pattern'].format(year=y))))
    assert len(hurs_files) > 0, f"No humidity files found in {basepath}/{'/'.join(rh_cfg['subdirs'])}/"
    hurs = xr.open_mfdataset(hurs_files, chunks=chunks)['hurs']
    if 'valid_time' in hurs.dims:
        hurs = hurs.rename({'valid_time': 'time'})
    print(f'Loaded Humidity Data (lazy): {hurs.dims}, chunks={hurs.chunks}')

    # --- Align all variables on their common time axis ---
    tas = tas.assign_coords(time=tas.indexes['time'].normalize())
    pr = pr.assign_coords(time=pr.indexes['time'].normalize())
    ws = ws.assign_coords(time=ws.indexes['time'].normalize())
    hurs = hurs.assign_coords(time=hurs.indexes['time'].normalize())
    tas, pr, ws, hurs = xr.align(tas, pr, ws, hurs, join='inner')
    print(f'Aligned time dimensions: tas={tas.time.size}, pr={pr.time.size}, ws={ws.time.size}, hurs={hurs.time.size}')

    if tas.time.size == 0:
        raise ValueError(
            "No overlapping dates after alignment. "
            f"tas[0:3]={tas.time.values[:3]}, "
            f"pr[0:3]={pr.time.values[:3]}, "
            f"ws[0:3]={ws.time.values[:3]}, "
            f"hurs[0:3]={hurs.time.values[:3]}"
        )


    # --- Forward-fill NaN values in inputs ---
    # ERA5 wind speed has sporadic NaN days (e.g. ~30 days/year for some regions).
    # Because FWI is sequential (each day depends on previous moisture codes),
    # a single NaN propagates forward permanently. Forward-fill is safe here:
    # a missing wind observation is best approximated by the previous day's value.
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
    print(
        "Post-rechunk time chunks: "
        f"tas={tas.chunks[0]}, pr={pr.chunks[0]}, ws={ws.chunks[0]}, hurs={hurs.chunks[0]}"
    )
    print(f"tas dtype: {tas.dtype}, pr dtype: {pr.dtype}, ws dtype: {ws.dtype}, hurs dtype: {hurs.dtype}")
    print(f"tas shape: {tas.shape}, pr shape: {pr.shape}, ws shape: {ws.shape}, hurs shape: {hurs.shape}")

    # --- Persist inputs to materialise unit conversions in worker memory ---
    # This prevents the -273.15 / *1000 operations from bloating the FWI task graph.
    print("Persisting inputs to worker memory...")
    tas, pr, ws, hurs = client.persist([tas, pr, ws, hurs])
    wait([tas, pr, ws, hurs])
    print("Inputs persisted.")

    # --- Compute FWI for the full period ---
    print(f"Computing FWI for {start_year}-{end_year}...")
    dc, dmc, ffmc, isi, bui, fwi = xc.indices.cffwis_indices(
        tas=tas,
        pr=pr,
        sfcWind=ws,
        hurs=hurs,
        lat=tas.latitude,
        initial_start_up=True
    )
    print(f"FWI dtype: {fwi.dtype}, shape: {fwi.shape}, chunks: {getattr(fwi, 'chunks', None)}")

    # --- Save one file per selected sub-index ---
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
        out_path = os.path.join(out_dir, f'era5_{idx_name}_{RUN_LABEL}_{start_year}-{end_year}.nc')
        enc = {idx_name: {'chunksizes': (min(n_times, 365), SPATIAL_CHUNK, SPATIAL_CHUNK)}}
        ds = xr.Dataset({idx_name: da})
        print(ds)
        ds.to_netcdf(out_path, encoding=enc)
        print(f"Saved {idx_name} to {out_path}")
    print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
    client.close()
    cluster.close()
    print('Finished')
    print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))