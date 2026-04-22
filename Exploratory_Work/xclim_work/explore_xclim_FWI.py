import cftime
import xclim as xc
import xarray as xr
import os
import glob
import warnings
import time
import numpy as np
import cftime
warnings.filterwarnings("ignore", category=FutureWarning)
#Due to its depth, the DC is the slowest-changing moisture code with a time lag of 52 d (Van Wagner, 1987).


start_year = 2016
end_year = 2025   #(inclusive)
#~50-100Gb of active memory at a given time (~7yrs * 12 months * 60-300mb/file*4 varis)
#9yrs of wind = 15gb 
#9yrs of temp - 8gb
#'' of RH = 34gb
#'' of precip = 7gb



#2m max temp 
#total daily precip (sum)
#mean daily Wind speed
#mean realtive humidity

basepath = '/data/users/appldata/Data/OBS-ERA5/daily'
out_dir = '/data/scratch/bob.potts/sowf/test_output/ERA5_FWI'

start_time = time.time()
# --- Merge all input files for each variable over the full period ---
years = range(start_year, end_year + 1)

# Temperature
tas_files = []
for y in years:
    tas_files += sorted(glob.glob(os.path.join(basepath, '2m_temperature', 'daily_maximum', f'era5_daily_maximum_2m_temperature_{y}*.nc')))
tas_list = [xr.open_dataset(f)['t2m'] - 273.15 for f in tas_files]
tas = xr.concat(tas_list, dim='valid_time' if 'valid_time' in tas_list[0].dims else 'time')
print('Loaded Temp Data',tas.shape,tas.dims)
if 'valid_time' in tas.dims:
    tas = tas.rename({'valid_time': 'time'})

# Precipitation
pr_files = []
for y in years:
    pr_files += sorted(glob.glob(os.path.join(basepath, 'total_precipitation', 'daily_sum', f'era5_daily_sum_total_precipitation_{y}*.nc')))
pr_list = [xr.open_dataset(f)['tp'] * 1000 for f in pr_files]  # m to mm
pr = xr.concat(pr_list, dim='valid_time' if 'valid_time' in pr_list[0].dims else 'time')
print('Loaded Precip Data',pr.shape,pr.dims)
if 'valid_time' in pr.dims:
    pr = pr.rename({'valid_time': 'time'})
pr.attrs['units'] = 'mm/day'


# Wind (robust loading and time fix)
wind_files = []
for y in years:
    wind_files += sorted(glob.glob(os.path.join(basepath, '10m_mean_wind_speed', f'era5_daily_mean_10m_wind_speed_{y}-*.nc')))
wind_list = [xr.open_dataset(f,decode_times=False)['wind_speed_mean'] for f in wind_files]
ws = xr.concat(wind_list, dim='time')
print('Loaded Wind Data',ws.dims)




# Humidity
hurs_files = []
for y in years:
    hurs_files += sorted(glob.glob(os.path.join(basepath, 'relative_humidity', 'mean', f'era5_daily_mean_relative_humidity_{y}*.nc')))
hurs_list = [xr.open_dataset(f)['hurs'] for f in hurs_files]
hurs = xr.concat(hurs_list, dim='valid_time' if 'valid_time' in hurs_list[0].dims else 'time')
print('Loaded Humidity Data')
if 'valid_time' in hurs.dims:
    hurs = hurs.rename({'valid_time': 'time'})

# --- Align time coordinates if needed ---
if 'time' in hurs.dims and 'time' in tas.dims:
    hurs = hurs.assign_coords(time=tas.time)
    pr = pr.assign_coords(time=tas.time)
    ws = ws.assign_coords(time=tas.time)

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

# --- Save as a single file ---
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'era5_fwi_{start_year}-{end_year}.nc')
fwi.attrs['long_name'] = 'Fire Weather Index'
fwi.attrs['units'] = 'FWI'
fwi.to_dataset(name='fwi').to_netcdf(out_path)
print(f"Saved merged FWI to {out_path}")
print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
