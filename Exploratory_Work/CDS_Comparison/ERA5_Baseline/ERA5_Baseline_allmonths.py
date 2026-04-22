import iris
import numpy as np
import time
import glob
import iris.coord_categorisation as icc
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *

############# User inputs here #############
Country = os.environ.get("CYLC_TASK_PARAM_country", 'Korea')
INDEX = os.environ.get("CYLC_TASK_PARAM_index", 'canadian_fire_weather_index')
START_YEAR = 1980
END_YEAR = 2013
percentile = 95

############# User inputs end here #############
baseline_dir = '/data/scratch/bob.potts/sowf/test_output/AllMonths_Baseline'
folder = '/data/scratch/chantelle.burton/SoW2526/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
index_dict = {
    'canadian_fire_weather_index': 'FWI',
    'fine_fuel_moisture_content': 'FFMC',
    'duff_moisture_content': 'DMC',
    'drought_code': 'DC',
    'initial_spread_index': 'ISI',
    'build_up_index': 'BUI'
}

REGION_CONFIGS = {
    'Iberia':   {'shape_name': 'Northwest Iberia'},
    'Korea':    {'shape_name': 'Southeast South Korea'},
    'Scotland': {'shape_name': 'Scottish Highlands'},
    'Chile':    {'shape_name': 'Chilean Temperate Forests and Matorral'},
    'Canada':   {'shape_name': 'Midwestern Canadian Shield forests'},
}

shape_name = REGION_CONFIGS[Country]['shape_name']
print(f'Running {Country} ({shape_name})')

start_time = time.time()

# Load all historical files for ALL 12 months within year range
hist_files = []
for m in range(1, 13):
    hist_pattern = folder + f'/historicalFWI/ERA5/FWI_era5_era5_era5_*{m:02d}01-*.nc'
    all_files = sorted(glob.glob(hist_pattern))

    pattern = re.compile(rf'_(\d{{4}}){m:02d}01-\d{{8}}_')
    for f in all_files:
        match = pattern.search(f)
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                hist_files.append(f)

hist_files = sorted(set(hist_files))
print(f"Found {len(hist_files)} files for years {START_YEAR}-{END_YEAR} (all months)")

if not hist_files:
    raise FileNotFoundError(f"No historical ERA5 files found in year range {START_YEAR}-{END_YEAR}")

cubes = iris.load(hist_files, INDEX)
for cube in cubes:
    for coord_name in ("year", "season_year", 'month', 'month_number', 'season'):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)

ERA5_hist_all = cubes.concatenate_cube()

# Cut to shapefile
print("Applying shapefile")
ERA5_hist_all = apply_shapefile_inclusive(shp_file, shape_name, ERA5_hist_all)

# Add year and month_number coordinates
try:
    icc.add_year(ERA5_hist_all, 'time')
except ValueError:
    pass
try:
    icc.add_month_number(ERA5_hist_all, 'time')
except ValueError:
    pass

# Compute monthly spatial percentile for each year-month
# 1) Aggregate by (year, month_number) — temporal percentile within each month
print("Computing temporal percentile by year-month...")
monthly_time_p = ERA5_hist_all.aggregated_by(['year', 'month_number'],
                                              iris.analysis.PERCENTILE, percent=percentile)

# 2) Spatial percentile for each year-month slice
print("Computing spatial percentile for each year-month...")
monthly_country_p = monthly_time_p.collapsed(['latitude', 'longitude'],
                                              iris.analysis.PERCENTILE, percent=percentile)

# Extract arrays
years = monthly_country_p.coord('year').points
months = monthly_country_p.coord('month_number').points
values = np.ravel(monthly_country_p.data)

# Build date strings as YYYY-MM
date_strs = [f'{int(y)}-{int(m):02d}' for y, m in zip(years, months)]

# Save to CSV
index_short = index_dict[INDEX]
output_file = os.path.join(baseline_dir, f'ERA5_{index_short}_{START_YEAR}-{END_YEAR}_{Country}_allmonths_{percentile}%.csv')

with open(output_file, 'w') as f:
    f.write(f'Date,{index_short}\n')
    for d, v in zip(date_strs, values):
        f.write(f'{d},{v:.6f}\n')
print(f"Saved to: {output_file}")

print('Finished')
print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
print(f"Data shape: {values.shape} ({len(np.unique(years))} years x {len(np.unique(months))} months)")
