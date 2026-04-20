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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *

############# User inputs here #############
Country = os.environ.get("CYLC_TASK_PARAM_country", 'Korea') #fallback to user input if not running in cylc wrapped
INDEX = os.environ.get("CYLC_TASK_PARAM_index", 'canadian_fire_weather_index') #fallback to user input if not running in cylc wrapped. Options: 'canadian_fire_weather_index', 'fine_fuel_moisture_code', 'duff_moisture_code', 'drought_code', 'initial_spread_index', 'build_up_index'
#Country = 'Korea' # Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7), 'Chile' (1,2), 'Canada' (7,8)
START_YEAR = 1980
END_YEAR = 2013
CSV_EXPORT = True #True for CSV, False for .dat

############# User inputs end here #############

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


# Set up the 2025 files and months automatically
if Country == 'Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    shape_name = 'Southeast South Korea'

elif Country == 'Iberia':
    print('Running Iberia')
    Month = 8
    month = 'Aug'
    percentile = 95
    shape_name = 'Northwest Iberia'

elif Country == 'Scotland':
    print('Running Scotland')
    Month = 6,7
    month = 'June-July'
    percentile = 95
    shape_name = 'Scottish Highlands'

elif Country == 'Chile':
    print('Running Chile')
    Month = 1,2
    month = 'January-February'
    percentile = 95
    shape_name = 'Chilean Temperate Forests and Matorral'

elif Country == 'Canada':
    print('Running Canada')
    Month = 7,8
    month = 'July-August'
    percentile = 95
    shape_name = 'Midwestern Canadian Shield forests'

start_time = time.time()

# Handle single or multiple months
if isinstance(Month, tuple):
    months = Month
else:
    months = (Month,)

# Load all historical files for all relevant months
hist_files = []
for m in months:
    hist_pattern = folder + f'/historicalFWI/ERA5/FWI_era5_era5_era5_*{m:02d}01-*.nc'
    all_files = sorted(glob.glob(hist_pattern))
    
    # Filter files by year range
    pattern = re.compile(rf'_(\d{{4}}){m:02d}01-\d{{8}}_')
    for f in all_files:
        match = pattern.search(f)
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                hist_files.append(f)

hist_files = sorted(set(hist_files))  # Remove duplicates and sort
print(f"Found {len(hist_files)} files for years {START_YEAR}-{END_YEAR}")

if not hist_files:
    raise FileNotFoundError(f"No historical ERA5 files found in year range {START_YEAR}-{END_YEAR}")


cubes = iris.load(hist_files, INDEX)
for cube in cubes:
    
    for coord_name in ("year", "season_year",'month','month_number'): #scalar coords prevent concat so drop them - datetime integrity maintained by regular coords and year coord below.
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)

ERA5_hist_all = cubes.concatenate_cube()
# Cut to shapefile 
print("Applying shapefile")
ERA5_hist_all = apply_shapefile_inclusive(shp_file, shape_name, ERA5_hist_all) 

# Add year coordinate
try:
    icc.add_year(ERA5_hist_all, 'time')
except ValueError:
    pass  # already exists

# 1) Percentile over time within each year
print("Computing time percentile by year...")
yr_time_p = ERA5_hist_all.aggregated_by('year', iris.analysis.PERCENTILE, percent=percentile)

# 2) Percentile over space (lat/lon) for each year
print("Computing spatial percentile for each year...")
yr_country_p = yr_time_p.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=percentile)

# Final 1D array by year
ERA5_ImpactsToolBox_Arr = np.ravel(yr_country_p.data)

# Save ERA5 out to a text file
output_file = f'/data/scratch/bob.potts/sowf/test_output/Baseline/ERA5_{index_dict[INDEX]}_{START_YEAR}-{END_YEAR}_{Country}_{percentile}%' 
                            #it's bad practise to have a % sign in a file name but here we are.

if CSV_EXPORT:
    # Get the years from the cube
    years = yr_country_p.coord('year').points
    
    # Create YEAR-MONTH strings (handle single or multi-month)
    if isinstance(Month, tuple):
        month_str = '/'.join(f'{m:02d}' for m in Month)  # e.g., "01/02"
    else:
        month_str = f'{Month:02d}'
    
    year_month = [f'{int(y)}-{month_str}' for y in years]

    # Save ERA5 out to a text file with YEAR-MONTH,VALUE format
    
    with open(f'{output_file}.csv', 'w') as f:
        f.write('Date,FWI\n')
        for ym, value in zip(year_month, ERA5_ImpactsToolBox_Arr):
            f.write(f'{ym},{value:.6f}\n')
    print(f"Saved to: {output_file}.csv")

else:
    np.savetxt(f'{output_file}.dat', ERA5_ImpactsToolBox_Arr)
    print(f"Saved to: {output_file}.dat")

print('Finished')
print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
print(f"Data shape: {ERA5_ImpactsToolBox_Arr.shape}")
