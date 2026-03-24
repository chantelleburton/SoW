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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *

############# User inputs here #############
Country = 'Scotland'
START_YEAR = 1960
END_YEAR = 2013

# Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############

folder = '/data/scratch/chantelle.burton/SoW2526/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
OUTPUT_DIR = '/data/scratch/bob.potts/sowf/ERA5_Checks/'
# Set up the 2025 files and months automatically
if Country == 'Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    shape_name = 'South Korea'
    #ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Scotland':
    print('Running Scotland')
    Month = 7
    month = 'July'
    percentile = 95
    shape_name = 'Scottish Highlands'
    #ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')


start_time = time.time()
#Month = np.arange(1,13).tolist()  # Set to all months for historical investigation 
# Handle single or multiple months

if isinstance(Month, (tuple, list, np.ndarray)):
    months = Month
else:
    months = [Month]

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

cubes = iris.load(hist_files, 'canadian_fire_weather_index')
for cube in cubes:
    for coord_name in ("year", "season_year",'month','month_number','season'): #scalar coords prevent concat so drop them - datetime integrity maintained by regular coords and year coord below.
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)

ERA5_hist_all = cubes.concatenate_cube()
# Cut to shapefile 
print("Applying shapefile")
ERA5_hist_all = contrain_to_sow_shapefile(ERA5_hist_all, shp_file, shape_name) #SLOW operation, keep out of loop. Could be replaced with the new iris.util.mask_cube_from_shape but works as is.

# Add year coordinate
try:
    icc.add_year(ERA5_hist_all, 'time')
except ValueError:
    pass  # already exists

# Export the final cube to NetCDF
ERA5_hist_all['time'] = ERA5_hist_all['time'].astype(np.float64)
output_nc = os.path.join(OUTPUT_DIR, f"ERA5_{Country}_{START_YEAR}_{END_YEAR}_clipped.nc")
iris.save(ERA5_hist_all, output_nc)
print(f"Exported clipped cube to {output_nc}")