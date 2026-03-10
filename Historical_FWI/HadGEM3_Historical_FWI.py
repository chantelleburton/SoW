# Create .dat files for unbias-corrected historical data, then plot  PDFs

#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-

import numpy as np
import iris
import time
#matplotlib.use('Agg')
import warnings
import os
import glob
import iris.coord_categorisation as icc
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

############# User inputs here #############
Country = 'Iberia'
START_YEAR = 1960
END_YEAR = 2013
CSV_EXPORT = True #True for CSV, False for .dat
# Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############


folder = '/data/scratch/chantelle.burton/SoW2526/'

#Set up the 2025 files and months automatically
if Country == 'South Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    shape_name = 'Southeast South Korea'
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-31_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')
      
elif Country == 'Iberia':
    print('Running Iberia')
    Month = 8
    month = 'Aug'
    percentile = 95
    shape_name = 'Northwest Iberia'
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Scotland':
    print('Running Scotland')
    Month = 7
    month = 'July'
    percentile = 95
    shape_name = 'Scottish Highlands'
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

member = 10#os.environ["CYLC_TASK_PARAM_member"] #when running in cylc wrapped, use this to enable all 16 members to be run in parallel.
 #may not be need anymore as cut member proc time from 8 mins to 16 seconds so 16* 16 so under 5 mins for all members assuming lin scale (Should be true)

start_time = time.time()    
# Load all historical files once (for selected month and year range)
hist_pattern = folder + f'/historicalFWI/HadGEM/FWI_HadGEM3-A-N216_r1i1p{member}_historical_gwl*{Month:02d}01*.nc'
all_files = sorted(glob.glob(hist_pattern))

# Filter files by year range
hist_files = []
pattern = re.compile(rf'_historical_gwl(\d{{4}}){Month:02d}01')
for f in all_files:
    match = pattern.search(f)
    if match:
        year = int(match.group(1))
        if START_YEAR <= year <= END_YEAR:
            hist_files.append(f)

print(f"Found {len(hist_files)} files for years {START_YEAR}-{END_YEAR}")

if not hist_files:
    raise FileNotFoundError("No HadGEM3 historical files found in year range 1960-2013")

# Load + concatenate
cubes = iris.load(hist_files, 'canadian_fire_weather_index')

for cube in cubes:
    for coord_name in ("year", "season_year"):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)

HadGEM3_all = cubes.concatenate_cube()

# Constrain once
HadGEM3_all = contrain_to_sow_shapefile(
    HadGEM3_all,
    '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp',
    shape_name)

# Add year coordinate
try:
    icc.add_year(HadGEM3_all, 'time')
except ValueError:
    pass

# 1) Percentile over time within each year
yr_time_p = HadGEM3_all.aggregated_by('year', iris.analysis.PERCENTILE, percent=percentile)

# 2) Percentile over space (lat/lon) for each year
yr_country_p = yr_time_p.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=percentile)

# Final 1D array by year
HadGEM3_Arr = np.ravel(yr_country_p.data)

# Save HadGEM3 text out to a file
output_file = f'/data/scratch/bob.potts/sowf/test_output/HadGEM3_FWI_{START_YEAR}-{END_YEAR}_{Country}_{member}_{percentile}%'

if CSV_EXPORT:
    # Get the years from the cube
    years = yr_country_p.coord('year').points
    
    # Create YEAR-MONTH strings
    year_month = [f'{int(y)}-{Month:02d}' for y in years]

    # Save HadGEM3 out to a text file with YEAR-MONTH,VALUE format
    with open(f'{output_file}.csv', 'w') as f:
        f.write('Date,FWI\n')
        for ym, value in zip(year_month, HadGEM3_Arr):
            f.write(f'{ym},{value:.6f}\n')
    print(f"Saved to: {output_file}.csv")

else:
    np.savetxt(f'{output_file}.dat', HadGEM3_Arr)
    print(f"Saved to: {output_file}.dat")
    
print('Finished')
print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
print(f"Data shape: {HadGEM3_Arr.shape}")
#single member takes approx 8 minutes.