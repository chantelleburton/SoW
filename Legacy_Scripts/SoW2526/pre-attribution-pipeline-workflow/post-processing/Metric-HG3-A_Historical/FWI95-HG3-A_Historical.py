
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

############# User inputs here #############
START_YEAR = 1980
END_YEAR = 2013

# Options: 'Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############
member = int(os.environ.get("CYLC_TASK_PARAM_member", '1')) #when running in cylc wrapped, use this to enable all 16 members to be run in parallel.
Country = os.environ.get("CYLC_TASK_PARAM_country", 'Iberia') #fallback to user input if not running in cylc wrapped
INDEX_NAMES = {
    'ffmc': 'Fine Fuel Moisture Content',
    'fwi':  'Canadian Fire Weather Index',
    'isi':  'Initial Spread Index',
    'bui':  'Build Up Index',
    'dmc':  'Duff Moisture Content',
    'dc':   'Drought Code',
}
index_code = os.environ.get("CYLC_TASK_PARAM_index", 'fwi')
print(index_code)
index = INDEX_NAMES[index_code]
print(index)
folder = '/data/scratch/bob.potts/sowf/fwi-calculation-pipeline/HadGEM3-A_Historical/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
out_folder ='/data/scratch/bob.potts/sowf/fwi-calculation-pipeline/Historical_Metrics'

if Country == 'Iberia':
    print('Running Iberia')
    Month = 8
    month = 'Aug'
    percentile = 95
    shape_name = 'Northwest Iberia'

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

if isinstance(Month, tuple):  # handles multi-month events
    months = Month
else:
    months = (Month,)

# One file per member/index covers the whole computed period, e.g.
# hadgem3a_isi_historical_r1i1p15_1980-2013.nc
hist_pattern = folder + f'hadgem3a_{index_code}_historical_r1i1p{member}_*_modified.nc'
hist_files = sorted(glob.glob(hist_pattern))
if not hist_files:
    raise FileNotFoundError(f"No HadGEM3 historical {index_code} file found: {hist_pattern}")
assert len(hist_files) == 1, f"Expected exactly one file for member {member}, found {len(hist_files)}: {hist_files}"
print(f"Loading {hist_files[0]}")

HadGEM3_all = iris.load_cube(hist_files[0], iris.NameConstraint(var_name=index_code))

for coord_name in ("year", "season_year"):
    if HadGEM3_all.coords(coord_name):
        HadGEM3_all.remove_coord(coord_name)

# Clip to the requested months and year range (previously done via filename
# filtering; now done on the loaded cube's time coordinate).
clip_constraint = iris.Constraint(
    time=lambda cell: cell.point.month in months and START_YEAR <= cell.point.year <= END_YEAR
)
HadGEM3_all = HadGEM3_all.extract(clip_constraint)
if HadGEM3_all is None:
    raise ValueError(f"No data left after clipping to months={months}, years {START_YEAR}-{END_YEAR}")
print(f"Clipped to months={months}, years {START_YEAR}-{END_YEAR}: \n {HadGEM3_all.summary(shorten=True)}")

# Constrain once
HadGEM3_all = apply_shapefile_inclusive(shp_file, shape_name, HadGEM3_all)

# Add year coordinate
try:
    icc.add_year(HadGEM3_all, 'time')
except ValueError:
    pass
#iris.save(HadGEM3_all, f'/data/scratch/bob.potts/sowf/test_output/Zenodo_Interim/HadGEM3-A_FWI_{START_YEAR}-{END_YEAR}_{Country}_member{member}.nc')
# 1) Percentile over time within each year
yr_time_p = HadGEM3_all.aggregated_by('year', iris.analysis.PERCENTILE, percent=percentile)

# 2) Percentile over space (lat/lon) for each year
yr_country_p = yr_time_p.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=percentile)

# Final 1D array by year
HadGEM3_Arr = np.ravel(yr_country_p.data)

# Save HadGEM3 text out to a file
output_file = f'{out_folder}/HadGEM3_{index_code.upper()}{percentile}_{START_YEAR}-{END_YEAR}_{Country}_{member}_{percentile}%_modified'

# Get the years from the cube
years = yr_country_p.coord('year').points
if isinstance(Month, tuple):
    month_str = '/'.join(f'{m:02d}' for m in Month)  # e.g., "01/02"
else:
    month_str = f'{Month:02d}'
# Create YEAR-MONTH strings
year_month = [f'{int(y)}-{month_str}' for y in years]

# Save HadGEM3 out to a text file with YEAR-MONTH,VALUE format
with open(f'{output_file}.csv', 'w') as f:
    f.write(f'Date,{index}\n')
    for ym, value in zip(year_month, HadGEM3_Arr):
        f.write(f'{ym},{value:.6f}\n')
print(f"Saved to: {output_file}.csv")

print('Finished')
print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
print(f"Data shape: {HadGEM3_Arr.shape}")
