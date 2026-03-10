import glob
import re
import iris
import iris
import numpy as np
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='iris')


############# User inputs here #############
Country = 'Iberia'
YEAR = 2024
# Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############

folder = '/data/scratch/chantelle.burton/SoW2526/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
data_folder = '/data/scratch/chantelle.burton/SoW2526/Y2526FWI/'

# Set up the 2025 files and months automatically
if Country == 'South Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    shape_name = 'South Korea'
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-31_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Iberia':
    print('Running Iberia')
    Month = 8
    month = 'Aug'
    percentile = 95
    shape_name = 'Northwest Iberia'
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Scotland':
    print('Running Scotland')
    Month = 7
    month = 'July'
    percentile = 95
    shape_name = 'Scottish Highlands'
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

start_time = time.time()
index_name = 'canadian_fire_weather_index'

print("\n=== Processing historicalExt ===")
hist_files = sorted(glob.glob(data_folder + 'FWI_HadGEM3-A-N216_r*_historicalExt_20230601-20250201_global_day.nc'))
print(f"Found {len(hist_files)} historicalExt files")

# Regex to extract member and realization from filename
pattern = r'_r(\d+)i1p(\d+)_'

# Step 1: Load all files and add coordinates BEFORE merging
print("\n=== Loading all cubes ===")
hist_cubes = iris.cube.CubeList()

for hist_file in hist_files:
    try:
        # Extract member and realization from filename
        match = re.search(pattern, hist_file)
        if not match:
            print(f"Warning: Could not extract member/realization from {hist_file}")
            continue
        
        member = int(match.group(1))
        realization = int(match.group(2))
        
        print(f"Loading Member: {member}, Realization: {realization})")
        hist = iris.load_cube(hist_file, index_name)
        
        # Add member and realization as scalar coordinates BEFORE merging
        hist.add_aux_coord(iris.coords.AuxCoord(member, long_name='ensemble_member', units='1'))
        hist.add_aux_coord(iris.coords.AuxCoord(realization, long_name='realization', units='1'))
        
        hist_cubes.append(hist)
        
    except IOError as e:
        print(f"Error loading {hist_file}: {e}")

print(f"Loaded {len(hist_cubes)} cubes")    

# Step 2: Merge all cubes along member/realization dimensions
print("\n=== Merging cubes ===")
iris.util.equalise_attributes(hist_cubes)

hist_merged = hist_cubes.merge_cube()
#this is approx 350gb of data
print(f"Merged cube shape: {hist_merged.shape}")
print(hist_merged)

# Step 3: Apply operations to the merged cube (single operation on 525 ensemble members)
print("\n=== Applying constraints and percentiles ===")
hist_merged = contrain_to_sow_shapefile(hist_merged, shp_file, shape_name)
hist_merged = ConstrainToYear(hist_merged, YEAR)
hist_merged = CountryPercentile(hist_merged, percentile)
hist_merged = TimePercentile(hist_merged, percentile)

print(f"Processed cube shape: {hist_merged.shape}")
print(hist_merged)

# Step 4: Save and extract
print("\n=== Saving merged cube ===")
output_file = f'/data/scratch/bob.potts/sowf/test_output/{Country}_Uncorrected_hist_EXT{percentile}%.nc'


print(f"Saved to: {output_file}")
