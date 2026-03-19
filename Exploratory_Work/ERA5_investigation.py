import glob
import matplotlib.pyplot as plt
import iris
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import contrain_to_sow_shapefile
# Directory containing 1990s ERA5 files for Korea
data_dir = '/data/scratch/bob.potts/sowf/ERA5_Checks/2m_temperature/daily_mean/'
file_pattern = data_dir + 'era5_daily_mean_2m_temperature_199*.nc'
shapefile = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
region_name = 'Southeast South Korea'
PLOT_DIR = '/data/scratch/bob.potts/sowf/test_output/Plots'
files = sorted(glob.glob(file_pattern))

# Load all cubes and concatenate
cubelist = iris.cube.CubeList()
for f in files:
    cube = iris.load_cube(f)
    cubelist.append(cube)

# Equalise attributes and concatenate
iris.util.equalise_attributes(cubelist)
# Manually stack cubes along time axis
data = np.concatenate([cube.data for cube in cubelist], axis=0)
valid_times = np.concatenate([cube.coord('valid_time').points for cube in cubelist])

# Sort by valid_times to ensure monotonicity
sort_idx = np.argsort(valid_times)
valid_times_sorted = valid_times[sort_idx]
data_sorted = data[sort_idx]

latitude = cubelist[0].coord('latitude')
longitude = cubelist[0].coord('longitude')

# Create new valid_time coordinate
valid_time_coord = iris.coords.DimCoord(
    valid_times_sorted,
    standard_name='time',
    units=cubelist[0].coord('valid_time').units
)

# Create new cube
cube_all = iris.cube.Cube(
    data_sorted,
    dim_coords_and_dims=[(valid_time_coord, 0), (latitude, 1), (longitude, 2)],
    attributes=cubelist[0].attributes,
    long_name=cubelist[0].long_name,
    units=cubelist[0].units
)

# Apply shapefile mask once to the concatenated cube
cube_korea = contrain_to_sow_shapefile(cube_all, shapefile, region_name)

# Get time coordinate and daily spatial mean
mean_cube_korea = cube_korea.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
dates = mean_cube_korea.coord('valid_time').units.num2date(mean_cube_korea.coord('time').points)
daily_means = mean_cube_korea.data

plt.figure(figsize=(14,5))
plt.plot(dates, daily_means, marker='.', linestyle='-', color='teal')
plt.xlabel('Date')
plt.ylabel('Mean 2m Temperature (K)')
plt.title('ERA5 Daily Mean 2m Temperature (1990s, Korea region via shapefile)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'era5_daily_mean_2m_temperature_korea.png'), dpi=300)
plt.show()