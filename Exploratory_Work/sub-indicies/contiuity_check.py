import warnings

warnings.filterwarnings("ignore",module="iris")
import iris
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.constrain_cubes_standard import contrain_to_sow_shapefile
import glob
import re

# --- Subindex dictionary ---
index_dict = {
	# 'canadian_fire_weather_index': 'FWI',
	# 'fine_fuel_moisture_content': 'FFMC',
	# 'duff_moisture_content': 'DMC',
	# 'drought_code': 'DC',
	'initial_spread_index': 'ISI',
	'build_up_index': 'BUI'
}

# --- Country/region config ---
country_config = {
	'Korea': {
		'shape_name': 'South Korea',
	},
	'Iberia': {
		'shape_name': 'Northwest Iberia',
	},
	'Scotland': {
		'shape_name': 'Scottish Highlands',
	},
	'Chile': {
		'shape_name': 'Chilean Temperate Forests and Matorral',
	},
	'Canada': {
		'shape_name': 'Midwestern Canadian Shield forests',
	},
}

shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
data_dir = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI/'
file_prefix = 'FWI_ERA5_global_day_'
output_dir = '/data/scratch/bob.potts/sowf/test_output/Exports/ContinuityCubes'
os.makedirs(output_dir, exist_ok=True)

# Helper to extract year from filename
def extract_year_from_filename(fname):
	match = re.search(r'_(\d{4})\d{2}\d{2}-', fname)
	if match:
		return int(match.group(1))
	return None

# Find all files for 2023-2026
all_files = sorted(glob.glob(os.path.join(data_dir, file_prefix + '*.nc')))
selected_files = [f for f in all_files if extract_year_from_filename(f) in [2023, 2024, 2025, 2026]]


# --- Efficient: load all cubes per file, crop all indices, then merge per index and region ---

# --- Efficient: concat first, then crop ---
from collections import defaultdict

index_cubes = defaultdict(list)
reference_units = {}

for f in selected_files:
    print(f"Processing file: {os.path.basename(f)}")
    try:
        cubes = iris.load(f)
    except Exception as e:
        print(f"  Could not load file {f}: {e}")
        continue
    for subindex_name, short_label in index_dict.items():
        try:
            cube = cubes.extract(subindex_name)[0]
            # Remove all scalar coordinates related to time
            time_scalar_names = ['month', 'month_number', 'season', 'season_year', 'year']
            for coord in list(cube.coords()):
                if coord.long_name in time_scalar_names:
                    cube.remove_coord(coord)
            # Set reference unit from the first cube for each index
            if subindex_name not in reference_units:
                reference_units[subindex_name] = cube.coord('time').units
            # Standardize time units and remove attributes
            if cube.coords('time'):
                time_coord = cube.coord('time')
                time_coord.convert_units(reference_units[subindex_name])
                time_coord.attributes = {}
                time_coord.var_name = None
                time_coord.long_name = None
                time_coord.standard_name = 'time'
                
            index_cubes[subindex_name].append(cube)
        except Exception as e:
            print(f"    Failed {subindex_name} in {f}: {e}")

# Now, for each index, merge, then crop to each region
for subindex_name, cubes in index_cubes.items():
	short_label = index_dict[subindex_name]
	if not cubes:
		print(f"No data for {short_label}")
		continue
	try:
		# print(cubes[0].coord('time'))  # Check time coordinate before merging
		# print(cubes[1].coord('time'))
		# print(cubes[2].coord('time'))
		# print(cubes[3].coord('time'))
		merged = iris.cube.CubeList(cubes).concatenate_cube()
	except Exception as e:
		print(f"Failed to merge {short_label}: {e}")
		continue
	for region, reg_cfg in country_config.items():
		print(f"  Cropping {short_label} to {region}")
		try:
			cropped = contrain_to_sow_shapefile(merged, shp_file, reg_cfg['shape_name'])
			out_path = os.path.join(output_dir, f"{short_label}_{region}_2023-2026.nc")
			iris.save(cropped, out_path)
			print(f"    Saved: {out_path}")
		except Exception as e:
			print(f"    Failed to crop/save {short_label} for {region}: {e}")
