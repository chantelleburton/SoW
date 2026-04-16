import warnings

warnings.filterwarnings("ignore", module="iris")
import iris
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.constrain_cubes_standard import contrain_to_sow_shapefile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import iris.quickplot as qplt
import cartopy.crs as ccrs
# --- Configurable country settings ---
country_config = {
    'Korea': {
        'months': [3],
        'month_label': 'March',
        'percentile': 95,
        'shape_name': 'South Korea',
        'title': 'Korea March 2025',
        'file_prefix': 'Korea_March2025',
    },
    'Iberia': {
        'months': [8],
        'month_label': 'Aug',
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
        'title': 'Iberia August 2025',
        'file_prefix': 'Iberia_Aug2025',
    },
    'Scotland': {
        'months': [6, 7],
        'month_label': 'June-July',
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
        'title': 'Scotland June-July 2025',
        'file_prefix': 'Scotland_JuneJuly2025',
    },
    'Chile': {
        'months': [1, 2],
        'month_label': 'January-February',
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
        'title': 'Chile Jan-Feb 2025',
        'file_prefix': 'Chile_JanFeb2025',
    },
    'Canada': {
        'months': [7, 8],
        'month_label': 'July-August',
        'percentile': 95,
        'shape_name': 'Midwestern Canadian Shield forests',
        'title': 'Canada July-August 2025',
        'file_prefix': 'Canada_JulyAug2025',
    },
}

# --- User toggle ---
Country = 'Scotland'  # <--- CHANGE THIS TO SELECT COUNTRY

cfg = country_config[Country]


plot_dir = '/data/scratch/bob.potts/sowf/test_output/Plots/Sub-Indices'
export_dir = '/data/scratch/bob.potts/sowf/test_output/Exports/Sub-Indices'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'


# --- Subindex dictionary ---
index_dict = {
    'canadian_fire_weather_index': 'FWI',
    'fine_fuel_moisture_content': 'FFMC',
    'duff_moisture_content': 'DMC',
    'drought_code': 'DC',
    'initial_spread_index': 'ISI',
    'build_up_index': 'BUI'
}

# --- Data directory and file pattern ---
data_dir = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI/'
file_prefix = 'FWI_ERA5_global_day_'

import glob
import re

# Helper to extract month from filename
def extract_month_from_filename(fname):
    # Example: ..._20250301-20250401.nc -> 202503
    match = re.search(r'_(\d{6})\d{2}-', fname)
    if match:
        return int(match.group(1)[4:6])  # MM
    return None

# Helper to extract year from filename
def extract_year_from_filename(fname):
    match = re.search(r'_(\d{4})\d{4}\d{2}-', fname)
    if match:
        return int(match.group(1))
    match = re.search(r'_(\d{4})\d{2}\d{2}-', fname)
    if match:
        return int(match.group(1))
    return None

# Find all files
all_files = sorted(glob.glob(os.path.join(data_dir, file_prefix + '*.nc')))

# Filter files for the months in cfg['months']
selected_files = []
for f in all_files:
    month = extract_month_from_filename(f)
    if month in cfg['months']:
        selected_files.append(f)

def process_and_plot_subindex(subindex_name, cube_file, cfg, month, year):
    print(f"Processing {subindex_name} for {Country} month {month} year {year}")
    cubes = iris.load(cube_file)
    try:
        cube = cubes.extract(subindex_name)[0]
    except Exception as e:
        print(f"  Could not extract {subindex_name} from {cube_file}: {e}")
        return
    # Cut to shapefile
    cube_cut = contrain_to_sow_shapefile(cube, shp_file, cfg['shape_name'])
    # Collapse to percentile
    cube_pct = cube_cut.collapsed(['time'], iris.analysis.PERCENTILE, percent=cfg['percentile'])
    # Plot
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    qplt.pcolormesh(cube_pct)
    ax.coastlines()
    label = index_dict.get(subindex_name, subindex_name)
    plt.title(f"{cfg['title']} {label} {cfg['percentile']}th Percentile {year}-{month:02d}")
    plt.tight_layout()
    fname = f"{cfg['file_prefix']}_{label}_{year}{month:02d}_{cfg['percentile']}th_percentile.png"
    plt.savefig(os.path.join(plot_dir, Country, fname))
    plt.close()
    print(f"Saved: {fname}")

# Loop over all selected files and all indices
for cube_file in selected_files:
    month = extract_month_from_filename(cube_file)
    year = extract_year_from_filename(cube_file)
    for subindex_name in index_dict.keys():
        process_and_plot_subindex(subindex_name, cube_file, cfg, month, year)


