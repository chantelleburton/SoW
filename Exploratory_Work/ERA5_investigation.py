import glob
import matplotlib.pyplot as plt
import iris
import os
import sys
import pandas as pd
import numpy as np
import datetime
try:
    import cftime
except ImportError:
    cftime = None
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import contrain_to_sow_shapefile


COUNTRY_CONFIG = {
    'Korea': {
        'region_name': 'Southeast South Korea',
        'shapefile': '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp',
        'plot_dir': '/data/scratch/bob.potts/sowf/test_output/Plots',
        'file_patterns': {
            'temperature': '/data/scratch/bob.potts/sowf/ERA5_Checks/2m_temperature/daily_mean/era5_daily_mean_2m_temperature_199*.nc',
            'precipitation': '/data/scratch/bob.potts/sowf/ERA5_Checks/precipitation/daily_sum/era5_daily_sum_total_precipitation_199*.nc',
            'relative_humidity': '/data/scratch/bob.potts/sowf/ERA5_Checks/relative_humidity/daily_mean/era5_daily_mean_relative_humidity_199*.nc',
        },
        'plot_titles': {
            'temperature': 'ERA5 Daily Mean 2m Temperature 1990s Korea',
            'precipitation': 'ERA5 Daily Total Precipitation 1990s Korea',
            'relative_humidity': 'ERA5 Daily Mean Relative Humidity 1990s Korea',
        },
        'plot_filenames': {
            'temperature': 'mean_monthly_era5_daily_mean_2m_temperature_korea.png',
            'precipitation': 'mean_monthly_era5_sum_total_precipitation_korea.png',
            'relative_humidity': 'mean_monthly_era5_relative_humidity_korea.png',
        },
        'plot_labels': {
            'temperature': 'Mean 2m Temperature (K)',
            'precipitation': 'Total Precipitation (mm)',
            'relative_humidity': 'Mean Relative Humidity (%)',
        },
    },
    'Scotland': {
        'region_name': 'Scottish Highlands',
        'shapefile': '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp',
        'plot_dir': '/data/scratch/bob.potts/sowf/test_output/Plots',
        'file_patterns': {
            'temperature': '/data/scratch/bob.potts/sowf/ERA5_Checks/2m_temperature/daily_mean/era5_daily_mean_2m_temperature_199*.nc',
            'precipitation': '/data/scratch/bob.potts/sowf/ERA5_Checks/precipitation/daily_sum/era5_daily_sum_total_precipitation_199*.nc',
            'relative_humidity': '/data/scratch/bob.potts/sowf/ERA5_Checks/relative_humidity/daily_mean/era5_daily_mean_relative_humidity_199*.nc',
        },
        'plot_titles': {
            'temperature': 'ERA5 Daily Mean 2m Temperature 1990s Scotland',
            'precipitation': 'ERA5 Daily Total Precipitation 1990s Scotland',
            'relative_humidity': 'ERA5 Daily Mean Relative Humidity 1990s Scotland',
        },
        'plot_filenames': {
            'temperature': 'mean_monthly_era5_daily_mean_2m_temperature_scotland.png',
            'precipitation': 'mean_monthly_era5_sum_total_precipitation_scotland.png',
            'relative_humidity': 'mean_monthly_era5_relative_humidity_scotland.png',
        },
        'plot_labels': {
            'temperature': 'Mean 2m Temperature (K)',
            'precipitation': 'Total Precipitation (mm)',
            'relative_humidity': 'Mean Relative Humidity (%)',
        },
    },
    'Iberia': {
        'region_name': 'Iberia',
        'shapefile': '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp',
        'plot_dir': '/data/scratch/bob.potts/sowf/test_output/Plots',
        'file_patterns': {
            'temperature': '/data/scratch/bob.potts/sowf/ERA5_Checks/2m_temperature/daily_mean/era5_daily_mean_2m_temperature_199*.nc',
            'precipitation': '/data/scratch/bob.potts/sowf/ERA5_Checks/precipitation/daily_sum/era5_daily_sum_total_precipitation_199*.nc',
            'relative_humidity': '/data/scratch/bob.potts/sowf/ERA5_Checks/relative_humidity/daily_mean/era5_daily_mean_relative_humidity_199*.nc',
        },
        'plot_titles': {
            'temperature': 'ERA5 Daily Mean 2m Temperature 1990s Iberia',
            'precipitation': 'ERA5 Daily Total Precipitation 1990s Iberia',
            'relative_humidity': 'ERA5 Daily Mean Relative Humidity 1990s Iberia',
        },
        'plot_filenames': {
            'temperature': 'mean_monthly_era5_daily_mean_2m_temperature_iberia.png',
            'precipitation': 'mean_monthly_era5_sum_total_precipitation_iberia.png',
            'relative_humidity': 'mean_monthly_era5_relative_humidity_iberia.png',
        },
        'plot_labels': {
            'temperature': 'Mean 2m Temperature (K)',
            'precipitation': 'Total Precipitation (mm)',
            'relative_humidity': 'Mean Relative Humidity (%)',
        },
    },
}

def convert_to_datetime(d):
    if hasattr(d, 'to_datetime'):
        return d.to_datetime()
    elif cftime and isinstance(d, cftime.datetime):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, np.datetime64):
        return pd.to_datetime(d).to_pydatetime()
    else:
        return d

def process_era5(variable, config):
    shapefile = config['shapefile']
    region_name = config['region_name']
    PLOT_DIR = config['plot_dir']
    file_pattern = config['file_patterns'][variable]
    plot_label = config['plot_labels'][variable]
    plot_filename = config['plot_filenames'][variable]
    plot_title = config['plot_titles'][variable]
    files = sorted(glob.glob(file_pattern))

    all_dates = []
    all_means = []

    for f in files:
        cube = iris.load_cube(f)
        #print(cube)
        cube_region = contrain_to_sow_shapefile(cube, shapefile, region_name)
        mean_cube = cube_region.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

        try:
            mean_cube = mean_cube.collapsed('time', iris.analysis.MEAN)
            time_coord = mean_cube.coord('time')
        except iris.exceptions.CoordinateNotFoundError:
            try:
                mean_cube = mean_cube.collapsed('valid_time', iris.analysis.MEAN)
                time_coord = mean_cube.coord('valid_time')
            except iris.exceptions.CoordinateNotFoundError:
                exit(f"Error: Neither 'time' nor 'valid_time' coordinate found in cube from file {f}.")
        dates = time_coord.units.num2date(time_coord.points)
        all_dates.extend(dates)
        # Multiply rainfall by 1000 if variable is precipitation (convert m to mm)
        data = mean_cube.data * 1000 if variable.lower() in ["precipitation", "rainfall", "total_precipitation"] else mean_cube.data
        # Always treat data as array for extension
        data_arr = np.atleast_1d(data)
        all_means.extend(data_arr.tolist())

    all_dates_py = [convert_to_datetime(d) for d in all_dates]
    sorted_data = sorted(zip(all_dates_py, all_means), key=lambda x: x[0])
    all_dates_py, all_means = zip(*sorted_data)

    plt.figure(figsize=(14,5))
    plt.plot(all_dates_py, all_means, marker='.', linestyle='-', color='teal')
    plt.xlabel('Date')
    plt.ylabel(plot_label)
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, plot_filename), dpi=300)
    #plt.show()

# Run for selected country


# ================== CONFIGURATION ==================
Country = 'Scotland'  # Options: 'Korea', 'Scotland', 'Iberia'

config = COUNTRY_CONFIG[Country]
for variable in ['temperature', 'precipitation', 'relative_humidity']:
    process_era5(variable, config)