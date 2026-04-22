import pandas as pd
import numpy as np
import iris
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

############# Get parameters from Cylc (or defaults for local testing) #############
Country = os.environ.get("CYLC_TASK_PARAM_country", "Iberia") #fallback value of iberia
run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")  # 'hist' or 'histnat'
output_folder = '/data/scratch/bob.potts/sowf/test_output/Uncorrected_Attribution_Ensembles/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
print(f'Processing Country: {Country}, run type: {run_type}')


folder = '/data/scratch/chantelle.burton/SoW2526/'
DATA_YEAR = 2024

#Set up the 2025 files and months automatically
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
    shape_name = 'Scottish Highlands'
    percentile = 95

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

else:
    raise ValueError(f"Unknown Country: {Country}. Expected one of: Korea, Iberia, Scotland, Chile, Canada")


############## 1) Create .dat files and save out to save time in plotting #################

folder = '/data/scratch/chantelle.burton/SoW2526/'
index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'



# Unified block for both hist and histnat, exporting to a single CSV (no bias correction)
index_filestem = index_filestem1 if run_type == 'hist' else index_filestem2
ensemble_members = np.arange(1, 106)  # 105 ensemble members
realisations = np.arange(1, 6)        # 5 realisations per ensemble member

# Assume all cubes have the same year range, get years from first cube
n_cols = len(ensemble_members) * len(realisations)
data_matrix = None
col_names = []

for e_idx, ensemble_member in enumerate(ensemble_members):
    for r_idx, realisation in enumerate(realisations):
        col_idx = e_idx * len(realisations) + r_idx
        col_names.append(f"Ens{ensemble_member}_Real{realisation}")
        try:
            if ensemble_member < 10:
                cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
            elif ensemble_member < 100:
                cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
            else:
                cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
            cube = apply_shapefile_inclusive(shp_file, shape_name, cube)
            cube = ConstrainToYear(cube, DATA_YEAR)
            cube = CountryPercentile(cube, percentile)
            cube = TimePercentile(cube, percentile)
            data = np.ravel(cube.data)
            # On first successful cube, set up years and matrix
            
            if data_matrix is None:
                years = np.array([dt.year for dt in cube.coord('time').units.num2date(cube.coord('time').points)])
                n_years = len(years)
                data_matrix = np.full((n_years, n_cols), np.nan)
            if len(data) == n_years:
                data_matrix[:, col_idx] = data
            else:
                print(f"Warning: Data length mismatch for Ens{ensemble_member} Real{realisation}")        
        except IOError:
            print(f"Missing data for Ens{ensemble_member} Real{realisation}")
            continue

# Export to CSV if any data was found
if data_matrix is not None:
    df_out = pd.DataFrame(data_matrix, columns=col_names)
    df_out.insert(0, "Year", years)
    output_file = f"{output_folder}{Country}_NoCorrection_{run_type}_{percentile}percent_DataYear_{DATA_YEAR}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"Wrote output to {output_file}")
else:
    print("No data found to export.")