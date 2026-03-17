# Plot PDFs for 2023 HadGEM3 ALL and NAT, plus ERA5 line
######### NOTE: Need to run Supplement2.pr first to get df_obs and df_sim files ######


#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-


import numpy as np
import iris
import statsmodels.api as sm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

############# Get parameters from Cylc (or defaults for local testing) #############
Country = os.environ.get("CYLC_TASK_PARAM_country", "Iberia") #fallback value of iberia
run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")  # 'hist' or 'histnat'
output_folder = '/data/scratch/bob.potts/sowf/test_output/Historical_Ensembles/'



folder = '/data/scratch/chantelle.burton/SoW2526/'
YEAR = 2024
#Set up the 2025 files and months automatically
if Country == 'Korea':
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
    shape_name = 'Scottish Highlands'
    percentile = 95
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Chile':
    print('Running Chile')
    Month = 1,2
    month = 'January-February'
    percentile = 95
    shape_name = 'Chilean Temperate Forests and Matorral'
    daterange = iris.Constraint(time=lambda cell: cell.point.month in Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-11-01-2026-02-28_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Canada':
    print('Running Canada')
    Month = 7,8
    month = 'July-August'
    percentile = 95
    shape_name = 'Midwestern Canadian Shield forests'
    daterange = iris.Constraint(time=lambda cell: cell.point.month in Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

else:
    raise ValueError(f"Unknown Country: {Country}. Expected one of: Korea, Iberia, Scotland, Chile, Canada")


############## 1) Create .dat files and save out to save time in plotting #################

folder = '/data/scratch/chantelle.burton/SoW2526/'
index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'


#### Process hist array ####
if run_type == 'hist':
    index_filestem = index_filestem1
    ensemble_members = np.arange(1, 106)  # 105 ensemble members
    histarray = []
    for ensemble_member in ensemble_members:
        for realisation in np.arange(1, 6):  # 5 realisations per ensemble member
            try:
                if ensemble_member < 10:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                elif ensemble_member < 100:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                else:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)           
                cube = contrain_to_sow_shapefile(cube, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', shape_name)
                cube = ConstrainToYear(cube,YEAR) 
                cube = CountryPercentile(cube, percentile)
                cube = TimePercentile(cube, percentile)
                data = np.ravel(cube.data)
                # Output file matches PDF_95%.py format:
                filename = output_folder + f'{Country}_Uncorrected_hist{percentile}%.dat'
                with open(filename, 'a') as f:
                    np.savetxt(f, data, newline=',', fmt='%s')
                    f.write('\n')
                histarray.append(data)
            except IOError:
                pass 
     
    histarray = np.array(histarray)
    histarray = np.ravel(histarray)
    print(repr(histarray)) 

#### Process histnat array ####
elif run_type == 'histnat':
    index_filestem = index_filestem2
    histnatarray = []
    ensemble_members = np.arange(1, 106)
    for ensemble_member in ensemble_members:
        for realisation in np.arange(1, 6):
            try:
                if ensemble_member < 10:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                elif ensemble_member < 100:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                else:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)           
                cube = contrain_to_sow_shapefile(cube, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', shape_name)
                cube = ConstrainToYear(cube,YEAR)  
                cube = CountryPercentile(cube, percentile)
                cube = TimePercentile(cube, percentile) 
                data = np.ravel(cube.data)
                # Output file matches PDF_95%.py format:
                filename = output_folder + f'{Country}_Uncorrected_histnat{percentile}%.dat'
                with open(filename, 'a') as f:
                    np.savetxt(f, data, newline=',', fmt='%s')
                    f.write('\n')
                histnatarray.append(data)
            except IOError:
                pass 
        
    histnatarray = np.array(histnatarray)
    histnatarray = np.ravel(histnatarray)
    print(repr(histnatarray))

else:
    raise ValueError(f"Unknown run_type: {run_type}. Expected 'hist' or 'histnat'")