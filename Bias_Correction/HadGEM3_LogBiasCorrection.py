# Plot PDFs for 2023 HadGEM3 ALL and NAT, plus ERA5 line
######### NOTE: Need to run Supplement2.pr first to get df_obs and df_sim files ######


#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-


import numpy as np
import iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
baseline_member = int(os.environ.get("CYLC_TASK_PARAM_member", 1)) #if in doubt, use just ensemble member 1.
run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")  # 'hist' or 'histnat'

print(f'Processing Country: {Country}, baseline member: {baseline_member}, run type: {run_type}')

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
    raise ValueError(f"Unknown Country: {Country}. Expected one of: SouthKorea, Iberia, Scotland, Chile, Canada")


############## 1) Create .dat files and save out to save time in plotting #################

folder = '/data/scratch/chantelle.burton/SoW2526/'
baseline_folder = '/data/scratch/bob.potts/sowf/test_output/Baseline/'
index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'

## Bias correct 525 ensemble members for 2024/2025 using this baseline member
BiasCorrDict = {}

# Step 0: Load fwi data from CSV using pandas
df_obs = pd.read_csv(baseline_folder+'ERA5_FWI_1960-2013_'+Country+'_'+str(percentile)+'%.dat')  # Historical ERA5 array
df_sim = pd.read_csv(baseline_folder+'HadGEM3_FWI_1960-2013_'+Country+'_'+str(baseline_member)+'_'+str(percentile)+'%.dat')  # Historical HadGEM array
df_obs[np.isnan(df_obs)] = 0.000000000001 
df_sim[np.isnan(df_sim)] = 0.000000000001 

#### Log transform the data here #### 
df_obs = np.log(np.exp(df_obs)-1)
df_sim = np.log(np.exp(df_sim)-1)

# Extract years and FWI values
years = np.arange(1960, 2013)
fwi_sim = df_sim.values
fwi_sim = fwi_sim[:, 0]
fwi_obs = df_obs.values
fwi_obs = fwi_obs[:, 0]

# Step 1a: Fit a linear regression model to obs and sim
t = years - 2025  # shift years to be relative to 2025
X = sm.add_constant(t)  # add a constant term for intercept

def find_regression_parameters(fwi):
    model = sm.OLS(fwi, X)
    results = model.fit()
    # Step 1b: Get the coefficients (slope and intercept)
    fwi0, delta = results.params
    return fwi0, delta, np.std(fwi - delta * t) 

fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim)
fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs)


#### Process hist array ####
if run_type == 'hist':
    index_filestem = index_filestem1
    ensemble_members = np.arange(1, 106)  # 105 ensemble members
    histarray = []
    for ensemble_member in ensemble_members:
        print(f'hist ensemble_member={ensemble_member}, baseline_member={baseline_member}, country={Country}')
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

                #### Log transform the data here #### 
                data = np.log(np.exp(data)-1)

                # Step 2: Detrend the sim and scale to obs
                data_corrected = fwi0_obs + (data - delta_sim * t - fwi0_sim)
               
                #### Inverse log (exponential) transform here ####      
                data_corrected = np.log(np.exp(data_corrected)+1)

                f = open('/data/scratch/bob.potts/sowf/test_output/Log_Transforms/'+Country+'_baseline'+str(baseline_member)+'_ens'+str(ensemble_member)+'_hist'+str(percentile)+'percent_LogTransform.dat', 'a')
                np.savetxt(f, (data_corrected), newline=',', fmt='%s')
                f.write('\n')
                f.close()
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
        print(f'histnat ensemble_member={ensemble_member}, baseline_member={baseline_member}, country={Country}')
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

                #### Log transform the data here #### 
                data = np.log(np.exp(data)-1)

                # Step 2: Detrend the sim and scale to obs
                data_corrected = fwi0_obs + (data - delta_sim * t - fwi0_sim)

                #### Inverse log (exponential) transform here ####      
                data_corrected = np.log(np.exp(data_corrected)+1)

                f = open('/data/scratch/bob.potts/sowf/test_output/Log_Transforms/'+Country+'_baseline'+str(baseline_member)+'_ens'+str(ensemble_member)+'_histnat'+str(percentile)+'percent_LogTransform.dat', 'a')
                np.savetxt(f, (data_corrected), newline=',', fmt='%s')
                f.write('\n')
                f.close()
                histnatarray.append(data)
            except IOError:
                pass 
        
    histnatarray = np.array(histnatarray)
    histnatarray = np.ravel(histnatarray)

else:
    raise ValueError(f"Unknown run_type: {run_type}. Expected 'hist' or 'histnat'")