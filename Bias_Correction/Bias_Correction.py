
import numpy as np
import iris
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import statsmodels.api as sm
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import time
import warnings
start_time = time.time()
############# User inputs here #############
Country = 'Iberia'
folder = '/data/scratch/chantelle.burton/SoW2526/'
cube_folder = '/data/scratch/bob.potts/sowf/test_output/Historical_Ensembles'
output_folder = '/data/scratch/bob.potts/sowf/test_output/Log_Transforms/'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
index_name = 'canadian_fire_weather_index'
# Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############





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





## For each historical member, bias correct 525 members for 2024/2025 and save out

#for member in np.arange(1,16):
for histmember in np.arange(10,11): #TEST
    print(histmember)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv(folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat')#This is the historical ERA5 array made in Historical_FWI.py/ Supplement.py
    df_sim = pd.read_csv(folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(histmember)+'_'+str(percentile)+'%.dat')#This is the historical HadGEM array made in Historical_FWI.py/ Supplement.py
    df_obs[np.isnan(df_obs)] = 0.000000000001 
    df_sim[np.isnan(df_sim)] = 0.000000000001 

    ####Log transform the data here#### 
    df_obs = np.log(np.exp(df_obs)-1)
    df_sim = np.log(np.exp(df_sim)-1)

    # Extract years and FWI values
    years = np.arange(1960,2013)
    fwi_sim = df_sim.values
    fwi_sim = fwi_sim[:,0]
    fwi_obs = df_obs.values
    fwi_obs = fwi_obs[:,0]

    # Step 1a: Fit a linear regression model to obs and sim
    t = years - 2025 # shift years to be relative to 2025
    X = sm.add_constant(t)  # add a constant term for intercept
    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()

        # Step 1b: Get the coefficients (slope and intercept)
        fwi0, delta = results.params
    
        return fwi0, delta, np.std(fwi - delta * t) 

    fwi0_sim, delta_sim, std_sim =  find_regression_parameters(fwi_sim)
    fwi0_obs, delta_obs, std_obs =  find_regression_parameters(fwi_obs)
    # The regression parameters (fwi0_obs, fwi0_sim, delta_sim) were fitted to shift data TO 2025
    # currently only have 2024 data, t_year = -1 for 2024, when have 2025 data set t_year = 0
    t_year = -1 #(scalar value)
    for Forcing in ['hist','histnat']:
        #/data/scratch/bob.potts/sowf/test_output/Historical_Ensembles/Iberia_UNCORRECTED_histnat_95_percent.nc
        historical_cube = iris.load_cube(os.path.join(cube_folder,f'{Country}_UNCORRECTED_{Forcing}_{percentile}_percent.nc'), index_name)
        print(historical_cube)
        #### Log transform the data here #### 
        historical_cube.data = np.log(np.exp(historical_cube.data) - 1)
        print(historical_cube.data.shape)
        # Step 2: Detrend the sim and scale to obs
        endhist = fwi0_obs + (historical_cube.data - delta_sim * t_year - fwi0_sim)

        #### Inverse Log (exponential) transform here ####      
        endhist = np.log(np.exp(endhist) + 1)

        # Create new cube with corrected data
        export_cube = historical_cube.copy()
        export_cube.data = endhist

        # Save to file Iberia_1_hist95%_LogTransform.dat
        iris.save(export_cube,os.path.join(output_folder, f'Testing_{Country}_{histmember}_{Forcing}_{percentile}_percent_LogTransform.nc'))
        np.savetxt(os.path.join(output_folder, f'Testing_{Country}_{histmember}_{Forcing}_{percentile}_percent_LogTransform.csv'), endhist, fmt='%s', delimiter=',')

        histnatarray = endhist.ravel()
        
print('Completed in',np.round(time.time() - start_time, 2), 'seconds')