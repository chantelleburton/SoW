import numpy as np
import iris
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
DATA_YEARS = [2024]#[2020, 2021, 2022, 2023, 2024] # List of years to process. Currently set to just 2024 untill 2020-2024 attribtution ensemble runs are done.
TARGET_YEAR = 2024 # this is the year we want the regression to be relative to (i.e. the year we want to bias correct to). 
BASELINE_START_YEAR = 1960 # start of the regression baseline period (inclusive)
BASELINE_END_YEAR = 2013 # end of the regression baseline period (inclusive)

#Set up the 2025 files and months automatically
if Country == 'Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    shape_name = 'Southeast South Korea'
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')
      
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
df_obs = pd.read_csv(baseline_folder+'ERA5_FWI_1960-2013_'+Country+'_'+str(percentile)+'%.dat',header=None)  # Historical ERA5 array
df_sim = pd.read_csv(baseline_folder+'HadGEM3_FWI_1960-2013_'+Country+'_'+str(baseline_member)+'_'+str(percentile)+'%.dat',header=None)  # Historical HadGEM array
df_obs[np.isnan(df_obs)] = 0.000000000001 #nans break the soft log transform so fill val ~ 0 to avoid issues.
df_sim[np.isnan(df_sim)] = 0.000000000001 

#### Log transform the data here #### 
df_obs = np.log(np.exp(df_obs)-1)
df_sim = np.log(np.exp(df_sim)-1)

# Extract years and FWI values
years = np.arange(BASELINE_START_YEAR, (BASELINE_END_YEAR + 1)) #arange uses the stop value as exclusive, so add 1 to include the end year.
fwi_sim = df_sim.values
fwi_sim = fwi_sim[:, 0]
fwi_obs = df_obs.values
fwi_obs = fwi_obs[:, 0]

# Step 1a: Fit a linear regression model to obs and sim
t = years - TARGET_YEAR #shift years relative to the target year.
X = sm.add_constant(t)  # add a constant term for intercept

def find_regression_parameters(fwi):
    model = sm.OLS(fwi, X)
    results = model.fit()
    # Step 1b: Get the coefficients (slope and intercept)
    fwi0, delta = results.params
    return fwi0, delta, np.std(fwi - delta * t) 

fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim)
fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs)



index_filestem = index_filestem1 if run_type == 'hist' else index_filestem2 #pull cylc run_type parameter in and do a hist or histnat run. 

# Loop over each DATA_YEAR
for DATA_YEAR in DATA_YEARS:
    print(f"Processing DATA_YEAR: {DATA_YEAR}")
    ensemble_members = np.arange(1, 106)  # 105 ensemble members
    realisations = np.arange(1, 6)        # 5 realisations per ensemble member
    n_years = len(years)
    n_cols = len(ensemble_members) * len(realisations)
    data_matrix = np.full((n_years, n_cols), np.nan)
    col_names = []

    for e_idx, ensemble_member in enumerate(ensemble_members): # loop through all 105 ensemble members
        for r_idx, realisation in enumerate(realisations): #loop through physics realisations 1-5 for each of the 105 ensemble members
            col_idx = e_idx * len(realisations) + r_idx #number of columns (should be 5 * 105)
            col_names.append(f"Ens{ensemble_member}_Real{realisation}")
            try:
                if ensemble_member < 10:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name) 
                elif ensemble_member < 100:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                else:
                    cube = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(ensemble_member)+'i1p'+str(realisation)+'_'+index_filestem+'_20230601-20250201_global_day.nc', index_name)
                cube = contrain_to_sow_shapefile(cube, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', shape_name)
                cube = ConstrainToYear(cube, DATA_YEAR)
                cube = CountryPercentile(cube, percentile)
                cube = TimePercentile(cube, percentile)
                data = np.ravel(cube.data)

                # Soft Log transform
                data = np.log(np.exp(data)-1)
                # Detrend and scale
                data_corrected = fwi0_obs + (data - delta_sim * t - fwi0_sim)
                # Inverse soft log transform
                data_corrected = np.log(np.exp(data_corrected)+1)

                # Store in matrix
                if len(data_corrected) == n_years: #check that the length of the data matches the number of years (BASELINE_START_YEAR to BASELINE_END_YEAR)
                    data_matrix[:, col_idx] = data_corrected
                else:
                    print(f"Warning: Data length mismatch for Ens{ensemble_member} Real{realisation}")
            except IOError:
                print(f"Missing data for Ens{ensemble_member} Real{realisation}")
                continue

    # Build DataFrame and write CSV
    df_out = pd.DataFrame(data_matrix, columns=col_names)
    df_out.insert(0, "Year", years)
    output_dir = '/data/scratch/bob.potts/sowf/test_output/Condensed_Log_Transforms/'
    output_file = f"{output_dir}{Country}_baseline{baseline_member}_{run_type}{percentile}percent_LogTransform_Target_{TARGET_YEAR}_DataYear_{DATA_YEAR}_BaselinePeriod_{BASELINE_START_YEAR}_{BASELINE_END_YEAR}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"Wrote output to {output_file}")