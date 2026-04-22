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
Country = os.environ.get("CYLC_TASK_PARAM_country", "Korea") #fallback value of Iberia
baseline_member = int(os.environ.get("CYLC_TASK_PARAM_member", 1)) #if in doubt, use just ensemble member 1.
run_type = os.environ.get("CYLC_TASK_PARAM_runtype", "hist")  # 'hist' or 'histnat'

print(f'Processing Country: {Country}, baseline member: {baseline_member}, run type: {run_type}')
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
folder = '/data/scratch/chantelle.burton/SoW2526/'
DATA_YEARS = [2024]#[2020, 2021, 2022, 2023, 2024] # List of years to process. Currently set to just 2024 untill 2020-2024 attribtution ensemble runs are done.
TARGET_YEAR = 2024 # this is the year we want the regression to be relative to (i.e. the year we want to bias correct to). 
BASELINE_START_YEAR = 1997 # start of the regression baseline period (inclusive)
BASELINE_END_YEAR = 2013 # end of the regression baseline period (inclusive)

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
    raise ValueError(f"Unknown Country: {Country}. Expected one of: SouthKorea, Iberia, Scotland, Chile, Canada")


############## 1) Create .dat files and save out to save time in plotting #################

folder = '/data/scratch/chantelle.burton/SoW2526/'
baseline_folder = '/data/scratch/bob.potts/sowf/test_output/Baseline/'
index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'

## Bias correct 525 ensemble members for 2024/2025 using this baseline member
BiasCorrDict = {}

# Step 0: Load FWI data from new CSVs using pandas
df_obs = pd.read_csv(baseline_folder+f'ERA5_FWI_{BASELINE_START_YEAR}-{BASELINE_END_YEAR}_{Country}_{percentile}%.csv')
df_sim = pd.read_csv(baseline_folder+f'HadGEM3_FWI_{BASELINE_START_YEAR}-{BASELINE_END_YEAR}_{Country}_{baseline_member}_{percentile}%.csv')

# Replace NaNs with small value to avoid log issues
df_obs['FWI'] = df_obs['FWI'].replace(np.nan, 0.000000000001)
df_sim['FWI'] = df_sim['FWI'].replace(np.nan, 0.000000000001)

#### Log transform the data here ####
df_obs_log = np.log(np.exp(df_obs['FWI'])-1)
df_sim_log = np.log(np.exp(df_sim['FWI'])-1)

# Extract years from the 'Date' column (assumes format YYYY-MM or YYYY-MM/MM) #technically reducdant with the file read but double checks.
all_years = df_obs['Date'].apply(lambda x: int(x.split('-')[0]))
baseline_mask = (all_years >= BASELINE_START_YEAR) & (all_years <= BASELINE_END_YEAR)
years = all_years[baseline_mask].values
fwi_obs = df_obs_log[baseline_mask].values
fwi_sim = df_sim_log[baseline_mask].values

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
                cube = apply_shapefile_inclusive(shp_file, shape_name, cube)
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