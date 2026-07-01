import numpy as np
import iris
import pandas as pd
import statsmodels.api as sm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

############# Get parameters from Cylc (or defaults for local testing) #############
Country = os.environ.get("CYLC_TASK_PARAM_country", None)
if Country is None:
    Country = "Iberia"
    print(f"WARNING: CYLC_TASK_PARAM_country not set, falling back to '{Country}'")

baseline_member_str = os.environ.get("CYLC_TASK_PARAM_member", None)
if baseline_member_str is None:
    baseline_member = 1
    print(f"WARNING: CYLC_TASK_PARAM_member not set, falling back to {baseline_member}")
else:
    baseline_member = int(baseline_member_str)

run_type = os.environ.get("CYLC_TASK_PARAM_runtype", None)
if run_type is None:
    run_type = "hist"
    print(f"WARNING: CYLC_TASK_PARAM_runtype not set, falling back to '{run_type}'")

print(f'Processing Country: {Country}, baseline member: {baseline_member}, run type: {run_type}')



shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
attribution_folder = '/data/scratch/chantelle.burton/SoW2526/'
output_dir = '/data/scratch/bob.potts/sowf/test_output/Condensed_Log_Transforms/'
baseline_folder = '/data/scratch/bob.potts/sowf/test_output/Baseline/'


DATA_YEARS = [2024]#[2021, 2022, 2023, 2024]# List of years to process. Currently set to just 2024 until 2020-2024 attribtution ensemble runs are processed.
BASELINE_START_YEAR = 1980 # start of the regression baseline period (inclusive)
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


############## 1) Create .csv files and save out to save time in plotting #################

index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'

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

# Step 1a: Regression helper (takes t as parameter so it can be recomputed per data year)
def find_regression_parameters(fwi, t):
    X = sm.add_constant(t)
    model = sm.OLS(fwi, X)
    results = model.fit()
    fwi0, delta = results.params
    return fwi0, delta, np.std(fwi - delta * t)

index_filestem = index_filestem1 if run_type == 'hist' else index_filestem2 #pull cylc run_type parameter in and do a hist or histnat run. 

# Loop over each DATA_YEAR (target year = data year)
for DATA_YEAR in DATA_YEARS:
    print(f"Processing DATA_YEAR: {DATA_YEAR} (target year = {DATA_YEAR})")

    # Fit regression centred on this data year
    t = years - DATA_YEAR
    fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim, t)
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs, t)
    ensemble_members = np.arange(1, 106)  # 105 ensemble members
    realisations = np.arange(1, 6)        # 5 realisations per ensemble member
    n_years = len(years)
    n_cols = len(ensemble_members) * len(realisations)
    data_matrix = np.full((n_years, n_cols), np.nan)
    col_names = []
    successful = []  # list of (ensemble_member, realisation, filepath)
    missing = []     # list of (ensemble_member, realisation, filepath)
    errors = []      # list of (ensemble_member, realisation, filepath, error_message)

    for e_idx, ensemble_member in enumerate(ensemble_members): # loop through all 105 ensemble members
        for r_idx, realisation in enumerate(realisations): #loop through physics realisations 1-5 for each of the 105 ensemble members
            col_idx = e_idx * len(realisations) + r_idx #number of columns (should be 5 * 105)
            col_names.append(f"Ens{ensemble_member}_Real{realisation}")

            # Build filepath once for logging
            member_str = f"r{ensemble_member:03d}"
            filename = f"FWI_HadGEM3-A-N216_{member_str}i1p{realisation}_{index_filestem}_20230601-20250201_global_day.nc"
            filepath = os.path.join(attribution_folder, 'Y2526FWI', filename)

            try:
                cube = iris.load_cube(filepath, index_name)
                cube = apply_shapefile_inclusive(shp_file, shape_name, cube)
                cube = ConstrainToYear(cube, DATA_YEAR)
                cube = CountryPercentile(cube, percentile)
                cube = TimePercentile(cube, percentile)
                data = np.ravel(cube.data)

                # Soft Log transform
                data = np.log(np.exp(data)-1)
                # Detrend and scale
                #data_corrected = fwi0_obs + delta_obs * t + (data - delta_sim * t - fwi0_sim)
                data_corrected = fwi0_obs + (data - delta_sim * t - fwi0_sim)
                # Inverse soft log transform
                data_corrected = np.log(np.exp(data_corrected)+1)

                # Store in matrix
                if len(data_corrected) == n_years: #check that the length of the data matches the number of years (BASELINE_START_YEAR to BASELINE_END_YEAR)
                    data_matrix[:, col_idx] = data_corrected
                    successful.append((ensemble_member, realisation, filepath))
                else:
                    errors.append((ensemble_member, realisation, filepath, f"Data length mismatch: got {len(data_corrected)}, expected {n_years}"))
            except (IOError, OSError):
                missing.append((ensemble_member, realisation, filepath))
                continue
            except Exception as e:
                errors.append((ensemble_member, realisation, filepath, str(e)))
                continue

    # Build DataFrame and write CSV
    df_out = pd.DataFrame(data_matrix, columns=col_names)
    df_out.insert(0, "Year", years)
    
    output_file = f"{output_dir}{Country}_baseline{baseline_member}_{run_type}{percentile}percent_LogTransform_Target_{DATA_YEAR}_DataYear_{DATA_YEAR}_BaselinePeriod_{BASELINE_START_YEAR}_{BASELINE_END_YEAR}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"Wrote output to {output_file}")

    # Write log file #this area is just for logging purposes.
    total = len(ensemble_members) * len(realisations)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_file_dir, exist_ok=True)
    log_file = f"{log_file_dir}/{Country}_baseline{baseline_member}_{run_type}_Target_{DATA_YEAR}_file_status_{run_timestamp}.txt"
    with open(log_file, 'w') as lf:
        lf.write(f"File Status Log\n")
        lf.write(f"===============\n")
        lf.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"Country: {Country}\n")
        lf.write(f"Baseline member: {baseline_member}\n")
        lf.write(f"Run type: {run_type}\n")
        lf.write(f"Data year: {DATA_YEAR}\n")
        lf.write(f"Baseline period: {BASELINE_START_YEAR}-{BASELINE_END_YEAR}\n")
        lf.write(f"Percentile: {percentile}\n\n")
        lf.write(f"Summary\n")
        lf.write(f"-------\n")
        lf.write(f"Successful: {len(successful)}/{total}\n")
        lf.write(f"Missing:    {len(missing)}/{total}\n")
        lf.write(f"Errors:     {len(errors)}/{total}\n\n")

        if missing:
            lf.write(f"Missing Files ({len(missing)})\n")
            lf.write(f"-" * 40 + "\n")
            for ens, real, fpath in missing:
                lf.write(f"  Ens{ens}_Real{real}: {fpath}\n")
            lf.write("\n")

        if errors:
            lf.write(f"Errors ({len(errors)})\n")
            lf.write(f"-" * 40 + "\n")
            for ens, real, fpath, msg in errors:
                lf.write(f"  Ens{ens}_Real{real}: {fpath}\n")
                lf.write(f"    Error: {msg}\n")
            lf.write("\n")

        if successful:
            lf.write(f"Successful ({len(successful)})\n")
            lf.write(f"-" * 40 + "\n")
            for ens, real, fpath in successful:
                lf.write(f"  Ens{ens}_Real{real}: {fpath}\n")

    print(f"Wrote file status log to {log_file}")
    print(f"Summary: {len(successful)}/{total} successful, {len(missing)}/{total} missing, {len(errors)}/{total} errors")