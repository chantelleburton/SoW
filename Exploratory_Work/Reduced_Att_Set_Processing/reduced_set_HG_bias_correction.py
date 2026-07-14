"""
TEMPORARY / EXPLORATORY bias-correction script.

Based on Bias_Correction/HadGEM3_LogBiasCorrection_MultiYear.py, but instead of processing the full
105 x 5 attribution ensemble grid, it processes only the REDUCED set of members that have complete
data over the 5-year period (2020-2024), as listed in reduced_set_complete_attr_members.csv.

Bias correction is still performed against all 15 baseline members (one per cylc `member` param).
The purpose is to see how the final probability ratio changes when the reduced member set is used.

Differences vs HadGEM3_LogBiasCorrection_MultiYear.py:
  * Members come from the reduced-set CSV (HistExt column for `hist`, HistNatExt for `histnat`).
  * Source data are the per-member directories of MONTHLY netCDF files (andrew.hartley attribution
    ensemble), so only the event month(s) are loaded per target year (consistent with the
    month-specific baseline CSVs).
  * DATA_YEARS covers the full 2020-2024 period.
"""

import numpy as np
import iris
import pandas as pd
import statsmodels.api as sm
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
from find_matching_members import get_complete_member_dirs
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
output_dir = '/data/scratch/bob.potts/sowf/test_output/Reduced_Set_Log_Transforms/'
baseline_folder = '/data/scratch/bob.potts/sowf/test_output/Baseline/'
zenodo_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Export/'

HIST_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalExt'
)
HISTNAT_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalNatExt'
)


DATA_YEARS = [2020, 2021, 2022, 2023, 2024]  # Full 5-year period the reduced set is complete for.
BASELINE_START_YEAR = 1980  # start of the regression baseline period (inclusive)
BASELINE_END_YEAR = 2013  # end of the regression baseline period (inclusive)

#Set up the months automatically
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
    Month = 6, 7
    month = 'June-July'
    shape_name = 'Scottish Highlands'
    percentile = 95

elif Country == 'Chile':
    print('Running Chile')
    Month = 1, 2
    month = 'January-February'
    percentile = 95
    shape_name = 'Chilean Temperate Forests and Matorral'

elif Country == 'Canada':
    print('Running Canada')
    Month = 7, 8
    month = 'July-August'
    percentile = 95
    shape_name = 'Midwestern Canadian Shield forests'

else:
    raise ValueError(f"Unknown Country: {Country}. Expected one of: Korea, Iberia, Scotland, Chile, Canada")

# Normalise Month to a tuple of month numbers
months = Month if isinstance(Month, tuple) else (Month,)

############## 1) Create .csv files and save out to save time in plotting #################

index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'

index_filestem = index_filestem1 if run_type == 'hist' else index_filestem2  # hist or histnat run
run_name = 'Factual' if run_type == 'hist' else 'Counterfactual'
# Step 0: Load FWI baseline data from CSVs using pandas (unchanged from MultiYear.py)
df_obs = pd.read_csv(baseline_folder+f'ERA5_FWI_{BASELINE_START_YEAR}-{BASELINE_END_YEAR}_{Country}_{percentile}%.csv')
df_sim = pd.read_csv(baseline_folder+f'HadGEM3_FWI_{BASELINE_START_YEAR}-{BASELINE_END_YEAR}_{Country}_{baseline_member}_{percentile}%.csv')

# Replace NaNs with small value to avoid log issues
df_obs['FWI'] = df_obs['FWI'].replace(np.nan, 0.000000000001)
df_sim['FWI'] = df_sim['FWI'].replace(np.nan, 0.000000000001)

#### Log transform the data here ####
df_obs_log = np.log(np.exp(df_obs['FWI'])-1)
df_sim_log = np.log(np.exp(df_sim['FWI'])-1)

# Extract years from the 'Date' column (assumes format YYYY-MM or YYYY-MM/MM)
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


# Step 1b: Dynamically find complete members for this run type
run_dir = HIST_DIR if run_type == 'hist' else HISTNAT_DIR
member_dirs = get_complete_member_dirs(run_dir)
print(f"Reduced set: {len(member_dirs)} complete members found in {run_dir}")


def build_member_month_files(member_dir, year, months):
    """Build the monthly netCDF file paths for a member directory / year / event month(s)."""
    member_id = os.path.basename(member_dir.rstrip('/'))
    files = []
    for m in months:
        start = date(year, m, 1)
        end = date(year + 1, 1, 1) if m == 12 else date(year, m + 1, 1)
        fname = (f"FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{index_filestem}_{member_id}"
                 f"_global_day_{start:%Y%m%d}-{end:%Y%m%d}.nc")
        files.append(os.path.join(member_dir, fname))
    return files


def load_member_event_month(member_dir, year, months):
    """Load and concatenate the event-month file(s) for a member/year into a single cube."""
    files = build_member_month_files(member_dir, year, months)

    cubes = iris.cube.CubeList()
    reference_time_units = None
    for f in files:
        cube = iris.load_cube(f, index_name)
        if len(files) > 1:
            # Remove scalar time-related coordinates that differ between months (prevent concat issues)
            for coord in list(cube.coords()):
                if coord.long_name in ('month', 'month_number', 'season', 'season_year', 'year'):
                    cube.remove_coord(coord)
            # Standardise time units to the first cube's reference time
            if cube.coords('time'):
                time_coord = cube.coord('time')
                if reference_time_units is None:
                    reference_time_units = time_coord.units
                else:
                    time_coord.convert_units(reference_time_units)
                time_coord.attributes = {}
                time_coord.var_name = None
                time_coord.long_name = None
                time_coord.standard_name = 'time'
        cubes.append(cube)

    if len(cubes) > 1:
        return cubes.concatenate_cube()
    return cubes[0]


# Loop over each DATA_YEAR (target year = data year)
for DATA_YEAR in DATA_YEARS:
    print(f"Processing DATA_YEAR: {DATA_YEAR} (target year = {DATA_YEAR})")

    # Fit regression centred on this data year
    t = years - DATA_YEAR
    fwi0_sim, delta_sim, std_sim = find_regression_parameters(fwi_sim, t)
    fwi0_obs, delta_obs, std_obs = find_regression_parameters(fwi_obs, t)

    n_years = len(years)
    n_cols = len(member_dirs)
    data_matrix = np.full((n_years, n_cols), np.nan)
    col_names = []
    successful = []  # list of (member_id, filepath)
    missing = []     # list of (member_id, filepath)
    errors = []      # list of (member_id, filepath, error_message)

    for col_idx, member_dir in enumerate(member_dirs):
        member_id = os.path.basename(member_dir.rstrip('/'))
        col_names.append(member_id)
        filepaths = build_member_month_files(member_dir, DATA_YEAR, months)

        try:
            cube = load_member_event_month(member_dir, DATA_YEAR, months)
            cube = apply_shapefile_inclusive(shp_file, shape_name, cube)

            cube = CountryPercentile(cube, percentile)
            cube = TimePercentile(cube, percentile)
            data = np.ravel(cube.data)

            # Soft Log transform
            data = np.log(np.exp(data)-1)
            # Detrend and scale (bias correction against baseline member)
            data_corrected = fwi0_obs + (data - delta_sim * t - fwi0_sim)
            # Inverse soft log transform
            data_corrected = np.log(np.exp(data_corrected)+1)

            # Store in matrix
            if len(data_corrected) == n_years:
                data_matrix[:, col_idx] = data_corrected
                successful.append((member_id, filepaths[0]))
            else:
                errors.append((member_id, filepaths[0],
                               f"Data length mismatch: got {len(data_corrected)}, expected {n_years}"))
        except (IOError, OSError):
            missing.append((member_id, filepaths[0]))
            continue
        except Exception as e:
            errors.append((member_id, filepaths[0], str(e)))
            continue

    # Build DataFrame and write CSV
    df_out = pd.DataFrame(data_matrix, columns=col_names)
    df_out.insert(0, "Year", years)

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}{Country}_baseline{baseline_member}_{run_type}{percentile}percent_LogTransform_Target_{DATA_YEAR}_DataYear_{DATA_YEAR}_BaselinePeriod_{BASELINE_START_YEAR}_{BASELINE_END_YEAR}.csv"
    df_out.to_csv(output_file, index=False)
    print(f"Wrote output to {output_file}")

    # Write log file #this area is just for logging purposes.
    total = len(member_dirs)
 
    print(f"Summary: {len(successful)}/{total} successful, {len(missing)}/{total} missing, {len(errors)}/{total} errors")
