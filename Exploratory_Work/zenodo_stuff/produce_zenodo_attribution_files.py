"""
Produce Zenodo attribution NetCDF files from the HadGEM3-A attribution ensemble.

For each complete ensemble member, loads all monthly FWI data across the
full 2020-2024 period (Jan 2020 – Dec 2024, 60 months), applies the
relevant shapefile mask, and saves a single NetCDF per member.

Only members returned by get_complete_member_dirs() are included – i.e.
members that have the full Nov 2019 – Jan 2025 time period available.

Output filename format:
    {zenodo_folder}/HadGEM3-A_FWI_{member_id}_{Country}_{run_name}.nc
"""

import iris
import iris.util
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Reduced_Att_Set_Processing'))

from utils.cubefuncs import apply_shapefile_inclusive
from find_matching_members import get_both_complete_member_ids, count_matching_files, N_EXPECTED, HIST_DIR as _HIST_DIR, HISTNAT_DIR as _HISTNAT_DIR
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

############# Get parameters from Cylc (or defaults for local testing) #############
Country = os.environ.get("CYLC_TASK_PARAM_country", None)
if Country is None:
    Country = "Iberia"
    print(f"WARNING: CYLC_TASK_PARAM_country not set, falling back to '{Country}'")

run_type = os.environ.get("CYLC_TASK_PARAM_runtype", None)
if run_type is None:
    run_type = "hist"
    print(f"WARNING: CYLC_TASK_PARAM_runtype not set, falling back to '{run_type}'")

# Single-member mode: cylc passes a flat member index 1..525.
# Index maps to r{run:03d}i1p{physics} where:
#   run     = (member - 1) // 5 + 1   (1..105)
#   physics = (member - 1) % 5 + 1    (1..5)
# If not set, fall back to looping over all complete members (local testing).
_member_num = os.environ.get("CYLC_TASK_PARAM_member", None)
if _member_num is not None:
    _n = int(_member_num)
    _run = (_n - 1) // 5 + 1
    _physics = (_n - 1) % 5 + 1
    SINGLE_MEMBER_ID = f"r{_run:03d}i1p{_physics}"
else:
    SINGLE_MEMBER_ID = None

print(f"Processing Country: {Country}, run type: {run_type}, member: {SINGLE_MEMBER_ID or 'ALL'}")

shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
zenodo_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Interim/'

HIST_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalExt'
)
HISTNAT_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalNatExt'
)

############# Country / region configuration #############
if Country == 'Korea':
    shape_name = 'Southeast South Korea'

elif Country == 'Iberia':
    shape_name = 'Northwest Iberia'

elif Country == 'Scotland':
    shape_name = 'Scottish Highlands'

elif Country == 'Chile':
    shape_name = 'Chilean Temperate Forests and Matorral'

elif Country == 'Canada':
    shape_name = 'Midwestern Canadian Shield forests'

else:
    raise ValueError(f"Unknown Country: {Country}. Expected one of: Korea, Iberia, Scotland, Chile, Canada")

index_filestem = 'historicalExt' if run_type == 'hist' else 'historicalNatExt'
run_name = 'Factual' if run_type == 'hist' else 'Counterfactual'
run_dir = HIST_DIR if run_type == 'hist' else HISTNAT_DIR

index_name = 'canadian_fire_weather_index'

############# Identify members to process #############
# Only members complete in BOTH historicalExt and historicalNatExt are used,
# so that hist and histnat outputs always form a matched pair.
if SINGLE_MEMBER_ID is not None:
    hist_path = os.path.join(_HIST_DIR, SINGLE_MEMBER_ID)
    histnat_path = os.path.join(_HISTNAT_DIR, SINGLE_MEMBER_ID)
    if (count_matching_files(hist_path) < N_EXPECTED or
            count_matching_files(histnat_path) < N_EXPECTED):
        print(f"Member {SINGLE_MEMBER_ID} is not complete in both run directories, nothing to do.")
        sys.exit(0)
    member_dirs = [os.path.join(run_dir, SINGLE_MEMBER_ID)]
    print(f"Single-member mode: {SINGLE_MEMBER_ID}")
else:
    both_complete_ids = get_both_complete_member_ids()
    member_dirs = [os.path.join(run_dir, m) for m in both_complete_ids]
    print(f"Found {len(member_dirs)} members complete in both run directories")


def _all_2020_2024_stamps():
    """Return the 60 monthly date-stamp strings for Jan 2020 – Dec 2024."""
    stamps = []
    for year in range(2020, 2025):
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            stamps.append(f"{start:%Y%m%d}-{end:%Y%m%d}")
    return stamps


MONTHLY_STAMPS = _all_2020_2024_stamps()


def load_member_full_period(member_dir):
    """Load and concatenate all 60 monthly files (Jan 2020 – Dec 2024) for a member."""
    member_id = os.path.basename(member_dir.rstrip('/'))
    cubes = iris.cube.CubeList()
    reference_time_units = None
    for stamp in MONTHLY_STAMPS:
        fname = (f"FWI_ATTRIBUTION_ENSEMBLE_MOHC_HadGEM3-A-N216_{index_filestem}_{member_id}"
                 f"_global_day_{stamp}.nc")
        cube = iris.load_cube(os.path.join(member_dir, fname), index_name)
        # Remove scalar time-derived coordinates that vary between monthly files
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
    iris.util.equalise_attributes(cubes)
    return cubes.concatenate_cube()


############# Main loop over members #############
os.makedirs(zenodo_folder, exist_ok=True)

n_success = 0
n_error = 0

for member_dir in member_dirs:
    member_id = os.path.basename(member_dir.rstrip('/'))
    print(f"\nProcessing member: {member_id}")

    try:
        cube = load_member_full_period(member_dir)
        cube = apply_shapefile_inclusive(shp_file, shape_name, cube)
    except (IOError, OSError) as e:
        print(f"  Missing file(s): {e}")
        n_error += 1
        continue
    except Exception as e:
        print(f"  ERROR: {e}")
        n_error += 1
        continue

    output_file = os.path.join(zenodo_folder, f"HadGEM3-A_FWI_2020-2024_{member_id}_{Country}_{run_name}.nc")
    iris.save(cube, output_file)
    print(f"  Saved: {output_file}")
    n_success += 1

print(f"\nDone. {n_success} members saved, {n_error} errors.")
