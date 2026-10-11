"""
Produce per-member Zenodo interim NetCDF files from the HadGEM3-A attribution ensemble.

One task per (run_type, member): reads the member's xclim FWI file once
(attribution_pipeline raw_fwi/hg3_attribution output), trims it to Jan 2020 -
Dec 2024, and writes one shapefile-masked file per region in REGIONS.

Only members with a file for BOTH historicalExt and historicalNatExt are
processed. Partial time axes are saved as they are; attribution_file_merging.py
fills the missing days with NaN.

Output filename format:
    {zenodo_folder}/HadGEM3-A_FWI_2020-2024_{member_id}_{Country}_{run_name}.nc
"""

import iris
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from attribution_pipeline._utils import apply_shapefile_inclusive

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

INDEX = "fwi"

RAW_FWI_DIR = '/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_attribution'
SHAPEFILE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../Shapefiles/SoW2526_Focal_MASTER_20260218.shp',
))
SHAPEFILE_NAME_COLUMN = 'name'

# Country label (used in output filenames) -> shape name in SHAPEFILE
REGIONS = {
    'Iberia': 'Northwest Iberia',
    'Chile': 'Chilean Temperate Forests and Matorral',
    'Canada': 'Midwestern Canadian Shield forests',
}

# Cylc runtype -> (raw filename token, output run name)
RUN_TYPES = {
    'hist': ('historicalExt', 'Factual'),
    'histnat': ('historicalNatExt', 'Counterfactual'),
}

zenodo_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Interim/'

############# Get parameters from Cylc (or defaults for local testing) #############
run_type = os.environ.get("CYLC_TASK_PARAM_runtype", None)
if run_type is None:
    run_type = "hist"
    print(f"WARNING: CYLC_TASK_PARAM_runtype not set, falling back to '{run_type}'")

# Single-member mode: cylc passes a flat member index 1..525.
# Index maps to r{run:03d}i1p{physics} where:
#   run     = (member - 1) // 5 + 1   (1..105)
#   physics = (member - 1) % 5 + 1    (1..5)
# If not set, fall back to looping over all 525 members (local testing).
_member_num = os.environ.get("CYLC_TASK_PARAM_member", None)
if _member_num is not None:
    _n = int(_member_num)
    _run = (_n - 1) // 5 + 1
    _physics = (_n - 1) % 5 + 1
    member_ids = [f"r{_run:03d}i1p{_physics}"]
else:
    member_ids = [f"r{r:03d}i1p{p}" for r in range(1, 106) for p in range(1, 6)]

run_token, run_name = RUN_TYPES[run_type]
print(f"Processing run type: {run_type}, members: {member_ids[0] if len(member_ids) == 1 else 'ALL'}")


def _member_file(token, member_id):
    return os.path.join(RAW_FWI_DIR, f"hadgem3a_{INDEX}_{token}_{member_id}.nc")


def _is_paired(member_id):
    return all(os.path.exists(_member_file(tok, member_id)) for tok, _ in RUN_TYPES.values())


def _load_2020_2024(member_id):
    cube = iris.load_cube(_member_file(run_token, member_id), iris.NameConstraint(var_name=INDEX))
    for coord_name in ("year", "season_year"):
        if cube.coords(coord_name):
            cube.remove_coord(coord_name)
    return cube.extract(iris.Constraint(time=lambda cell: 2020 <= cell.point.year <= 2024))


############# Main loop over members #############
os.makedirs(zenodo_folder, exist_ok=True)

n_saved = 0
n_skipped = 0
n_error = 0

for member_id in member_ids:
    print(f"\nProcessing member: {member_id}")

    if not _is_paired(member_id):
        print("  Not present in both historicalExt and historicalNatExt, skipping.")
        n_skipped += 1
        continue

    try:
        cube = _load_2020_2024(member_id)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        n_error += 1
        continue
    if cube is None:
        print("  No data in 2020-2024, skipping.")
        n_skipped += 1
        continue
    print(f"  {cube.coord('time').shape[0]} days in 2020-2024")

    for country, shape_name in REGIONS.items():
        try:
            region_cube = apply_shapefile_inclusive(
                SHAPEFILE, shape_name, cube, name_column=SHAPEFILE_NAME_COLUMN,
            )
        except Exception as e:
            print(f"  ERROR masking {country}: {e}")
            n_error += 1
            continue
        output_file = os.path.join(
            zenodo_folder, f"HadGEM3-A_FWI_2020-2024_{member_id}_{country}_{run_name}.nc"
        )
        iris.save(region_cube, output_file)
        print(f"  Saved: {output_file}")
        n_saved += 1

print(f"\nDone. {n_saved} files saved, {n_skipped} members skipped, {n_error} errors.")
