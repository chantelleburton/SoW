"""
Merge per-member Zenodo attribution NetCDF files into one file per region/run-type.

Reads all HadGEM3-A_FWI_2020-2024_*_{Region}_{RunType}.nc files from the
Zenodo export directory, stacks them along a new 'member_id' dimension, and
saves one merged file per (Region, RunType) combination.

Output: {zenodo_folder}/HadGEM3-A_FWI_{Region}_{RunType}.nc
"""

import glob
import os
import re
from collections import defaultdict

import iris
import iris.cube
import iris.coords
import iris.util

zenodo_input_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Interim/'
zenodo_output_folder = '/data/scratch/bob.potts/sowf/test_output/Zenodo_Export/'
# ---- Discover per-member files ----
pattern = os.path.join(zenodo_input_folder, 'HadGEM3-A_FWI_2020-2024_*.nc')
all_files = sorted(glob.glob(pattern))

fname_re = re.compile(
    r'HadGEM3-A_FWI_2020-2024_(r\d+i1p\d+)_([A-Za-z]+)_(Factual|Counterfactual)\.nc$'
)

# Group by (region, run_type)
groups = defaultdict(list)
for fpath in all_files:
    m = fname_re.match(os.path.basename(fpath))
    if m:
        member_id, region, run_type = m.group(1), m.group(2), m.group(3)
        groups[(region, run_type)].append((member_id, fpath))

print(f"Found {len(all_files)} per-member files in {len(groups)} region/run-type groups")

# ---- Merge each group along a new member_id dimension ----
for (region, run_type), members in sorted(groups.items()):
    members = sorted(members)  # sort by member_id for reproducibility
    print(f"\nMerging {len(members)} members for {region} / {run_type}")

    cubes = iris.cube.CubeList()
    for member_id, fpath in members:
        cube = iris.load_cube(fpath)
        cube.add_aux_coord(
            iris.coords.AuxCoord(member_id, long_name='member_id')
        )
        cubes.append(cube)

    iris.util.equalise_attributes(cubes)
    merged = cubes.merge_cube()

    output_file = os.path.join(zenodo_output_folder, f"HadGEM3-A_FWI_2020-2024_{region}_{run_type}.nc")
    iris.save(merged, output_file)
    print(f"  Saved: {output_file} ({len(members)} members)")

print("\nDone.")
