"""
Merge per-member baseline NetCDF files into one file per region.

Reads all HadGEM3-A_FWI_1980-2013_{Region}_member{N}.nc files from the
Zenodo export directory, stacks them along a new 'member' dimension, and
saves one merged file per region.

Output: {zenodo_folder}/HadGEM3-A_FWI_1980-2013_{Region}.nc
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
# ---- Discover per-member baseline files ----
pattern = os.path.join(zenodo_input_folder, 'HadGEM3-A_FWI_1980-2013_*.nc')
all_files = sorted(glob.glob(pattern))

fname_re = re.compile(
    r'HadGEM3-A_FWI_1980-2013_([A-Za-z]+)_member(\d+)\.nc$'
)

# Group by region, keyed by integer member number for correct ordering
groups = defaultdict(list)
for fpath in all_files:
    m = fname_re.match(os.path.basename(fpath))
    if m:
        region, member_num = m.group(1), int(m.group(2))
        groups[region].append((member_num, fpath))

print(f"Found {len(all_files)} per-member files across {len(groups)} regions")

# ---- Merge each region along a new member dimension ----
for region, members in sorted(groups.items()):
    members = sorted(members)  # sort by member number
    print(f"\nMerging {len(members)} members for {region}")

    cubes = iris.cube.CubeList()
    for member_num, fpath in members:
        cube = iris.load_cube(fpath)
        cube.add_aux_coord(
            iris.coords.AuxCoord(member_num, long_name='member', units='1')
        )
        cubes.append(cube)

    iris.util.equalise_attributes(cubes)
    merged = cubes.merge_cube()

    output_file = os.path.join(zenodo_output_folder, f"HadGEM3-A_FWI_1980-2013_{region}.nc")
    iris.save(merged, output_file)
    print(f"  Saved: {output_file} ({len(members)} members)")

print("\nDone.")
