import glob
import re
import iris
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")

############# User inputs here #############
Country = "Iberia"
YEAR = 2024
FORCINGS = ["historicalExt", "historicalNatExt"]  # set to one or both
# Options: 'South Korea', 'Iberia', 'Scotland'
############# User inputs end here #############

folder = "/data/scratch/chantelle.burton/SoW2526/"
shp_file = "/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp"
data_folder = "/data/scratch/chantelle.burton/SoW2526/Y2526FWI/"
index_name = "canadian_fire_weather_index"
start_time = time.time()

if Country == "South Korea":
    print("Running South Korea")
    percentile = 95
    shape_name = "South Korea"

elif Country == "Iberia":
    print("Running Iberia")
    percentile = 95
    shape_name = "Northwest Iberia"

elif Country == "Scotland":
    print("Running Scotland")
    percentile = 95
    shape_name = "Scottish Highlands"

else:
    raise ValueError(f"Unknown Country: {Country}")

pattern = re.compile(r"_r(\d+)i1p(\d+)_")
forcing_suffix = {"historicalExt": "EXT", "historicalNatExt": "NATEXT"}

def member_realization_key(path):
    m = pattern.search(path)
    if not m:
        return (10**9, 10**9) # chuck unmatched files to the end
    return (int(m.group(1)), int(m.group(2)))

for forcing in FORCINGS:
    print(f"\n=== Processing {forcing} ===")
    files = sorted(
        glob.glob(f"{data_folder}FWI_HadGEM3-A-N216_r*_{forcing}_20230601-20250201_global_day.nc"),
        key=member_realization_key
    )
    print(f"Found {len(files)} files")

    cubes = iris.cube.CubeList()
    for f in files:
        try:
            match = pattern.search(f)
            if not match:
                print(f"Skipping (no member/realization match): {f}")
                continue

            member = int(match.group(1))
            realization = int(match.group(2))
            print(f"Loading member={member}, realization={realization}")

            cube = iris.load_cube(f, index_name)
            cube.add_aux_coord(iris.coords.AuxCoord(member, long_name="ensemble_member", units="1"))
            cube.add_aux_coord(iris.coords.AuxCoord(realization, long_name="realization_number", units="1"))
            cubes.append(cube)

        except IOError as e:
            print(f"Error loading {f}: {e}")

    print(f"Loaded {len(cubes)} cubes")

    iris.util.equalise_attributes(cubes)
    merged = cubes.merge_cube()
    print(f"Merged shape: {merged.shape}")

    merged = contrain_to_sow_shapefile(merged, shp_file, shape_name)
    merged = ConstrainToYear(merged, YEAR)
    merged = CountryPercentile(merged, percentile)
    merged = TimePercentile(merged, percentile)

    out = f"/data/scratch/bob.potts/sowf/test_output/{Country}_Uncorrected_hist_{forcing_suffix.get(forcing, forcing)}{percentile}%.nc"
    iris.save(merged, out)
    print(f"Saved: {out}")

print(f"--- {np.round(time.time() - start_time, 2)} seconds ---")
