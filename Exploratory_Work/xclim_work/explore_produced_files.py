import xarray 
import iris
import numpy as np
import iris.quickplot as qplt
cubes = iris.load('/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_2021-2021.nc')
print(cubes)
cube = cubes.extract(iris.Constraint(name='Fire Weather Index'))[0]
print("Cube coordinates:", cube.coords)
# Remove only the auxiliary 'time' coordinate (not the dimension coordinate)
import iris.coords
for coord in cube.coords('time'):
	if isinstance(coord, iris.coords.AuxCoord):
		print(f"Removing auxiliary time coord: {coord}")
		cube.remove_coord(coord)
# Print available time points
time_coord = cube.coord('time')
print("Time points available:", time_coord.points)

# Select a short period (e.g., first 10 time steps)
short_period = cube[:10]
print(short_period)
fwi_data = short_period.data
fwi_min = np.min(fwi_data)
fwi_max = np.max(fwi_data)
print(f"FWI min (first 10): {fwi_min}")
print(f"FWI max (first 10): {fwi_max}")

# Count NaNs and non-NaNs
n_nans = np.isnan(fwi_data).sum()
n_non_nans = np.count_nonzero(~np.isnan(fwi_data))
print(f"Number of NaNs in FWI data (first 10): {n_nans}")
print(f"Number of non-NaNs in FWI data (first 10): {n_non_nans}")

# --- Additional diagnostics for correct slicing and NaN counting ---
print("\n--- Diagnostics for full cube ---")
print("Cube data shape:", cube.data.shape)
n_nans_full = np.isnan(cube.data).sum()
n_non_nans_full = np.count_nonzero(~np.isnan(cube.data))
print(f"Number of NaNs in full FWI data: {n_nans_full}")
print(f"Number of non-NaNs in full FWI data: {n_non_nans_full}")

# Try counting for the first time step (assuming time is last axis)
try:
    fwi_first_time = cube.data[..., 0]
    print("First time step shape:", fwi_first_time.shape)
    n_nans_first_time = np.isnan(fwi_first_time).sum()
    n_non_nans_first_time = np.count_nonzero(~np.isnan(fwi_first_time))
    print(f"Number of NaNs in FWI data (first time step): {n_nans_first_time}")
    print(f"Number of non-NaNs in FWI data (first time step): {n_non_nans_first_time}")
except Exception as e:
    print("Error extracting first time step:", e)
except Exception as e:
	print("Error extracting short period:", e)