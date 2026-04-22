import iris
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from datetime import datetime

# === TOGGLE: Set variable to analyze ===
# Options: 'humidity' or 'temperature'
VAR_TO_ANALYZE = 'humidity'  # Change to 'temperature' for temperature analysis
PLOT_FOLDER = '/data/scratch/bob.potts/sowf/test_output/Plots'

# Set file pattern and variable search string
if VAR_TO_ANALYZE == 'humidity':
    file_pattern = "/data/scratch/bob.potts/sowf/JPA-3Q/*spfh-hyb-an-gauss*.199*.nc"
    var_search = 'humidity'
    ylabel = 'Mean Specific Humidity'
    title = 'Monthly Mean Specific Humidity (1990s, all spfh files)'
elif VAR_TO_ANALYZE == 'temperature':
    file_pattern = "/data/scratch/bob.potts/sowf/JPA-3Q/*tmp-hyb-an-gauss*.199*.nc"
    var_search = 'temperature'
    ylabel = 'Mean Temperature (K)'
    title = 'Monthly Mean Temperature (1990s, all tmp files)'
else:
    raise ValueError("VAR_TO_ANALYZE must be 'humidity' or 'temperature'")

files = sorted(glob.glob(file_pattern))

print(f"Found {len(files)} files for 1990s ({var_search}):")
for f in files:
    print(os.path.basename(f))

all_times = []
all_means = []

for f in files:
    cubes = iris.load(f)
    # Find the relevant cube
    target_cube = None
    for cube in cubes:
        if var_search in cube.name().lower():
            target_cube = cube
            break
    if target_cube is None:
        print(f"No {var_search} cube found in {f}")
        continue
    # Try to collapse over all spatial and vertical dims to get mean for each time
    collapse_dims = []
    for dim in ['model_level_number', 'atmosphere_hybrid_sigma_pressure_coordinate', 'latitude', 'longitude']:
        try:
            target_cube.coord(dim)
            collapse_dims.append(dim)
        except Exception:
            continue
    if target_cube.ndim > 1 and collapse_dims:
        try:
            mean_cube = target_cube.collapsed(collapse_dims, iris.analysis.MEAN)
        except Exception as e:
            print(f"Collapse failed for {f}: {e}")
            continue
        # Get time points
        try:
            time_coord = mean_cube.coord('time')
            times = time_coord.units.num2date(time_coord.points)
            all_times.extend(times)
            all_means.extend(mean_cube.data)
        except Exception as e:
            print(f"Time extraction failed for {f}: {e}")
    else:
        print(f"Cube in {f} is not suitable for collapse, skipping.")

# Convert to numpy arrays and sort by time
all_times = np.array(all_times)
all_means = np.array(all_means)
if len(all_times) == 0:
    print("No time series data found.")
else:
    # Sort by time
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_means = all_means[sort_idx]
    # Group by month and compute monthly means
    months = np.array([datetime(t.year, t.month, 1) for t in all_times])
    unique_months = np.unique(months)
    monthly_means = []
    for m in unique_months:
        monthly_means.append(np.mean(all_means[months == m]))
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(unique_months, monthly_means, marker='o')
    plt.xlabel('Month')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER + f"/JPA_SK_monthly_mean_{var_search}_1990s.png", dpi=300)
    plt.show()
