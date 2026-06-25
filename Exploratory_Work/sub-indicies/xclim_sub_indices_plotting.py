import os
import sys
import re
import glob
import warnings

warnings.filterwarnings("ignore", module="iris")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import iris
import iris.analysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geopandas as gpd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.cubefuncs import apply_shapefile_inclusive, CountryPercentile, TimePercentile
from utils.constrain_cubes_standard import contrain_coords


INPUT_DIR = "/data/scratch/bob.potts/sowf/test_output/XClim_FWI"
PLOT_DIR = "/data/scratch/bob.potts/sowf/test_output/Plots/Sub-Indicies/XClim_4Permutations"
EXPORT_DIR = "/data/scratch/bob.potts/sowf/test_output/Exports/Sub-Indicies/XClim_4Permutations"
SHP_FILE = "/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp"

PERCENTILE = 95

REGION_CONFIGS = {
	"Iberia": {"shape_name": "Northwest Iberia"},
	"Korea": {"shape_name": "Southeast South Korea"},
	"Scotland": {"shape_name": "Scottish Highlands"},
	"Chile": {"shape_name": "Chilean Temperate Forests and Matorral"},
	"Canada": {"shape_name": "Midwestern Canadian Shield forests"},
}

INDEX_ORDER = ["ffmc", "dmc", "dc", "isi", "bui"]#"fwi"
INDEX_LABELS = {
	"fwi": "FWI",
	"ffmc": "FFMC",
	"dmc": "DMC",
	"dc": "DC",
	"isi": "ISI",
	"bui": "BUI",
}

PERMUTATIONS = [
	("Mean_RH_Mean_Wind", "Mean RH + Mean Wind", "#1f77b4"),
	("Mean_RH_Max_Wind", "Mean RH + Max Wind", "#ff7f0e"),
	("Minimum_RH_Mean_Wind", "Minimum RH + Mean Wind", "#2ca02c"),
	("Minimum_RH_Max_Wind", "Minimum RH + Max Wind", "#d62728"),
]

# Legacy FWI naming for backward compatibility.
LEGACY_FWI_SUFFIX = {
	"Mean_RH_Mean_Wind": "RH_Mean_MeanWS",
	"Mean_RH_Max_Wind": "Mean_RH_MaxWS",
	"Minimum_RH_Mean_Wind": "Minimum_RH_MeanWS",
	"Minimum_RH_Max_Wind": "Minimum_RH_MaxWS",
}


def parse_year_start(path):
	"""Sort files by the first year in their filename."""
	fname = os.path.basename(path)
	m = re.search(r"_(\d{4})-(\d{4})\.nc$", fname)
	if m:
		return int(m.group(1))
	m = re.search(r"_(\d{4})-(\d{4})_[A-Za-z_]+\.nc$", fname)
	if m:
		return int(m.group(1))
	return 9999


def find_input_files(index_name, run_label):
	"""Find decade files for one index + permutation label."""
	modern = sorted(
		glob.glob(os.path.join(INPUT_DIR, f"era5_{index_name}_{run_label}_*.nc")),
		key=parse_year_start,
	)
	if modern:
		return modern

	if index_name == "fwi":
		legacy_suffix = LEGACY_FWI_SUFFIX[run_label]
		legacy = sorted(
			glob.glob(os.path.join(INPUT_DIR, f"era5_fwi_*_{legacy_suffix}.nc")),
			key=parse_year_start,
		)
		return legacy

	return []


def load_crop_and_merge(files, bbox):
	"""Load decade files, crop to region bbox, realise, then concatenate."""
	cubes = iris.cube.CubeList()
	reference_time_units = None
	seen_max_time = None

	for f in files:
		cube = iris.load_cube(f)
		# Remove duplicate / problematic aux coords that block concatenation
		for coord in list(cube.coords(dim_coords=False)):
			if coord.standard_name == 'time' or coord.long_name in (
				'time', 'day_of_month', 'month', 'month_number',
				'season', 'season_year', 'year',
				'original GRIB coordinate for key: level(surface)',
			):
				cube.remove_coord(coord)
		if cube.coords('realization'):
			cube.remove_coord('realization')
		# Standardise time units across decade chunks
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

		# Trim overlapping time steps from decade boundaries
		if seen_max_time is not None:
			constraint = iris.Constraint(time=lambda cell: cell.point > seen_max_time)
			cube = cube.extract(constraint)
			if cube is None:
				continue

		t = cube.coord('time')
		seen_max_time = t.units.num2date(t.points.max())

		# Crop to region bbox and realise into numpy while data is small
		cube = contrain_coords(cube, bbox)
		_ = cube.data
		cube.attributes = {}
		cubes.append(cube)

	if not cubes:
		return None

	iris.util.equalise_attributes(cubes)
	merged = cubes.concatenate_cube()
	return merged


def monthly_95_series(cube):
	"""Compute monthly 95th percentile: time then space, using cubefuncs."""
	if cube is None:
		return pd.DataFrame(columns=["Date", "Value"])

	t = cube.coord('time')
	dates = t.units.num2date(t.points)
	year_months = sorted({(dt.year, dt.month) for dt in dates})

	rows = []
	for year, month in year_months:
		constraint = iris.Constraint(
			time=lambda cell: cell.point.year == year and cell.point.month == month
		)
		month_cube = cube.extract(constraint)
		if month_cube is None:
			continue

		time_pct = TimePercentile(month_cube, PERCENTILE)
		spatial_pct = CountryPercentile(time_pct, PERCENTILE)
		rows.append({"Date": pd.Timestamp(year=year, month=month, day=1),
		             "Value": float(np.array(spatial_pct.data))})

	out = pd.DataFrame(rows)
	if not out.empty:
		out = out.sort_values("Date").reset_index(drop=True)
	return out


def build_all_series():
	"""Load data, mask to regions, compute monthly 95th for all combos."""
	series_map = {}
	total = len(REGION_CONFIGS) * len(INDEX_ORDER) * len(PERMUTATIONS)
	done = 0

	# Pre-read region bounding boxes once
	shp = gpd.read_file(SHP_FILE)
	region_bounds = {}
	for region, cfg in REGION_CONFIGS.items():
		geom = shp[shp['name'] == cfg['shape_name']].geometry.values[0]
		minx, miny, maxx, maxy = geom.bounds
		region_bounds[region] = (minx, maxx, miny, maxy)

	# Discover files upfront (no loading yet)
	file_map = {}
	for index_name in INDEX_ORDER:
		for run_label, _, _ in PERMUTATIONS:
			file_map[(index_name, run_label)] = find_input_files(index_name, run_label)

	for region, cfg in REGION_CONFIGS.items():
		print(f"\nProcessing region: {region}")
		bbox = region_bounds[region]

		for index_name in INDEX_ORDER:
			for run_label, _, _ in PERMUTATIONS:
				files = file_map[(index_name, run_label)]
				if not files:
					series_map[(region, index_name, run_label)] = pd.DataFrame(columns=["Date", "Value"])
					done += 1
					continue

				# Load, crop to region bbox, realise, merge, then mask
				merged = load_crop_and_merge(files, bbox)
				if merged is None:
					series_map[(region, index_name, run_label)] = pd.DataFrame(columns=["Date", "Value"])
				else:
					# Force numpy — concatenate_cube re-lazifies into dask
					_ = merged.data
					# Reorder to (time, lat, lon) so mask_cube_from_shape can broadcast
					time_dim = merged.coord_dims('time')[0]
					lat_dim = merged.coord_dims('latitude')[0]
					lon_dim = merged.coord_dims('longitude')[0]
					merged.transpose([time_dim, lat_dim, lon_dim])
					masked = apply_shapefile_inclusive(SHP_FILE, cfg['shape_name'], merged)
					series_map[(region, index_name, run_label)] = monthly_95_series(masked)

				done += 1
				if done % 10 == 0 or done == total:
					print(f"  Computed {done}/{total} region/index/permutation series")

	return series_map


def save_plot_and_csv(region, index_name, series_map):
	"""Save one plot + one CSV for a region/index, with all permutations overlaid."""
	os.makedirs(PLOT_DIR, exist_ok=True)
	os.makedirs(EXPORT_DIR, exist_ok=True)

	fig, ax = plt.subplots(figsize=(14, 5))
	merged = None
	plotted = 0
	missing = []

	for run_label, display, color in PERMUTATIONS:
		df = series_map[(region, index_name, run_label)]
		if df.empty:
			missing.append(display)
			continue

		ax.plot(df["Date"], df["Value"], linewidth=1.2, color=color, label=display)
		plotted += 1

		col_name = run_label
		one = df.rename(columns={"Value": col_name})[["Date", col_name]]
		if merged is None:
			merged = one
		else:
			merged = merged.merge(one, on="Date", how="outer")

	idx_label = INDEX_LABELS[index_name]
	ax.set_title(f"XClim {idx_label} Monthly 95th Percentile - {region}")
	ax.set_xlabel("Date")
	ax.set_ylabel(f"{idx_label} 95th percentile")
	ax.grid(True, alpha=0.3)

	if plotted > 0:
		ax.legend(loc="best", fontsize=9)
	else:
		ax.text(
			0.5, 0.5, "No data found for any permutation",
			ha="center", va="center", transform=ax.transAxes, fontsize=10,
		)

	if missing:
		ax.text(
			0.01, 0.01, "Missing: " + ", ".join(missing),
			transform=ax.transAxes, fontsize=8, alpha=0.8,
		)

	plt.tight_layout()

	stem = f"XClim_{idx_label}_{region}_95pct_4permutations"
	plot_path = os.path.join(PLOT_DIR, f"{stem}.png")
	plt.savefig(plot_path, dpi=180, bbox_inches="tight")
	plt.close(fig)

	if merged is None:
		merged = pd.DataFrame(columns=["Date"] + [p[0] for p in PERMUTATIONS])
	else:
		merged = merged.sort_values("Date").reset_index(drop=True)

	csv_path = os.path.join(EXPORT_DIR, f"{stem}.csv")
	merged.to_csv(csv_path, index=False)
	print(f"Saved plot: {plot_path}")
	print(f"Saved csv:  {csv_path}")


def main():
	print("Building monthly 95th percentile timeseries...")
	series_map = build_all_series()

	print("\nSaving 30 plots (6 indices x 5 regions)...")
	for region in REGION_CONFIGS:
		for index_name in INDEX_ORDER:
			save_plot_and_csv(region, index_name, series_map)

	print("Done.")


if __name__ == "__main__":
	main()
