"""Plot region-wise overlap-year spinup checks for chunked XClim FWI files.

Compares the overlap year at each boundary between consecutive chunk files
(e.g., 2009-2019 vs 2019-2025, comparing 2019 against 2019).

Outputs one daily and one monthly panel figure per region.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import iris
import iris.analysis.cartography
import iris.coords
import iris.util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from utils.cubefuncs import apply_shapefile_inclusive


DEFAULT_INPUT_FILES = [
	"/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_Mean_RH_Mean_Wind_1979-1989.nc",
	"/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_Mean_RH_Mean_Wind_1989-1999.nc",
	"/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_Mean_RH_Mean_Wind_1999-2009.nc",
	"/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_Mean_RH_Mean_Wind_2009-2019.nc",
	"/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_Mean_RH_Mean_Wind_2019-2025.nc",
]

DEFAULT_SHAPEFILE = (
	"/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/"
	"SoW2526_Focal_MASTER_20260218.shp"
)

DEFAULT_OUTPUT_DIR = "/data/scratch/bob.potts/sowf/test_output/Plots/XClim_Spinup_Review"

REGION_CONFIGS = {
	"Iberia": {"shape_name": "Northwest Iberia"},
	"Korea": {"shape_name": "Southeast South Korea"},
	"Scotland": {"shape_name": "Scottish Highlands"},
	"Chile": {"shape_name": "Chilean Temperate Forests and Matorral"},
	"Canada": {"shape_name": "Midwestern Canadian Shield forests"},
}


def parse_year_range(file_path: str) -> tuple[int, int]:
	match = re.search(r"_(\d{4})-(\d{4})\.nc$", os.path.basename(file_path))
	if not match:
		raise ValueError(f"Could not parse year range from filename: {file_path}")
	return int(match.group(1)), int(match.group(2))


def build_boundary_pairs(file_paths: list[str]) -> list[dict]:
	file_info = sorted(
		[{"path": p, "start": parse_year_range(p)[0], "end": parse_year_range(p)[1]} for p in file_paths],
		key=lambda x: x["start"],
	)
	boundaries = []
	for left, right in zip(file_info[:-1], file_info[1:]):
		if left["end"] != right["start"]:
			raise ValueError(
				f"No shared overlap year between {left['path']} and {right['path']}"
			)
		boundaries.append({
			"prev_path": left["path"],
			"next_path": right["path"],
			"prev_label": f"{left['start']}-{left['end']}",
			"next_label": f"{right['start']}-{right['end']}",
			"overlap_year": left["end"],
		})
	return boundaries


def extract_region_series(file_path, overlap_year, shapefile_path, shape_name, agg):
	"""Load one year from file via Iris, mask with cubefuncs, return daily+monthly pd.Series."""

	# Load directly with Iris — avoids all xarray-to-iris conversion problems.
	cube = iris.load_cube(file_path)

	# These files have a duplicate AuxCoord named 'time' (from valid_time).
	# Remove it so coord('time') is unambiguous.
	for coord in cube.coords('time'):
		if isinstance(coord, iris.coords.AuxCoord):
			cube.remove_coord(coord)
			break

	# Remove scalar dimension coords (surface, realization) that add length-1 dims.
	for coord in list(cube.coords()):
		if coord.shape == (1,) and coord.name() not in ('time', 'latitude', 'longitude'):
			try:
				cube = cube.slices_over(coord).next() if False else iris.util.squeeze(cube)
				break
			except Exception:
				pass
	# Use squeeze to collapse any remaining length-1 dimensions.
	cube = iris.util.squeeze(cube)

	# Constrain to the overlap year.
	time_coord = cube.coord('time')
	dates = time_coord.units.num2date(time_coord.points)
	year_indices = [i for i, d in enumerate(dates) if d.year == overlap_year]
	if not year_indices:
		raise ValueError(f"Year {overlap_year} not found in {file_path}")

	# Subset along time dimension.
	time_dim = cube.coord_dims('time')[0]
	cube = cube[tuple(
		slice(None) if dim != time_dim else year_indices
		for dim in range(cube.ndim)
	)]

	# Realise data into memory — the subset is small (1 year, will be cropped
	# to region bbox) and mask_cube_from_shape needs concrete arrays.
	_ = cube.data

	# mask_cube_from_shape broadcasts a 2D (lat, lon) mask against the cube.
	# NumPy requires extra dims to be leading, so time must come first.
	time_dim = cube.coord_dims('time')[0]
	lat_dim = cube.coord_dims('latitude')[0]
	lon_dim = cube.coord_dims('longitude')[0]
	cube.transpose([time_dim, lat_dim, lon_dim])

	# Apply shapefile mask (bbox crop + inclusive rasterised mask).
	masked_cube = apply_shapefile_inclusive(shapefile_path, shape_name, cube)

	# Spatial reduction.
	coords = ('longitude', 'latitude')
	for coord in coords:
		if not masked_cube.coord(coord).has_bounds():
			masked_cube.coord(coord).guess_bounds()

	if agg == "mean":
		weights = iris.analysis.cartography.area_weights(masked_cube)
		collapsed = masked_cube.collapsed(coords, iris.analysis.MEAN, weights=weights)
	else:
		collapsed = masked_cube.collapsed(coords, iris.analysis.PERCENTILE, percent=95)

	# Extract time axis and values as pandas Series.
	time_coord = collapsed.coord('time')
	out_dates = time_coord.units.num2date(time_coord.points)
	dates_idx = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, d.day) for d in out_dates])
	daily = pd.Series(collapsed.data.flatten().astype(float), index=dates_idx, name="fwi")

	monthly = daily.resample("MS").mean()
	return daily, monthly


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def region_slug(name):
	return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def month_ticks_for_daily():
	starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
	return [int(ts.dayofyear) for ts in starts], [ts.strftime("%b") for ts in starts]


def plot_region_panels(region_name, boundary_results, resolution, output_dir):
	n = len(boundary_results)
	n_cols = 2 if n > 1 else 1
	n_rows = math.ceil(n / n_cols)

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
	axes = np.atleast_1d(axes).ravel()

	day_positions, day_labels = month_ticks_for_daily()
	mon_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
				  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

	handles = None
	for idx, b in enumerate(boundary_results):
		ax = axes[idx]
		prev = b[f"prev_{resolution}"]
		nxt = b[f"next_{resolution}"]
		diff = nxt - prev

		h1 = ax.plot(prev.index, prev.values, color="#1f77b4", lw=1.8, label="Previous chunk")[0]
		h2 = ax.plot(nxt.index, nxt.values, color="#d62728", lw=1.8, label="Next chunk")[0]

		ax2 = ax.twinx()
		h3 = ax2.plot(diff.index, diff.values, color="#2f2f2f", lw=1.2, ls="--", label="Difference")[0]
		ax2.axhline(0, color="#666", lw=0.8, alpha=0.6)

		ax.set_title(f"Overlap {b['overlap_year']}: {b['prev_label']} vs {b['next_label']}", fontsize=10)
		ax.grid(alpha=0.25, ls=":")
		ax.set_ylabel("FWI")
		ax2.set_ylabel("Difference")

		if resolution == "daily":
			ax.set_xlabel("Day of year")
			ax.set_xlim(1, 366)
			ax.set_xticks(day_positions)
			ax.set_xticklabels(day_labels)
		else:
			ax.set_xlabel("Month")
			ax.set_xlim(1, 12)
			ax.set_xticks(range(1, 13))
			ax.set_xticklabels(mon_labels)

		if handles is None:
			handles = [h1, h2, h3]

	for ax in axes[n:]:
		ax.remove()

	fig.suptitle(f"{region_name} — overlap-year spinup check ({resolution})", fontsize=14, y=1.02)
	if handles:
		fig.legend(handles, ["Previous chunk", "Next chunk", "Difference"],
				   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
	fig.tight_layout(rect=[0, 0, 1, 0.95])

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	out_file = Path(output_dir) / f"spinup_{resolution}_{region_slug(region_name)}.png"
	fig.savefig(out_file, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return out_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
	parser = argparse.ArgumentParser(description="Spinup overlap-year diagnostics.")
	parser.add_argument("--input-files", nargs="+", default=DEFAULT_INPUT_FILES)
	parser.add_argument("--shapefile", default=DEFAULT_SHAPEFILE)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--regions", default=",".join(REGION_CONFIGS.keys()))
	parser.add_argument("--agg", choices=["mean", "p95"], default="mean")
	args = parser.parse_args()

	selected = [r.strip() for r in args.regions.split(",") if r.strip()]
	for r in selected:
		if r not in REGION_CONFIGS:
			raise ValueError(f"Unknown region '{r}'. Available: {list(REGION_CONFIGS.keys())}")

	missing = [f for f in args.input_files if not os.path.exists(f)]
	if missing:
		raise FileNotFoundError(f"Missing: {missing}")

	boundary_pairs = build_boundary_pairs(args.input_files)

	print("Boundary pairs:")
	for p in boundary_pairs:
		print(f"  {p['prev_label']} -> {p['next_label']} (overlap {p['overlap_year']})")

	output_dir = Path(args.output_dir)

	for region in selected:
		print(f"\nProcessing region: {region}")
		shape_name = REGION_CONFIGS[region]["shape_name"]
		boundary_results = []

		for pair in boundary_pairs:
			oy = pair["overlap_year"]

			prev_daily, prev_monthly = extract_region_series(
				pair["prev_path"], oy, args.shapefile, shape_name, args.agg)
			next_daily, next_monthly = extract_region_series(
				pair["next_path"], oy, args.shapefile, shape_name, args.agg)

			# Convert to day-of-year / month index for plotting.
			prev_d = pd.Series(prev_daily.values, index=prev_daily.index.dayofyear)
			next_d = pd.Series(next_daily.values, index=next_daily.index.dayofyear)
			prev_m = pd.Series(prev_monthly.values, index=prev_monthly.index.month)
			next_m = pd.Series(next_monthly.values, index=next_monthly.index.month)

			# Align on common indices.
			common_d = prev_d.index.intersection(next_d.index)
			common_m = prev_m.index.intersection(next_m.index)

			boundary_results.append({
				"overlap_year": oy,
				"prev_label": pair["prev_label"],
				"next_label": pair["next_label"],
				"prev_daily": prev_d.loc[common_d],
				"next_daily": next_d.loc[common_d],
				"prev_monthly": prev_m.loc[common_m],
				"next_monthly": next_m.loc[common_m],
			})

			print(f"  {pair['prev_label']} -> {pair['next_label']}: "
				  f"daily={len(common_d)}, monthly={len(common_m)}")

		plot_region_panels(region, boundary_results, "daily", output_dir)
		plot_region_panels(region, boundary_results, "monthly", output_dir)
		print(f"  Figures saved to {output_dir}")

	print("\nDone.")


if __name__ == "__main__":
	main()
