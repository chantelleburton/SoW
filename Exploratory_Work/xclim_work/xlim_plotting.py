#plotting xarray data
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
# Generalise to read and plot FWI for all years 2020-2024
import glob

fwi_files = sorted(glob.glob('/data/scratch/bob.potts/sowf/test_output/ERA5_FWI/era5_fwi_202403.nc'))
fwi_files = [f for f in fwi_files if any(str(y) in f for y in range(2020, 2025))]

for fwi_path in fwi_files:
	ds = xr.open_dataset(fwi_path)
	# Subset to Korea bounding box (approx 33–39N, 124–131E)
	fwi_korea = ds['fwi'].sel(latitude=slice(39, 33), longitude=slice(124, 131))
	fwi_day = fwi_korea.isel(time=0)
	print(f"Plotting Korea region from {fwi_path}")
	ccss = ccrs.PlateCarree()
	fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccss})
	# Plot FWI on Cartopy axes
	im = fwi_day.plot(ax=ax, cmap='hot', vmin=0, transform=ccrs.PlateCarree(), add_colorbar=True)
	# Add coastlines and features
	ax.coastlines(resolution='10m', color='black', linewidth=1)
	ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
	ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.5)
	ax.set_extent([124, 131, 33, 39], crs=ccrs.PlateCarree())
	plt.title(f"FWI (Korea) on {str(fwi_day['time'].values)[:10]} from {os.path.basename(fwi_path)}")
	plt.tight_layout()
	plt.show()
	ds.close()
