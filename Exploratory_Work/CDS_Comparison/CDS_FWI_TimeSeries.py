import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os

plot_dir = '/data/scratch/bob.potts/sowf/test_output/Plots'
export_dir = '/data/scratch/bob.potts/sowf/test_output/Exports'
baseline_dir = '/data/scratch/bob.potts/sowf/test_output/AllMonths_Baseline'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'

Country = os.environ.get("CYLC_TASK_PARAM_country", 'Korea')

REGION_CONFIGS = {
    'Iberia':   {'percentile': 95, 'shape_name': 'Northwest Iberia'},
    'Korea':    {'percentile': 95, 'shape_name': 'Southeast South Korea'},
    'Scotland': {'percentile': 95, 'shape_name': 'Scottish Highlands'},
    'Chile':    {'percentile': 95, 'shape_name': 'Chilean Temperate Forests and Matorral'},
    'Canada':   {'percentile': 95, 'shape_name': 'Midwestern Canadian Shield forests'},
}


def apply_shapefile_inclusive(ds, shp_file, shape_name):
    """Mask an xarray dataset to a shapefile region.
    Returns (masked_ds, mask_da, region_gdf) so the mask can be inspected.
    """
    from rasterio.features import geometry_mask
    from affine import Affine

    shapefile = gpd.read_file(shp_file)
    region_gdf = shapefile[shapefile['name'] == shape_name]
    if region_gdf.empty:
        raise ValueError(f"Shape '{shape_name}' not found in {shp_file}. "
                         f"Available: {shapefile['name'].tolist()}")
    region_geom = region_gdf.geometry.unary_union
    minx, miny, maxx, maxy = region_geom.bounds

    # Convert 0-360 longitudes to -180/180 if needed
    if ds.longitude.values.max() > 180:
        ds = ds.assign_coords(longitude=(ds.longitude.values + 180) % 360 - 180)
        ds = ds.sortby('longitude')

    # Step 1: Crop to bounding box (with small buffer)
    buf = 0.5
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        lat_slice = slice(maxy + buf, miny - buf)
    else:
        lat_slice = slice(miny - buf, maxy + buf)
    ds_bbox = ds.sel(latitude=lat_slice, longitude=slice(minx - buf, maxx + buf))

    # Step 2: Build rasterised mask (same approach as iris.util.mask_cube_from_shape)
    lats = ds_bbox.latitude.values
    lons = ds_bbox.longitude.values
    res_lon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.25
    res_lat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25

    if lats[0] > lats[-1]:
        origin_lat = lats[0] + res_lat / 2
    else:
        origin_lat = lats[-1] + res_lat / 2

    transform = Affine(res_lon, 0, lons[0] - res_lon / 2,
                       0, -res_lat, origin_lat)

    mask = geometry_mask(
        [region_geom],
        out_shape=(len(lats), len(lons)),
        transform=transform,
        all_touched=True,
        invert=True
    )

    if lats[0] < lats[-1]:
        mask = mask[::-1]

    mask_da = xr.DataArray(mask, dims=['latitude', 'longitude'],
                           coords={'latitude': lats, 'longitude': lons})

    return ds_bbox.where(mask), mask_da, region_gdf


# --- Load and concatenate CDS-FWI GRIB files ---
grib_files = [
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1973-1981.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1982-1990.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1991-1999.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-2000-2008.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-2009-2017.grib',
]

print("Loading GRIB files...")
datasets = [xr.open_dataset(f, engine='cfgrib') for f in grib_files]
ds = xr.concat(datasets, dim='time')
print(f"Full time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# --- Subset to 1980-2013 to match ERA5 baseline ---
ds = ds.sel(time=slice('1980-01-01', '2013-12-31'))
print(f"After 1980-2013 subset: {ds.time.values[0]} to {ds.time.values[-1]}")

# --- Process selected region ---
cfg = REGION_CONFIGS[Country]
region_name = Country
pct = cfg['percentile'] / 100.0

print(f"\n{'='*60}")
print(f"Processing: {region_name} (all months)")
print(f"{'='*60}")

# Mask to shapefile region
region_ds, region_mask, region_gdf = apply_shapefile_inclusive(ds, shp_file, cfg['shape_name'])
print(f"  Region bbox: lat {region_ds.latitude.values[0]:.1f}-{region_ds.latitude.values[-1]:.1f}, "
      f"lon {region_ds.longitude.values[0]:.1f}-{region_ds.longitude.values[-1]:.1f}")

# --- Diagnostic plot: verify the rasterised mask ---
minx, miny, maxx, maxy = region_gdf.geometry.unary_union.bounds
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
region_mask.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', alpha=0.6,
                 add_colorbar=False)
# Grid cell edges
lons_m = region_mask.longitude.values
lats_m = region_mask.latitude.values
res_lon = abs(lons_m[1] - lons_m[0]) if len(lons_m) > 1 else 0.25
res_lat = abs(lats_m[1] - lats_m[0]) if len(lats_m) > 1 else 0.25
for lon in lons_m:
    ax.axvline(lon - res_lon / 2, color='grey', linewidth=0.3, alpha=0.5)
ax.axvline(lons_m[-1] + res_lon / 2, color='grey', linewidth=0.3, alpha=0.5)
for lat in lats_m:
    ax.axhline(lat - res_lat / 2, color='grey', linewidth=0.3, alpha=0.5)
ax.axhline(lats_m[-1] + res_lat / 2, color='grey', linewidth=0.3, alpha=0.5)
# Shapefile boundary overlay
region_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, transform=ccrs.PlateCarree())

ax.set_extent([minx - 1, maxx + 1, miny - 1, maxy + 1], crs=ccrs.PlateCarree())
ax.set_title(f'{region_name} — Rasterised mask check ({cfg["shape_name"]})\n'
             f'Blue = included cells, Red = shapefile boundary')
ax.gridlines(draw_labels=True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'CDS_allmonths_mask_check_{region_name}.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Mask check plot saved: CDS_allmonths_mask_check_{region_name}.png")

# --- Compute monthly FWI percentile for every year-month ---
years_available = np.unique(region_ds.time.dt.year.values)
cds_results = {}

for year in years_available:
    for month in range(1, 13):
        subset = region_ds.sel(time=(region_ds.time.dt.year == year) &
                                    (region_ds.time.dt.month == month))
        fwi = subset['fwinx']

        if fwi.size == 0 or np.all(np.isnan(fwi.values)):
            continue

        spatial_pct = fwi.quantile(pct, dim=['latitude', 'longitude'])
        temporal_pct = float(spatial_pct.quantile(pct, dim='time').values)

        date_key = f'{year}-{month:02d}'
        cds_results[date_key] = temporal_pct

if not cds_results:
    raise RuntimeError(f"No results for {region_name}")

print(f"  Computed {len(cds_results)} monthly values")

# Save CDS results
cds_df = pd.DataFrame({
    'Date': list(cds_results.keys()),
    'CDS_FWI_95': list(cds_results.values())
})
csv_path = os.path.join(export_dir, f'CDS_FWI_AllMonths_{region_name}.csv')
cds_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# --- Load ERA5 comparison CSV ---
era5_csv_path = os.path.join(baseline_dir,
                             f'ERA5_FWI_1980-2013_{region_name}_allmonths_{cfg["percentile"]}%.csv')
if not os.path.exists(era5_csv_path):
    print(f"  ERA5 CSV not found: {era5_csv_path}, skipping comparison.")
else:
    era5_csv = pd.read_csv(era5_csv_path)

    # Merge on Date
    comparison = pd.merge(cds_df, era5_csv, on='Date')
    comparison.rename(columns={'FWI': 'ERA5_FWI_95'}, inplace=True)
    comparison['Difference'] = comparison['CDS_FWI_95'] - comparison['ERA5_FWI_95']
    comparison['Pct_Diff'] = 100 * comparison['Difference'] / comparison['ERA5_FWI_95']

    # Parse dates for plotting
    comparison['DateParsed'] = pd.to_datetime(comparison['Date'], format='%Y-%m')

    print(f"\n--- {region_name} All-Months Comparison ---")
    print(f"  Matched months: {len(comparison)}")
    print(f"  Mean absolute difference: {comparison['Difference'].abs().mean():.6f}")
    print(f"  Max absolute difference:  {comparison['Difference'].abs().max():.6f}")
    print(f"  Mean % difference:        {comparison['Pct_Diff'].mean():.2f}%")
    print(f"  Correlation:              {comparison['CDS_FWI_95'].corr(comparison['ERA5_FWI_95']):.6f}")

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(comparison['DateParsed'], comparison['ERA5_FWI_95'], '-', label='ImpactTB 95%',
             color='blue', linewidth=0.8, alpha=0.8)
    ax1.plot(comparison['DateParsed'], comparison['CDS_FWI_95'], '-', label='CDS 95%',
             color='red', linewidth=0.8, alpha=0.8)
    ax1.set_ylabel(f'FWI {cfg["percentile"]}th Percentile')
    ax1.set_title(f'{region_name} Monthly FWI {cfg["percentile"]}th Percentile: '
                  f'CDS-FWI vs ERA5 Pipeline (1980-2013)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(comparison['DateParsed'], comparison['Difference'], color='grey', alpha=0.7, width=20)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Difference (CDS - ERA5)')
    ax2.set_title('Difference')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'CDS_vs_ERA5_FWI_allmonths_{region_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: CDS_vs_ERA5_FWI_allmonths_{region_name}.png")
