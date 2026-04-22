import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os

explore_dir = '/data/scratch/bob.potts/sowf/test_output/Exports'
plot_dir = '/data/scratch/bob.potts/sowf/test_output/Plots'
baseline_dir = '/data/scratch/bob.potts/sowf/test_output/Baseline'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'

REGION_CONFIGS = {
    'Iberia': {
        'Month': 8,
        'month_name': 'Aug',
        'event_year': 2025,
        'percentile': 95,
        'shape_name': 'Northwest Iberia',
    },
    'Korea': {
        'Month': 3,
        'month_name': 'March',
        'event_year': 2025,
        'percentile': 95,
        'shape_name': 'Southeast South Korea',
    },
    'Scotland': {
        'Month': (6, 7),
        'month_name': 'June-July',
        'event_year': 2026,
        'percentile': 95,
        'shape_name': 'Scottish Highlands',
    },
    'Chile': {
        'Month': (1, 2),
        'month_name': 'January-February',
        'event_year': 2026,
        'percentile': 95,
        'shape_name': 'Chilean Temperate Forests and Matorral',
    },
    'Canada': {
        'Month': (7, 8),
        'month_name': 'July-August',
        'percentile': 95,
        'event_year': 2025,
        'shape_name': 'Midwestern Canadian Shield forests',
    }
}


def get_months(month_cfg):
    """Return a list of month numbers from config (handles int or tuple)."""
    if isinstance(month_cfg, (list, tuple)):
        return list(month_cfg)
    return [month_cfg]


def apply_shapefile_inclusive(ds, shp_file, shape_name):
    """Mask an xarray dataset to a shapefile region.

    Mirrors the iris-based apply_shapefile_inclusive from cubefuncs.py:
      1. Read shapefile and look up region geometry by name
      2. Crop to bounding box to reduce data volume
      3. Apply rasterised mask (matching iris.util.mask_cube_from_shape behaviour)
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

    # Affine transform: origin at top-left corner of the grid
    if lats[0] > lats[-1]:
        origin_lat = lats[0] + res_lat / 2
        lat_step = -res_lat
    else:
        origin_lat = lats[-1] + res_lat / 2
        lat_step = -res_lat
        lats_for_mask = lats[::-1]

    transform = Affine(res_lon, 0, lons[0] - res_lon / 2,
                       0, lat_step, origin_lat)

    mask = geometry_mask(
        [region_geom],
        out_shape=(len(lats), len(lons)),
        transform=transform,
        all_touched=True,
        invert=True  # True = inside geometry is True
    )

    # If we reversed lats for the transform, flip the mask back
    if lats[0] < lats[-1]:
        mask = mask[::-1]

    # Build an xarray DataArray for the mask on the same grid
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
print(f"Combined time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# --- Subset to 1980-2013 ---
ds = ds.sel(time=slice('1980-01-01', '2013-12-31'))
print(f"After 1980-2013 subset: {ds.time.values[0]} to {ds.time.values[-1]}")

# --- Loop over regions ---
for region_name, cfg in REGION_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Processing: {region_name} ({cfg['month_name']})")
    print(f"{'='*60}")

    months = get_months(cfg['Month'])
    pct = cfg['percentile'] / 100.0

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
    lons = region_mask.longitude.values
    lats = region_mask.latitude.values
    res_lon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.25
    res_lat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25
    for lon in lons:
        ax.axvline(lon - res_lon / 2, color='grey', linewidth=0.3, alpha=0.5)
    ax.axvline(lons[-1] + res_lon / 2, color='grey', linewidth=0.3, alpha=0.5)
    for lat in lats:
        ax.axhline(lat - res_lat / 2, color='grey', linewidth=0.3, alpha=0.5)
    ax.axhline(lats[-1] + res_lat / 2, color='grey', linewidth=0.3, alpha=0.5)
    # Shapefile boundary overlay
    region_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, transform=ccrs.PlateCarree())
    ax.set_extent([minx - 1, maxx + 1, miny - 1, maxy + 1], crs=ccrs.PlateCarree())
    ax.set_title(f'{region_name} — Rasterised mask check ({cfg["shape_name"]})\n'
                 f'Blue = included cells, Red = shapefile boundary')
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'CDS_mask_check_{region_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Mask check plot saved: CDS_mask_check_{region_name}.png")

    # For each year, select target months, compute spatial then temporal percentile
    years_available = np.unique(region_ds.time.dt.year.values)
    cds_results = {}
    for year in years_available:
        subset = region_ds.sel(time=(region_ds.time.dt.year == year) &
                                    (region_ds.time.dt.month.isin(months)))
        fwi = subset['fwinx']

        if fwi.size == 0 or np.all(np.isnan(fwi.values)):
            print(f"  {year}: No data for {cfg['month_name']}, skipping")
            continue

        spatial_pct = fwi.quantile(pct, dim=['latitude', 'longitude'])
        temporal_pct = float(spatial_pct.quantile(pct, dim='time').values)

        cds_results[year] = temporal_pct
        print(f"  {year}: CDS FWI {cfg['percentile']}th = {temporal_pct:.6f}")

    if not cds_results:
        print(f"  No CDS results for {region_name}, skipping comparison.")
        continue

    # --- Load ERA5 comparison CSV ---
    era5_csv_path = os.path.join(baseline_dir, f'ERA5_FWI_1980-2013_{region_name}_{cfg["percentile"]}%.csv')
    if not os.path.exists(era5_csv_path):
        print(f"  ERA5 CSV not found: {era5_csv_path}, skipping comparison.")
        continue

    era5_csv = pd.read_csv(era5_csv_path)
    era5_csv['Year'] = era5_csv['Date'].str[:4].astype(int)

    # --- Build comparison DataFrame ---
    cds_df = pd.DataFrame({
        'Year': list(cds_results.keys()),
        'CDS_FWI_95': list(cds_results.values())
    })
    comparison = pd.merge(cds_df, era5_csv[['Year', 'FWI']], on='Year')
    comparison.rename(columns={'FWI': 'ERA5_FWI_95'}, inplace=True)
    comparison['Difference'] = comparison['CDS_FWI_95'] - comparison['ERA5_FWI_95']
    comparison['Pct_Diff'] = 100 * comparison['Difference'] / comparison['ERA5_FWI_95']

    print(f"\n--- {region_name} Comparison ---")
    print(comparison.to_string(index=False))
    print(f"\nMean absolute difference: {comparison['Difference'].abs().mean():.6f}")
    print(f"Max absolute difference:  {comparison['Difference'].abs().max():.6f}")
    print(f"Mean % difference:        {comparison['Pct_Diff'].mean():.2f}%")
    print(f"Correlation:              {comparison['CDS_FWI_95'].corr(comparison['ERA5_FWI_95']):.6f}")

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(comparison['Year'], comparison['ERA5_FWI_95'], 'o-', label='ImpactTB 95%', color='blue')
    ax1.plot(comparison['Year'], comparison['CDS_FWI_95'], 's--', label='CDS 95%', color='red')
    ax1.set_ylabel(f'FWI {cfg["percentile"]}th Percentile')
    ax1.set_title(f'{region_name} {cfg["month_name"]} FWI {cfg["percentile"]}th Percentile: CDS-FWI vs ERA5 Pipeline (1980-2013)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(comparison['Year'], comparison['Difference'], color='grey', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Difference (CDS - ERA5)')
    ax2.set_title('Difference')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'CDS_vs_ERA5_FWI_comparison_{region_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: CDS_vs_ERA5_FWI_comparison_{region_name}.png")

