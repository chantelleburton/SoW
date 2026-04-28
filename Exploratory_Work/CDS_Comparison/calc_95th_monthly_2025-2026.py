import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import os

# Path to the GRIB file containing FWI and all sub-indices for 2025 + Jan 2026
grib_file = '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI+-2025-2026.grib'

# Output directory for results
export_dir = '/data/scratch/bob.potts/sowf/test_output/Exports'
os.makedirs(export_dir, exist_ok=True)

shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'

REGION_CONFIGS = {
    'Iberia':   {'percentile': 95, 'shape_name': 'Northwest Iberia'},
    'Korea':    {'percentile': 95, 'shape_name': 'Southeast South Korea'},
    'Scotland': {'percentile': 95, 'shape_name': 'Scottish Highlands'},
    'Chile':    {'percentile': 95, 'shape_name': 'Chilean Temperate Forests and Matorral'},
    'Canada':   {'percentile': 95, 'shape_name': 'Midwestern Canadian Shield forests'},
}

# Months of interest: Jan-Dec 2025 and Jan 2026
months_of_interest = [(2025, m) for m in range(1, 13)] + [(2026, 1)]


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

    # Step 2: Build rasterised mask
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


# --- Load GRIB file ---
print(f"Opening GRIB file: {grib_file}")
ds = xr.open_dataset(grib_file, engine='cfgrib')

# List all variables (FWI and sub-indices)
variables = list(ds.data_vars)
print(f"Variables found: {variables}")

# --- Process all regions, all variables -> single CSV ---
all_rows = []

for country, cfg in REGION_CONFIGS.items():
    pct = cfg['percentile'] / 100.0

    print(f"\n{'='*60}")
    print(f"Processing: {country}")
    print(f"{'='*60}")

    # Mask to shapefile region
    region_ds, region_mask, region_gdf = apply_shapefile_inclusive(ds, shp_file, cfg['shape_name'])
    print(f"  Region bbox: lat {region_ds.latitude.values[0]:.1f}-{region_ds.latitude.values[-1]:.1f}, "
          f"lon {region_ds.longitude.values[0]:.1f}-{region_ds.longitude.values[-1]:.1f}")

    for year, month in months_of_interest:
        date_key = f'{year}-{month:02d}'
        row = {'Country': country, 'Date': date_key}

        for var in variables:
            subset = region_ds[var].sel(
                time=(region_ds.time.dt.year == year) & (region_ds.time.dt.month == month)
            )

            if subset.size == 0 or np.all(np.isnan(subset.values)):
                row[var] = np.nan
                continue

            spatial_pct = subset.quantile(pct, dim=['latitude', 'longitude'])
            temporal_pct = float(spatial_pct.quantile(pct, dim='time').values)
            row[var] = temporal_pct

        all_rows.append(row)

    print(f"  Done: {country}")

# Save single CSV with all countries and all indices
df = pd.DataFrame(all_rows)
csv_path = os.path.join(export_dir, 'CDS_FWI_all_indices_2025-2026.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print("Done.")
