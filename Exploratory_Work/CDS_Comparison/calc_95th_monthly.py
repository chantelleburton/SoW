import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import os

baseline_dir = '/data/scratch/bob.potts/sowf/test_output/AllMonths_Baseline'
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'

Dataset = os.environ.get("CYLC_TASK_PARAM_dataset", 'CDS')
Country = os.environ.get("CYLC_TASK_PARAM_country", 'Korea')

REGION_CONFIGS = {
    'Iberia':   {'percentile': 95, 'shape_name': 'Northwest Iberia'},
    'Korea':    {'percentile': 95, 'shape_name': 'Southeast South Korea'},
    'Scotland': {'percentile': 95, 'shape_name': 'Scottish Highlands'},
    'Chile':    {'percentile': 95, 'shape_name': 'Chilean Temperate Forests and Matorral'},
    'Canada':   {'percentile': 95, 'shape_name': 'Midwestern Canadian Shield forests'},
}

CDS_GRIB_FILES = [
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1973-1981.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1982-1990.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-1991-1999.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-2000-2008.grib',
    '/data/scratch/bob.potts/sowf/CDS-FWI/CDS-FWI-2009-2017.grib',
]

XCLIM_NC_FILES = [
    '/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_1979-1989.nc',
    '/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_1989-1999.nc',
    '/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_1999-2009.nc',
    '/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_2009-2019.nc',
    '/data/scratch/bob.potts/sowf/test_output/XClim_FWI/era5_fwi_2019-2025.nc',
]


def get_region_bbox(shp_file, shape_name):
    """Read shapefile, return (region_geom, region_gdf, bounds)."""
    shapefile = gpd.read_file(shp_file)
    region_gdf = shapefile[shapefile['name'] == shape_name]
    if region_gdf.empty:
        raise ValueError(f"Shape '{shape_name}' not found in {shp_file}. "
                         f"Available: {shapefile['name'].tolist()}")
    region_geom = region_gdf.geometry.union_all()
    return region_geom, region_gdf, region_geom.bounds


def crop_to_bbox(ds, bounds, buf=0.5):
    """Crop an xarray Dataset to a bounding box (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds

    # Convert 0-360 longitudes to -180/180 if needed
    if ds.longitude.values.max() > 180:
        ds = ds.assign_coords(longitude=(ds.longitude.values + 180) % 360 - 180)
        ds = ds.sortby('longitude')

    if ds.latitude.values[0] > ds.latitude.values[-1]:
        lat_slice = slice(maxy + buf, miny - buf)
    else:
        lat_slice = slice(miny - buf, maxy + buf)

    return ds.sel(latitude=lat_slice, longitude=slice(minx - buf, maxx + buf))


def apply_shapefile_mask(ds, region_geom):
    """Apply a rasterised shapefile mask to an already-cropped dataset."""
    from rasterio.features import geometry_mask
    from affine import Affine

    lats = ds.latitude.values
    lons = ds.longitude.values
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
        invert=True,
    )

    if lats[0] < lats[-1]:
        mask = mask[::-1]

    mask_da = xr.DataArray(mask, dims=['latitude', 'longitude'],
                           coords={'latitude': lats, 'longitude': lons})
    return ds.where(mask_da)


def load_cds(bounds):
    """Load CDS GRIB files, cropping each to bbox before concatenation."""
    parts = []
    for fpath in CDS_GRIB_FILES:
        ds = xr.open_dataset(fpath, engine='cfgrib')
        ds = crop_to_bbox(ds, bounds)
        parts.append(ds)
        print(f"  {os.path.basename(fpath)}: {int(ds.time.dt.year.values[0])}-"
              f"{int(ds.time.dt.year.values[-1])}")
    return xr.concat(parts, dim='time')


def load_xclim(bounds):
    """Load xclim decade files, cropping each to bbox before concatenation.
    Skips first year of non-first files (spin-up artefacts) and deduplicates times.
    """
    parts = []
    for i, fpath in enumerate(XCLIM_NC_FILES):
        ds = xr.open_dataset(fpath)
        ds = crop_to_bbox(ds, bounds)
        file_start_year = int(ds.time.dt.year.values[0])
        if i > 0:
            ds = ds.sel(time=ds.time.dt.year > file_start_year)
            print(f"  {os.path.basename(fpath)}: skipped {file_start_year}, "
                  f"using {int(ds.time.dt.year.values[0])}-{int(ds.time.dt.year.values[-1])}")
        else:
            print(f"  {os.path.basename(fpath)}: using {file_start_year}-"
                  f"{int(ds.time.dt.year.values[-1])} (first file, no skip)")
        parts.append(ds)

    merged = xr.concat(parts, dim='time')
    _, unique_idx = np.unique(merged.time.values, return_index=True)
    merged = merged.isel(time=np.sort(unique_idx))
    return merged


def compute_monthly_95th(ds, var_name, percentile):
    """Compute monthly 95th percentile: temporal-then-spatial (matches ERA5 baseline).

    For each year-month:
      1) 95th percentile over time (per grid cell) within that month
      2) 95th percentile over lat/lon → single scalar
    """
    pct = percentile / 100.0
    results = {}

    years = np.unique(ds.time.dt.year.values)
    for year in years:
        for month in range(1, 13):
            sel = ds.sel(time=(ds.time.dt.year == year) &
                              (ds.time.dt.month == month))
            fwi = sel[var_name]
            if fwi.size == 0 or np.all(np.isnan(fwi.values)):
                continue

            # Step 1: temporal percentile per grid cell
            temporal_pct = fwi.quantile(pct, dim='time')
            # Step 2: spatial percentile → scalar
            spatial_pct = float(temporal_pct.quantile(pct, dim=['latitude', 'longitude']).values)

            results[f'{year}-{month:02d}'] = spatial_pct

    return results


# ---- Main ----
cfg = REGION_CONFIGS[Country]
percentile = cfg['percentile']
shape_name = cfg['shape_name']

print(f"Dataset: {Dataset}, Country: {Country} ({shape_name})")

# 1. Get region bounding box from shapefile (before loading any data)
region_geom, region_gdf, bounds = get_region_bbox(shp_file, shape_name)
print(f"Region bbox: lon [{bounds[0]:.2f}, {bounds[2]:.2f}], lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")

# 2. Load data (cropped to bbox on load)
if Dataset == 'CDS':
    print("Loading CDS GRIB files (bbox-cropped)...")
    ds = load_cds(bounds)
    var_name = 'fwinx'
    prefix = 'CDS'
elif Dataset == 'XClim':
    print("Loading xclim decade files (bbox-cropped)...")
    ds = load_xclim(bounds)
    var_name = 'fwi'
    prefix = 'XClim'
else:
    raise ValueError(f"Unknown dataset: {Dataset}. Expected 'CDS' or 'XClim'.")

print(f"Loaded time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Grid size: {ds.dims}")

# 3. Apply shapefile mask
print("Applying shapefile mask...")
ds = apply_shapefile_mask(ds, region_geom)

# 4. Compute monthly 95th percentile (temporal-then-spatial)
print(f"Computing monthly {percentile}th percentile (temporal-then-spatial)...")
results = compute_monthly_95th(ds, var_name, percentile)

if not results:
    raise RuntimeError(f"No results computed for {Dataset} / {Country}")

print(f"  Computed {len(results)} monthly values")

# 5. Export CSV
output_file = os.path.join(baseline_dir, f'{prefix}_FWI_AllMonths_{Country}.csv')
df = pd.DataFrame({'Date': list(results.keys()), 'FWI': list(results.values())})
df.to_csv(output_file, index=False)
print(f"Saved: {output_file}")
print("Done.")
