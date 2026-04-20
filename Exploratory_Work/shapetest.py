import iris
import numpy as np
from shapely import intersects
import sys, os
import cartopy.crs as ccrs
import iris.quickplot as qplt
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import contrain_to_sow_shapefile
shp = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
import geopandas as gpd
shapefile = gpd.read_file(shp)
region_names = shapefile['name'].values
print(f"Available regions: {region_names}")
plot_dir = '/data/scratch/bob.potts/sowf/test_output/Plots'
file = '/data/scratch/andrew.hartley/impactstoolbox/Data/era5/Fire-Weather/FWI/FWI_ERA5_global_day_20230101-20230201.nc'
cubes = iris.load(file)
cube = cubes.extract('canadian_fire_weather_index')[0]

# Set coordinate system for iris.util.mask_cube_from_shape
cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

for region_name in region_names:
    print(f"\nProcessing: {region_name}")
    
    # Get geometry for this region
    region_gdf = shapefile[shapefile['name'] == region_name]
    region_geom = region_gdf['geometry'].values[0]
    
    # Method 1: Original custom masking function
    original_cut_cube = contrain_to_sow_shapefile(cube.copy(), shp, region_name)
    
    # Method 2: Iris built-in masking
    new_cut_cube = iris.util.mask_cube_from_shape(cube.copy(), region_geom)
    
    # Get bounds for plotting extent
    minx, miny, maxx, maxy = region_geom.bounds
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot original method
    ax1 = axes[0]
    qplt.contourf(original_cut_cube[0, :, :], cmap='viridis', axes=ax1)
    region_gdf.plot(ax=ax1, edgecolor='red', facecolor='none', linewidth=2, zorder=2, transform=ccrs.PlateCarree())
    ax1.set_xlim(minx - 1, maxx + 1)
    ax1.set_ylim(miny - 1, maxy + 1)
    ax1.set_title(f'{region_name} - Original Method (contrain_to_sow_shapefile)')
    ax1.coastlines()
    
    # Plot new iris method
    ax2 = axes[1]
    qplt.contourf(new_cut_cube[0, :, :], cmap='viridis', axes=ax2)
    region_gdf.plot(ax=ax2, edgecolor='red', facecolor='none', linewidth=2, zorder=2, transform=ccrs.PlateCarree())
    ax2.set_xlim(minx - 1, maxx + 1)
    ax2.set_ylim(miny - 1, maxy + 1)
    ax2.set_title(f'{region_name} - New Method (iris.util.mask_cube_from_shape)')
    ax2.coastlines()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,f'shapetest_{region_name.replace(" ", "_")}.png'), dpi=300)
    #plt.show()
