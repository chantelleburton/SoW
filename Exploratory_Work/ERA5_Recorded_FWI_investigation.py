import iris
import numpy as np
from datetime import datetime, timedelta
import iris.coords
import os
import sys
import geopandas
from pyproj import CRS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.constrain_cubes_standard import *

# --- CONFIG ---
shp_file = '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp'
shape_file = geopandas.read_file(shp_file)
shape = shape_file[shape_file['name'] == 'Scottish Highlands']

# Extract the geometry as a shapely object (MultiPolygon)
geometry = shape.geometry.values[0]
print(type(geometry))

f = '/data/scratch/bob.potts/sowf/ERA5_Checks/ERA5_FWI/FWI_2020s.nc'
cube = iris.load_cube(f)
#print(cube.attributes)
crs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

# Assign CRS to latitude and longitude coordinates
cube.coord('latitude').coord_system = crs
cube.coord('longitude').coord_system = crs

#print(cube.coord_system.CRS)

print(cube.coord('latitude').coord_system)
print(cube.coord('longitude').coord_system)

# --- MASKING USING AUXILIARY COORDS ---
from iris._shapefiles import create_shape_mask

# Create mask using auxiliary coordinates
mask = create_shape_mask(
	geometry,
	cube.coord('longitude').points,
	cube.coord('latitude').points,

)
# mask shape: (542080,)

# Broadcast mask to (time, points)
mask_2d = np.broadcast_to(mask, cube.shape)

# Apply mask to cube data (mask out points outside the shape)
cube.data = np.ma.masked_where(~mask_2d, cube.data)
print('Masked cube:', cube)
