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
shape = shape_file[shape_file['name'] == 'Scottish Highlands'] #hardcoded change later to be based on inpit type.

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
constrained_cube = iris.util.mask_cube_from_shape(cube, geometry)
print(constrained_cube)
