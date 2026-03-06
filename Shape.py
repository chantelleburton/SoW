import iris
from iris.analysis import geometry
import iris.coord_categorisation as icc
import cartopy.io.shapereader as shpreader

import shapely.geometry as sgeom
import shapely.ops as ops
from shapely.geometry import Point
from shapely.vectorized import contains
import numpy as np
import cartopy.crs as ccrs
import geopandas as gp

#import regionmask
from pdb import set_trace
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime



shp = gp.read_file('/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shx')
#print(shp.columns)
#print(shp.head())  
#print(shp.info())
#print(shp.geometry.head())   
#print(shp[["geometry"]].head())  
#shp.plot()
select = shp.iloc[[0]] 
print( select['name'] )
#geom = shp[shp['name'].str.contains(name, case=False, na=False)].geometry.unary_union
#select.plot()  
#plt.show()
