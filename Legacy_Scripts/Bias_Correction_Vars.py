# Plots bias for each variable

#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-


import numpy as np
import iris
import datetime
import matplotlib
#matplotlib.use('Agg')
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.analysis
import iris.plot as iplt
import iris.coord_categorisation
import os
import cartopy.io.shapereader as shpreader
from ascend import shape
import iris.analysis.stats
import scipy.stats
from scipy import stats
import ascend
from ascend import shape
import cf_units
import seaborn as sns



Country = 'Greece'
# 'Chile' (2), 'Canada' (6), 'Greece' (8), 'Bolivia' (11)
Month = 8
percentile = 90

def CountryConstrain(cube):
    shpfilename = str(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))
    natural_earth_file = shape.load_shp(str(shpfilename))
    CountryMask = shape.load_shp(shpfilename, Name=Country)
    Country_shape = CountryMask.unary_union()
    Country1 = Country_shape.mask_cube(cube)
    return Country1 

def CountryMean(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    grid_weights = iris.analysis.cartography.area_weights(cube)
    cube = cube.collapsed(coords, iris.analysis.MEAN, weights = grid_weights)
    return cube 

def ERATimeMean(cube):
    cube = cube.collapsed('valid_time', iris.analysis.MEAN)
    #cube = cube.collapsed('forecast_reference_time', iris.analysis.MEAN) #for RH
    return cube 

def CountryMax(cube):
    coords = ('longitude', 'latitude')
    cube = cube.collapsed(coords, iris.analysis.MAX)
    return cube 

def CountryPercentile(cube, percentile):
    coords = ('longitude', 'latitude')
    cube = cube.collapsed(coords, iris.analysis.PERCENTILE, percent=percentile)
    return cube 

def TimeMean(cube):
    cube = cube.collapsed('time', iris.analysis.MEAN)
    return cube 

def TimeMax(cube):
    cube = cube.collapsed('time', iris.analysis.MAX)
    return cube 

def TimePercentile(cube, percentile):
    cube = cube.collapsed(['time'], iris.analysis.PERCENTILE, percent=percentile)
    return cube 

'''
#Create historical arrays (regional annual mean) for each variable for ERA5
#Vars = ('10u', '10v', 'pr', 'rh', 'tasmax')
#vars = ('10m_u_component_of_wind/daily_mean', '10m_v_component_of_wind/daily_mean', 'total_precipitation/daily_sum', 'relative_humidity/mean', '2m_temperature/daily_maximum')
Vars = ('tasmax','10u')
vars = ('2m_temperature/daily_maximum','2m')
for i, var in enumerate(vars):
    print(var)
    ERA5_Arr = []
    for year in np.arange(1970, 2014):
        ERA5_ImpactsToolBox_Arr = []
        print('ERA5',year)
        ERA5 = iris.load_cube('/data/users/appldata/Data/OBS-ERA5/daily/'+var+'/*'+str(year)+'08.nc')
        ERA5 = ERATimeMean(ERA5)
        ERA5 = CountryConstrain(ERA5)
        ERA5 = CountryMean(ERA5)
        ERA5 = np.ravel(ERA5.data)
        ERA5_Arr.append(ERA5)
    #Save text out to a file
    f = open('/data/scratch/chantelle.burton/SoW2526/Vars/ERA5_'+Vars[i]+'_'+Country+'1970-2013_MEAN.dat','a')
    np.savetxt(f,(ERA5_Arr))
    f.close() 

print('Done ERA5')
exit()
'''
#Create historical arrays (regional annual mean) for each variable for HadGEM3
members = ('aojaa', 'aojab', 'aojac', 'aojad', 'aojae', 'aojaf', 'aojag', 'aojah', 'aojai', 'aojaj','dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')   
for member in members:
    #HadGEM3_Arr = []
    for year in np.arange(1961, 2014):
        print('HadGEM',member,year)
        HadGEM3 = iris.load('/scratch/cburton/scratch/FWI/2023/historical/Nov1960-2013/'+member+'_apa.pp')
        print("cubelist",HadGEM3)
        daterange = iris.Constraint(time=lambda cell: cell.point.year == year)
        HadGEM3 = HadGEM3.extract(daterange)
        print("One Year",HadGEM3)
        HadGEM3 = HadGEM3.extract(iris.Constraint(name="air_temperature"))
        desired_cell_method = iris.coords.CellMethod(method='maximum', coords='time', intervals='1 hour')
        cell_method_constraint = iris.Constraint(cube_func=lambda cube: desired_cell_method in cube.cell_methods)
        HadGEM3 = HadGEM3.extract(cell_method_constraint)[0]
        print("One Var",HadGEM3)
        daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)         
        HadGEM3 = HadGEM3.extract(daterange)
        print("One Month",HadGEM3)
        HadGEM3 = TimePercentile(HadGEM3)
        print("Time Mean",HadGEM3)
        HadGEM3 = CountryConstrain(HadGEM3)
        HadGEM3 = CountryPercentile(HadGEM3)
        print("Country Mean",HadGEM3.data)
        HadGEM3 = np.ravel(HadGEM3.data)
        print(HadGEM3)
        #HadGEM3_Arr.append(HadGEM3.data)

        #Save text out to a file
        f = open('/scratch/cburton/scratch/FWI/2023/BiasCorr/HadGEM_Precip_'+Country+'1960-2013_'+member+'_MEAN_Terminal.dat','a')
        np.savetxt(f,(HadGEM3))
        f.close()  

exit()





#1)  Get the arrays from ERA5 and from HadGEM3 first, and save out to save time and memory
# ERA5 from toolbox
ERA5_ImpactsToolBox_Arr = []
for year in np.arange(1960, 2013):
    print('ERA5',year)
    ERA5_ImpactsToolBox = iris.load_cube('/scratch/cburton/scratch/FWI/2023/ERA5/FWI_ccra3_'+str(year)+'.nc')
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_ImpactsToolBox = ERA5_ImpactsToolBox.extract(daterange)
    ERA5_ImpactsToolBox = TimeMax(ERA5_ImpactsToolBox)#, percentile)
    ERA5_ImpactsToolBox = CountryConstrain(ERA5_ImpactsToolBox)
    ERA5_ImpactsToolBox = CountryMax(ERA5_ImpactsToolBox)#.data, percentile)
    ERA5_ImpactsToolBox = np.ravel(ERA5_ImpactsToolBox.data)
    ERA5_ImpactsToolBox_Arr.append(ERA5_ImpactsToolBox)

#Save text out to a file
f = open('/scratch/cburton/scratch/FWI/2023/ERA5/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_MAX.dat','a')
np.savetxt(f,(ERA5_ImpactsToolBox_Arr))
f.close()    


HadGEM3_Arr = []
members = ('aojaa', 'aojab', 'aojac', 'aojad', 'aojae', 'aojaf', 'aojag', 'aojah', 'aojai', 'aojaj','dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')   
for member in members:
    for year in np.arange(1960, 2014):
        print('HadGEM',member,year)
        HadGEM3 = iris.load_cube('/scratch/cburton/scratch/FWI/2023/historical/1960-2013/FWI_'+member+'.nc')
        print(HadGEM3)
        if HadGEM3 == None:
            pass
        else:
           daterange = iris.Constraint(time=lambda cell: cell.point.year == year)
        if HadGEM3 == None:
            pass
        else:
            HadGEM3 = HadGEM3.extract(daterange)
            daterange = iris.Constraint(time=lambda cell:  cell.point.month == Month)
        if HadGEM3 == None:
            pass
        else:            
            HadGEM3 = HadGEM3.extract(daterange)
        if HadGEM3 == None:
            pass
        else:
            print('Selected times :\n' + str(HadGEM3.coord('time')))
            HadGEM3 = TimeMax(HadGEM3)#, percentile)
            HadGEM3 = CountryConstrain(HadGEM3)
            HadGEM3 = CountryMax(HadGEM3)#, percentile).data
            HadGEM3 = np.ravel(HadGEM3.data)
            HadGEM3_Arr.append(HadGEM3)


            #Save text out to a file
            f = open('/scratch/cburton/scratch/FWI/2023/historical/HadGEM3_Arr_'+Country+'1960-2013_'+member+'_MAX.dat','a')
            np.savetxt(f,(HadGEM3))
            f.close()  

exit()



#2) To speed things up, read from the .dat file rather than calculate the array all over again:
ERA5_ImpactsToolBox_File = ('/scratch/cburton/scratch/FWI/2023/ERA5/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_MAX.dat')
data = []
with open(ERA5_ImpactsToolBox_File, 'r') as f:
    d = f.readlines()
    for i in d:
        data.append([float(i)]) 
ERA5_ImpactsToolBox_Arr = np.array(data, dtype='O')

members = ('aojaa', 'aojab', 'aojac', 'aojad', 'aojae', 'aojaf', 'aojag', 'aojah', 'aojai', 'aojaj', 'dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')
data = []
for member in members:
    HadGEM3_File = ('/scratch/cburton/scratch/FWI/2023/historical/HadGEM3_Arr_'+Country+'1960-2013_'+member+'_MAX.dat')
    with open(HadGEM3_File, 'r') as f:
         d = f.readlines()
         for i in d:
            data.append([float(i)]) 
HadGEM3_Arr = np.array(data, dtype='O')

'''
ERA5_Toolbox = iris.load_cube('/scratch/cburton/scratch/FWI/2023/ERA5/JF_2023_FWI_Toolbox.nc')
ERA5_Toolbox = CountryConstrain(ERA5_Toolbox)
ERA5_Toolbox = CountryMean(ERA5_Toolbox)
ERA5_Toolbox = TimeMean(ERA5_Toolbox)
ERA5_Toolbox = np.array(ERA5_Toolbox.data)
print(ERA5_Toolbox)
'''

import seaborn as sns
sns.distplot(HadGEM3_Arr, hist=True, kde=True, 
             color = 'yellow', fit_kws={"linewidth":2.5,"color":"orange"}, label='HadGEM3')

#sns.distplot(ERA5_ImpactsToolBox_Arr, hist=True, kde=True, 
#             color = 'grey', fit_kws={"linewidth":2.5,"color":"black"}, label='ERA5')

#plt.axvline(x=ERA5_Toolbox, color='black', linewidth=2.5, label='ERA5 JF 2023')


plt.xlabel('FWI')
plt.title(str(Country)+' Max. Fire weather, June 1960-2013')
plt.legend(loc='best')
plt.show()
exit()










#Maps comparing HadGEM3 hist with ERA5 toolbox
folder = '/scratch/cburton/scratch/FWI/2022/'
index_filestem1 = 'hist/'
index_name = 'Canadian_FWI'

members = np.arange(1,106)
#members = np.arange(1,2)
z=0
for member in members:
    print ('hist',member)
    for n in np.arange(1,6):
        try:
            if member < 10:
                hist = iris.load_cube(folder+index_filestem1+'FWI_00'+str(member)+'_'+str(n)+'-2022.nc', index_name)
            elif member > 9 and member < 100:
                hist = iris.load_cube(folder+index_filestem1+'FWI_0'+str(member)+'_'+str(n)+'-2022.nc', index_name)
            else:
                hist = iris.load_cube(folder+index_filestem1+'FWI_'+str(member)+'_'+str(n)+'-2022.nc', index_name)
            hist = CountryConstrain(hist)
            hist = TimeMean(hist)
            if z == 0:
                allhist = hist.copy()
                z=z+1
            if z != 0:
                allhist = allhist+hist
                z=z+1
        except IOError:
             pass 
print(z)  
print(hist)
hist = hist/z


ERA5 = iris.load_cube('/scratch/cburton/scratch/FWI/2022/ERA5/JJ_2022_FWI_Toolbox.nc')
#ERA5=ERA5.extract(iris.Constraint(latitude=lambda cell: (40.0) < cell < (60.0), longitude=lambda cell: (0) < cell < (360)))  
cs_new = iris.coord_systems.GeogCS(6371229.)
ERA5.coord('latitude').coord_system = cs_new
ERA5.coord('longitude').coord_system = cs_new
hist.coord('latitude').coord_system = cs_new
hist.coord('longitude').coord_system = cs_new
ERA5 = ERA5.regrid(hist, iris.analysis.Linear())
ERA5 = CountryConstrain(ERA5)
#ERA5 = ERA5.intersection(longitude=(-180, 180))
ERA5 = TimeMean(ERA5)


plt.subplot(1,3,1)
qplt.pcolormesh(hist, vmin = 0, vmax=20.0)
plt.title('FWI from HadGEM3') 

plt.subplot(1,3,2)
qplt.pcolormesh(ERA5, vmin = 0, vmax=20.0)
plt.title('FWI from ERA5') 

plt.subplot(1,3,3)
diff = ERA5.copy()
diff.data = hist.data - ERA5.data
qplt.pcolormesh(diff)
plt.title('Difference')


plt.show()
exit()













