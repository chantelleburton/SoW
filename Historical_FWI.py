# Create .dat files for unbias-corrected historical data, then plot  PDFs

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
import iris.analysis.stats
import scipy.stats
from scipy import stats
import cf_units
import seaborn as sns
from contrain_cubes_standard import *



############# User inputs here #############
Country = 'Iberia'
# Options: 'South Korea' (3), 'Iberia' (8), 'Scotland' (7)
############# User inputs end here #############


folder = '/data/scratch/chantelle.burton/SoW2526/'

#Set up the 2025 files and months automatically
if Country == 'South Korea':
    print('Running South Korea')
    Month = 3
    month = 'March'
    percentile = 95
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-01-01-2025-05-31_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')
      
elif Country == 'Iberia':
    print('Running Iberia')
    Month = 8
    month = 'Aug'
    percentile = 95
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')

elif Country == 'Scotland':
    print('Running Scotland')
    Month = 7
    month = 'July'
    percentile = 95
    daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
    ERA5_2025 = iris.load_cube(folder+'Y2526FWI/FWI_ERA5_std_reanalysis_2025-06-01-2025-10-01_global_day_initialise-from=previous-and-use-numpy=False-and-code-src=copernicus-and-save-input-data=True.nc', 'canadian_fire_weather_index')



## Functions
def CountryConstrain(cube, Country):
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

def TimeMean(cube):
    cube = cube.collapsed('time', iris.analysis.MEAN)
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

#1)  Get the historical FWI arrays from ERA5 and from HadGEM3 first, and save out to save time and memory (only need to do this once)

# ERA5 from toolbox - create historical array of country percentile, month percentile FWI 
ERA5_ImpactsToolBox_Arr = []
for year in np.arange(1960, 2014):
    print('ERA5',year)
    ERA5_ImpactsToolBox = iris.load_cube(folder+'/historicalFWI/ERA5/FWI_era5_era5_era5_'+str(year)+'0'+str(Month)+'01*.nc', 'canadian_fire_weather_index')
    ERA5_ImpactsToolBox = TimePercentile(ERA5_ImpactsToolBox, percentile)
    ERA5_ImpactsToolBox = contrain_to_sow_shapefile(ERA5_ImpactsToolBox, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
    #ERA5_ImpactsToolBox = CountryConstrain(ERA5_ImpactsToolBox, Country)
    ERA5_ImpactsToolBox = CountryPercentile(ERA5_ImpactsToolBox, percentile)
    ERA5_ImpactsToolBox = np.ravel(ERA5_ImpactsToolBox.data)
    ERA5_ImpactsToolBox_Arr.append(ERA5_ImpactsToolBox)
    print(ERA5_ImpactsToolBox)

#Save ERA5 out to a text file
f = open(folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat','a')
np.savetxt(f,(ERA5_ImpactsToolBox_Arr))
f.close()    
print('finished')
exit()

# HadGEM3
#for member in np.arange(1,16):
'''
for member in np.arange(10,11): #TEST
    HadGEM3_Arr = []
    for year in np.arange(1960, 2014):
        print('HadGEM',member,year)
        HadGEM3 = iris.load_cube(folder+'/historicalFWI/HadGEM/FWI_HadGEM3-A-N216_r1i1p'+str(member)+'_historical_gwl'+str(year)+'0'+str(Month)+'01*.nc', 'canadian_fire_weather_index')
        #HadGEM3 = iris.load_cube(folder+'/historicalFWI/HadGEM/FWI_HadGEM3-A-N216_r1i1p10_historical_gwl'+str(year)+'0'+str(Month)+'01*.nc', 'canadian_fire_weather_index')#TEST
        if HadGEM3 == None:
            pass
        else:
           daterange = iris.Constraint(time=lambda cell: cell.point.year == year)
        if HadGEM3 == None:
            pass
        else:
            HadGEM3 = HadGEM3.extract(daterange)
        if HadGEM3 == None:
            pass
        else:            
            HadGEM3 = HadGEM3.extract(daterange)
        if HadGEM3 == None:
            pass
        else:
            print('Selected times :\n' + str(HadGEM3.coord('time')))
            HadGEM3 = TimePercentile(HadGEM3, percentile)
            HadGEM3 = contrain_to_sow_shapefile(HadGEM3, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp','Northwest Iberia')
            #HadGEM3 = CountryConstrain(HadGEM3, Country)
            HadGEM3 = CountryPercentile(HadGEM3, percentile)
            HadGEM3 = np.ravel(HadGEM3.data)
            HadGEM3_Arr.append(HadGEM3)


    #Save HaGEM3 text out to a file
    print(HadGEM3_Arr)
    f = open(folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(member)+'_'+str(percentile)+'%.dat','a')
    np.savetxt(f,(HadGEM3_Arr))
    f.close()  
print('finished')
exit()
'''
print('starting')

#2) To speed things up, read from the .dat file rather than calculate the array all over again:
ERA5_ImpactsToolBox_File = (folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat')
data = []
with open(ERA5_ImpactsToolBox_File, 'r') as f:
    d = f.readlines()
    for i in d:
        data.append([float(i)]) 
    ERA5_ImpactsToolBox_Arr = np.array(data, dtype='O')
print(ERA5_ImpactsToolBox_Arr)
print('done ERA5')

data = []
#for member in np.arange(1,16):
for member in np.arange(10,11):
    HadGEM3_File = (folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(member)+'_'+str(percentile)+'%.dat')
    with open(HadGEM3_File, 'r') as f:
         d = f.readlines()
         for i in d:
            data.append([float(i)]) 
HadGEM3_Arr = np.array(data, dtype='O')

print('done HG')
ERA5_2025 = ERA5_2025.extract(daterange)
#Get the ERA5 2025 data for the threshold line
ERA5_2025 = contrain_to_sow_shapefile(ERA5_2025, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
ERA5_2025 = CountryPercentile(ERA5_2025, percentile)
ERA5_2025 = TimePercentile(ERA5_2025, percentile)
ERA5_2025 = np.array(ERA5_2025.data)
print(ERA5_2025)


##Make the plot
import seaborn as sns
sns.distplot(HadGEM3_Arr, hist=True, kde=True, 
             color = 'yellow', fit_kws={"linewidth":2.5,"color":"orange"}, label='HadGEM3')

sns.distplot(ERA5_ImpactsToolBox_Arr, hist=True, kde=True, 
             color = 'grey', fit_kws={"linewidth":2.5,"color":"black"}, label='ERA5')

plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5 '+month+' 2025')


plt.xlabel('FWI')
plt.title('Northwest Iberia '+str(percentile)+'th percentile Fire weather, '+month+' 1960-2013 (Uncorrected)')
plt.legend(loc='best')
plt.show()
exit()

'''












