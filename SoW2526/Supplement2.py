
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
import pandas as pd
import statsmodels.api as sm
from constrain_cubes_standard import *


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




#########  Subplot (a) - Historical PDF uncorrectd ######### 
ERA5_ImpactsToolBox_File = (folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat')
data = []
with open(ERA5_ImpactsToolBox_File, 'r') as f:
    d = f.readlines()
    for i in d:
        data.append([float(i)]) 
ERA5_ImpactsToolBox_Arr = np.array(data, dtype='O')

data = []
#for member in np.arange(1,16):
for member in np.arange(10,11):
    HadGEM3_File = (folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(member)+'_'+str(percentile)+'%.dat')
    with open(HadGEM3_File, 'r') as f:
         d = f.readlines()
         for i in d:
            data.append([float(i)]) 
HadGEM3_Arr = np.array(data, dtype='O')

ERA5_2025 = ERA5_2025.extract(daterange)
#Get the ERA5 2025 data for the threshold line
ERA5_2025 = contrain_to_sow_shapefile(ERA5_2025, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
ERA5_2025 = CountryPercentile(ERA5_2025, percentile)
ERA5_2025 = TimePercentile(ERA5_2025, percentile)
ERA5_2025 = np.array(ERA5_2025.data)
print(ERA5_2025)


##Make the plot
plt.subplot(2,2,1)
sns.distplot(HadGEM3_Arr, hist=True, kde=True, 
             color = 'yellow', fit_kws={"linewidth":2.5,"color":"orange"}, label='HadGEM3')

sns.distplot(ERA5_ImpactsToolBox_Arr, hist=True, kde=True, 
             color = 'grey', fit_kws={"linewidth":2.5,"color":"black"}, label='ERA5')

plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5 '+month+' 2025')

plt.xlabel(' ')
plt.title('a) '+month+' 1960-2013 (Uncorrected)')
plt.legend(loc='best')


######### Subplot (b) - Historical PDF bias-correctd and transformed ######### 
BiasCorrDict = {}
FWI_SIM = {}
#for member in np.arange(1,16):
for member in np.arange(10,11):
    print(member)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv(folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat')
    df_sim = pd.read_csv(folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(member)+'_'+str(percentile)+'%.dat')

    df_obs[np.isnan(df_obs)] = 0.000000000001
    df_sim[np.isnan(df_sim)] = 0.000000000001 

    ####Log transform the data here#### 
    df_obs = np.log(np.exp(df_obs)-1)
    df_sim = np.log(np.exp(df_sim)-1)

    # Extract years and FWI values
    years = np.arange(1960,2013)
    fwi_sim = df_sim.values
    fwi_sim = fwi_sim[:,0]
    fwi_obs = df_obs.values
    fwi_obs = fwi_obs[:,0]
    #fwi_sim[np.isnan(fwi_sim)] = 0

    print(len(fwi_sim))
    print(len(fwi_obs))
    print(len(years))

    # Step 1a: Fit a linear regression model to obs and sim
    t = years - 2025  # shift years to be relative to 2025 #CB
    X = sm.add_constant(t)  # add a constant term for intercept
    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()

        # Step 1b: Get the coefficients (slope and intercept)
        fwi0, delta = results.params
        return fwi0, delta, np.std(fwi - delta * t) 

    fwi0_sim, delta_sim, std_sim =  find_regression_parameters(fwi_sim)
    fwi0_obs, delta_obs, std_obs =  find_regression_parameters(fwi_obs)

    # Step 2: Detrend the sim and scale to obs
    BiasCorrDict[str(member)] = fwi0_obs + (fwi_sim - delta_sim * t - fwi0_sim) 

    fwi_detrended_sim = BiasCorrDict[str(member)]

fwi_detrended_sim_ENSEMBLE =[BiasCorrDict['10']] #[BiasCorrDict['10'],BiasCorrDict['aojab'],BiasCorrDict['aojac'],BiasCorrDict['aojad'],BiasCorrDict['aojae'],BiasCorrDict['aojaf'],BiasCorrDict['aojag'],BiasCorrDict['aojah'],BiasCorrDict['aojai'],BiasCorrDict['aojaj'],BiasCorrDict['dlrja'],BiasCorrDict['dlrjb'],BiasCorrDict['dlrjc'],BiasCorrDict['dlrjd'],BiasCorrDict['dlrje']]
fwi_detrended_sim_ENSEMBLE = np.ravel(fwi_detrended_sim_ENSEMBLE)

###Inverse of log transform here###
fwi_detrended_sim_ENSEMBLE = np.log(np.exp(fwi_detrended_sim_ENSEMBLE)+1)
fwi_obs = np.log(np.exp(fwi_obs)+1)
fwi_sim = np.log(np.exp(fwi_sim)+1)

##Make the PDF plot
plt.subplot(2,2,2)
sns.distplot(fwi_detrended_sim_ENSEMBLE, kde_kws={"clip": (0, None)}, color = 'yellow',   label='HadGEM3')
sns.distplot(fwi_obs, kde=True, kde_kws={"clip": (0, None)}, color = 'grey',  label='ERA5')
plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5 '+month+' 2025')
plt.xlabel(' ')
plt.title('b) '+month+' 1960-2013 (Corrected)')
plt.legend(loc='best')


######### Subplot (c) - Timeseries of bias correction ######### 
plt.subplot(2,2,3)
plt.plot(years, fwi_obs, label='ERA5', color='blue')
plt.plot(years, fwi_sim, label='HadGEM3', color='red')
plt.plot(years, fwi_detrended_sim, label='Detrended & Shifted FWI', color='purple')
plt.xlabel('Year')
plt.ylabel('FWI')
plt.title('c) Time Series Plot of FWI and Detrended & Shifted FWI')
plt.legend()
plt.grid(True)


######### Subplot (d) - Uncorrected 2025 ######### 
ALL = '/data/scratch/chantelle.burton/SoW2526/output/'+Country+'_UNCORRECTED_hist'+str(percentile)+'%.dat'
NAT = '/data/scratch/chantelle.burton/SoW2526/output/'+Country+'_UNCORRECTED_histnat'+str(percentile)+'%.dat'

All_array = []
Nat_array = []
with open(ALL) as f:
     all_lines=f.readlines()
All_array += [float(line.rstrip(',\n')) for line in all_lines]

with open(NAT) as f:
     nat_lines=f.readlines()
Nat_array += [float(line.rstrip(',\n')) for line in nat_lines]

NatDict = np.array(Nat_array)
AllDict = np.array(All_array)


plt.subplot(2,2,4)
sns.distplot(AllDict, hist=True, kde=True, 
             color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='ALL')
sns.distplot(NatDict, hist=True, kde=True, 
             color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='NAT')
plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5 '+month+' 2025')
plt.xlabel('FWI')
plt.title('d) '+month+' 2025 Uncorrected')
plt.legend()

plt.suptitle(str(Country)+' '+str(percentile)+'th percentile FWI')
plt.show()










