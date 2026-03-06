#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-

import sys 
sys.path.append('/home/users/zhongwei/Downloads/Sow_fireAttr/')
from constrain_cubes_standard import *

import numpy as np
import iris
import datetime
import matplotlib
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.analysis
import iris.plot as iplt
import iris.coord_categorisation
import os
import cartopy.io.shapereader as shpreader
#from ascend import shape
import iris.analysis.stats
import scipy.stats
from scipy import stats
# import ascend
# from ascend import shape
import cf_units
import seaborn as sns
import pandas as pd
import statsmodels.api as sm


############# User inputs here #############
Country = 'Amazon forest northeast of the Amazon and Rio Negro rivers'
# Country = 'Pantanal'
#Country = 'Congo basin'
# Options: 'Greater Pantanal basin plus Chiquitano forests' (6), 'Greece' (8), 'SAM' (9+10)
############# User inputs end here #############


#Set up the 2024 files and months automatically
if Country == 'Amazon forest northeast of the Amazon and Rio Negro rivers':
    print('Running Amazon and Rio Negro rivers Jan-Mar')
    daterange = iris.Constraint(time=lambda cell: 1<= cell.point.month <=3)
    month = 'Jan-Mar'
    percentile = 95
    ERA5_2024 = iris.load_cube('/work/scratch-pw2/zhongwei/SoW/ERA5/FWI_ERA5_2024-01-01-2024-03-31_-74_-50_-3_9_day_initialise-from=previous-and-save-input-data=False.nc','canadian_fire_weather_index')#Amazon and Rio Negro rivers
elif Country == 'Pantanal':
    print('Running Pantanal')
    daterange = iris.Constraint(time=lambda cell: 8<= cell.point.month <=9)
    month = 'Aug-Sept'
    percentile = 95
    ERA5_2024 = iris.load_cube('/work/scratch-pw2/zhongwei/SoW/ERA5/FWI_ERA5_2024-08-01-2024-09-31_-64_-53_-22_-12_day_initialise-from=previous-and-save-input-data=False.nc','canadian_fire_weather_index')
elif Country == 'Congo basin':
    print('Running Congo basin')
    daterange = iris.Constraint(time=lambda cell: 6<= cell.point.month <=8)
    month = 'Jun-Aug'
    percentile = 95
    ERA5_2024 = iris.load_cube('/work/scratch-pw2/zhongwei/SoW/ERA5/FWI_ERA5_2024-06-01-2024-08-31_9_31_-6_7_day_initialise-from=previous-and-save-input-data=False.nc','canadian_fire_weather_index')
#elif Country == 'SAM':
#    print('Running SAM')
#    daterange = iris.Constraint(time=lambda cell: 9<= cell.point.month <=10)
#    month = 'Sept-Oct'
#    percentile = 95
#    ERA5_2023 = iris.load_cube('/scratch/cburton/impactstoolbox/Data/era5/Fire-Weather/FWI-2-day/FWI-2-day_ERA5_std_reanalysis_2023-09-01-2023-10-31_-77.75.-9.75.-56.0.2.25_day_initialise-from-copernicus:True-and-use-numpy=False.nc')#SAM



### Functions
def CountryMean(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(cpyflakesoord).guess_bounds()
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


#############  Subplot (a) - Historical PDF uncorrected ######### 
### Read the .dat files  ###
ERA5_ImpactsToolBox_File = ('/work/scratch-pw2/zhongwei/SoW/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat') 
data = []
with open(ERA5_ImpactsToolBox_File, 'r') as f:
    d = f.readlines()
    for i in d:
        data.append([float(i)]) 
ERA5_ImpactsToolBox_Arr = np.array(data, dtype='O')

members = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15') 
data = []
for member in members:
    HadGEM3_File = ('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+member+'_'+str(percentile)+'%.dat')
    with open(HadGEM3_File, 'r') as f:
         d = f.readlines()
         for i in d:
            data.append([float(i)]) 
HadGEM3_Arr = np.array(data, dtype='O')

#Get the ERA5 2024 data for the threshold line
if Country != 'SAM': #(Already constrained to box before making the data for SAM)
    ERA5_2024.coord("longitude").circular = True
    ERA5_2024 = ERA5_2024.intersection(longitude=(-180, 180))

    ERA5_2024 = contrain_to_sow_shapefile(ERA5_2024, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp', Country)
    ERA5_2024 = CountryPercentile(ERA5_2024, percentile)
    ERA5_2024 = TimePercentile(ERA5_2024, percentile)
    ERA5_2024 = np.array(ERA5_2024.data)
    print(ERA5_2024)


###  Make the plot  ###
plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
plt.subplot(2,2,1)
sns.distplot(HadGEM3_Arr, hist=True, kde=True, 
             color = 'yellow', fit_kws={"linewidth":2.5,"color":"orange"}, label='HadGEM3')

sns.distplot(ERA5_ImpactsToolBox_Arr, hist=True, kde=True, 
             color = 'grey', fit_kws={"linewidth":2.5,"color":"black"}, label='ERA5')

plt.axvline(x=ERA5_2024, color='black', linewidth=2.5, label='ERA5 '+month+' 2024')


plt.xlabel(' ')
plt.title('a) '+month+' 1960-2013 (Uncorrected)')
plt.legend(loc='best')



############ Subplot (b) - Historical PDF bias-correctd and transformed ########### 
BiasCorrDict = {}
FWI_SIM = {}
members = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15') 
for member in members:
    print(member)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat')
    df_sim = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+member+'_'+str(percentile)+'%.dat')
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

    print(len(fwi_sim))
    print(len(fwi_obs))
    print(len(years))

    # Step 1a: Fit a linear regression model to obs and sim
    t = years - 2024  # shift years to be relative to 2024
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
    BiasCorrDict[member] = fwi0_obs + (fwi_sim - delta_sim * t - fwi0_sim) 
    fwi_detrended_sim = BiasCorrDict[member]

fwi_detrended_sim_ENSEMBLE = [BiasCorrDict['r1i1p1'],BiasCorrDict['r1i1p2'],BiasCorrDict['r1i1p3'],BiasCorrDict['r1i1p4'],BiasCorrDict['r1i1p5'],BiasCorrDict['r1i1p6'],BiasCorrDict['r1i1p7'],BiasCorrDict['r1i1p8'],BiasCorrDict['r1i1p9'],BiasCorrDict['r1i1p10'],BiasCorrDict['r1i1p11'],BiasCorrDict['r1i1p12'],BiasCorrDict['r1i1p13'],BiasCorrDict['r1i1p14'],BiasCorrDict['r1i1p15']]
fwi_detrended_sim_ENSEMBLE = np.ravel(fwi_detrended_sim_ENSEMBLE)

###Inverse of log transform here###
fwi_detrended_sim_ENSEMBLE = np.log(np.exp(fwi_detrended_sim_ENSEMBLE)+1)
fwi_obs = np.log(np.exp(fwi_obs)+1)
fwi_sim = np.log(np.exp(fwi_sim)+1)

##Make the PDF plot
plt.subplot(2,2,2)
sns.distplot(fwi_detrended_sim_ENSEMBLE, kde_kws={"clip": (0, None)}, color = 'yellow',   label='HadGEM3')
sns.distplot(fwi_obs, kde=True, kde_kws={"clip": (0, None)}, color = 'grey',  label='ERA5')
plt.axvline(x=ERA5_2024, color='black', linewidth=2.5, label='ERA5 '+month+' 2024')
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

######### Subplot (d) - Unbiased NAT & ALL ######### 
### Then read in the .dat files and make the plot ###
ALL = '/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalExt/Array/FWI'+Country+'_UNCORRECTED_hist'+str(percentile)+'%.dat'
NAT = '/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalNatExt/Array/FWI'+Country+'_UNCORRECTED_histnat'+str(percentile)+'%.dat'

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
plt.axvline(x=ERA5_2024, color='black', linewidth=2.5, label='ERA5 '+month+' 2024')
plt.xlabel('FWI')
plt.title('d) '+month+' 2024 Uncorrected')
plt.legend()

plt.suptitle(str(Country)+' '+str(percentile)+'th percentile FWI')
#plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
plt.savefig(str(Country)+'_S2.pdf')
plt.savefig(str(Country)+'_S2.png')
plt.show()