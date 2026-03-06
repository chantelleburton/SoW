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
# from ascend import shape
import iris.analysis.stats
import scipy.stats
from scipy import stats
# import ascend
# from ascend import shape
import cf_units
import seaborn as sns
from scipy.stats import genextreme as gev, kstest
import pandas as pd
import statsmodels.api as sm
from pdb import set_trace

############# User inputs here #############
Country = 'Amazon forest northeast of the Amazon and Rio Negro rivers'
# Country = 'Pantanal'
# Country = 'Congo basin'
############# User inputs end here #############

def SetCountry(Country):
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
    return month,percentile,daterange,ERA5_2024


def CountryMean(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    grid_weights = iris.analysis.cartography.area_weights(cube)
    cube = cube.collapsed(coords, iris.analysis.MEAN, weights = grid_weights)
    return cube 

def CountryMax(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    grid_weights = iris.analysis.cartography.area_weights(cube)
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

def RiskRatio(Alldata,Natdata, Threshold):
    ALL = (np.count_nonzero(Alldata > ERA5_2023))
    NAT = (np.count_nonzero(Natdata > ERA5_2023))
    RR = ALL/NAT
    return RR

def draw_bs_replicates(ALL, NAT, ERA5, func, size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    RR_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        ALL_sample = np.random.choice(ALL,size=(int(np.round(len(ALL)-(0.1*len(ALL))))), replace=False)
        ALL_sample = np.random.choice(ALL_sample,size=len(ALL), replace=True)
        NAT_sample = np.random.choice(NAT,size=(int(np.round(len(NAT)-(0.1*len(NAT))))), replace=False)
        NAT_sample = np.random.choice(NAT_sample,size=len(ALL), replace=True)
        # Get bootstrap replicate and append to bs_replicates
        RR_replicates[i] = func(ALL_sample, NAT_sample, ERA5)  
    return RR_replicates

def GetERA5(ERA5_2024,Country):
#Get the ERA5 2024 data for the threshold line
    if Country != 'SAM': #(Already constrained to box before making the data for SAM)
        ERA5_2024.coord("longitude").circular = True
        ERA5_2024 = ERA5_2024.intersection(longitude=(-180, 180))
        
        ERA5_2024 = contrain_to_sow_shapefile(ERA5_2024, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp',Country)
        ERA5_2024 = CountryPercentile(ERA5_ERA5_20242023, percentile)
        ERA5_2024 = TimePercentile(ERA5_2024, percentile)
        ERA5_2024 = np.array(ERA5_2024.data)
    return ERA5_2024



############## Create .dat files and save out to save time in plotting #################

folder = '/work/scratch-pw2/zhongwei/SoW/HadGEM3/'
index_filestem1 = 'historicalExt/'
index_filestem2 = 'historicalNatExt/'
index_name = 'canadian_fire_weather_index'

### 1) Make the .dat files for ERA5 ###
ERA5_ImpactsToolBox_Arr = []
for year in np.arange(1960, 2014):
    print('ERA5',year)
    ERA5_ImpactsToolBox = iris.load_cube('/work/scratch-pw2/zhongwei/SoW/ERA5/merged/FWI_era5_era5_era5_mergetime_'+str(year)+'.nc','canadian_fire_weather_index')
    #FWI_era5_era5_era5_'+str(year)+'_global_day_initialise-from=previous-and-save-input-data=False.nc
    ERA5_ImpactsToolBox = ERA5_ImpactsToolBox.extract(daterange)
    ERA5_ImpactsToolBox = TimePercentile(ERA5_ImpactsToolBox, percentile)

    ERA5_ImpactsToolBox.coord("longitude").circular = True
    ERA5_ImpactsToolBox = ERA5_ImpactsToolBox.intersection(longitude=(-180, 180))
    
    ERA5_ImpactsToolBox = contrain_to_sow_shapefile(ERA5_ImpactsToolBox, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp',Country)
    ERA5_ImpactsToolBox = CountryPercentile(ERA5_ImpactsToolBox, percentile)
    ERA5_ImpactsToolBox = np.ravel(ERA5_ImpactsToolBox.data)
    ERA5_ImpactsToolBox_Arr.append(ERA5_ImpactsToolBox)
    print(ERA5_ImpactsToolBox)

#Save ERA5 text out to a file
f = open('/work/scratch-pw2/zhongwei/SoW/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat','a')
np.savetxt(f,(ERA5_ImpactsToolBox_Arr))
f.close() 



### 2) Make the .dat files for HadGEM3 ###
# HadGEM3
members = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15') 

for member in members:
    HadGEM3_Arr = []
    for year in np.arange(1960, 2014):
        print('HadGEM',member,year)
        HadGEM3 = iris.load_cube('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/1960-2013/FWI_HadGEM3-A-N216_'+member+'_historical_all.nc','canadian_fire_weather_index')
        print(HadGEM3,year)
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
            
            HadGEM3.coord("longitude").circular = True
            HadGEM3 = HadGEM3.intersection(longitude=(-180, 180))
            
            HadGEM3 = contrain_to_sow_shapefile(HadGEM3, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp',Country)
            HadGEM3 = CountryPercentile(HadGEM3, percentile)
            HadGEM3 = np.ravel(HadGEM3.data)
            HadGEM3_Arr.append(HadGEM3)


    #Save HadGEM3 text out to a file
    f = open('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+member+'_'+str(percentile)+'%.dat','a')
    np.savetxt(f,(HadGEM3_Arr))
    f.close()  

exit()


## 3) For each historical member, bias correct 525 members for 2024 and save out
BiasCorrDict = {}
histmembers = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15') 
percentile = 95
for histmember in histmembers:
    print(histmember)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat')
    df_sim = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+histmember+'_'+str(percentile)+'%.dat')
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

    
    #### First do for hist array  ####
    members = np.arange(1,106)
    histarray = []
    for member in members:
        print ('hist',member)
        for n in np.arange(1,6):
            try:
                if member < 10:
                    hist = iris.load_cube(folder+index_filestem1+'/fireSeason/FWI_HadGEM3-A-N216_r00'+str(member)+'i1p'+str(n)+'_historicalExt_20230601-20250201_global_day_fireSeason.nc', index_name)
                elif member > 9 and member < 100:
                    hist = iris.load_cube(folder+index_filestem1+'/fireSeason/FWI_HadGEM3-A-N216_r0'+str(member)+'i1p'+str(n)+'_historicalExt_20230601-20250201_global_day_fireSeason.nc', index_name)
                else:
                    hist = iris.load_cube(folder+index_filestem1+'/fireSeason/FWI_HadGEM3-A-N216_r'+str(member)+'i1p'+str(n)+'_historicalExt_20230601-20250201_global_day_fireSeason.nc', index_name)
                print(hist)
                hist.coord("longitude").circular = True
                hist = hist.intersection(longitude=(-180, 180))
                
                hist = contrain_to_sow_shapefile(hist, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp', Country)
                #hist = CountryConstrain(hist, Country)
                hist = CountryPercentile(hist, percentile)
                hist = TimePercentile(hist, percentile)
                hist = np.ravel(hist.data)

                ####Log transform the data here#### 
                hist = np.log(np.exp(hist)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (hist - delta_sim * 0 - fwi0_sim)
                print(Endhist)

                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)
 
                f = open('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalExt/Array/FWI_'+Country+'_2022code+BC1'+histmember+'_hist'+str(percentile)+'%_LogTransform_-1+1_fireSeason.dat','a')
                np.savetxt(f,(Endhist),newline=',',fmt='%s')
                f.write('\n')
                f.close()
                histarray.append(hist)
            except IOError:
                 pass 
     
    histarray = np.array(histarray)
    histarray = np.ravel(histarray)
    print(repr(histarray)) 
#exit()

exit()



## 4) For each historicalNat member, bias correct 525 members for 2023 and save out
BiasCorrDict = {}
histmembers = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15') 
percentile = 95
for histmember in histmembers:
    print(histmember)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat')
    df_sim = pd.read_csv('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+histmember+'_'+str(percentile)+'%.dat')
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

    # Step 1a: Fit a linear regression model to obs and sim
    t = years - 2024  # shift years to be relative to 2023
    X = sm.add_constant(t)  # add a constant term for intercept
    def find_regression_parameters(fwi):
        model = sm.OLS(fwi, X)
        results = model.fit()

        # Step 1b: Get the coefficients (slope and intercept)
        fwi0, delta = results.params
    
        return fwi0, delta, np.std(fwi - delta * t) 

    fwi0_sim, delta_sim, std_sim =  find_regression_parameters(fwi_sim)
    fwi0_obs, delta_obs, std_obs =  find_regression_parameters(fwi_obs)   
              
   
    ##### Repeat for histnat array (can run this separately in paralell to save time) ####
    histnatarray = []
    members = np.arange(1,106)
    for member in members:
        print ('histnat',member)
        for n in np.arange(1,6):
            try:
                if member < 10:
                    histnat = iris.load_cube(folder+index_filestem2+'/fireSeason/FWI_HadGEM3-A-N216_r00'+str(member)+'i1p'+str(n)+'_historicalNatExt_20230601-20250201_global_day_fireSeason.nc', index_name)
                elif member > 9 and member < 100:
                    histnat = iris.load_cube(folder+index_filestem2+'/fireSeason/FWI_HadGEM3-A-N216_r0'+str(member)+'i1p'+str(n)+'_historicalNatExt_20230601-20250201_global_day_fireSeason.nc', index_name)
                else:
                    histnat = iris.load_cube(folder+index_filestem2+'/fireSeason/FWI_HadGEM3-A-N216_r'+str(member)+'i1p'+str(n)+'_historicalNatExt_20230601-20250201_global_day_fireSeason.nc', index_name)    

                histnat.coord("longitude").circular = True
                histnat = histnat.intersection(longitude=(-180, 180))
                
                histnat = contrain_to_sow_shapefile(histnat, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp', Country)
                
                #histnat = CountryConstrain(histnat, Country)
                histnat = CountryPercentile(histnat, percentile)
                histnat = TimePercentile(histnat, percentile)
                histnat = np.ravel(histnat.data)

                ####Log transform the data here#### 
                histnat = np.log(np.exp(histnat)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (histnat - delta_sim * 0 - fwi0_sim)

                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)

                f = open('/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalNatExt/Array/FWI_'+Country+'_2022code+BC1'+histmember+'_histnat'+str(percentile)+'%_LogTransform_-1+1_fireSeason.dat','a')
                np.savetxt(f,(Endhist),newline=',',fmt='  %s')
                f.write('\n')
                f.close()
                histnatarray.append(histnat)
            except IOError:
                pass 
        
    histnatarray = np.array(histnatarray)
    histnatarray = np.ravel(histnatarray)

exit()