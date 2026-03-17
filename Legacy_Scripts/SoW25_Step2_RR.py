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
# Country = 'Amazon forest northeast of the Amazon and Rio Negro rivers'
# Country = 'Pantanal'
# Country = 'Congo basin'
# Options: 'Greater Pantanal basin plus Chiquitano forests' (6), 'Greece' (8), 'SAM' (9+10)
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

### Functions
#def CountryConstrain(cube, Country):
#    if Country != 'SAM':
#       shpfilename = str(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))
#       natural_earth_file = shape.load_shp(str(shpfilename))
#       CountryMask = shape.load_shp(shpfilename, Name=Country)
#       Country_shape = CountryMask.unary_union()
#       Country1 = Country_shape.mask_cube(cube)
#    elif Country == 'SAM':
       # Region = 2.25 N, -77.75 W, -56 E, -9.75 S
#       Country1=cube.extract(iris.Constraint(latitude=lambda cell: (-9.75) < cell < (2.25), longitude=lambda cell: (282.85) < cell < (304)))#  SAM region
#    return Country1 

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
    #ALL = (np.count_nonzero(Alldata > np.percentile(Alldata,99)))
    #NAT = (np.count_nonzero(Natdata > np.percentile(Natdata,99)))
    ALL = (np.count_nonzero(Alldata > ERA5_2024))
    NAT = (np.count_nonzero(Natdata > ERA5_2024))
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
#Get the ERA5 2023 data for the threshold line
    if Country != 'SAM': #(Already constrained to box before making the data for SAM)
        ERA5_2024.coord("longitude").circular = True
        ERA5_2024 = ERA5_2024.intersection(longitude=(-180, 180))
        
        ERA5_2024 = contrain_to_sow_shapefile(ERA5_2024, 'Focal_Regions-20250402T143811Z-001/Focal_Regions/SoW2425_Focal_MASTER_20250221.shp',Country)
        #ERA5_2023 = CountryConstrain(ERA5_2023, Country)
        if Country == 'Amazon forest northeast of the Amazon and Rio Negro rivers':
            percentile = 89
        elif Country == 'Pantanal':
            percentile = 95
        elif Country == 'Congo basin':
            percentile = 85
            
        ERA5_2024 = CountryPercentile(ERA5_2024, percentile)
        ERA5_2024 = TimePercentile(ERA5_2024, percentile)
        ERA5_2024 = np.array(ERA5_2024.data)
    return ERA5_2024

############## 2) Create 3 subplots #################

plt.subplots(1,3, figsize=(14,4),constrained_layout=True)

Countries = ('Amazon forest northeast of the Amazon and Rio Negro rivers','Pantanal','Congo basin')
n = 1

import csv
with open('SoW25_RR_main_plot_data_4Jul.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Country", "Month", "Percentile", "DateRange", "ALL", "NAT", "RR","ERA5_2024"])  # Header   
    for Country in Countries:
        print(Country)
        month,percentile,daterange,ERA5_2024 = SetCountry(Country)
        ERA5_2024 = GetERA5(ERA5_2024,Country)

        #Read in data files for each historical member (each one has 525 values for the 2023 data), for ALL and NAT
        AllDict = {}
        NatDict = {}
        AllDict[Country] = []  
        NatDict[Country] = []  

        # members = ('aojaa', 'aojab', 'aojac', 'aojad', 'aojae', 'aojaf', 'aojag', 'aojah', 'aojai', 'aojaj','dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')   
        members = ('r1i1p1', 'r1i1p2', 'r1i1p3', 'r1i1p4', 'r1i1p5', 'r1i1p6', 'r1i1p7', 'r1i1p8', 'r1i1p9', 'r1i1p10','r1i1p11', 'r1i1p12', 'r1i1p13', 'r1i1p14', 'r1i1p15')
        for member in members:
            print(member)
            all_file = '/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalExt/Array/FWI_'+Country+'_2022code+BC1'+member+'_hist'+str(percentile)+'%_LogTransform_-1+1_fireSeason.dat'
            nat_file = '/work/scratch-pw2/zhongwei/SoW/HadGEM3/historicalNatExt/Array/FWI'+Country+'_2022code+BC1'+member+'_histnat'+str(percentile)+'%_LogTransform_-1+1_fireSeason.dat'
            with open(all_file) as f:
                 all_lines=f.readlines()
            AllDict[Country] += [float(line.rstrip(',\n')) for line in all_lines]
            with open(nat_file) as f:
                 nat_lines=f.readlines()
            NatDict[Country] += [float(line.rstrip(',\n')) for line in nat_lines]

        #Make sure they are arrays, so we can plot them
        NatDict[Country] = np.array(NatDict[Country])
        AllDict[Country] = np.array(AllDict[Country])
        print(len(NatDict[Country]))

        ### Bootstrap and print the Risk Ratio Results when cycling through each country ###
        RR = draw_bs_replicates(AllDict[Country], NatDict[Country], ERA5_2024, RiskRatio, 10000)
        print(len(RR))
        print(np.percentile(RR, 5))
        print(np.percentile(RR, 50))
        print(np.percentile(RR, 95))

        all_values = AllDict[Country]
        nat_values = NatDict[Country]
        RR_values  = RR
        era5 = ERA5_2024
        for all_val, nat_val, rr_val in zip(all_values, nat_values, RR_values):
            writer.writerow([Country, month, percentile, daterange, all_val, nat_val, rr_val, era5])
            
        #Make the plot
        plt.subplot(1,3,n)
        #sns.distplot(AllDict[Country], hist=True, kde_kws={"clip": (0, None)}, 
                 #color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='factual')
        sns.distplot(AllDict[Country], hist=True, kde_kws={"clip": (0, None)}, 
                 color = '#C7403D', fit_kws={"linewidth":2.5,"color":"#C7403D"}, label='factual')
        #sns.distplot(NatDict[Country], hist=True, kde_kws={"clip": (0, None)},
                 #color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='counterfactual')
        sns.distplot(NatDict[Country], hist=True, kde_kws={"clip": (0, None)},
                 color = '#008787', fit_kws={"linewidth":2.5,"color":"#008787"}, label='counterfactual')
        plt.axvline(x=ERA5_2024, color='black', linewidth=2.5, label='ERA5')
        plt.ylabel(' ')
        if Country == 'Amazon forest northeast of the Amazon and Rio Negro rivers':
            Country = 'Northeast Amazonia'
        elif Country == 'Pantanal':
            Country = 'Pantanal-Chiquitano'
        elif Country == 'Congo basin':
            Country = 'Congo Basin'
        plt.title(Country+' FWI '+month+' 2024')
        if n == 1:
            plt.ylabel('Density')
        if n == 2:
            plt.xlabel('Fire Weather Index')
        if n == 3:
            plt.legend()
        n = n+1
    
plt.savefig('SoW25_RR_main_new_4Jul.pdf')
plt.savefig('SoW25_RR_main_new_4Jul.png')
plt.show()

exit()


