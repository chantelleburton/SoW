# Plot PDFs for 2023 HadGEM3 ALL and NAT, plus ERA5 line
######### NOTE: Need to run Supplement2.pr first to get df_obs and df_sim files ######


#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-


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
import iris.analysis.stats
import scipy.stats
from scipy import stats
import cf_units
import seaborn as sns
from scipy.stats import genextreme as gev, kstest
import pandas as pd
import statsmodels.api as sm
from pdb import set_trace
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="iris")
warnings.filterwarnings("ignore", category=FutureWarning, module="iris")

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

### Functions

def ConstrainToYear(cube):
    year = iris.Constraint(time=lambda cell: cell.point.year == 2024)#because we don't have 2025 data
    cube = cube.extract(year)
    return cube  

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
    ALL = (np.count_nonzero(Alldata > ERA5_2025))
    NAT = (np.count_nonzero(Natdata > ERA5_2025))
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

def GetERA5(ERA5_2025,Country):
#Get the ERA5 2025 data for the threshold line
    ERA5_2025 = contrain_to_sow_shapefile(ERA5_2025, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
    ERA5_2025 = CountryPercentile(ERA5_2025, percentile)
    ERA5_2025 = TimePercentile(ERA5_2025, percentile)
    ERA5_2025 = np.array(ERA5_2025.data)
    return ERA5_2025



############## 1) Create .dat files and save out to save time in plotting #################

folder = '/data/scratch/chantelle.burton/SoW2526/'
index_filestem1 = 'historicalExt'
index_filestem2 = 'historicalNatExt'
index_name = 'canadian_fire_weather_index'


## For each historical member, bias correct 525 members for 2024/2025 and save out
BiasCorrDict = {}
#for member in np.arange(1,16):
for histmember in np.arange(10,11): #TEST
    print(histmember)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv(folder+'/output/ERA5_FWI_1960-2013_'+Country+str(percentile)+'%.dat')#This is the historical ERA5 array made in Historical_FWI.py/ Supplement.py
    df_sim = pd.read_csv(folder+'output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(histmember)+'_'+str(percentile)+'%.dat')#This is the historical HadGEM array made in Historical_FWI.py/ Supplement.py
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
    t = years - 2025  # shift years to be relative to 2025
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
    members = np.arange(1,2)
    histarray = []
    for member in members:
        print ('hist',member)
        for n in np.arange(1,6):
            try:
                if member < 10:
                    hist = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(member)+'i1p'+str(n)+'_'+index_filestem1+'_20230601-20250201_global_day.nc', index_name)
                elif member > 9 and member < 100:
                    hist = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(member)+'i1p'+str(n)+'_'+index_filestem1+'_20230601-20250201_global_day.nc', index_name)
                else:
                    hist = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(member)+'i1p'+str(n)+'_'+index_filestem1+'_20230601-20250201_global_day.nc', index_name)           
                hist = contrain_to_sow_shapefile(hist, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
                hist = ConstrainToYear(hist) 
                hist = CountryPercentile(hist, percentile)
                hist = TimePercentile(hist, percentile)
                iris.save(hist, '/data/scratch/bob.potts/sowf/test_output/'+'Interm_TEST'+Country+'_'+str(member)+'_hist'+str(percentile)+'%_LogTransform.nc')
                hist = np.ravel(hist.data)

                ####Log transform the data here#### 
                hist = np.log(np.exp(hist)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (hist - delta_sim * t - fwi0_sim)
               
                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)
                #/data/scratch/bob.potts/sowf/test_output/Iberia_Uncorrected_hist_EXT95%.nc
                f = open('/data/scratch/bob.potts/sowf/test_output/'+Country+'_'+str(member)+'_hist'+str(percentile)+'%_LogTransform.dat','a')
                np.savetxt(f,(Endhist),newline=',',fmt='%s')
                f.write('\n')
                f.close()
                histarray.append(hist)
            except IOError:
                pass 
     
    histarray = np.array(histarray)
    histarray = np.ravel(histarray)
    print(repr(histarray)) 
    exit()

'''

    ##### Repeat for histnat array (can run this separately in parallel to save time) ####
    histnatarray = []
    members = np.arange(1,106)
    for member in members:
        print ('histnat',member)
        for n in np.arange(1,6):
            try:
                if member < 10:
                    histnat = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r00'+str(member)+'i1p'+str(n)+'_'+index_filestem2+'_20230601-20250201_global_day.nc', index_name)
                elif member > 9 and member < 100:
                    histnat = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r0'+str(member)+'i1p'+str(n)+'_'+index_filestem2+'_20230601-20250201_global_day.nc', index_name)
                else:
                    histnat = iris.load_cube(folder+'Y2526FWI/FWI_HadGEM3-A-N216_r'+str(member)+'i1p'+str(n)+'_'+index_filestem2+'_20230601-20250201_global_day.nc', index_name)           
                histnat = contrain_to_sow_shapefile(histnat, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp', 'Northwest Iberia')
                histnat = ConstrainToYear(histnat)  
                histnat = CountryPercentile(histnat, percentile)
                histnat = TimePercentile(histnat, percentile)
                histnat = np.ravel(histnat.data)

                ####Log transform the data here#### 
                histnat = np.log(np.exp(histnat)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (histnat - delta_sim * t - fwi0_sim)

                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)

                f = open(folder+'output/'+Country+'_'+str(member)+'_histnat'+str(percentile)+'%_LogTransform.dat','a')
                np.savetxt(f,(Endhist),newline=',',fmt='  %s')
                f.write('\n')
                f.close()
                histnatarray.append(histnat)
            except IOError:
                pass 
        
    histnatarray = np.array(histnatarray)
    histnatarray = np.ravel(histnatarray)




############## 2) Create 3 subplots #################
Countries = ('Iberia', 'Iberia', 'Iberia')
#Countries = ('Iberia', 'South Korea', 'Scotland')
n = 1
ERA5_2025 = GetERA5(ERA5_2025,Country)
for Country in Countries:
    print(Country)
    #month,percentile,daterange,ERA5_2025 = Country
    

    #Read in data files for each historical member (each one has 525 values for the 2023 data), for ALL and NAT
    AllDict = {}
    NatDict = {}
    AllDict[Country] = []  
    NatDict[Country] = []  

    members = np.arange(1,106)
    for member in members:
        print(member)                       
        all_file = folder+'output/'+Country+'_'+str(member)+'_hist'+str(percentile)+'%_LogTransform.dat'
        nat_file = folder+'output/'+Country+'_'+str(member)+'_histnat'+str(percentile)+'%_LogTransform.dat'
        with open(all_file) as f:
             all_lines=f.readlines()

        for line in all_lines:
            numbers = line.strip().split(',')   # split by comma
            AllDict[Country] += [float(num) for num in numbers if num]

        with open(nat_file) as f:
            nat_lines = f.readlines()

        for line in nat_lines:
            numbers = line.strip().split(',')   # split by comma
            NatDict[Country] += [float(num) for num in numbers if num]

    #Make sure they are arrays, so we can plot them
    NatDict[Country] = np.array(NatDict[Country])
    AllDict[Country] = np.array(AllDict[Country])
    print(len(NatDict[Country]))

    ### Bootstrap and print the Risk Ratio Results when cycling through each country ###
    RR = draw_bs_replicates(AllDict[Country], NatDict[Country], ERA5_2025, RiskRatio, 10000)
    print(len(RR))
    print(np.percentile(RR, 5))
    print(np.percentile(RR, 95))

    #Make the plot
    plt.subplot(1,3,n)
    sns.distplot(AllDict[Country], hist=True, kde_kws={"clip": (0, None)}, 
             color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='ALL')
    sns.distplot(NatDict[Country], hist=True, kde_kws={"clip": (0, None)},
             color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='NAT')
    plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5')
    plt.ylabel(' ')
    #if Country == 'SAM':
    #    Country = 'Western Amazonia'
    plt.title(Country+' FWI '+month+' 2025')
    if n == 1:
        plt.ylabel('Density')
    if n == 2:
        plt.xlabel('Fire Weather Index')
    if n == 3:
        plt.legend()
    n = n+1

plt.show()

exit()




PRINTED RESULTS


- Transformed & Corrected Canada  
2.8517158325517946
3.59375


- Transformed and corrected Greece 
1.8518518518518519
4.130434782608695

- Transformed & corrected SAM
20.03417354773287
28.542385822359204

'''

