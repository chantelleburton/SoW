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
from ascend import shape
import iris.analysis.stats
import scipy.stats
from scipy import stats
import ascend
from ascend import shape
import cf_units
import seaborn as sns
from scipy.stats import genextreme as gev, kstest
import pandas as pd
import statsmodels.api as sm
from pdb import set_trace



############# User inputs here #############
Country = 'Canada'
# Options: 'Canada' (6), 'Greece' (8), 'SAM' (9+10)
############# User inputs end here #############

def SetCountry(Country):
#Set up the 2023 files and months automatically
    if Country == 'Canada':
        print('Running Canada')
        Month = 6
        month = 'June'
        percentile = 95
        daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
        ERA5_2023 = iris.load_cube('/scratch/cburton/impactstoolbox/Data/era5/Fire-Weather/FWI-2-day/FWI-2-day_ERA5_std_reanalysis_2023-06-01-2023-06-30_-144.0.47.0.-42.0.85.0_day_initialise-from-copernicus:True-and-use-numpy=False.nc')#Canada
    elif Country == 'Greece':
        print('Running Greece')
        Month = 8
        month = 'Aug'
        percentile = 90
        daterange = iris.Constraint(time=lambda cell: cell.point.month == Month)
        ERA5_2023 = iris.load_cube('/scratch/cburton/impactstoolbox/Data/era5/Fire-Weather/FWI-2-day/FWI-2-day_ERA5_std_reanalysis_2023-08-01-2023-08-31_4.0.33.0.43.0.47.0_day_initialise-from-copernicus:True-and-use-numpy=False.nc')#Greece
    elif Country == 'SAM':
        print('Running SAM')
        daterange = iris.Constraint(time=lambda cell: 9<= cell.point.month <=10)
        month = 'Sept-Oct'
        percentile = 95
        ERA5_2023 = iris.load_cube('/scratch/cburton/impactstoolbox/Data/era5/Fire-Weather/FWI-2-day/FWI-2-day_ERA5_std_reanalysis_2023-09-01-2023-10-31_-77.75.-9.75.-56.0.2.25_day_initialise-from-copernicus:True-and-use-numpy=False.nc')#SAM
    return month,percentile,daterange,ERA5_2023

### Functions
def CountryConstrain(cube, Country):
    if Country != 'SAM':
       shpfilename = str(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))
       natural_earth_file = shape.load_shp(str(shpfilename))
       CountryMask = shape.load_shp(shpfilename, Name=Country)
       Country_shape = CountryMask.unary_union()
       Country1 = Country_shape.mask_cube(cube)
    elif Country == 'SAM':
       # Region = 2.25 N, -77.75 W, -56 E, -9.75 S
       Country1=cube.extract(iris.Constraint(latitude=lambda cell: (-9.75) < cell < (2.25), longitude=lambda cell: (282.85) < cell < (304)))#  SAM region
    return Country1 

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

def GetERA5(ERA5_2023,Country):
#Get the ERA5 2023 data for the threshold line
    if Country != 'SAM': #(Already constrained to box before making the data for SAM)
        ERA5_2023 = CountryConstrain(ERA5_2023, Country)
    ERA5_2023 = CountryPercentile(ERA5_2023, percentile)
    ERA5_2023 = TimePercentile(ERA5_2023, percentile)
    ERA5_2023 = np.array(ERA5_2023.data)
    return ERA5_2023




############## 1) Create .dat files and save out to save time in plotting #################

folder = '/scratch/cburton/scratch/FWI/2023/'
index_filestem1 = 'hist/'
index_filestem2 = 'histnat/'
index_name = 'Canadian_FWI'


## For each historical member, bias correct 525 members for 2023 and save out
BiasCorrDict = {}
histmembers = ('aojaa','aojab', 'aojac', 'aojad', 'aojae', 'aojaf','aojag', 'aojah', 'aojai', 'aojaj','dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')   #
for histmember in histmembers:
    print(histmember)
    # Step 0; Load fwi data from CSV using pandas
    df_obs = pd.read_csv('/scratch/cburton/scratch/FWI/2023/ERA5/Array/ERA5_ImpactsToolBox_Arr_'+Country+'1960-2013_'+str(percentile)+'%.dat')
    df_sim = pd.read_csv('/scratch/cburton/scratch/FWI/2023/historical/Array/HadGEM3_Arr_'+Country+'Jan1960-2013_'+histmember+'_'+str(percentile)+'%.dat')
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
    t = years - 2023  # shift years to be relative to 2023
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
                    hist = iris.load_cube(folder+index_filestem1+Month+'/FWI_00'+str(member)+'_'+str(n)+'.nc', index_name)
                elif member > 9 and member < 100:
                    hist = iris.load_cube(folder+index_filestem1+Month+'/FWI_0'+str(member)+'_'+str(n)+'.nc', index_name)
                else:
                    hist = iris.load_cube(folder+index_filestem1+Month+'/FWI_'+str(member)+'_'+str(n)+'.nc', index_name)
                print(hist)
                hist = CountryConstrain(hist, Country)
                hist = CountryPercentile(hist, percentile)
                hist = TimePercentile(hist, percentile)
                hist = np.ravel(hist.data)

                ####Log transform the data here#### 
                hist = np.log(np.exp(hist)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (hist - delta_sim * t - fwi0_sim)
                print(Endhist)

                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)
 
                f = open('/scratch/cburton/scratch/FWI/2023/hist/Array/'+Country+'_2022code+BC1'+histmember+'_hist'+str(percentile)+'%_LogTransform_-1+1.dat','a')
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
         
              
   
    ##### Repeat for histnat array (can run this separately in parallel to save time) ####
    histnatarray = []
    members = np.arange(1,106)
    for member in members:
        print ('histnat',member)
        for n in np.arange(1,6):
            try:
                if member < 10:
                    histnat = iris.load_cube(folder+index_filestem2+Month+'/FWI_00'+str(member)+'_'+str(n)+'.nc', index_name)
                elif member > 9 and member < 100:
                    histnat = iris.load_cube(folder+index_filestem2+Month+'/FWI_0'+str(member)+'_'+str(n)+'.nc', index_name)
                else:
                    histnat = iris.load_cube(folder+index_filestem2+Month+'/FWI_'+str(member)+'_'+str(n)+'.nc', index_name)    
                histnat = CountryConstrain(histnat, Country)
                histnat = CountryPercentile(histnat, percentile)
                histnat = TimePercentile(histnat, percentile)
                histnat = np.ravel(histnat.data)

                ####Log transform the data here#### 
                histnat = np.log(np.exp(histnat)-1)

                # Step 2: Detrend the sim and scale to obs
                Endhist = fwi0_obs + (histnat - delta_sim * t - fwi0_sim)

                ####inverse Log (exponential) transform here####      
                Endhist = np.log(np.exp(Endhist)+1)

                f = open('/scratch/cburton/scratch/FWI/2023/histnat/Array/'+Country+'_2022code+BC1'+histmember+'_histnat'+str(percentile)+'%_LogTransform_-1+1.dat','a')
                np.savetxt(f,(Endhist),newline=',',fmt='  %s')
                f.write('\n')
                f.close()
                histnatarray.append(histnat)
            except IOError:
                pass 
        
    histnatarray = np.array(histnatarray)
    histnatarray = np.ravel(histnatarray)

exit()




############## 2) Create 3 subplots #################

Countries = ('Canada', 'Greece', 'SAM')
n = 1
for Country in Countries:
    print(Country)
    month,percentile,daterange,ERA5_2023 = SetCountry(Country)
    ERA5_2023 = GetERA5(ERA5_2023,Country)

    #Read in data files for each historical member (each one has 525 values for the 2023 data), for ALL and NAT
    AllDict = {}
    NatDict = {}
    AllDict[Country] = []  
    NatDict[Country] = []  

    members = ('aojaa', 'aojab', 'aojac', 'aojad', 'aojae', 'aojaf', 'aojag', 'aojah', 'aojai', 'aojaj','dlrja', 'dlrjb', 'dlrjc', 'dlrjd', 'dlrje')   
    for member in members:
        print(member)
        all_file = '/scratch/cburton/scratch/FWI/2023/hist/Array/'+Country+'_2022code+BC1'+member+'_hist'+str(percentile)+'%_LogTransform_-1+1.dat'
        nat_file = '/scratch/cburton/scratch/FWI/2023/histnat/Array/'+Country+'_2022code+BC1'+member+'_histnat'+str(percentile)+'%_LogTransform_-1+1.dat'
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
    RR = draw_bs_replicates(AllDict[Country], NatDict[Country], ERA5_2023, RiskRatio, 10000)
    print(len(RR))
    print(np.percentile(RR, 5))
    print(np.percentile(RR, 95))

    #Make the plot
    plt.subplot(1,3,n)
    sns.distplot(AllDict[Country], hist=True, kde_kws={"clip": (0, None)}, 
             color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='ALL')
    sns.distplot(NatDict[Country], hist=True, kde_kws={"clip": (0, None)},
             color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='NAT')
    plt.axvline(x=ERA5_2023, color='black', linewidth=2.5, label='ERA5')
    plt.ylabel(' ')
    if Country == 'SAM':
        Country = 'Western Amazonia'
    plt.title(Country+' FWI '+month+' 2023')
    if n == 1:
        plt.ylabel('Density')
    if n == 2:
        plt.xlabel('Fire Weather Index')
    if n == 3:
        plt.legend()
    n = n+1

plt.show()

exit()




'''
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

