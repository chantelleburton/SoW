# Plot PDFs for 2023 HadGEM3 ALL and NAT, plus ERA5 line

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



'''
##Uncorrected
#ALL = '/scratch/cburton/scratch/FWI/2023/hist/Array/Greece_2022code_hist95%.dat'
#NAT = '/scratch/cburton/scratch/FWI/2023/histnat/Array/Greece_2022code_histnat95%.dat'
ALL = '/scratch/cburton/scratch/FWI/2023/hist/Array/'+Country+'_UNCORRECTED_hist'+str(percentile)+'%.dat'
NAT = '/scratch/cburton/scratch/FWI/2023/histnat/Array/'+Country+'_UNCORRECTED_histnat'+str(percentile)+'%.dat'


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

print(AllDict)
print(NatDict)
sns.distplot(AllDict, hist=True, kde=True, 
             color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='ALL')
sns.distplot(NatDict, hist=True, kde=True, 
             color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='NAT')
plt.axvline(x=ERA5_2023, color='black', linewidth=2.5, label='ERA5 '+month+' 2023')
plt.title(Country+' FWI '+month+' 2023')
plt.legend()
plt.show()

exit()
'''


###Corrected - Just ONE region / plot

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
ERA5_2025 = GetERA5(ERA5_2025, Country)
RR = draw_bs_replicates(AllDict[Country], NatDict[Country], ERA5_2025, RiskRatio, 10000)
print(len(RR))
print(np.percentile(RR, 5))
print(np.percentile(RR, 95))

#Make the plot
sns.distplot(AllDict[Country], hist=True, kde_kws={"clip": (0, None)}, 
         color = 'orange', fit_kws={"linewidth":2.5,"color":"orange"}, label='ALL')
sns.distplot(NatDict[Country], hist=True, kde_kws={"clip": (0, None)},
         color = 'blue', fit_kws={"linewidth":2.5,"color":"blue"}, label='NAT')
plt.axvline(x=ERA5_2025, color='black', linewidth=2.5, label='ERA5')
plt.title(Country+' FWI '+month+' 2025')
plt.legend()
plt.show()





#Bootstrap to get Risk Ratio

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

# Draw bootstrap replicates

RR = draw_bs_replicates(AllDict, NatDict, ERA5_2025, RiskRatio, 10000)


print(len(RR))
print(np.percentile(RR, 5))
print(np.percentile(RR, 95))


exit()




'''
RESULTS
- Uncorrected Canada
121
61
Risk Ratio =  1.9836065573770492

- Uncorrected Greece
172
70
Risk Ratio =  2.4571428571428573

'/scratch/cburton/scratch/FWI/2023/'+EXP+'/Array/Uncorrected'+Country+EXP+percentile+'%.dat'

- Corrected Canada  (one mem)
64
33
Risk Ratio =  1.9393939393939394

Canada with all members (95%):
1497
725
Risk Ratio =  2.0648275862068965

Bootstrapped: 1000
1.9327628281910427
2.2217466266866563

Bootstrapped with replacement and 1000
1.60938999167347
1.8394014264571188

Bootstrapped with replacement and 10000
1.6019326689950548
1.841659117756872

- Corrected Greece 

95%:
0/0

90%:
53
14
Risk Ratio =  3.7857142857142856

Bootstrapped: 1000
2.3973913043478263
7.0

Bootstrapped with replacement and 10000
2.3157894736842106
6.857142857142857


- Corected SAM
Bootstrapped with replacement and 10000
8.598040758676351
10.298037084609556


'''

