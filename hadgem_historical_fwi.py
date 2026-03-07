# Create .dat files for unbias-corrected historical data, then plot  PDFs

#module load scitools/default-current
#python3
#-*- coding: iso-8859-1 -*-

import numpy as np
import iris
import time
#matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
from utils.constrain_cubes_standard import *
from utils.cubefuncs import *


############# User inputs here #############
Country = 'South Korea'
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

member = os.environ["CYLC_TASK_PARAM_member"]

HadGEM3_Arr = []
start_time = time.time()
for year in np.arange(1960, 2014):
    print('Member number:',member,'\n Year:',year)
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
        tcoord = HadGEM3.coord('time')
        dates = tcoord.units.num2date(tcoord.points)
        print("Selected times:\n", dates[0], "to", dates[-1])
        HadGEM3 = TimePercentile(HadGEM3, percentile)
        HadGEM3 = contrain_to_sow_shapefile(HadGEM3, '/data/users/chantelle.burton/Attribution/StateOfFires_2025-26/SoW2526_Focal_MASTER_20260218.shp','Northwest Iberia')
        #HadGEM3 = CountryConstrain(HadGEM3, Country)
        HadGEM3 = CountryPercentile(HadGEM3, percentile)
        HadGEM3 = np.ravel(HadGEM3.data)
        HadGEM3_Arr.append(HadGEM3)


#Save HaGEM3 text out to a file
f = open('/data/scratch/bob.potts/sowf/test_output/HadGEM3_FWI_1960-2013_'+Country+'_'+str(member)+'_'+str(percentile)+'%.dat','w')
np.savetxt(f,(HadGEM3_Arr))
f.close()  
print('Finished')
#exit()

print("--- %s seconds ---" % (np.round(time.time() - start_time, 2)))
#single member takes approx 8 minutes.