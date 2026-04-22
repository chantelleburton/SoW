import iris
import numpy as np
import geopandas as gpd
from .constrain_cubes_standard import contrain_to_sow_shapefile

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

def ConstrainToYear(cube, target_year):
    year_constraint = iris.Constraint(time=lambda cell: cell.point.year == target_year)
    out = cube.extract(year_constraint)
    if out is None:
        t = cube.coord('time')
        dts = t.units.num2date(t.points)
        years_present = sorted({dt.year for dt in dts})
        raise ValueError(f"No data for year {target_year}. Years present: {years_present[:5]} ... {years_present[-5:]}")
    return out 
def RiskRatio(Alldata, Natdata, Threshold):
    """
    Calculate the Risk Ratio between ALL (anthropogenic) and NAT (natural) scenarios.
    
    Parameters
    ----------
    Alldata : array-like
        FWI values from ALL forcing scenario
    Natdata : array-like
        FWI values from NAT (natural-only) forcing scenario
    Threshold : float
        The threshold value (e.g., ERA5 2025 observed value)
    
    Returns
    -------
    float
        Risk Ratio (ALL exceedance count / NAT exceedance count)
    """
    ALL_count = np.count_nonzero(Alldata > Threshold)
    NAT_count = np.count_nonzero(Natdata > Threshold)
    
    if NAT_count == 0:
        return np.inf  # Handle division by zero
    
    return ALL_count / NAT_count


def draw_bs_replicates(ALL, NAT, threshold, func, size):
    """
    Create bootstrap replicates for uncertainty estimation.
    
    Uses a two-step resampling: first subsample 90% without replacement,
    then resample to original size with replacement.
    
    Parameters
    ----------
    ALL : array-like
        FWI values from ALL forcing scenario
    NAT : array-like
        FWI values from NAT forcing scenario
    threshold : float
        The threshold value for Risk Ratio calculation
    func : callable
        Function to compute statistic (e.g., RiskRatio)
    size : int
        Number of bootstrap replicates to generate
    
    Returns
    -------
    np.ndarray
        Array of bootstrap replicates
    """
    RR_replicates = np.empty(size)
    
    ALL_subsample_size = int(np.round(len(ALL) * 0.9))
    NAT_subsample_size = int(np.round(len(NAT) * 0.9))
    
    for i in range(size):
        # Step 1: Subsample 90% without replacement
        ALL_subsample = np.random.choice(ALL, size=ALL_subsample_size, replace=False)
        NAT_subsample = np.random.choice(NAT, size=NAT_subsample_size, replace=False)
        
        # Step 2: Resample to original size with replacement
        ALL_sample = np.random.choice(ALL_subsample, size=len(ALL), replace=True)
        NAT_sample = np.random.choice(NAT_subsample, size=len(NAT), replace=True)
        
        # Compute statistic
        RR_replicates[i] = func(ALL_sample, NAT_sample, threshold)
    
    return RR_replicates


def GetERA5Threshold(era5_file, shp_file, shape_name, month, percentile):
    """
    Compute ERA5 2025 threshold matching original Supplement2.py methodology.
    
    Order: extract month -> shapefile mask -> spatial percentile -> temporal percentile
    
    Parameters
    ----------
    era5_file : str
        Path to the ERA5 netCDF file
    shp_file : str
        Path to the shapefile for region masking
    shape_name : str
        Name of the shape/region within the shapefile
    month : int or tuple
        Month(s) to extract (e.g., 7 for July, or (7, 8) for July-August)
    percentile : float
        Percentile to compute (e.g., 95)
    
    Returns
    -------
    float
        Scalar threshold value
    """
    # Load ERA5 cube
    era5_cube = iris.load_cube(era5_file, 'canadian_fire_weather_index')
    
    # 1. Apply month constraint FIRST (matching original)
    if isinstance(month, tuple):
        daterange = iris.Constraint(time=lambda cell: cell.point.month in month)
    else:
        daterange = iris.Constraint(time=lambda cell: cell.point.month == month)
    
    era5_cube = era5_cube.extract(daterange)
    
    # 2. Apply shapefile mask
    era5_cube = apply_shapefile_inclusive(shp_file, shape_name, era5_cube)
    
    # 3. Spatial percentile
    era5_cube = CountryPercentile(era5_cube, percentile)
    
    # 4. Temporal percentile
    era5_cube = TimePercentile(era5_cube, percentile)
    
    return float(np.array(era5_cube.data))

def apply_shapefile_inclusive(shp_file, shape_name, cube):
    
    shapefile = gpd.read_file(shp_file)
    
    # Set coordinate system for iris.util.mask_cube_from_shape
    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    
    # Get geometry for this region
    region_gdf = shapefile[shapefile['name'] == shape_name]
    region_geom = region_gdf['geometry'].values[0]
    
    # Step 1: Crop to bounding box to reduce data volume
    from .constrain_cubes_standard import contrain_coords
    minx, miny, maxx, maxy = region_geom.bounds
    cube = contrain_coords(cube, (minx, maxx, miny, maxy))
    
    # Step 2: Apply inclusive mask via iris
    masked_cube = iris.util.mask_cube_from_shape(cube, region_geom)
    
    return masked_cube