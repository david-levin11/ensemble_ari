from osgeo import osr
import numpy as np
import xarray as xr
import os

# Define the projection string
proj_string = """PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",
DATUM["unnamed",SPHEROID["Sphere",6371200,0]],
PRIMEM["Greenwich",0],
UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
PROJECTION["Polar_Stereographic"],
PARAMETER["latitude_of_origin",60],
PARAMETER["central_meridian",210],
PARAMETER["false_easting",0],
PARAMETER["false_northing",0],
UNIT["Metre",1],
AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]"""

# Define the geotransform values
geotransform = [-2619384.0459885043, 2976.56, 0.0, -1523959.1167387813, 0.0, -2976.56]
# Create the SpatialReference object
srs = osr.SpatialReference()
srs.ImportFromWkt(proj_string)

# Define grid size (e.g., 100x100 grid)
nrows = 1105  # number of rows
ncols = 1649  # number of columns

# Initialize the latitude and longitude arrays
latitudes = np.zeros((nrows, ncols))
longitudes = np.zeros((nrows, ncols))

# Define the starting point (first grid point latitude and longitude)
first_lat = 40.530000
first_lon = 181.429000

# Initialize the coordinate transformation object to convert between the projection and geographic coordinates
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(4326)  # WGS84 (latitude/longitude)

transform = osr.CoordinateTransformation(srs, target_srs)


# Loop through the grid and calculate the latitude and longitude for each pixel
for i in range(nrows):
    for j in range(ncols):
        # Adjust the row index for bottom-left origin (flip the rows)
        #row_index = nrows - 1 - i
        
        # Convert pixel (row_index, j) to projected coordinates (Easting, Northing)
        easting = geotransform[0] + j * geotransform[1]  # x is affected by the column (j)
        northing = geotransform[3] + i * geotransform[5]  # y is affected by the row (row_index)
        
        # Transform projected coordinates to geographic coordinates (latitude, longitude)
        lat,lon, _ = transform.TransformPoint(easting, northing)
        
        # Store in the arrays
        latitudes[i, j] = lat
        longitudes[i, j] = lon
# Reverse the latitude grid to start at the bottom-left   
latitudes = np.flipud(latitudes)
longitudes = (longitudes + 360) % 360
longitudes = np.flipud(longitudes)

# Now, latitudes and longitudes contain the 2D grids for your specified projection
print(f"Latitudes are: {latitudes}")
print(f"Longitudes are: {longitudes}")

ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'
ds = 'base_nbmak.grib2'
# extracting lat and lon arrays for later
with xr.open_dataset(os.path.join(ensemble_dir, ds)) as orig_grid:
    lats = orig_grid.latitude.values
    lons = orig_grid.longitude.values

print(f"Latitude array: {lats}")
print(f"Longitude array: {lons}")
print(f"Latitude shape: {lats.shape}")
print(f"Longitude shape: {lons.shape}")


# # Get the top-left latitude and longitude (first element in first row and first column)
# top_left_lat = lats[-1, 0]  # First latitude
# top_left_lon = lons[0, 0]  # First longitude

# print("Top-left latitude:", top_left_lat)
# print("Top-left longitude:", top_left_lon)