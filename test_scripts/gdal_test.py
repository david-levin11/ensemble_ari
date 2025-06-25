from osgeo import gdal, osr
from pyproj import Proj, transform
import numpy as np
import json
import math

# Path to your JSON configuration file
config_file = r"C:\Users\David.Levin\ensemble_ari\ensemble_ari_config.json"

# Load JSON configuration
with open(config_file, "r") as file:
    config = json.load(file)

ari_region = "hi"

region_config = config["ensemble"]["name"]["nbm"]["ari_regions"][ari_region]
# get projection information
projection_info = region_config["projection_info"]
geotransform = region_config["geotransform"]
print(geotransform)
grib_file = r"C:\Users\David.Levin\ensemble_ari\ensemble_data\base_nbmhi.grib2"

# Path to save the output GeoTIFF
output_tiff = r"C:\Users\David.Levin\ensemble_ari\ensemble_data\base_nbmhi.tif"

def latlon_to_mercator(radius, lat, lon, central_meridian):
    """
    Converts latitude and longitude to Mercator projected coordinates (meters).
    
    Parameters:
        radius (float): Radius of the Earth (in meters).
        lat (float): Latitude of the point (in degrees).
        lon (float): Longitude of the point (in degrees).
        central_meridian (float): Central meridian of the Mercator projection (in degrees).
        
    Returns:
        tuple: (x, y) coordinates in Mercator projection (meters).
    """
    # Convert longitude to Mercator X
    x = radius * (lon - central_meridian) * math.pi / 180

    # Convert latitude to Mercator Y
    # Ensure the latitude is within valid range for Mercator projection
    if lat > 89.9:
        lat = 89.9
    elif lat < -89.9:
        lat = -89.9

    y = radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    
    return x, y

# Open the GRIB file in read-only mode
source_dataset = gdal.Open(grib_file, gdal.GA_ReadOnly)

# Read raster dimensions and geotransform information
cols = source_dataset.RasterXSize
rows = source_dataset.RasterYSize
bands = source_dataset.RasterCount
data_type = source_dataset.GetRasterBand(1).DataType

projection = source_dataset.GetProjection()
print(projection)

# Define projection
proj_merc = Proj("+proj=merc +a=6371200 +b=6371200 +lat_ts=20 +lon_0=-160 +units=m +no_defs")
proj_latlon = Proj(proj="latlong", datum="WGS84")

# Convert Mercator to lat/lon
x, y = -552600.5347875714, 3102675.6406176295
lon, lat = transform(proj_merc, proj_latlon, x, y)
new_lon = -164.9695
new_lat = 26.8605
print(f"Upper-left corner in lat/lon: {lon}, {lat}")

x, y = transform(proj_latlon, proj_merc, new_lon, new_lat)

print(f"Expected upper-left corner in Mercator: {x}, {y}")
source_dataset = None
# llur = 26.8605
# lnur = 195.0305-360
# central_meridian = -160
# R = 6371200
# x, y = latlon_to_mercator(R, llur, lnur, central_meridian)
# print(x)
# print(y)
# rotation_angle = -160
# theta = math.radians(rotation_angle)
# x_scale = 2500*math.cos(theta)
# x_rotation = -2500*math.sin(theta)

# y_rotation = 2500*math.sin(theta)
# y_scale = -2500*math.cos(theta)

# print(f"{x_scale}, {x_rotation}, {y_rotation}, {y_scale}")
# Define the projection
# srs = osr.SpatialReference()
# srs.ImportFromProj4("+proj=merc +a=6371200 +b=6371200 +lat_ts=20 +lon_0=-160 +units=m +no_defs")
# srs.SetProjCS(projection_info["type"])
# srs.SetMercator(
#     projection_info["LaD"],
#     projection_info["orientation"],
#     projection_info["scale1"],
#     projection_info["scale2"],
#     projection_info["scale3"]
# )
# srs.SetGeogCS(
#     projection_info["GCS"],
#     projection_info["GCS"],
#     projection_info["GCS"],
#     projection_info["radius"],
#     projection_info["inverse_flattening"]
# )
# projection_wkt = srs.ExportToWkt()

# # Create a new GeoTIFF dataset
# driver = gdal.GetDriverByName("GTiff")
# output_dataset = driver.Create(output_tiff, cols, rows, bands, data_type)

# # Set geotransform and projection on the new dataset
# output_dataset.SetGeoTransform(geotransform)
# output_dataset.SetProjection(projection_wkt)

# # Copy data from the GRIB dataset to the GeoTIFF
# for band_index in range(1, bands + 1):
#     source_band = source_dataset.GetRasterBand(band_index)
#     output_band = output_dataset.GetRasterBand(band_index)
#     data = source_band.ReadAsArray()
#     #data = np.flipud(data)
#     output_band.WriteArray(data)

# # Clean up
# output_dataset.FlushCache()
# output_dataset = None
# source_dataset = None

# print(f"Successfully saved projected dataset to {output_tiff}")
