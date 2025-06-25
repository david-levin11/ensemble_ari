import os
from osgeo import gdal, osr

grib_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'
gefs_file = 'base_gefs.grib2'

source_dataset = gdal.Open(os.path.join(grib_dir, gefs_file))

# Read raster dimensions and geotransform information
cols = source_dataset.RasterXSize
rows = source_dataset.RasterYSize
geotransform = source_dataset.GetGeoTransform()
projection = source_dataset.GetProjection()
print(f"Projection is: {projection}")
print(f"Geo transform is {geotransform}")
print(f"X size is: {cols}")
print(f"Y size is: {rows}")