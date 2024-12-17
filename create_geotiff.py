import os
from osgeo import gdal
import xarray as xr
import numpy as np

input_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_exceedance_data'

model_longname = 'nbmqmd'

model_shortname = 'nbm'

region = 'ak'

output_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_exceedance_rasters'

graphicdir = os.path.join(os.path.join(output_dir, model_longname), region)

nc_dir = os.path.join(os.path.join(input_dir, model_longname), region)

base_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_data'

base_model = f"base_{model_shortname}{region}.grib2"

base_grid = os.path.join(base_dir, base_model)



# making sure our directory exists
os.makedirs(nc_dir, exist_ok=True)
os.makedirs(graphicdir, exist_ok=True)

with gdal.Open(base_grid) as grib_ds:
    # Extract GeoTransform and Projection (CRS) from GRIB file
    geo_transform = grib_ds.GetGeoTransform()
    projection_wkt = grib_ds.GetProjection()
    print("GeoTransform:", geo_transform)
    print("Projection (WKT):", projection_wkt)



for fl in os.listdir(nc_dir):
    # Input and output file paths
    input_nc = os.path.join(nc_dir, fl)
    input_fl = fl.split(".")[0]
    output_tiff = os.path.join(graphicdir, f"{input_fl}.tif")

    # Specify the variable you want to export
    variable = "exceedance_perc"

    ds_xr = xr.open_dataset(input_nc)

    # Assuming the variable of interest is "exceedance_perc"
    data_var = ds_xr["exceedance_perc"].values  # Extract data array
    # flip so gdal writes it correctly
    data_var = np.flipud(data_var)
    nodata_value = np.nan
    # Step 3: Write the data to a GeoTIFF with CRS and GeoTransform
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = data_var.shape  # Get the dimensions of the data

    # Create the GeoTIFF file
    tiff_ds = driver.Create(output_tiff, cols, rows, 1, gdal.GDT_Float32)

    # Assign GeoTransform and Projection to the GeoTIFF
    tiff_ds.SetGeoTransform(geo_transform)
    tiff_ds.SetProjection(projection_wkt)

    # Step 4: Write data to the GeoTIFF
    band = tiff_ds.GetRasterBand(1)
    band.WriteArray(data_var)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()

    # Clean up
    tiff_ds = None
    print(f"GeoTIFF saved to {output_tiff}")
