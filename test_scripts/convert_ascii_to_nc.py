import os
from osgeo import gdal

## ToDo
## Use the variable name from the ascii file when saving to .nc in gdal
## Alternatively use "global" like "gl2yr24ha" for the variable name
## Use gdal to stitch the created .nc files together with a bounding box that includes HI, AK, and CONUS

ascii_dir = r"C:\Users\David.Levin\ensemble_ari\ascii_data"
ncdir = r"C:\Users\David.Levin\ensemble_ari\nc_data"

ri_areas = ["sw", "orb", "mw", "se", "ne", "tx", "inw", "ak", "hi"]
#ri_areas = ['gl']
ri_lengths = ["2","5","10","25","50","100"]
ri_durations = ["24"]

if not os.path.exists(ncdir):
    os.makedirs(ncdir, exist_ok=False)

for area in ri_areas:
    for length in ri_lengths:
        for duration in ri_durations:

            # Input and output file paths
            #varname = f"{area}{length}yr{duration}ha"
            varname = f"gl{length}yr{duration}ha"
            input_asc_file = f"{area}{length}yr{duration}ha.asc"  # Replace with the path to your .asc file
            output_nc_file = f"{area}{length}yr{duration}ha.nc" # Replace with the desired output NetCDF file name
            
            # Open the ASCII Grid file
            asc_dataset = gdal.Open(os.path.join(ascii_dir,input_asc_file))
            if asc_dataset is None:
                raise Exception(f"Failed to open input file: {input_asc_file}")

            # Translate to NetCDF
            gdal.Translate(
                os.path.join(ncdir,output_nc_file),             # Output file
                asc_dataset,                # Input dataset
                format="netCDF",             # Specify NetCDF format
                outputSRS="EPSG:4269",
                creationOptions=[f"BAND_NAMES={varname}"]
            )

            print(f"Successfully converted {input_asc_file} to {output_nc_file}")
            asc_dataset=None