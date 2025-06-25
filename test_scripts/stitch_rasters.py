import os
from osgeo import gdal

ncdir = r"C:\Users\David.Levin\ensemble_ari\nc_data"
ascii_dir = r"C:\Users\David.Levin\ensemble_ari\ascii_data"
ri_areas = ["sw", "orb", "mw", "se", "ne", "tx", "inw", "ak", "hi"]
ri_lengths = ["2","5","10","25","50","100"]
ri_duration = "24"


if not os.path.exists(ncdir):
    os.makedirs(ncdir, exist_ok=False)


# for length in ri_lengths:
#     # List of input NetCDF files
#     input_files = [
#         os.path.join(ascii_dir, f"{area}{length}yr{ri_duration}ha.asc") 
#         for area in ri_areas
#     ]
    
#     # Check if all input files exist
#     missing_files = [f for f in input_files if not os.path.exists(f)]
#     if missing_files:
#         raise FileNotFoundError(f"Missing input files: {missing_files}")
    
#     mosaic_file = f"gl{length}yr{ri_duration}ha.tif"
#     output_mosaic_nc = os.path.join(ncdir, mosaic_file)
#     print(f"Merging files into: {output_mosaic_nc}")

#     # Use gdal.WarpOptions to specify additional parameters
#     warp_options = gdal.WarpOptions(
#         dstSRS="EPSG:4269",             # Reproject to EPSG:4269
#     )

#     # Use gdal.Warp to merge the files
#     warp = gdal.Warp(
#         destNameOrDestDS=output_mosaic_nc,  # Output file path
#         srcDSOrSrcDSTab=input_files,       # List of input files
#         options=warp_options               # Warp options
#     )

#     if warp is None:
#         raise RuntimeError(f"Failed to merge files for {length}-year duration")

#     print(f"Successfully merged files into {output_mosaic_nc}")
# now resampling to something more managable
for ri_length in ri_lengths:
    mosaic_file = os.path.join(ncdir, f"gl{ri_length}yr{ri_duration}ha.tif")
    output_raster = os.path.join(ncdir, f"gl{ri_length}yr{ri_duration}ha_resampled.tif")
    # Define the target resolution in degrees (for ~2.5 km resolution)
    target_resolution = 0.0225  # Approximate 2.5 km resolution in degrees

    # Define the spatial reference (EPSG:4326 for Lat/Lon)
    target_srs = 'EPSG:4326'

    # Perform the resampling using GDAL Warp 
    gdal.Warp(output_raster, mosaic_file,
            format='GTiff',  # Output format
            xRes=target_resolution,  # Resolution in X (Longitude)
            yRes=target_resolution,  # Resolution in Y (Latitude)
            dstSRS=target_srs,  # Target Spatial Reference
            #outputBounds=output_bounds_west,  # Define the bounding box
            targetAlignedPixels=True,  # Ensure pixel alignment
            warpOptions=['DATELINEOFFSET=180'])  # Handle dateline properly
    
    print("Resampling and reprojection complete!")

        
# now converting to netcdf for use with python applications like xesmf
for length in ri_lengths:
    # Input and output file paths
    #varname = f"{area}{length}yr{duration}ha"
    varname = f"gl{length}yr{ri_duration}ha"
    input_asc_file = f"gl{length}yr{ri_duration}ha_resampled.tif"  # Replace with the path to your .asc file
    output_nc_file = f"gl{length}yr{ri_duration}ha.nc" # Replace with the desired output NetCDF file name
    
    # Open the ASCII Grid file
    asc_dataset = gdal.Open(os.path.join(ncdir,input_asc_file))
    if asc_dataset is None:
        raise Exception(f"Failed to open input file: {input_asc_file}")

    # Translate to NetCDF
    gdal.Translate(
        os.path.join(ncdir,output_nc_file),             # Output file
        asc_dataset,                # Input dataset
        format="netCDF",             # Specify NetCDF format
        outputSRS="EPSG:4326",
        creationOptions=[f"BAND_NAMES={varname}"]
    )

    print(f"Successfully converted {input_asc_file} to {output_nc_file}")
    asc_dataset=None