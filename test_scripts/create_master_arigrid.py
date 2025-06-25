import os
import requests
import zipfile
import xarray as xr
from osgeo import gdal

# config
ncdir = r"C:\Users\David.Levin\ensemble_ari\nc_data"
ascii_dir = r"C:\Users\David.Levin\ensemble_ari\ascii_data"
ri_areas = ["sw", "orb", "mw", "se", "ne", "tx", "inw", "ak", "hi"]
ri_lengths = ["2","5","10","25","50","100"]
ri_duration = "24"
hdsc_base_url = "https://hdsc.nws.noaa.gov/pub/hdsc/data/"
ari_regions = ["sw", "orb", "mw", "se", "ne", "tx", "inw", "ak", "hi"]
recurrence_intervals = [int(x) for x in ri_lengths]
durations = [int(ri_duration)]

def unzip_and_cleanup(zip_path, extract_to):
        """
        Extract the contents of a ZIP file to a specified directory and delete the ZIP file afterward.

        Args:
            zip_path (str): The full path to the ZIP file to be extracted.
            extract_to (str): The directory where the contents of the ZIP file should be extracted.

        """
        # Unzip the file
        print(f"Unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted contents to: {extract_to}")

        # Delete the .zip file
        print(f"Deleting .zip file: {zip_path}")
        os.remove(zip_path)
        print(f"Deleted: {zip_path}")

def download_ari_files(hdsc_base_url, output_dir, ari_regions, recurrence_intervals, durations, overwrite=False):
        """
        Download ASCII ARI (Average Recurrence Interval) grids from the Hydrometeorological Design Studies Center (HDSC) 
        and extract them for further processing.

        This function fetches zipped ARI grids for specified regions, recurrence intervals, and durations, unzips them, 
        and stores the extracted files in the configured output directory.

        Configuration Keys:
            - base_url: Base URL of the HDSC server.
            - output_dir: Directory to save downloaded and extracted files.
            - overwrite_existing: Whether to overwrite existing files.
            - ari_regions: Region-specific prefixes for ARI datasets.
            - recurrence_intervals_years: ARI recurrence intervals in years.
            - durations_hours: ARI durations in hours.

        Notes:
            - The function assumes HDSC files follow a specific naming convention (e.g., `{state}{ri}yr{duration:02}ha.zip`).
            - Downloaded files are unzipped immediately, and the .zip files are removed to save space.
            - Ensures required directories exist before processing.

        Output:
            - Extracted ASCII ARI files for each combination of region, recurrence interval, and duration.
            - Logs detailing download and extraction statuses.

        """
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        #state = self.config["ari_settings"]["region"]   # this may be problematic for states other than AK
        # looping through all ARI regions
        for state in ari_regions:
            base_url = f"{hdsc_base_url}{state}/"
            for ri in recurrence_intervals:
                for duration in durations:
                    filename = f"{state}{ri}yr{duration:02}ha.zip"
                    check_file = f"{state}{ri}yr{duration:02}ha.asc"
                    file_url = f"{base_url}{filename}"
                    output_path = os.path.join(output_dir, filename)
                    check_file_path = os.path.join(output_dir, check_file)
                    # Skip if file exists and overwrite is False
                    if os.path.exists(check_file_path) and not overwrite:
                        print(f"File already exists, skipping: {output_path}")
                        continue

                    print(f"Downloading: {file_url}")
                    try:
                        response = requests.get(file_url, stream=True)
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    f.write(chunk)
                            print(f"Saved to: {output_path}")
                        else:
                            print(f"Failed to download {file_url}: {response.status_code}")
                    except Exception as e:
                        print(f"Error downloading {file_url}: {e}")

                    # Unzip and delete the .zip file
                    try:
                        unzip_and_cleanup(output_path, output_dir)
                    except Exception as e:
                        print(f"Error unzipping {output_path}: {e}")
# # checking to see if our ascii files exists and if not, grabbing them from the HDSC server
print(f"Now processing grids for {ri_duration}hr ARI...")
download_ari_files(hdsc_base_url, ascii_dir, ari_regions, recurrence_intervals, durations)
# creating our netcdf directory
if not os.path.exists(ncdir):
    os.makedirs(ncdir, exist_ok=False)

# processing ASCII files
for length in ri_lengths:
    # List of input NetCDF files
    input_files = [
        os.path.join(ascii_dir, f"{area}{length}yr{ri_duration}ha.asc") 
        for area in ri_areas
    ]
    
    # Check if all input files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing input files: {missing_files}")
    
    mosaic_file = f"gl{length}yr{ri_duration}ha.tif"
    output_mosaic = os.path.join(ncdir, mosaic_file)
    print(f"Merging files into: {output_mosaic}")

    # Use gdal.WarpOptions to specify additional parameters
    warp_options = gdal.WarpOptions(
        dstSRS="EPSG:4269",             # Reproject to EPSG:4269
    )

    # Use gdal.Warp to merge the files
    warp = gdal.Warp(
        destNameOrDestDS=output_mosaic,  # Output file path
        srcDSOrSrcDSTab=input_files,       # List of input files
        options=warp_options               # Warp options
    )

    if warp is None:
        raise RuntimeError(f"Failed to merge files for {length}-year duration")

    print(f"Successfully merged files into {output_mosaic}")
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
    varname = f"gl{ri_duration}ha"
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

# creating muiltidimensional datasets
datasets = []
for length in ri_lengths:
    in_file = os.path.join(ncdir,f"gl{length}yr{ri_duration}ha.nc")
    with xr.open_dataset(in_file) as ds:
        # Add a new coordinate for the recurrence interval
        ds = ds.expand_dims({"ARI": [int(length)]})
        # Rename dimensions
        updated_ds = ds.rename_dims({"lat": "y", "lon": "x"})
        # Drop the crs variable
        cleaned_ds = updated_ds.drop_vars("crs")
        # Append to the list
        datasets.append(cleaned_ds)
# Combine all datasets along the recurrence_interval dimension
combined_ds = xr.concat(datasets, dim="ARI")
outfile = os.path.join(ncdir, f"global_ari_{ri_duration}hr.nc")
# Save the combined dataset to a new .netcdf file (optional)
combined_ds.to_netcdf(outfile)

print(combined_ds)