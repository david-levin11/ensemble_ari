import os
import sys
import time
import re
import requests
import logging
import json
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
from datetime import datetime, timedelta, timezone
from osgeo import gdal, osr


class Ensemble_ARI:
    def __init__(self, model, region, config_path='ensemble_ari_config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up logging
        if self.config["logging"]["log_to_file"]:
            logging.basicConfig(
                filename=self.config["logging"]["log_file"],
                level=getattr(logging, self.config["logging"]["log_level"]),
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(
                level=getattr(logging, self.config["logging"]["log_level"]),
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger()

        # what model are we processing
        self.model = model
        # what region (global or subset)
        self.region = region
        # checks for wrong regions
        model_config = self.config["ensemble"]["name"][self.model]
        if self.region not in model_config["valid_regions"]:
            self.logger.error(f"{self.region} is not a valid domain for {self.model}.")
            self.logger.info(f"Valid regions for {self.model} are: {model_config['valid_regions']}")
            self.logger.info(f"Switching domain to {model_config['valid_regions'][0]}")
            self.region = model_config["valid_regions"][0]

    def download_ari_files(self):
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
        hdsc_base_url = self.config["download"]["base_url"]
        output_dir = self.config["download"]["output_dir"]
        overwrite = self.config["download"]["overwrite_existing"]
        regrid_dir = self.config["ensemble"]["regrid_dir"]
        ari_regions = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["ari_prefixes"]
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Ensure regrid directory exists
        os.makedirs(regrid_dir, exist_ok=True)

        #state = self.config["ari_settings"]["region"]   # this may be problematic for states other than AK
        recurrence_intervals = self.config["ari_settings"]["recurrence_intervals_years"]
        durations = self.config["ari_settings"]["durations_hours"]
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
                        self.logger.info(f"File already exists, skipping: {output_path}")
                        continue

                    self.logger.info(f"Downloading: {file_url}")
                    try:
                        response = requests.get(file_url, stream=True)
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    f.write(chunk)
                            self.logger.info(f"Saved to: {output_path}")
                        else:
                            self.logger.warning(f"Failed to download {file_url}: {response.status_code}")
                    except Exception as e:
                        self.logger.error(f"Error downloading {file_url}: {e}")

                    # Unzip and delete the .zip file
                    try:
                        self.unzip_and_cleanup(output_path, output_dir)
                    except Exception as e:
                        self.logger.error(f"Error unzipping {output_path}: {e}")

    def unzip_and_cleanup(self, zip_path, extract_to):
        """
        Extract the contents of a ZIP file to a specified directory and delete the ZIP file afterward.

        Args:
            zip_path (str): The full path to the ZIP file to be extracted.
            extract_to (str): The directory where the contents of the ZIP file should be extracted.

        """
        # Unzip the file
        self.logger.info(f"Unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            self.logger.info(f"Extracted contents to: {extract_to}")

        # Delete the .zip file
        self.logger.info(f"Deleting .zip file: {zip_path}")
        os.remove(zip_path)
        self.logger.info(f"Deleted: {zip_path}")

    def cleanup_ascii(self, directory, delete_file):
        """
        Delete all files in the specified directory that start with the given prefix.

        :param directory: Path to the directory to search for files.
        :param delete_file: filename to match files against.
        """
        try:
            for filename in os.listdir(directory):
                prefix = delete_file.split(".")[0]
                if filename.startswith(prefix):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.logger.info(f"Deleted: {file_path}")
                    else:
                        self.logger.info(f"Skipped (not a file): {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def regrid_to_base_dataset(self):
        """
        Preprocess and regrid ARI (Average Recurrence Interval) grids to match the base ensemble dataset's spatial 
        resolution and projection. This function handles the conversion of ASCII ARI grids into NetCDF format, 
        regridding them to align with the ensemble dataset specifications.

        The function currently supports two models:
            - NBM (National Blend of Models)
            - EPS (ECMWF Ensemble Prediction System)

        Workflow:
            1. Determine the base grid specification (projection, geotransform, dimensions) for the selected model.
            2. Loop through ARI regions, recurrence intervals, and durations to:
                a. Read ASCII input files.
                b. Reproject and resample them to the target grid using GDAL.
                c. Save the resampled data as NetCDF files.
            3. Optionally, merge ARI datasets for specific regions (e.g., CONUS for NBM).

        Configuration Keys:
            - ensemble_dir: Directory containing ensemble-specific data.
            - regrid_dir: Output directory for regridded ARI files.
            - ari_regions: Region-specific prefixes for ARI datasets.
            - overwrite_existing: Whether to overwrite existing files.
            - durations_hours: ARI durations in hours.
            - recurrence_intervals_years: ARI recurrence intervals in years.
            - projection_info: Spatial reference and grid configuration for the target ensemble dataset.

        Notes:
            - The function assumes ASCII ARI files follow HDSC's specific naming convention (e.g., `{state}{ri}yr{duration}ha.asc`).
            - Regridding uses bilinear resampling and aligns pixels with the target grid.
            - Flipping raster data along the y-axis may be necessary for compatibility with certain models (e.g., NBM).

        Output:
            - NetCDF files for each combination of ARI region, recurrence interval, and duration, regridded to the target ensemble dataset's grid.
            - Logs containing information about regridding processes, including file paths, dimensions, and errors.
        """
        ari_regions = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["ari_prefixes"]
        region_config = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        regrid_dir = self.config["ensemble"]["regrid_dir"]
        ascii_dir = self.config["download"]["output_dir"]
        overwrite = self.config["ensemble"]["overwrite_existing"]
        recurrence_intervals = self.config["ari_settings"]["recurrence_intervals_years"]
        durations = self.config["ari_settings"]["durations_hours"]
        if self.model == "nbm":
            # creating our regional grids from projection and transform/grid spacing in config
            grid_nbm = self.create_nbm_grid(region_config["projection_info"]["proj_string"], region_config["geotransform"], region_config["ysize"], region_config["xsize"])
            lats = grid_nbm[0]
            lons = grid_nbm[1]
            coordinates={
                            "latitude": (["y", "x"], lats), #using original dataset coordinates
                            "longitude": (["y", "x"], lons),
                        }
        elif self.model == "eps":
            # creating our global grid from the geotransform
            grid_global = self.create_global_lat_lon_grid(region_config["geotransform"], region_config["ysize"], region_config["xsize"])
            lats = grid_global[0]
            lons = grid_global[1]
            coordinates={
                            "latitude": lats, #using original dataset coordinates
                            "longitude": lons,
                        }
        else:
            self.logger.error(f"Cannot find grid specs for {self.model}. Check your config file for projection, grid spacing, and transformations")
            sys.exit()
        # Looping through the ARI "states/regions" for the ensemble to regrid
        for i, state in enumerate(ari_regions):
            for ri in recurrence_intervals:
                for duration in durations:
                    ensemble_name = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["ensemble_name"]
                    input_filename = f"{state}{ri}yr{duration:02}ha.asc"
                    if self.region == "co" and self.model == "nbm":
                        varname = f"{self.region}{ri}yr{duration:02}ha"
                    elif self.model == "eps":
                        varname = f"{self.region}{ri}yr{duration:02}ha"
                    else:
                        varname = f"{state}{ri}yr{duration:02}ha"
                    output_filename = f"{state}{ri}yr{duration:02}ha_regridded_to{ensemble_name}.nc"
                    input_file = os.path.join(ascii_dir, input_filename)
                    output_file = os.path.join(regrid_dir, output_filename)
                    if self.model == "nbm" and self.region == "co":
                        checkfile = os.path.join(regrid_dir, f"{varname}_regridded_to{ensemble_name}.nc")
                    elif self.model == "eps":
                        checkfile = os.path.join(regrid_dir, f"{varname}_regridded_to{ensemble_name}.nc")
                    else:
                        checkfile = output_file
                    # Skip if file exists and overwrite is False
                    if os.path.exists(checkfile) and not overwrite:
                        self.logger.info(f"File already exists, skipping: {output_file}")
                        continue
                    # Logging information for each combination of dataset, recurrence interval, and duration
                    self.logger.info(f"Regridding for Recurrence Interval: {ri} years, Duration: {duration:02} hours for region: {state}")
                    # have to read in the nbm hawaii projection info from config
                    if self.model == "nbm":
                        srs = osr.SpatialReference()
                        srs.ImportFromProj4(region_config["projection_info"]["proj_string"])
                        grib_projection = srs.ExportToWkt()
                        grib_geotransform = region_config["geotransform"]
                        grib_xsize = region_config["xsize"]
                        grib_ysize = region_config["ysize"]
                    elif self.model == "eps":
                        srs = osr.SpatialReference()
                        srs.ImportFromWkt(region_config["projection_info"]["proj_string"])
                        grib_projection = srs.ExportToWkt()
                        grib_geotransform = region_config["geotransform"]
                        grib_xsize = region_config["xsize"]
                        grib_ysize = region_config["ysize"]
                    else:
                        self.logger.error(f"Regridding is only working for NBM and EPS. Sorry!")
                        sys.exit()
                    self.logger.info(f"Base grid projection: {grib_projection}")
                    self.logger.info(f"Base grid geotransform: {grib_geotransform}")
                    self.logger.info(f"Base grid size: {grib_xsize} x {grib_ysize}")
                    # Compute output bounds (extent) from target geotransform and dimensions
                    xmin = grib_geotransform[0]
                    xmax = xmin + grib_xsize * grib_geotransform[1]
                    ymax = grib_geotransform[3]
                    ymin = ymax + grib_ysize * grib_geotransform[5]
                    output_bounds = (xmin, ymin, xmax, ymax)

                    self.logger.info(f"Regridding from {input_file} to {output_file}")

                    # Perform reprojection and resampling
                    try:
                        resampled_ds = gdal.Warp(
                            output_file,           # Output file
                            input_file,            # Input raster
                            format="NetCDF",       # Output format
                            dstSRS=grib_projection,  # Target spatial reference
                            xRes=grib_geotransform[1],  # X resolution
                            yRes=grib_geotransform[5],  # Y resolution
                            outputBounds=output_bounds,  # Match the spatial extent of the target grid
                            width=grib_xsize,            # Force output raster width
                            height=grib_ysize,           # Force output raster height
                            targetAlignedPixels=True,    # Align pixels with the target grid
                            resampleAlg="bilinear"       # Resampling algorithm
                        )
                        self.logger.info(f"Resampled raster saved to {output_file}")
                    except Exception as e:
                        self.logger.error(f"Error during regridding: {e}")
                        continue
                    #Closing datasets
                    resampled_ds = None
                    
                    # deleting unnecessary ascii files
                    #self.cleanup_ascii(ascii_dir, input_filename)
                    # Load the resampled dataset
                    try:
                        resampled_ds = gdal.Open(output_file)
                        # Get dimensions and geotransform
                        x_size = resampled_ds.RasterXSize
                        y_size = resampled_ds.RasterYSize
                        geotransform = resampled_ds.GetGeoTransform()

                        self.logger.info(f"Resampled dataset dimensions: {x_size} x {y_size}")
                        self.logger.info(f"Geotransform: {geotransform}")
                        # Extract the raster data
                        band = resampled_ds.GetRasterBand(1)
                        data = band.ReadAsArray()
                        # flipping the data along the y axis due to the way the raster is stored in GDAL
                        # Not sure if this is always the case...need to test on multiple datasets
                        if self.model == "nbm":
                            data = np.flipud(data)
                        self.logger.info(f'Data size is: {data.shape}')
                        #closing dataset
                        resampled_ds = None
                    except Exception as e:
                        self.logger.error(f"Error opening resampled dataset: {output_file}, {e}")
                        continue

                    # Create xarray dataset
                    output_ds = xr.Dataset(
                        {
                            varname: (["y", "x"], data)  # Rename data variable
                        },
                        coords=coordinates,
                        attrs={
                            "title": "Resampled Data with Lat/Lon",
                            "crs": grib_projection,
                        }
                    )
                    # Create xarray dataset without explicit y, x dimensions
                    # Save as NetCDF
                    try:
                        output_ds.to_netcdf(output_file)
                        self.logger.info(f"NetCDF saved to {output_file}")
                    except Exception as e:
                        self.logger.error(f"Error saving NetCDF: {output_file}, {e}")
        # At the end merge all the conus datasets together
        if self.model == "nbm" and self.region == "co":
            self.merge_ari()
        if self.model == "eps":
            self.merge_ari()

    def calc_ari_from_nbm_grib(self, nbmfilepath):
        """
        Calculate the ensemble percent exceedance of Average Recurrence Interval (ARI) grids 
        using NBM Probabilitistic QPF (PQPF) GRIB data.

        This function processes NBM (National Blend of Models) GRIB files to calculate exceedance percentages
        relative to predefined ARI grids. The results are saved as NetCDF files for further analysis.

        Args:
            nbmfilepath (str): Path to the NBM QMD GRIB file containing precipitation percentile data.

        Notes:
            - The ARI grids are assumed to be preprocessed and regridded to match the NBM grid structure.
            - GRIB data is accessed using `cfgrib`, and ARI data is accessed using `xarray`.

        Attributes:
            - The function dynamically constructs input and output paths based on the configuration settings.
            - Supports both spatial dimensions and percentiles for vectorized calculations.
            - Handles regional differences such as Alaska/Hawaii-specific latitude and longitude formatting.

        Output:
            The function generates NetCDF files containing the exceedance percentage grids for each combination
            of ARI duration and recurrence interval.

        """
        if self.model != "nbm":
            self.logger.error(f"This function only works for NBM gribs, not {self.model}.")
            sys.exit()
        # pulling vars from config
        ri_filepath = self.config["ensemble"]["regrid_dir"]
        base_exceedance_dir = self.config["ensemble"]["base_exceedance_dir"]
        ensemble = self.config["ensemble"]["name"][self.model]["longname"]
        ensemble_shortname = self.config["ensemble"]["name"][self.model]["shortname"]
        #region = self.config["ari_settings"]["region"]
        ri_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        ri_durations = self.config["ari_settings"]["durations_hours"]
        percentiles = self.config["ensemble"]["name"][self.model]["percentiles"]
        # working with the vars to create additional dynamic ones
        exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), self.region)
        # looping through the various ARI durations and computing percent exceedance grids
        for ri_duration in ri_durations:
            # looping through the RI grids
            for ri_length in ri_lengths:
                self.logger.info(f"Now working on {ri_length} ARI...")
                ri_file = f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}{self.region}.nc'
                self.logger.info(f"ARI file is: {ri_file}")
                
                # Step 1 Load QMD file and extract time information
                self.logger.info(f"Now loading {nbmfilepath}")
                time_info = self.extract_forecast_details(nbmfilepath, int(ri_duration))
                output_filename = f"{self.region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{time_info['step_string']}.nc"
                # Initializing our cube
                percentile_cube = []

                # Step 2 Extract each percentile at the appropriate step range (24hr)
                try:
                    for percentile in percentiles:
                        # Load the file for the current percentile
                        with xr.open_dataset(
                            nbmfilepath,
                            engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepRange': time_info["step_range"], 'percentileValue': percentile}}
                        ) as ds:
                            percentile_cube.append(ds.tp)  # Assuming `tp` is the variable
                except Exception as e:
                    self.logger.error(f"Error has occurred {e}")
                    self.logger.info(f"One possible issue may be that {nbmfilepath} doesn't seem to exist. Skipping this time step...")
                    continue

                # Step 3 Combine into a single cube
                percentile_cube = xr.concat(percentile_cube, dim='percentileValue')
                percentile_cube = percentile_cube.assign_coords(percentileValue=percentiles)
                percentile_cube = percentile_cube*0.03937

                # Step 4 Load regridded ARI data at the same duration as the step Range
                try:
                    with xr.open_dataset(os.path.join(ri_filepath, ri_file)) as ri_ds:
                        # replacing -9 with nan
                        ri_ds = ri_ds.where(ri_ds != -9, other=np.nan)
                        # ARI data is in 1000s of inches per HDSC metadata
                        ri_ds = ri_ds/1000
                        # pulling out the RI values
                        ri_da = ri_ds[f'{self.region}{ri_length}yr{int(ri_duration):02d}ha'].values
                        # percentiles 
                        selected_percentiles = np.array(percentiles)
                        # pulling out our percentile values
                        percentile_cube_data = percentile_cube.values
                        # Reshape for vectorized interpolation
                        reshaped_cube = np.moveaxis(percentile_cube_data, 0, -1)  # Shape: (y, x, 7) for easier indexing
                        # Flatten the spatial dimensions for interpolation
                        flat_random_precip = ri_da.flatten()
                        flat_cube = reshaped_cube.reshape(-1, reshaped_cube.shape[-1])
                        # Perform vectorized interpolation for each (y, x) point
                        flat_rank_array = np.array(
                            [
                                np.interp(value, flat_cube[i], selected_percentiles, left=0, right=100)
                                for i, value in enumerate(flat_random_precip)
                            ]
                        )
                        # Reshape back to the original 2D shape
                        rank_array = flat_rank_array.reshape(ri_da.shape)
                        # need the exceedance percentage not the rank
                        rank_array = 100 - rank_array
                        # nbm hawaii lat/lons come in 1d flattened arrays
                        if self.region == "hi":
                            lat = ri_ds["latitude"].data.reshape(rank_array.shape)
                            lon = ri_ds['longitude'].data.reshape(rank_array.shape)
                        else:
                            lat =  ri_ds['latitude'].data
                            lon =  ri_ds['longitude'].data
                except Exception as e:
                    self.logger.error(f"Error has occurred {e}")
                    self.logger.info(f"One possible issue may be that {ri_file} doesn't seem to exist in {ri_filepath}.  Make sure you have downloaded the ARIs and regridded to this model")
                    self.logger.info(f"Skipping {ri_length} ARI...")
                    continue
                    
                # Step 6: Save the rank array to NetCDF
                os.makedirs(exceedance_dir, exist_ok=True)  # Ensure the output directory exists
                # Construct the output file name dynamically
                output_file = os.path.join(exceedance_dir, output_filename)
                # Create an xarray Dataset for saving
                rank_ds = xr.Dataset(
                    {
                        "exceedance_perc": (["y", "x"], rank_array)  # Use the dimensions of the rank_array
                    },
                    coords={
                                    "latitude": (["y", "x"], lat),
                                    "longitude": (["y", "x"], lon),
                                },
                    attrs={
                        "title": f"Rank Percentile for {ri_length}-yr ARI at step {time_info['step_string']}",
                        "description": f"Rank computed from ARI and {ensemble} precipitation percentiles",
                        "units": "rank (percentile index)"
                    }
                )
                # Save to NetCDF
                rank_ds.to_netcdf(output_file)
                self.logger.info(f"Rank array saved to {output_file}")

    def calc_ari_from_eps_grib(self, epsfilepath_f, epsfilepath_b, ri_duration="24"):
        """
        Calculate the ensemble percent exceedance of Average Recurrence Interval (ARI) grids  
        using ECMWF Ensemble System (EPS) GRIB data.

        This function processes EPS GRIB files to calculate exceedance percentages of precipitation relative 
        to predefined ARI grids. It uses two EPS files representing consecutive time steps to compute accumulated 
        precipitation and evaluates its percentile rank relative to ARI data. The results are saved as NetCDF files.

        Args:
            epsfilepath_f (str): Path to the forward EPS GRIB file (current time step).
            epsfilepath_b (str): Path to the backward EPS GRIB file (previous time step).
            ri_duration (int): Duration (in hours) corresponding to the recurrence interval.

        Notes:
            - This function will calculate the precipitation interval represented by the two EPS grib files regardless
            of what the ARI duration is.  You should make sure that they match! (i.e. 24hr ARI for 24hr accumulated precipitation)
            - The input GRIB files should match the expected structure, including ensemble members and variable naming conventions.
            - ARI grids must be preprocessed and regridded to the EPS spatial resolution.

        Output:
            The function generates NetCDF files containing the exceedance percentage grids for each combination
            of ARI duration and recurrence interval.

        """
        if self.model != "eps":
            self.logger.error(f"This function only works with EPS grib files, not {self.model}.")
            sys.exit()
        # pulling vars from config
        ri_filepath = self.config["ensemble"]["regrid_dir"]
        base_exceedance_dir = self.config["ensemble"]["base_exceedance_dir"]
        ensemble = self.config["ensemble"]["name"][self.model]["longname"]
        ensemble_shortname = self.config["ensemble"]["name"][self.model]["shortname"]
        #region = self.config["ari_settings"]["region"]
        ri_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        percentiles = [p/100.0 for p in self.config["ensemble"]["name"][self.model]["percentiles"]]
        # working with the vars to create additional dynamic ones
        exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), self.region)
        # looping through the various ARI durations and computing percent exceedance grids
        # Step 1 Load EPS file and extract time information
        self.logger.info(f"Now loading {epsfilepath_f}")
        time_info = self.extract_forecast_details(epsfilepath_f, int(ri_duration))
        # Step 2 Create the appropriate accumulation period by opening valid and previous time steps
        try:
            # opening the first time step
            with xr.open_dataset(epsfilepath_f, engine="cfgrib", filter_by_keys = {'dataType': 'pf'}) as ds:
                #extract the precipitation
                tp_end = ds.tp 

            # opening the previous time step
            with xr.open_dataset(epsfilepath_b, engine="cfgrib", filter_by_keys = {'dataType': 'pf'}) as ds:
                #extract the precipitation
                tp_begin = ds.tp 

        except Exception as e:
            self.logger.error(f"Error has occurred {e}")
            self.logger.info(f"One possible issue may be that {epsfilepath_b} or {epsfilepath_f} doesn't seem to exist. Skipping this time step...")
        # Step 3 calculate the accumulated precip and create percentiles
        # calculate the 24hr precip (and converting to inches)
        tp_accum = (tp_end-tp_begin)*39.3701
        # calculating the percentiles
        percentile_cube = tp_accum.quantile(q=percentiles, dim="number")
    
        for ri_length in ri_lengths:
            self.logger.info(f"Now working on {ri_length} ARI...")
            ri_file = f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}.nc'
            self.logger.info(f"ARI file is: {ri_file}")
            output_filename = f"{self.region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{time_info['step_string']}.nc"
            # Step 4 Load regridded ARI data at the same duration as the step Range
            try:
                with xr.open_dataset(os.path.join(ri_filepath, ri_file)) as ri_ds:
                    # replacing -9 with nan
                    ri_ds = ri_ds.where(ri_ds != -9, other=np.nan)
                    # ARI data is in 1000s of inches per HDSC metadata
                    ri_ds = ri_ds/1000
                    # pulling out the RI values
                    ri_da = ri_ds[f'{self.region}{ri_length}yr{int(ri_duration):02d}ha'].values
                    # percentiles 
                    # Back to whole numbers for the percentiles for the interp
                    corrected_percentiles = [p*100 for p in percentiles]
                    selected_percentiles = np.array(corrected_percentiles)
                    # pulling out our percentile values
                    percentile_cube_data = percentile_cube.values
                    # Reshape for vectorized interpolation
                    reshaped_cube = np.moveaxis(percentile_cube_data, 0, -1)  # Shape: (y, x, 7) for easier indexing
                    # Flatten the spatial dimensions for interpolation
                    flat_random_precip = ri_da.flatten()
                    flat_cube = reshaped_cube.reshape(-1, reshaped_cube.shape[-1])
                    # Perform vectorized interpolation for each (y, x) point
                    flat_rank_array = np.array(
                        [
                            np.interp(value, flat_cube[i], selected_percentiles, left=0, right=100)
                            for i, value in enumerate(flat_random_precip)
                        ]
                    )
                    # Reshape back to the original 2D shape
                    rank_array = flat_rank_array.reshape(ri_da.shape)
                    # need the exceedance percentage not the rank
                    rank_array = 100 - rank_array
                    lat =  ri_ds['latitude'].data
                    lon =  ri_ds['longitude'].data
            except Exception as e:
                self.logger.error(f"Error has occurred {e}")
                self.logger.info(f"One possible issue may be that {ri_file} doesn't seem to exist in {ri_filepath}.  Make sure you have downloaded the ARIs and regridded to this model")
                self.logger.info(f"Skipping {ri_length} ARI...")
                continue
                
            # Step 6: Save the rank array to NetCDF
            os.makedirs(exceedance_dir, exist_ok=True)  # Ensure the output directory exists
            # Construct the output file name dynamically
            output_file = os.path.join(exceedance_dir, output_filename)
            # Create an xarray Dataset for saving
            rank_ds = xr.Dataset(
                {
                    "exceedance_perc": (["y", "x"], rank_array)  # Use the dimensions of the rank_array
                },
                coords={
                                "latitude": (["y"], lat),
                                "longitude": (["x"], lon),
                            },
                attrs={
                    "title": f"Rank Percentile for {ri_length}-yr ARI at step {time_info['step_string']}",
                    "description": f"Rank computed from ARI and {ensemble} precipitation percentiles",
                    "units": "rank (percentile index)"
                }
            )
            # Save to NetCDF
            rank_ds.to_netcdf(output_file)
            self.logger.info(f"Rank array saved to {output_file}")

    def extract_forecast_details(self, file_path, hours_back):
        """
        Extract forecast details from a GRIB file.

        :param file_path: Path to the GRIB file.
        :param hours_back: Number of hours to go back for step range calculation.
        :return: Dictionary with extracted details.
        """
        if self.model == "nbm":
            keys = {}
        elif self.model == "eps":
            keys = {'dataType': 'cf'}
        with xr.open_dataset(file_path, engine="cfgrib", filter_by_keys=keys) as ds:
            # Convert "time" to datetime objects
            time_value = pd.to_datetime(ds["time"].values)

            # Convert "step" (nanoseconds) to hours
            if "step" in ds.variables:
                step_values = ds["step"].values
                step_hours = (step_values / np.timedelta64(1, 'h')).astype(int)
                step_string = f"F{step_hours:03}"

                # Create forecast valid times by adding step to time
                valid_time = time_value + pd.Timedelta(hours=step_hours)

                # Calculate the step range dynamically
                min_step = max(0, step_hours - hours_back)
                max_step = step_hours
                step_range = f"{min_step}-{max_step}"

                return {
                    "time_value": time_value,
                    "step_hours": step_hours,
                    "step_string": step_string,
                    "valid_time": valid_time,
                    "step_range": step_range
                }
            else:
                raise ValueError("The 'step' variable is not present in the GRIB file.")

    def create_global_lat_lon_grid(self, geotransform, n_lat, n_lon):
        """
        Create global 1D latitude and longitude arrays based on a geotransform.

        Parameters:
        - geotransform (list or tuple): Geotransform with six values:
        [x_min, x_res, 0, y_max, 0, y_res]

        Returns:
        - lat_array (numpy.ndarray): 1D array of latitude values.
        - lon_array (numpy.ndarray): 1D array of longitude values.
        """
        x_min, x_res, _, y_max, _, y_res = geotransform

        # Generate the longitude and latitude 1D arrays
        lon_array = np.linspace(x_min, x_min + (n_lon - 1) * x_res, n_lon)
        lat_array = np.linspace(y_max, y_max + (n_lat - 1) * y_res, n_lat)

        return lat_array, lon_array
    
    def create_nbm_grid(self, proj_string, geotransform, nrows, ncols):
        """
        Convert projection and geotransform into 2D latitude and longitude grids.
        
        Parameters:
            proj_string (str): WKT string for the projection.
            geotransform (list): Geotransform values [top_left_x, pixel_width, 0, top_left_y, 0, pixel_height].
            nrows (int): Number of rows in the grid.
            ncols (int): Number of columns in the grid.
        
        Returns:
            tuple: 2D numpy arrays (latitudes, longitudes)
        """
        
        # Create the SpatialReference object
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_string)
        # Define the target spatial reference (WGS84: latitude, longitude)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WGS84 (latitude/longitude)

        # Create a transformation object
        transform = osr.CoordinateTransformation(srs, target_srs)

        # Initialize the latitude and longitude arrays
        latitudes = np.zeros((nrows, ncols))
        longitudes = np.zeros((nrows, ncols))

        # Loop through the grid and calculate latitude and longitude for each pixel
        for i in range(nrows):
            for j in range(ncols):
                # Calculate projected coordinates (Easting, Northing) from pixel (i, j)
                easting = geotransform[0] + j * geotransform[1]  # x is affected by the column (j)
                northing = geotransform[3] + i * geotransform[5]  # y is affected by the row (i)
                
                # Transform to geographic coordinates (latitude, longitude)
                lat, lon, _ = transform.TransformPoint(easting, northing)
                
                # Store the latitude and longitude
                latitudes[i, j] = lat
                longitudes[i, j] = lon

        # Flip the latitude grid to match bottom-left origin
        latitudes = np.flipud(latitudes)
        
        # Wrap longitudes into the [0, 360) range
        longitudes = (longitudes + 360) % 360
        
        # Flip the longitude grid to match bottom-left origin
        longitudes = np.flipud(longitudes)

        return latitudes, longitudes

    
    def merge_ari(self):
        """
        Merge regridded ARI (Average Recurrence Interval) regional datasets into a single dataset matching a larger model domain.

        This function takes individual regridded ARI datasets (e.g., for different regions of the CONUS), stitches them together
        to create a cohesive dataset for the entire domain, and outputs the merged dataset in NetCDF format. Once the merge
        is complete, the function cleans up individual regridded files to save disk space.

        Workflow:
            1. Check if the merged dataset already exists; skip processing if it does.
            2. Loop through recurrence intervals and durations.
            3. Open individual NetCDF files for the ARI regions, replace invalid data (-999) with NaN for seamless merging.
            4. Use a reduction operation to iteratively fill gaps by merging datasets.
            5. Save the merged dataset to a NetCDF file.
            6. Clean up individual regional files after successful merging.

        Configuration Keys:
            - ari_regions: Prefixes for ARI regions within the domain (e.g., state codes).
            - regrid_dir: Directory containing regridded NetCDF files.
            - recurrence_intervals_years: List of ARI recurrence intervals in years.
            - durations_hours: List of ARI durations in hours.
            - ensemble_name: Name of the ensemble or domain for merging.

        Notes:
            - This function assumes regional datasets are stored in the `regrid_dir` directory with specific naming conventions.
            - Invalid data (-999) in the datasets is replaced with NaN for accurate merging.
            - The final merged dataset is named using the region, recurrence interval, and duration.

        Output:
            - A single NetCDF file for each combination of recurrence interval and duration, covering the entire domain.
            - Logs indicating the status of merging and cleanup.

        """
        region_config=self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]
        ari_dir = self.config["ensemble"]["regrid_dir"]
        ari_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        ari_durations = self.config["ari_settings"]["durations_hours"]
        ensemble_name = region_config["ensemble_name"]
        # looping through the ari lengths and durations
        for year in ari_lengths:
            for duration in ari_durations:
                #checking to see if we already have conus files
                merged_filename = f"{self.region}{year}yr{duration}ha_regridded_to{ensemble_name}.nc"
                if os.path.exists(os.path.join(ari_dir, merged_filename)):
                    self.logger.info(f"Dataset {os.path.join(ari_dir, merged_filename)} exists. Skipping merge step")
                    continue
                # looping through the conus ari regions
                datasets = []
                for area in region_config["ari_prefixes"]:
                    # getting the appropriate datasets and adding them to the list
                    ari_filename = f"{area}{year}yr{duration}ha_regridded_to{ensemble_name}.nc"
                    self.logger.info(f"Attempting to open {ari_filename}")
                    with xr.open_dataset(os.path.join(ari_dir, ari_filename)) as ds:
                        # filling -999 data with NaN for easier stitching
                        self.logger.info(f"Appending dataset for {area}")
                        datasets.append(ds.where(ds[f"{self.region}{year}yr{duration}ha"]>=0))
                # now merging all the conus datasets into one
                
                # filling the first dataset with the second and so forth until we have a stitched grid...
                self.logger.info(f"Stitching together full datasets from {region_config['ari_prefixes']}")
                full_ds = reduce(lambda left, right: left.fillna(right), datasets)
                #conus_ds = xr.merge(datasets, compat="broadcast_equals")
                full_ds.to_netcdf(os.path.join(ari_dir, merged_filename))
                self.logger.info(f"Done merging full regridded ARI files for {year} year ARI and {duration} hr duration")
        # cleaning up
        for year in ari_lengths:
            for duration in ari_durations:
                for area in region_config["ari_prefixes"]:
                    # getting the appropriate datasets and adding them to the list
                    ari_filename = f"{area}{year}yr{duration}ha_regridded_to{ensemble_name}.nc"
                    self.logger.info(f"Attempting to delete {ari_filename}") 
                    try:
                        os.remove(os.path.join(ari_dir, ari_filename))
                        self.logger.info(f"Done deleting {ari_filename} from {ari_dir}")
                    except FileNotFoundError:
                        self.logger.error(f"{ari_filename} doesn't exist in {ari_dir}.  Skipping this file.")
                        continue
                    

#Example usage script
if __name__ == "__main__":
    # Specify the path to the JSON configuration file
    config_path = r"C:\Users\David.Levin\ensemble_ari\ensemble_ari_config.json"
    datapath = r'C:\Users\David.Levin\ensemble_ari\ensemble_data\ifs\20240925'
    nbmakpath = r'C:\Users\David.Levin\ensemble_ari\ensemble_data\nbmqmd\20201129'
    nbmcopath = r'C:\Users\David.Levin\ensemble_ari\ensemble_data\nbmqmd\20240925'
    nbmhipath = r'C:\Users\David.Levin\ensemble_ari\ensemble_data\nbmqmd\20240823'
    nbmakdatafile = 'blend.t12z.qmd.f060.ak.grib2'
    nbmcodatafile = 'blend.t12z.qmd.f048.co.grib2'
    nbmhidatafile = 'blend.t12z.qmd.f060.hi.grib2'
    datafile_f = '20240925120000-48h-enfo-ef.grib2'
    datafile_b = '20240925120000-24h-enfo-ef.grib2'
    ensfile_f = os.path.join(datapath, datafile_f)
    ensfile_b = os.path.join(datapath, datafile_b)
    # Initialize and run the script
    #ari_script = Ensemble_ARI("eps", "gl")
    #ari_script.download_ari_files()
    #ari_script.regrid_to_base_dataset()
    #ari_script.calc_ari_from_eps_grib(ensfile_f, ensfile_b, ri_duration="24")
    ##ari_script.calc_ari_from_nbm_grib(os.path.join(nbmcopath, nbmcodatafile))
