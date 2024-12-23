import os
import sys
import time
import re
import requests
import logging
import json
import zipfile
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from osgeo import gdal, osr

class Ensemble_ARI:
    def __init__(self, model, region, config_path):
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

    def download_ari_files(self):
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
        # Unzip the file
        self.logger.info(f"Unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            self.logger.info(f"Extracted contents to: {extract_to}")

        # Delete the .zip file
        self.logger.info(f"Deleting .zip file: {zip_path}")
        os.remove(zip_path)
        self.logger.info(f"Deleted: {zip_path}")

    def regrid_to_base_dataset(self):
        #region = self.config["ari_settings"]["region"]
        ari_regions = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["ari_prefixes"]
        region_config = self.config["ensemble"]["name"]["nbm"]["ari_regions"][self.region]
        #base_datasets = self.config["ensemble"]["name"][self.model]["base_datasets"][region]
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        regrid_dir = self.config["ensemble"]["regrid_dir"]
        ascii_dir = self.config["download"]["output_dir"]
        overwrite = self.config["ensemble"]["overwrite_existing"]
        #state = self.config["ari_settings"]["region"]
        recurrence_intervals = self.config["ari_settings"]["recurrence_intervals_years"]
        durations = self.config["ari_settings"]["durations_hours"]
        ds = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["base_dataset"]
        # getting base dataset if it doesn't exist on the system
        if not os.path.exists(os.path.join(ensemble_dir, ds)):
            self.get_base_dataset()
            # adding some time to make sure the process completes
            # curl...while nice to download subsets, isn't great about file locking
            time.sleep(10)
        self.logger.info(f"Processing base dataset: {ds}")
        # extracting lat and lon arrays for later
        with xr.open_dataset(os.path.join(ensemble_dir, ds)) as orig_grid:
            lats = orig_grid.latitude
            lons = orig_grid.longitude
            self.logger.info(f"Base grid shape is: {lats.shape} for latitude")
            self.logger.info(f"Base grid shape is: {lats.shape} for longitude")
        # Looping through the ARI "states/regions" for the ensemble to regrid
        for state in ari_regions:
            for ri in recurrence_intervals:
                for duration in durations:
                    ensemble_name = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["ensemble_name"]
                    input_filename = f"{state}{ri}yr{duration:02}ha.asc"
                    output_filename = f"{state}{ri}yr{duration:02}ha_regridded_to{ensemble_name}.nc"
                    input_file = os.path.join(ascii_dir, input_filename)
                    output_file = os.path.join(regrid_dir, output_filename)
                    # Skip if file exists and overwrite is False
                    if os.path.exists(output_file) and not overwrite:
                        self.logger.info(f"File already exists, skipping: {output_file}")
                        continue
                    # Logging information for each combination of dataset, recurrence interval, and duration
                    self.logger.info(f"Regridding for Recurrence Interval: {ri} years, Duration: {duration:02} hours for region: {state}")
                    base_grid = os.path.join(ensemble_dir, ds)
                    self.logger.info(f"Opening base grid file: {base_grid}")

                    # Open the target file to extract projection, geotransform, and resolution
                    try:
                        target_dataset = gdal.Open(base_grid)
                        if target_dataset is None:
                            self.logger.error(f"Failed to open the base grid file: {base_grid}")
                            continue
                        # have to read in the nbm hawaii projection info from config
                        if self.model == "nbm" and self.region == "hi":
                            srs = osr.SpatialReference()
                            srs.ImportFromProj4(region_config["projection_info"]["proj_string"])
                            grib_projection = srs.ExportToWkt()
                            grib_geotransform = region_config["geotransform"]
                        else:
                            grib_projection = target_dataset.GetProjection()
                            grib_geotransform = target_dataset.GetGeoTransform()
                        grib_xsize = target_dataset.RasterXSize
                        grib_ysize = target_dataset.RasterYSize

                        self.logger.info(f"Base grid projection: {grib_projection}")
                        self.logger.info(f"Base grid geotransform: {grib_geotransform}")
                        self.logger.info(f"Base grid size: {grib_xsize} x {grib_ysize}")
                    except Exception as e:
                        self.logger.error(f"Error opening base grid file: {base_grid}, {e}")
                        continue

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
                    target_dataset = None

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
                            f"{input_filename.split('.')[0]}": (["y", "x"], data)  # Rename data variable
                        },
                        coords={
                            "latitude": lats, #using original dataset coordinates
                            "longitude": lons,
                        },
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

    def calc_nbm_ari_full(self):
        # pulling vars from config
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        ri_filepath = self.config["ensemble"]["regrid_dir"]
        base_exceedance_dir = self.config["ensemble"]["base_exceedance_dir"]
        ensemble = self.config["ensemble"]["name"][self.model]["longname"]
        ensemble_shortname = self.config["ensemble"]["name"][self.model]["shortname"]
        valid_hours = self.config["ensemble"]["name"][self.model]["run_hours"]
        #region = self.config["ari_settings"]["region"]
        ri_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        ri_durations = self.config["ari_settings"]["durations_hours"]
        max_hour = self.config["ensemble"]["name"][self.model]["max_hour"] # this may need to be dynamic based on how far out PQPF06 goes
        rundt = self.config["ensemble"]["name"][self.model]["run_dt"]  # need to work around this somehow
        # working with the vars to create additional dynamic ones
        exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), self.region)
        # looping through the various ARI durations and computing percent exceedance grids
        for ri_duration in ri_durations:
            rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)
            # Generate forecast projections/steps/ranges for Herbie to download data
            forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
            steplist = [int(hour) for hour in forecast_projections]
            stepranges = [f'{int(float(fxx)-float(ri_duration))}-{fxx}' for fxx in steplist]
            # looping through the RI grids
            for ri_length in ri_lengths:
                print(f"Now working on {ri_length} ARI...")
                ri_file = f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}{self.region}.nc'
                print(f"ARI file is: {ri_file}")
                # looping through the QMD files in our model directory
                for trange, tstep in enumerate(steplist):
                    print(f"Time step is: {tstep}")

                    # Step 1 Load QMD file
                    efilename = f'blend.t{datetime.strptime(rundt, "%Y-%m-%d %H:%M").hour}z.qmd.f{str(tstep).zfill(3)}.{self.region}.grib2'
                    efilepath = f'{ensemble_dir}\{ensemble}\{datetime.strptime(rundt,"%Y-%m-%d %H:%M").strftime("%Y%m%d")}'
                    efile = os.path.join(efilepath, efilename)
                    print(f"Now loading {efilename} from {efilepath}")
                    # Creating list of percentiles (1-99)
                    percentiles = list(range(1,100))
                    # Initializing our cube
                    percentile_cube = []

                    # Step 2 Extract each percentile at the appropriate step range (24hr)
                    try:
                        for percentile in percentiles:
                            # Load the file for the current percentile
                            with xr.open_dataset(
                                efile,
                                engine='cfgrib',
                                backend_kwargs={'filter_by_keys': {'stepRange': stepranges[trange], 'percentileValue': percentile}}
                            ) as ds:
                                percentile_cube.append(ds.tp)  # Assuming `tp` is the variable
                    except Exception as e:
                        print(e)
                        print(f"{efilename} doesn't seem to exist in {efilepath}. Skipping this time step...")
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
                                                
                            # Step 5: Compute the rank and extract vals for our dataset
                            rank_array = (ri_ds < percentile_cube).sum(dim="percentileValue")
                            data = rank_array[f'{self.region}{ri_length}yr{int(ri_duration):02d}ha'].data
                            lat =  ri_ds['latitude'].data
                            lon =  ri_ds['longitude'].data
                    except Exception as e:
                        print(e)
                        print(f"{ri_file} doesn't seem to exist in {ri_filepath}.  Make sure you have downloaded the ARIs and regridded to this model")
                        print(f"Skipping {ri_length} ARI...")
                        continue
                    
                    # Step 6: Save the rank array to NetCDF
                    os.makedirs(exceedance_dir, exist_ok=True)  # Ensure the output directory exists
                    # Construct the output file name dynamically
                    output_file = os.path.join(exceedance_dir, f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{tstep:03d}.nc')
                    # Create an xarray Dataset for saving
                    rank_ds = xr.Dataset(
                        {
                            "exceedance_perc": (["y", "x"], data)  # Use the dimensions of the rank_array
                        },
                        coords={
                                        "latitude": (["y", "x"], lat),
                                        "longitude": (["y", "x"], lon),
                                    },
                        attrs={
                            "title": f"Rank Percentile for {ri_length}-yr ARI at step {tstep}",
                            "description": f"Rank computed from ARI and {ensemble} precipitation percentiles",
                            "units": "rank (percentile index)"
                        }
                    )
                    # Save to NetCDF
                    rank_ds.to_netcdf(output_file)
                    print(f"Rank array saved to {output_file}")
    
    def calc_nbm_ari_select(self):
        # pulling vars from config
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        ri_filepath = self.config["ensemble"]["regrid_dir"]
        base_exceedance_dir = self.config["ensemble"]["base_exceedance_dir"]
        ensemble = self.config["ensemble"]["name"][self.model]["longname"]
        ensemble_shortname = self.config["ensemble"]["name"][self.model]["shortname"]
        valid_hours = self.config["ensemble"]["name"][self.model]["run_hours"]
        #region = self.config["ari_settings"]["region"]
        ri_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        ri_durations = self.config["ari_settings"]["durations_hours"]
        max_hour = self.config["ensemble"]["name"][self.model]["max_hour"] # this may need to be dynamic based on how far out PQPF06 goes
        rundt = self.config["ensemble"]["name"][self.model]["run_dt"]  # need to work around this somehow
        percentiles = self.config["ensemble"]["name"][self.model]["percentiles"]
        # working with the vars to create additional dynamic ones
        exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), self.region)
        # looping through the various ARI durations and computing percent exceedance grids
        for ri_duration in ri_durations:
            rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)
            # Generate forecast projections/steps/ranges for Herbie to download data
            forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
            steplist = [int(hour) for hour in forecast_projections]
            stepranges = [f'{int(float(fxx)-float(ri_duration))}-{fxx}' for fxx in steplist]
            # looping through the RI grids
            for ri_length in ri_lengths:
                print(f"Now working on {ri_length} ARI...")
                ri_file = f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}{self.region}.nc'
                print(f"ARI file is: {ri_file}")
                # looping through the QMD files in our model directory
                for trange, tstep in enumerate(steplist):
                    print(f"Time step is: {tstep}")

                    # Step 1 Load QMD file
                    efilename = f'blend.t{datetime.strptime(rundt, "%Y-%m-%d %H:%M").hour}z.qmd.f{str(tstep).zfill(3)}.{self.region}.grib2'
                    efilepath = f'{ensemble_dir}\{ensemble}\{datetime.strptime(rundt,"%Y-%m-%d %H:%M").strftime("%Y%m%d")}'
                    efile = os.path.join(efilepath, efilename)
                    print(f"Now loading {efilename} from {efilepath}")
                    # Initializing our cube
                    percentile_cube = []

                    # Step 2 Extract each percentile at the appropriate step range (24hr)
                    try:
                        for percentile in percentiles:
                            # Load the file for the current percentile
                            with xr.open_dataset(
                                efile,
                                engine='cfgrib',
                                backend_kwargs={'filter_by_keys': {'stepRange': stepranges[trange], 'percentileValue': percentile}}
                            ) as ds:
                                percentile_cube.append(ds.tp)  # Assuming `tp` is the variable
                    except Exception as e:
                        print(e)
                        print(f"{efilename} doesn't seem to exist in {efilepath}. Skipping this time step...")
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
                        print(e)
                        print(f"{ri_file} doesn't seem to exist in {ri_filepath}.  Make sure you have downloaded the ARIs and regridded to this model")
                        print(f"Skipping {ri_length} ARI...")
                        continue
                    
                    # Step 6: Save the rank array to NetCDF
                    os.makedirs(exceedance_dir, exist_ok=True)  # Ensure the output directory exists
                    # Construct the output file name dynamically
                    output_file = os.path.join(exceedance_dir, f'{self.region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{tstep:03d}.nc')
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
                            "title": f"Rank Percentile for {ri_length}-yr ARI at step {tstep}",
                            "description": f"Rank computed from ARI and {ensemble} precipitation percentiles",
                            "units": "rank (percentile index)"
                        }
                    )
                    # Save to NetCDF
                    rank_ds.to_netcdf(output_file)
                    print(f"Rank array saved to {output_file}")
    
    def calc_ensemble_ari(self):
        if self.model == 'nbm' and self.config["ensemble"]["name"][self.model]["full_percentiles"] == "True":
            # running appropriate grid calculation script
            self.calc_nbm_ari_full()
        elif self.model == 'nbm' and self.config["ensemble"]["name"][self.model]["full_percentiles"] == "False":
            # running appropriate grid calculation script
            self.calc_nbm_ari_select()
        else:
            print(f"No ARI routine available for {self.model}!  Sorry!")

    def download_subset(self, remote_url, local_dir, local_filename, model, search_string):
        print(f"  > Downloading a subset of {model} gribs to {local_dir}")
        #making sure local dir exists
        os.makedirs(local_dir, exist_ok=True)
        local_file = os.path.join(local_dir, local_filename)
        idx = remote_url+".idx"
        r = requests.get(idx)
        if not r.ok:
            print('     ❌ SORRY! Status Code:', r.status_code, r.reason)
            print(f'      ❌ It does not look like the index file exists: {idx}')
            
        lines = r.text.split('\n')
        expr = re.compile(search_string)
        byte_ranges = {}
        for n, line in enumerate(lines, start=1):
        # n is the line number (starting from 1) so that when we call for
        # `lines[n]` it will give us the next line. (Clear as mud??)

            # Use the compiled regular expression to search the line
            if expr.search(line):
                # aka, if the line contains the string we are looking for...

                # Get the beginning byte in the line we found
                parts = line.split(':')
                rangestart = int(parts[1])

                # Get the beginning byte in the next line...
                if n+1 < len(lines):
                    # ...if there is a next line
                    parts = lines[n].split(':')
                    rangeend = int(parts[1])
                else:
                    # ...if there isn't a next line, then go to the end of the file.
                    rangeend = ''

                # Store the byte-range string in our dictionary,
                # and keep the line information too so we can refer back to it.
                byte_ranges[f'{rangestart}-{rangeend}'] = line
                #print(line)
        for i, (byteRange, line) in enumerate(byte_ranges.items()):

            if i == 0:
                # If we are working on the first item, overwrite the existing file.
                curl = f'curl -s --range {byteRange} {remote_url} > {local_file}'
            else:
                # If we are working on not the first item, append the existing file.
                curl = f'curl -s --range {byteRange} {remote_url} >> {local_file}'

            #print(f'  Downloading GRIB line [{num:>3}]: variable={var}, level={level}, forecast={forecast}')
            os.system(curl)

        if os.path.exists(local_file):
            print(f'      ✅ Success! Searched for [{search_string}] and got [{len(byte_ranges)}] GRIB fields and saved as {local_file}')
            return local_file
        else:
            print(print(f'      ❌ Unsuccessful! Searched for [{search_string}] and did not find anything!'))
    
    def download_base_tif(self, remote_url, local_filename, local_dir):
        try:
            response = requests.get(remote_url, stream=True)
            response.raise_for_status()
            with open(os.path.join(local_dir, local_filename), 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"{local_filename} successfully downloaded and saved to {local_dir}")
        except requests.exceptions.RequestException as e:
            print(f"An error occured while downloading {remote_url} to {local_filename}: {e}")


    def get_base_dataset(self):
        runtime = self.config["ensemble"]["name"][self.model]["base_runtime"]
        runprojection = self.config["ensemble"]["name"][self.model]["base_projection"]
        runprojection_string = f"{runprojection:03d}"
        base_url = self.config["ensemble"]["name"][self.model]["base_url"]
        ensemble_dir = self.config["ensemble"]["name"][self.model]["base_dir"]
        search_string = self.config["ensemble"]["name"][self.model]["base_search_string"] 
        attempts = 1   
        while attempts <= 10:
            print(f"Attempting to download {self.model} base file: try number {attempts}")
            utc_yesterday = datetime.now(timezone.utc) - timedelta(days=attempts)

            rundate = utc_yesterday.strftime('%Y%m%d')
            # nbm has different "domains" for ak and conus
            #if self.model == "nbm" and self.region != "hi":
            if self.model == "nbm":
                remote_url = f"{base_url}blend.{rundate}/{runtime}/core/blend.t{runtime}z.core.f{runprojection_string}.{self.region}.grib2"
                local_filename = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["base_dataset"]
                print(remote_url)
                try:
                    self.download_subset(remote_url, ensemble_dir, local_filename, self.model, search_string)
                    break
                except requests.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                    print(f"File not found for {remote_url}. Trying the previous day...")
            else:
                print(f"No url structure has been set up yet for {self.model}.  Check get_base_dataset() and add functionality!")
                sys.exit()
            attempts += 1
        

# Main script
if __name__ == "__main__":
    # Specify the path to the JSON configuration file
    config_path = r"C:\Users\David.Levin\ensemble_ari\ensemble_ari_config.json"

    # Initialize and run the script
    ari_script = Ensemble_ARI("nbm", "ak", config_path)
    ari_script.download_ari_files()
    ari_script.regrid_to_base_dataset()
    ari_script.calc_ensemble_ari()
