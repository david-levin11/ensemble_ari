import os
import requests
import logging
import json
import zipfile
import numpy as np
import xarray as xr
from datetime import datetime
from osgeo import gdal, osr

class Ensemble_ARI:
    def __init__(self, model, config_path):
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

    def download_ari_files(self):
        base_url = self.config["download"]["base_url"]
        output_dir = self.config["download"]["output_dir"]
        overwrite = self.config["download"]["overwrite_existing"]
        regrid_dir = self.config["ensemble"]["regrid_dir"]
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Ensure regrid directory exists
        os.makedirs(regrid_dir, exist_ok=True)

        state = self.config["ari_settings"]["region"]   # this may be problematic for states other than AK
        recurrence_intervals = self.config["ari_settings"]["recurrence_intervals_years"]
        durations = self.config["ari_settings"]["durations_hours"]

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
        base_datasets = self.config["ensemble"]["name"][self.model]["base_datasets"]
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        regrid_dir = self.config["ensemble"]["regrid_dir"]
        ascii_dir = self.config["download"]["output_dir"]
        overwrite = self.config["ensemble"]["overwrite_existing"]
        state = self.config["ari_settings"]["region"]
        recurrence_intervals = self.config["ari_settings"]["recurrence_intervals_years"]
        durations = self.config["ari_settings"]["durations_hours"]

        for ds in base_datasets:
            self.logger.info(f"Processing base dataset: {ds}")
            # extracting lat and lon arrays for later
            with xr.open_dataset(os.path.join(ensemble_dir, ds),
                                 filter_by_keys={'typeOfLevel': 'heightAboveGround', 'shortName': 'ws'}) as orig_grid:
                lats = orig_grid.latitude
                lons = orig_grid.longitude
                self.logger.info(f"Base grid shape is: {lats.shape} for latitude")
                self.logger.info(f"Base grid shape is: {lats.shape} for longitude")
            for ri in recurrence_intervals:
                for duration in durations:
                    ensemble_name = ds.split('.')[0].split('_')[1]
                    input_filename = f"{state}{ri}yr{duration:02}ha.asc"
                    output_filename = f"{state}{ri}yr{duration:02}ha_regridded_to{ensemble_name}.nc"
                    input_file = os.path.join(ascii_dir, input_filename)
                    output_file = os.path.join(regrid_dir, output_filename)
                    # Skip if file exists and overwrite is False
                    if os.path.exists(output_file) and not overwrite:
                        self.logger.info(f"File already exists, skipping: {output_file}")
                        continue
                    # Logging information for each combination of dataset, recurrence interval, and duration
                    self.logger.info(f"Regridding for Recurrence Interval: {ri} years, Duration: {duration:02} hours")
                    base_grid = os.path.join(ensemble_dir, ds)
                    self.logger.info(f"Opening base grid file: {base_grid}")

                    # Open the target file to extract projection, geotransform, and resolution
                    try:
                        target_dataset = gdal.Open(base_grid)
                        if target_dataset is None:
                            self.logger.error(f"Failed to open the base grid file: {base_grid}")
                            continue

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

    def calc_nbm_ari(self):
        # pulling vars from config
        ensemble_dir = self.config["ensemble"]["ensemble_dir"]
        ri_filepath = self.config["ensemble"]["regrid_dir"]
        base_exceedance_dir = self.config["ensemble"]["base_exceedance_dir"]
        ensemble = self.config["ensemble"]["name"][self.model]["longname"]
        ensemble_shortname = self.config["ensemble"]["name"][self.model]["shortname"]
        valid_hours = self.config["ensemble"]["name"][self.model]["run_hours"]
        region = self.config["ari_settings"]["region"]
        ri_lengths = self.config["ari_settings"]["recurrence_intervals_years"]
        ri_durations = self.config["ari_settings"]["durations_hours"]
        max_hour = self.config["ensemble"]["name"][self.model]["max_hour"] # this may need to be dynamic based on how far out PQPF06 goes
        rundt = self.config["ensemble"]["name"][self.model]["run_dt"]  # need to work around this somehow
        # working with the vars to create additional dynamic ones
        exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), region)
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
                ri_file = f'{region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}{region}.nc'
                print(f"ARI file is: {ri_file}")
                # looping through the QMD files in our model directory
                for trange, tstep in enumerate(steplist):
                    print(f"Time step is: {tstep}")

                    # Step 1 Load QMD file
                    efilename = f'blend.t{datetime.strptime(rundt, "%Y-%m-%d %H:%M").hour}z.qmd.f{str(tstep).zfill(3)}.{region}.grib2'
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
                            data = rank_array[f'{region}{ri_length}yr{int(ri_duration):02d}ha'].data
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
                    output_file = os.path.join(exceedance_dir, f'{region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{tstep:03d}.nc')
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
    
    def calc_ensemble_ari(self):
        if self.model == 'nbm':
            # running appropriate grid calculation script
            self.calc_nbm_ari()
        else:
            print(f"No ARI routing available for {self.model}!  Sorry!")
                        
# Main script
if __name__ == "__main__":
    # Specify the path to the JSON configuration file
    config_path = r"C:\Users\David.Levin\RFCVer\Regrid\Scripts\ensemble_ari_config.json"

    # Initialize and run the script
    ari_script = Ensemble_ARI("nbm", config_path)
    ari_script.download_ari_files()
    ari_script.regrid_to_base_dataset()
    ari_script.calc_ensemble_ari()
