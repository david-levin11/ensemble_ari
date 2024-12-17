Rank Exceedance Percentage Tool

This repository provides tools for processing precipitation forecast data, calculating rank exceedance percentages, and exporting results as NetCDF and GeoTIFF files. It is designed to work with NOAA/NWS datasets, such as GRIB2 files (e.g., QMD outputs) and ARI (Average Recurrence Interval) grids.

Overview
The project processes precipitation percentiles, compares them against ARI grids, and computes exceedance percentages for specific recurrence intervals. Key features include:

Precipitation Percentile Interpolation:
Interpolates observed precipitation against predefined percentiles to determine exceedance rank values.

Clipping and Bounds Handling:
Ranks below the 5th percentile are clipped to 0, while those above the 95th percentile are set to 100.

Projection Assignment:
Reads the coordinate reference system (CRS) and GeoTransform from original GRIB2 files and assigns them to GeoTIFF outputs.

Output Formats:
Outputs data as:

NetCDF: Retains spatially structured exceedance percentages.
GeoTIFF: Provides geospatially referenced raster data for GIS tools.
Dependencies
The project requires the following Python libraries:

xarray: For handling multi-dimensional NetCDF data.
numpy: For numerical operations.
gdal (via osgeo): For reading/writing GeoTIFFs and handling projections.
cfgrib: For extracting data from GRIB2 files.

Usage Instructions
1. Data Preparation
Almost all data preparation can be handled in ensemble_ari_config.json.

Currently this script is only set up to use the NWS National Blend of Models (NBM)

You will need a base NBM grid (I use a 2m temperature grid) for the region of choice.  It will need to be placed in your ensemble data directory (from the config).

ensemble_download.py will then download the NBM run of your choice to the specified directory in the config.  

From there you can regrid and interpolate the ARI data using ensemble_ari_final.py.  This script grabs the ASCII ARI data from https://hdsc.nws.noaa.gov/pfds/pfds_gis.html, 
regrids to the NBM domain using gdal and then interpolates the ARI grids into the NBM QMD precipiation percentile cube.

You can use the plotting scripts to view the output with the appropriate color scale.

NOTE:  This script is only set up for the Alaska domain for the NBM.  To modify it for the CONUS some additional work will need to be done to stitch the ARI grids together from 
the various CONUS regions.
