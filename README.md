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
