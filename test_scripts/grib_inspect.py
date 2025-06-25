import os
import cfgrib
import xarray as xr
import numpy as np
import pandas as pd


ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\nc_data'
ensemble_file_end = '20240925120000-48h-enfo-ef.grib2'
ensemble_file_begin = '20240925120000-24h-enfo-ef.grib2'
gefs_file = '20201129_t00z_60h_gefs.nc'
ari_file = "ak2yr24ha.nc"
rrfs_file = 
#percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# # opening the first time step
# with xr.open_dataset(os.path.join(ensemble_dir, ensemble_file_end), engine="cfgrib", filter_by_keys = {'dataType': 'pf'}) as ds:
#     #extract the precipitation
#     tp_end = ds.tp 

# # opening the previous time step
# with xr.open_dataset(os.path.join(ensemble_dir, ensemble_file_begin), engine="cfgrib", filter_by_keys = {'dataType': 'pf'}) as ds:
#     #extract the precipitation
#     tp_begin = ds.tp 

# # calculate the 24hr precip (and converting to inches)
# tp_accum = (tp_end-tp_begin)*39.3701
# print(tp_accum)
# # calculating the percentiles
# tp_percentiles = tp_accum.quantile(q=percentiles, dim="number")
# print(tp_percentiles)
with xr.open_dataset(os.path.join(ensemble_dir, ari_file)) as ds:
    print(ds)