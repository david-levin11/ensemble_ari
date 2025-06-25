import os
import cfgrib
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe

ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\nc_data'
base_dir = r"C:\Users\David.Levin\ensemble_ari\ensemble_data"
base_model = "nbmak"
base_file = f"base_{base_model}.grib2"
ari_len = 2
ari_dur = 24
ari_area = "ak"
ari_field = f"gl{ari_len}yr{ari_dur}ha"
ari_file = f"{ari_field}.nc"

with xr.open_dataset(os.path.join(ensemble_dir, ari_file)) as ds:
    print(ds)
    in_grid = {'lon': ds['lon'].values, 'lat': ds['lat'].values}
    arifield = ds[ari_field]
with xr.open_dataset(os.path.join(base_dir, base_file), engine="cfgrib") as base_ds:
    print(base_ds)
    out_grid = {'lon': base_ds['longitude'].values,'lat': base_ds['latitude'].values}

print(f'ARI grid is: {in_grid}')
print('\n\n')
print(f'NBM grid is: {out_grid}')
weights_file = "nbmak_weights.nc"
if not os.path.exists(weights_file):
    # creating the regridder
    regridder = xe.Regridder(in_grid, out_grid, 'bilinear', filename=weights_file, unmapped_to_nan=True)
else:
    regridder = xe.Regridder(in_grid, out_grid, 'bilinear', filename=weights_file, reuse_weights=True, unmapped_to_nan=True)

ds2 = regridder(arifield)

# Rename the variable in the Dataset
ds2 = ds2.rename(ari_field)

print(ds2)

# Define the relative path to the directory
output_dir = "./regridded_ari"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Construct the full output path
outfilename = os.path.join(output_dir, f"{ari_field}_regridded_to_{base_model}.nc")


ds2.to_netcdf(outfilename)


# print(f"Done regridding {ari_file} to {outfilename}")
