import numpy as np
import xarray as xr
import os


# GeoTransform values
x_min = -180   # Origin longitude
x_res = 0.5       # Longitude resolution
y_max = 90     # Origin latitude
y_res = -0.5      # Latitude resolution (negative since latitude decreases)

# Calculate the number of points for latitude and longitude
n_lon = 720  # Full longitude range (-180 to 180)
n_lat = 361  # Full latitude range (90 to -90)

# Generate the longitude and latitude 1D arrays
lon_array = np.linspace(x_min, x_min + (n_lon - 1) * x_res, n_lon)
lat_array = np.linspace(y_max, y_max + (n_lat - 1) * y_res, n_lat)

# Print results
print("Longitude array:", lon_array)
print("Latitude array:", lat_array)
print(f"Latitude shape: {lat_array.shape}")
print(f"Longitude shape: {lon_array.shape}")

ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'
ds = 'base_gefs.grib2'
# extracting lat and lon arrays for later
with xr.open_dataset(os.path.join(ensemble_dir, ds),filter_by_keys={'typeOfLevel': 'surface'}) as orig_grid:
    lats = orig_grid.latitude.values
    lons = orig_grid.longitude.values

print(f"Latitude array: {lats}")
print(f"Longitude array: {lons}")
print(f"Latitude shape: {lats.shape}")
print(f"Longitude shape: {lons.shape}")
