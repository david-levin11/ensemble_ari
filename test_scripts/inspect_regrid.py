import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cartopy import crs as ccrs
from cartopy import feature as cfeature

# File paths
input_dir = r"C:\Users\David.Levin\ensemble_ari\regridded_ari"
#input_dir = r"C:\Users\David.Levin\ensemble_ari\ensemble_data\gefs\20201129"
#input_dir = r"C:\Users\David.Levin\ensemble_ari\nc_data"
#nc_file = "20201129_t00z_60h_gefs.nc"
#nc_file = "ak2yr24ha.nc"
nc_file = 'ak2yr24ha_regridded_to_nbmak.nc'
file_path = os.path.join(input_dir, nc_file)

# Open the NetCDF file
ds = xr.open_dataset(file_path)

# Inspect dataset
print("Dataset dimensions:", ds.dims)
print("Dataset variables:", ds.data_vars)

#datacrs = ccrs.Mercator(central_longitude=-160, latitude_true_scale=20)
#datacrs = ccrs.LambertConformal(central_longitude=-95, central_latitude=25)
datacrs = ccrs.PlateCarree()
plotcrs = datacrs
#plotcrs = ccrs.NorthPolarStereo(central_longitude=-150, true_scale_latitude=60)
# Extract latitude, longitude, and first data variable for visualization
lats = ds["latitude"].values if "latitude" in ds else ds["lat"].values
lons = ds["longitude"].values if "longitude" in ds else ds["lon"].values
data_var_name = list(ds.data_vars.keys())[0]  # Take the first data variable
#data = ds[data_var_name][10,:,:].values
data = ds["ak2yr24ha"].values
norm = Normalize(vmin=0, vmax=4)
#lats = lats.reshape(data.shape)
#lons = lons.reshape(data.shape)
#lons, lats = np.meshgrid()
print(lats.shape)
print(lons.shape)
print(data.shape)
good_data = data
print(good_data)
# Plotting
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=plotcrs)

# Add geographical features
ax.add_feature(cfeature.LAND, edgecolor="black")
ax.add_feature(cfeature.COASTLINE)
#longrid, latgrid = np.meshgrid(lons, lats)
#ax.scatter(longrid, latgrid)
# Zoom on Hawaii (extent)
#ax.set_extent([-161, -154, 18, 23], crs=ccrs.PlateCarree())
ax.set_extent([-180, -100, 40, 78], crs=ccrs.PlateCarree())
#ax.set_extent([-170, -65, 18, 72])
#Plot the data
mesh = ax.pcolormesh(
    lons,
    lats,
    data,  # Assuming time as the first dimension; adjust as needed
    transform=ccrs.PlateCarree(),
    cmap="viridis"
)
plt.colorbar(mesh, ax=ax, orientation="horizontal", label=data_var_name)

# Add a title
plt.title(f"{data_var_name} over nbmak domain")
plt.show()
