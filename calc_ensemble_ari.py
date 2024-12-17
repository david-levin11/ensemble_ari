import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

############################## elements for config ##############################################################

ensemble_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_data'

ri_filepath = r'C:\Users\David.Levin\RFCVer\Regrid\regridded_ari'

base_exceedance_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_exceedance_data'

ensemble = 'nbmqmd'

ensemble_shortname = 'nbm'

region = 'ak'

valid_hours = ['00', '06', '12', '18']  # NBM QMD only runs at these hours

ri_duration = '24'

exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), region)

ri_lengths = ['2', '5', '10', '25', '50', '100']

rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)

max_hour = 132
# Generate forecast projections/steps/ranges for Herbie to download data
forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
steplist = [int(hour) for hour in forecast_projections]
stepranges = [f'{int(float(fxx)-float(ri_duration))}-{fxx}' for fxx in steplist]
# NBM model run datetime
rundt = '2020-11-29 12:00'  # how do we come up with the appropriate datetime to look for?

#################################### test functions ##############################################################

def plot_percentile_zoomed(percentile_cube, selected_percentile, alaska_extent=[-170, -130, 50, 72]):
    """
    Plot a selected percentile from the percentile cube, zoomed to Alaska.

    Parameters:
        percentile_cube (xarray.DataArray): The combined cube of percentiles.
        selected_percentile (float): The percentile to plot.
        alaska_extent (list): The geographical extent for Alaska [min_lon, max_lon, min_lat, max_lat].
    """
    plotcrs = ccrs.NorthPolarStereo(central_longitude=-150, true_scale_latitude=60)
    # Define a custom colormap (similar to your attached graphic)
    colors = [
        (0.0, "white"),    # Low precipitation
        (1.0, "green"),
        (2.0, "yellow"),
        (3.0, "orange"),
        (4.0, "red"),
        (5.0, "pink"),
        (7.0, "purple"),
        (10.0, "white")    # High precipitation
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    # Ensure the selected percentile is in the cube
    if selected_percentile not in percentile_cube.percentileValue.values:
        raise ValueError(f"Percentile {selected_percentile} not found in the data cube.")

    # Extract the data for the selected percentile
    data = percentile_cube.sel(percentileValue=selected_percentile)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': plotcrs})
    
    # Plot the data
    p = ax.pcolormesh(
        data.longitude, data.latitude, data,
        cmap=custom_cmap, shading="auto",
        transform=ccrs.PlateCarree()
    )

    # Add features and gridlines
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())

    # Add colorbar
    cbar = fig.colorbar(p, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Precipitation (inches)')

    # Add title
    ax.set_title(f"{selected_percentile}th Percentile Precipitation", fontsize=14)

    plt.show()

   
def plot_percentile_and_ari(percentile_cube, selected_percentile, ari_data, ari_lat, ari_lon, alaska_extent=[-170, -130, 50, 72]):
    """
    Plot a selected percentile from the percentile cube and overlay or compare it with the ARI data.

    Parameters:
        percentile_cube (xarray.DataArray): The combined cube of percentiles.
        selected_percentile (float): The percentile to plot.
        ari_data (numpy.ndarray): The recurrence interval (ARI) data.
        ari_lat (numpy.ndarray): Latitude values for the ARI data.
        ari_lon (numpy.ndarray): Longitude values for the ARI data.
        alaska_extent (list): The geographical extent for Alaska [min_lon, max_lon, min_lat, max_lat].
    """
    # Ensure the selected percentile is in the cube
    if selected_percentile not in percentile_cube.percentileValue.values:
        raise ValueError(f"Percentile {selected_percentile} not found in the data cube.")

    # Extract the data for the selected percentile
    data = percentile_cube.sel(percentileValue=selected_percentile)

    # Define a custom colormap
    colors = [
        (0.0, "green"),    # Low precipitation
        (0.2, "yellow"),
        (0.4, "orange"),
        (0.6, "red"),
        (0.8, "purple"),
        (1.0, "violet")    # High precipitation
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the selected percentile
    ax = axes[0]
    p = ax.pcolormesh(
        data.longitude, data.latitude, data,
        cmap=custom_cmap, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())
    cbar = fig.colorbar(p, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Percentile Precipitation (inches)')
    ax.set_title(f"{selected_percentile}th Percentile Precipitation", fontsize=14)

    # Plot the ARI data
    ax = axes[1]
    p2 = ax.pcolormesh(
        ari_lon, ari_lat, ari_data,
        cmap=custom_cmap, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())
    cbar2 = fig.colorbar(p2, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar2.set_label('Recurrence Interval (inches)')
    ax.set_title("Recurrence Interval (ARI)", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_percentile_ari_and_rank(percentile_cube, selected_percentile, ari_data, ari_lat, ari_lon, rank_array, alaska_extent=[-170, -130, 50, 72]):
    """
    Plot a selected percentile from the percentile cube, ARI data, and the ranked ARI array.

    Parameters:
        percentile_cube (xarray.DataArray): The combined cube of percentiles.
        selected_percentile (float): The percentile to plot.
        ari_data (xarray.DataArray): The recurrence interval (ARI) data.
        ari_lat (numpy.ndarray): Latitude values for the ARI data.
        ari_lon (numpy.ndarray): Longitude values for the ARI data.
        rank_array (xarray.DataArray): The ranked ARI array.
        alaska_extent (list): The geographical extent for Alaska [min_lon, max_lon, min_lat, max_lat].
    """
    # Ensure the selected percentile is in the cube
    if selected_percentile not in percentile_cube.percentileValue.values:
        raise ValueError(f"Percentile {selected_percentile} not found in the data cube.")

    # Extract the data for the selected percentile
    data = percentile_cube.sel(percentileValue=selected_percentile)

    # Define a custom colormap
    colors = [
        (0.0, "green"),    # Low precipitation
        (0.2, "yellow"),
        (0.4, "orange"),
        (0.6, "red"),
        (0.8, "purple"),
        (1.0, "violet")    # High precipitation
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the selected percentile
    ax = axes[0]
    p = ax.pcolormesh(
        data.longitude, data.latitude, data,
        cmap=custom_cmap, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())
    cbar = fig.colorbar(p, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Percentile Precipitation (inches)')
    ax.set_title(f"{selected_percentile}th Percentile Precipitation", fontsize=14)

    # Plot the ARI data
    ax = axes[1]
    p2 = ax.pcolormesh(
        ari_lon, ari_lat, ari_data,
        cmap=custom_cmap, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())
    cbar2 = fig.colorbar(p2, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar2.set_label('Recurrence Interval (inches)')
    ax.set_title("Recurrence Interval (ARI)", fontsize=14)


    colors = ['#38a800', '#ffff00', '#ffaa00', '#ff5500', '#ff0000', '#ff73ff', '#e600a9',
                  '#8400a8', '#a51cfc', '#ffffff']

    colorvals = [0,5,10,15,20,30,40,50,60,70,100]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(colorvals, ncolors=cmap.N)
    # Plot the ranked ARI array
    ax = axes[2]
    p3 = ax.pcolormesh(
        ari_lon, ari_lat, rank_array,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.LAKES, edgecolor='black', zorder=1)
    ax.set_extent(alaska_extent, crs=ccrs.PlateCarree())
    cbar3 = fig.colorbar(p3, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar3.set_label('Ranked ARI (Index)')
    ax.set_title("Ranked ARI Array", fontsize=14)

    plt.tight_layout()
    plt.show()

# Example usage
# plot_percentile_ari_and_rank(percentile_cube, selected_percentile=90, ari_data=ari_data_arr.data, ari_lat=lat, ari_lon=lon, rank_array=rank_array.data)


# ################################### process for nbm QMD files ####################################################
# # looping through the RI grids
# for ri_length in ri_lengths:
#     print(f"Now working on {ri_length} ARI...")
#     ri_file = f'{region}{ri_length}yr{int(ri_duration):02d}ha_regridded_to{ensemble_shortname}{region}.nc'
#     print(f"ARI file is: {ri_file}")
#     # looping through the QMD files in our model directory
#     for trange, tstep in enumerate(steplist):
#         print(f"Time step is: {tstep}")

#         # Step 1 Load QMD file
#         efilename = f'blend.t{datetime.strptime(rundt, "%Y-%m-%d %H:%M").hour}z.qmd.f{str(tstep).zfill(3)}.{region}.grib2'
#         efilepath = f'{ensemble_dir}\{ensemble}\{datetime.strptime(rundt,"%Y-%m-%d %H:%M").strftime("%Y%m%d")}'
#         efile = os.path.join(efilepath, efilename)
#         print(f"Now loading {efilename} from {efilepath}")
#         # Creating list of percentiles (1-99)
#         percentiles = list(range(1,100))
#         # Initializing our cube
#         percentile_cube = []

#         # Step 2 Extract each percentile at the appropriate step range (24hr)
#         try:
#             for percentile in percentiles:
#                 # Load the file for the current percentile
#                 with xr.open_dataset(
#                     efile,
#                     engine='cfgrib',
#                     backend_kwargs={'filter_by_keys': {'stepRange': stepranges[trange], 'percentileValue': percentile}}
#                 ) as ds:
#                     percentile_cube.append(ds.tp)  # Assuming `tp` is the variable
#         except Exception as e:
#             print(e)
#             print(f"{efilename} doesn't seem to exist in {efilepath}. Skipping this time step...")
#             continue

#         # Step 3 Combine into a single cube
#         percentile_cube = xr.concat(percentile_cube, dim='percentileValue')
#         percentile_cube = percentile_cube.assign_coords(percentileValue=percentiles)
#         percentile_cube = percentile_cube*0.03937

#         # Step 4 Load regridded ARI data at the same duration as the step Range
#         try:
#             with xr.open_dataset(os.path.join(ri_filepath, ri_file)) as ri_ds:
#                 # replacing -9 with nan
#                 ri_ds = ri_ds.where(ri_ds != -9, other=np.nan)
#                 # ARI data is in 1000s of inches per HDSC metadata
#                 ri_ds = ri_ds/1000
                                    
#                 # Step 5: Compute the rank and extract vals for our dataset
#                 rank_array = (ri_ds < percentile_cube).sum(dim="percentileValue")
#                 data = rank_array[f'{region}{ri_length}yr{int(ri_duration):02d}ha'].data
#                 lat =  ri_ds['latitude'].data
#                 lon =  ri_ds['longitude'].data
#         except Exception as e:
#             print(e)
#             print(f"{ri_file} doesn't seem to exist in {ri_filepath}.  Make sure you have downloaded the ARIs and regridded to this model")
#             print(f"Skipping {ri_length} ARI...")
#             continue
        
#         # Step 6: Save the rank array to NetCDF
#         os.makedirs(exceedance_dir, exist_ok=True)  # Ensure the output directory exists
#         # Construct the output file name dynamically
#         output_file = os.path.join(exceedance_dir, f'{region}{ri_length}yr{int(ri_duration):02d}ha_{ensemble}_{tstep:03d}.nc')
#         # Create an xarray Dataset for saving
#         rank_ds = xr.Dataset(
#             {
#                 "exceedance_perc": (["y", "x"], data)  # Use the dimensions of the rank_array
#             },
#             coords={
#                             "latitude": (["y", "x"], lat),
#                             "longitude": (["y", "x"], lon),
#                         },
#             attrs={
#                 "title": f"Rank Percentile for {ri_length}-yr ARI at step {tstep}",
#                 "description": f"Rank computed from ARI and {ensemble} precipitation percentiles",
#                 "units": "rank (percentile index)"
#             }
#         )
#         # Save to NetCDF
#         rank_ds.to_netcdf(output_file)
#         print(f"Rank array saved to {output_file}")


################################### process for nbm QMD files with limited percentiles ####################################################
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
        # only taking select percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
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
                ri_da = ri_ds[f'{region}{ri_length}yr{int(ri_duration):02d}ha'].values
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