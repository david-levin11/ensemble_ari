import xarray as xr
import os
#import rioxarray
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

############################## elements for config ##############################################################

base_exceedance_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_exceedance_data'

base_graphics_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_exceedance_graphics'

ensemble = 'nbmqmd'

ensemble_shortname = 'nbm'

region = 'ak'

valid_hours = ['00', '06', '12', '18']  # NBM QMD only runs at these hours

ri_duration = '24'

exceedance_dir = os.path.join(os.path.join(base_exceedance_dir, ensemble), region)

graphics_dir = os.path.join(os.path.join(base_graphics_dir, ensemble), region)

ri_lengths = ['2', '5', '10', '25', '50', '100']

rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)

max_hour = 132

extent = [-170, -130, 54, 72]

plotcrs = ccrs.NorthPolarStereo(central_longitude=-150, true_scale_latitude=60)

datacrs = ccrs.PlateCarree()

colors = ['#38a800', '#ffff00', '#ffaa00', '#ff5500', '#ff0000', '#ff73ff', '#e600a9',
                  '#8400a8', '#a51cfc', '#ffffff']

colorvals = [0,5,10,15,20,30,40,50,60,70,100]

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(colorvals, ncolors=cmap.N)
# Generate forecast projections/steps/ranges for Herbie to download data
forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
# creating our graphics dir if not already there
os.makedirs(graphics_dir, exist_ok=True)
# looping through the RI lengths
for ri in ri_lengths:
    # looping through the timesteps
    print(f"Now working on {ri}yr ARI...")
    for tstep in forecast_projections:
        print(f"Now working on time step f{tstep} for {ri}yr ARI...")
        exceedance_file = f"{region}{ri}yr{ri_duration}ha_{ensemble}_{tstep}.nc"
        # Load the resampled dataset
        with xr.open_dataset(os.path.join(exceedance_dir, exceedance_file)) as rank_array:
            # Extract the data variable and 2D coordinates
            data = rank_array["exceedance_perc"]
            latitude = rank_array["latitude"]
            longitude = rank_array["longitude"]

        # Set up the map projection
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={"projection": plotcrs}
        )

        # Plot the data
        c = ax.pcolormesh(
            longitude,
            latitude,
            data,
            transform=datacrs,
            cmap=cmap,
            norm=norm,
        )

        # Add coastlines and features
        ax.coastlines(resolution="10m", color="black")
        ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="gray")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

        # Set the extent to Southeast Alaska
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add a colorbar
        cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label(f"{ri}yr ARI Exceedance Percentage")

        # Add a title
        ax.set_title(f"{ensemble_shortname.upper()} {ri} Year {ri_duration}hr ARI Exceedance Percentage F{tstep}", fontsize=14)

        # Show the plot
        plt.tight_layout()
        graphicname = f"{region}{ri}yr{ri_duration}ha_{ensemble}_{tstep}_exceedancegraphic.png"
        plt.savefig(os.path.join(graphics_dir, graphicname))
        plt.close()

