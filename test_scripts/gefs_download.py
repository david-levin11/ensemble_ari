import os
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import xarray as xr

save_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'
base_runtime = '60'
model = 'gefs'
modelproduct = 'atmos.25'
search_string = ":APCP:"
members= [f"p{x:02}" for x in range(1,31)]
rundate = "2020-11-29 12:00"
rundate_dir = datetime.strptime(rundate, "%Y-%m-%d %H:00").strftime("%Y%m%d")
gefs_dir = rf'C:\Users\David.Levin\ensemble_ari\ensemble_data\{model}\{rundate_dir}'
timesteps = [int(base_runtime)-x for x in range(0,24,3) ]
print(timesteps)
memberlist = []
if not os.path.exists(gefs_dir):
    os.makedirs(gefs_dir, exist_ok=False)
for member in members:
    combined_dataset = None
    # GEFS comes with precip in 3 hourly accumulation time steps
    for timestep in timesteps:
        print(f"now working on member {member} for model {model} and time steps {timestep}")
        #try:
        H = Herbie(rundate, model=model, modelproduct=modelproduct, member=member, fxx=timestep)
        ds = H.xarray(search_string)
        print(ds)
        # Add to the combined dataset
        if combined_dataset is None:
            # Initialize with the first dataset
            combined_dataset = ds
        else:
            # Element-wise addition
            combined_dataset = combined_dataset + ds
            combined_dataset = combined_dataset * 0.03937008
    memberlist.append(combined_dataset)

# concatenate datasets along the "number" dimension
finalds = xr.concat(memberlist, dim="number")
outputfile = f"{rundate_dir}_t00z_{base_runtime}h_{model}.nc"
finalds.to_netcdf(os.path.join(gefs_dir, outputfile))
print(f"Final dataset saved as {outputfile}, in {gefs_dir}")
#     H.download(search_string, save_dir=save_dir, verbose=True)
#     H2 = Herbie(rundate, model=model, modelproduct=modelproduct, member=member, fxx=fxx2)
#     H2.download(search_string, save_dir=save_dir, verbose=True)
#         #break
#     #except Exception as e:
#         #print(e)
#         #print(f"\nNo {model} data found for {rundate}")
# for fl in os.listdir(gefs_dir):
#     flname = fl.split("__")[1]
#     os.rename(os.path.join(gefs_dir, fl), os.path.join(gefs_dir, flname))
#     print(f"Done renaming {fl} to {flname}")