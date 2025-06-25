import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from herbie import Herbie

ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'

ensemble = 'nbmqmd'

region = 'co'

ri_duration = '06'

rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)

max_hour = 132

# Generate forecast projections/steps/ranges for Herbie to download data
forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
steplist = [int(hour) for hour in forecast_projections]
stepranges = [f'{int(float(fxx)-float(ri_duration))}-{fxx}' for fxx in steplist]
# NBM model run datetime
rundt = '2024-09-25 12:00'
# Convert runtime to a datetime object
rundt_dt = datetime.strptime(rundt, '%Y-%m-%d %H:%M')
# Calculate valid times
valid_times = {hour: (rundt_dt + timedelta(hours=int(hour))).strftime('%Y-%m-%d %H:%M') for hour in forecast_projections}
# Percentiles we want
percentiles = [1,5,10,25,50,75,90,95,99]

for i, hour in enumerate(steplist):
    # #creating search strings
    # match_this = [f":APCP:surface:{stepranges[i]} hour acc fcst:{x}% level" for x in percentiles]
    # search = f"({'|'.join(match_this)})"
    print(f'Now downloading {ensemble} file for {rundt} at forecast projection {hour} valid at {valid_times[forecast_projections[i]]}')
    # downloading only the percentiles we want to save disk space
    H = Herbie(rundt, model=ensemble, fxx=hour, product=region)
    ensemble_file = H.download(save_dir=ensemble_dir, verbose=True)

