import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from herbie import Herbie

ensemble_dir = r'C:\Users\David.Levin\RFCVer\Regrid\ensemble_data'

ensemble = 'nbmqmd'

region = 'ak'

ri_duration = '24'

rolling_duration = 12 if int(ri_duration) >= 24 else int(ri_duration)

max_hour = 132

# Generate forecast projections/steps/ranges for Herbie to download data
forecast_projections = [f"{hour:03d}" for hour in range(int(ri_duration), max_hour + int(ri_duration), rolling_duration)]
steplist = [int(hour) for hour in forecast_projections]
stepranges = [f'{int(float(fxx)-float(ri_duration))}-{fxx}' for fxx in steplist]
# NBM model run datetime
rundt = '2020-11-29 12:00'
# Convert runtime to a datetime object
rundt_dt = datetime.strptime(rundt, '%Y-%m-%d %H:%M')
# Calculate valid times
valid_times = {hour: (rundt_dt + timedelta(hours=int(hour))).strftime('%Y-%m-%d %H:%M') for hour in forecast_projections}


for i, hour in enumerate(steplist):
    print(f'Now downloading {ensemble} file for {rundt} at forecast projection {hour} valid at {valid_times[forecast_projections[i]]}')
    H = Herbie(rundt, model=ensemble, fxx=hour, product=region)
    ensemble_file = H.download(save_dir=ensemble_dir, verbose=True)

