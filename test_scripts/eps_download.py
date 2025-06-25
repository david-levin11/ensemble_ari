import os
from datetime import datetime, timedelta, timezone
from herbie import Herbie

save_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'
base_runtime = '48'
model = 'eps'
modeltype = 'ifs'
modelproduct = 'enfo'
search_string = ":tp:"
attempts = 1   
while attempts <= 10:
    utc_yesterday = datetime.now(timezone.utc) - timedelta(days=attempts)
    
    # nbm has different "domains" for ak and conus
    #if self.model == "nbm" and self.region != "hi":
    if model == "eps":
        rundate = "2024-09-25 12:00"
        fxx = int(base_runtime)
        fxx2=fxx-24
        try:
            H = Herbie(rundate, model=modeltype, product = modelproduct, fxx=fxx)
            H.download(search_string, save_dir=save_dir, verbose=True)
            H2 = Herbie(rundate, model=modeltype, product = modelproduct, fxx=fxx2)
            H2.download(search_string, save_dir=save_dir, verbose=True)
            break
        except Exception as e:
            print(e)
            print(f"\nNo {model} data found for {rundate}")
    attempts += 1

        # remote_url = f"{base_url}blend.{rundate}/{runtime}/core/blend.t{runtime}z.core.f{runprojection_string}.{self.region}.grib2"
        # local_filename = self.config["ensemble"]["name"][self.model]["ari_regions"][self.region]["base_dataset"]
        # self.logger.info(f"Attempting to get data at {remote_url}")