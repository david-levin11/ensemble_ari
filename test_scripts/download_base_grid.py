import os
import re
import requests
import time
from datetime import datetime, timedelta, timezone


def download_subset(remote_url, local_dir, local_filename, model, search_string):
    print(f"  > Downloading a subset of {model} gribs to {local_dir}")
    #making sure local dir exists
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, local_filename)
    idx = remote_url+".idx"
    r = requests.get(idx)
    if not r.ok:
        print('     ❌ SORRY! Status Code:', r.status_code, r.reason)
        print(f'      ❌ It does not look like the index file exists: {idx}')
        
    lines = r.text.split('\n')
    expr = re.compile(search_string)
    byte_ranges = {}
    for n, line in enumerate(lines, start=1):
      # n is the line number (starting from 1) so that when we call for
      # `lines[n]` it will give us the next line. (Clear as mud??)

      # Use the compiled regular expression to search the line
      if expr.search(line):
          # aka, if the line contains the string we are looking for...

          # Get the beginning byte in the line we found
          parts = line.split(':')
          rangestart = int(parts[1])

          # Get the beginning byte in the next line...
          if n+1 < len(lines):
              # ...if there is a next line
              parts = lines[n].split(':')
              rangeend = int(parts[1])
          else:
              # ...if there isn't a next line, then go to the end of the file.
              rangeend = ''

          # Store the byte-range string in our dictionary,
          # and keep the line information too so we can refer back to it.
          byte_ranges[f'{rangestart}-{rangeend}'] = line
          #print(line)
    for i, (byteRange, line) in enumerate(byte_ranges.items()):

        if i == 0:
            # If we are working on the first item, overwrite the existing file.
            curl = f'curl -s --range {byteRange} {remote_url} > {local_file}'
        else:
            # If we are working on not the first item, append the existing file.
            curl = f'curl -s --range {byteRange} {remote_url} >> {local_file}'

        #print(f'  Downloading GRIB line [{num:>3}]: variable={var}, level={level}, forecast={forecast}')
        os.system(curl)

    if os.path.exists(local_file):
      print(f'      ✅ Success! Searched for [{search_string}] and got [{len(byte_ranges)}] GRIB fields and saved as {local_file}')
      return local_file
    else:
        print(print(f'      ❌ Unsuccessful! Searched for [{search_string}] and did not find anything!'))


if __name__ == "__main__":

    model = "nbm"

    region = "ak"

    

    runtime = "12"

    runprojection = 6

    runprojection_string = f"{runprojection:03d}"

    base_url = "https://noaa-nbm-grib2-pds.s3.amazonaws.com/"

    attempts = 1
    
    ensemble_dir = r'C:\Users\David.Levin\ensemble_ari\ensemble_data'

    search_string = ":TMP:2 m"
    while attempts <= 10:
        print(f"Attempting to download {model} base file: try number {attempts}")
        utc_yesterday = datetime.now(timezone.utc) - timedelta(days=attempts)

        rundate = utc_yesterday.strftime('%Y%m%d')

        if model == "nbm" and region == "ak":
            remote_url = f"{base_url}blend.{rundate}/{runtime}/core/blend.t{runtime}z.core.f{runprojection_string}.{region}.grib2"
            local_filename = "base_nbmak.grib2"
        print(remote_url)
        try:
            download_subset(remote_url, ensemble_dir, local_filename, model, search_string)
            break
        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"File not found for {remote_url}. Trying the previous day...")
        except Exception as err:
            print(f"An error occurred: {err}")
        # Go back one day
        attempts += 1    
    