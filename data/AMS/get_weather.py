import os 
import pandas as pd
import datetime
import pytz
from meteostat import Hourly
import pdb

path = './July/'
files_list = os.listdir(path)
id_ = '06240'
kmh2kts  = 1/1.852

for File in files_list:
    data         = pd.read_csv(path+File)
    timestamp1   = data['snapshot_id'].iloc[0]
    date_time    = datetime.datetime.fromtimestamp(timestamp1)
    local_tzinfo = pytz.timezone('Europe/Amsterdam')
    local_time   = date_time.astimezone(local_tzinfo)
    local_time   = local_time.replace(tzinfo=None)
    start        = local_time - datetime.timedelta(hours=1)
    #This will get the previous hour! Attenzione
    end          = local_time
    meteo_data   = Hourly(id_, start, end, 'Europe/Amsterdam')
    meteo_data   = meteo_data.fetch()
    meteo_new = {
                 'wind speed (kts)'     : meteo_data['wspd'].iloc[0]*kmh2kts,
                 'wind direction (deg)' : meteo_data['wdir'].iloc[0],
                 'temperature (°C)'     : meteo_data['temp'].iloc[0],
                 'pressure (hpa)'       : meteo_data['pres'].iloc[0]
                }
    meteo_save = pd.DataFrame(meteo_new, index=[0])
    get_id = File.split('_')[1]
    meteo_save.to_csv('meteo_'+get_id, index=False)

