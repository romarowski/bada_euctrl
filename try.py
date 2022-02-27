import numpy as np
import scipy
import pandas as pd
import pdb
import os
from modules.NoiseCraft import NoiseCraft
from model import aircraft_motion
import configparser
def main():

    #plot = [] 
    #while plot != 'y'  and plot != 'n':
    #    plot = input('plot? (y/n) ')
    #    plot_ = plot == 'y'

    plot_ = True

    equip = 'A320-211' # can be improved

    #folder = 'AMS_LGW'
    #flight_id = '520812451'

   #folder = 'ORY_to_BIA'

    #folder = 'AMS_TNG'
    #flight_id = '520831277'
    date = '20190720'

    #radar_fn  = date + '_' + flight_id + '.csv'
    path = 'input'
    folder_flights = 'flights'
    folder_meteo   = 'meteo'

    is_departure = True
    config = configparser.ConfigParser()
    config.read('config.ini')

    my_config_parser_dict = {s:dict(config.items(s))\
            for s in config.sections()}


    columns = my_config_parser_dict['column_names']

    column_names = {'Pressure'    : columns['pressure'],
                    'WindDir'     : columns['winddir'],
                    'WindSpeed'   : columns['windspeed'],
                    'Heading'     : columns['heading'],
                    'Altitude'    : columns['altitude'],
                    'Temperature' : columns['temperature'],
                    'GroundSpeed' : columns['groundspeed'],
                    'Lat'         : columns['lat'],
                    'Lon'         : columns['lon'],
                    'Time'        : columns['time']
                   }



    file_list = os.scandir(os.path.join(path, folder_flights))
    pdb.set_trace()
    for flight_fn in file_list:
            
            flight_fn = str(flight_fn.name)
            flight_id = flight_fn.split('_')[1].split('.')[0]
            

            meteo_fn = 'meteo_' + flight_id + '.csv'
            #radar_fn = 'flight_'+id_+'.csv'
            radar_fn =  flight_fn
            
            op_type = pd.read_csv(os.path.join('input', folder_meteo,
                meteo_fn))['op_type'].iloc[0]


            craft = NoiseCraft(equip, op_type)
            craft.load_meteo(folder_meteo, meteo_fn)
            #op_type = craft.op_type
            craft.load_radar(folder_flights, radar_fn)
            craft.correct_altitude(column_names, Eapt=craft.Eapt)

            craft.clean_radar_data(column_names, op_type = op_type)
            if op_type == 'D':
                origin = craft.Radar.iloc[0][[column_names['Lat'], column_names['Lon']]]
            else:
                origin = craft.Radar.iloc[-1][[column_names['Lat'], column_names['Lon']]]

            craft.lat_lon_to_m(origin.to_numpy(), column_names)
            craft.calculate_distance(op_type=op_type)
            craft.calculate_CAS(column_names)

            
            craft.load_data(column_names)

            after_TO = False

            mincount = 3
            maxcount = 10
            
            #try: 
            craft.segment(mincount, maxcount, wrt_to_time=True, after_TO=after_TO, 
                    normalize = False, op_type=op_type)
            
            #craft.recognize_steps()
            craft.get_mass_w_ANP_simple(op_type=op_type)

            
            if plot_:         
                craft.plot_segmented()
            
            craft.get_R_with_BADA_segmented(column_names)
            
            
            if after_TO:
                craft.extrapolate_TO_distance()
           #print('Steps: ' + str(craft.steps))
            
            #craft.new_vert_profile(column_names)
            #craft.new_vert_profile_pinv(column_names)
            
            if plot_:
                pass
                #craft.plot_segmented()
                #craft.plot_ANP_profile(column_names, flight_id)

            #craft.map_flaps()
            craft.thrust_FB_segmented(op_type=op_type)
            craft.plot_FB(op_type=op_type)

    return 0, craft

if __name__ == '__main__':
    success, craft = main()
    print('Exited: ' + str(bool(~success)))

