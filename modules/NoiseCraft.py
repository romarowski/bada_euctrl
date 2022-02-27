import pandas as pd
import os
import pdb
import numpy as np
from modules.ISA import air_density_ratio, pressure_ratio_, pressure_altitude,\
        temperature_profile, temperature_ratio_
from modules import inm_map_projection
from scipy import interpolate
from modules.haversine_distance import haversine
import matplotlib.pyplot as plt
from scipy import optimize, interpolate
from modules.ANP_steps import sin_of_climb_angle, TO_ground_roll_distance,\
        first_climb_CAS, distance_accelerate, corrected_net_thrust,\
        sin_of_climb_angle_pinv, first_climb_CAS_pinv, distance_accel_pinv,\
        corrected_net_thrust_pinv

import xml.etree.ElementTree as ET




class NoiseCraft:
    
    def __init__(self, ACFT_ID, op_type):
        self.ACFT_ID = ACFT_ID
        path = r'./data/ANP_Database/'
        dirs = ['Aerodynamic_coefficients.csv', 'Aircraft.csv', 
                'Default_approach_procedural_steps.csv', 
                'Default_departure_procedural_steps.csv', 'Default_weights.csv',
                'Jet_engine_coefficients.csv']
        #Working e.g. for A320-211, categories missing like propeller engine,etc.
        self.Aerodynamic_coefficients = pd.read_csv(os.path.join(path, dirs[0]),
                 sep=';')
        self.Aircraft = pd.read_csv(os.path.join(path, dirs[1]), sep=';')
        self.Approach_steps = pd.read_csv(os.path.join(path, dirs[2]), sep=';')
        self.Departure_steps = pd.read_csv(os.path.join(path, dirs[3]), sep=';')
        self.Default_weights = pd.read_csv(os.path.join(path, dirs[4]), sep=';')
        self.Jet_engine_coefficients = pd.read_csv(os.path.join(path, dirs[5]),
                sep=';')
        self._Meto_loaded = False
        self.Meteo = None
        self_Radar_loaded = False
        self.Radar = None

        #Adding BADA data
        tree = ET.parse(r'./data/A320-212/A320-212.xml')
        root = tree.getroot()
        AFCM = root.findall('./AFCM/')
        Ground = root.findall('./Ground/')[0]
        wing_span = np.float64(Ground.findall('span')[0].text)
    
        S = AFCM[0]
        self.aspect_ratio = wing_span**2 / np.float64(S.text)
        configurations = AFCM[1:] #First one is always Surface area S
        self.surf_area = np.float64(S.text)
        #TODO fix this to allow for any type of plane
        if op_type == 'D':
            configurations = [configurations[0], configurations[2], configurations[3]]
        else:
            configurations = [configurations[0], configurations[3], configurations[5]]


        speeds = np.array(
                [int(config.findall('vfe')[0].text) for config in configurations])
        
        self.flap_scheduling = speeds
        
        #List of speeds per config arranged from higher to lower speed
        configs_indexes = np.arange(speeds.size)
        
        self.current_config = lambda speed: configurations[max(configs_indexes[speeds>speed])]


        pass

    def load_meteo(self, folder, filename):
        path = 'input'
        #path = os.path.join('data', 'radar_tracks_n_meteo', folder) 
        self.Meteo = pd.read_csv(os.path.join(path, folder, filename))
        self._Meteo_loaded = True
        self.Eapt = self.Meteo['alt apt (ft)'].iloc[0]
        self.op_type = self.Meteo['op_type'].iloc[0]
        pass

    def load_radar(self, folder, filename):
        path = 'input'
        #path = os.path.join('data', 'radar_tracks_n_meteo', folder)
        self.Radar_raw = pd.read_csv(os.path.join(path, folder, filename))
        self._Radar_loaded = True
        pass

    def clean_radar_data(self, column_names, op_type='D'):
        max_height = 10000 #[ft]
        df = self.Radar_raw
        mask_h = np.flatnonzero(df[column_names['Altitude']] >= max_height)
        look_back_speed = 1 

        min_speed = 50 #[kts]

        #mask_s = np.flatnonzero(df[column_names['GroundSpeed']] >= min_speed)
        
        if op_type == 'A':
            loc_max = mask_h[-1]
            df = df.iloc[loc_max:].reset_index(drop=True)
            mask_s = np.flatnonzero(df[column_names['GroundSpeed']] >= min_speed)
            loc_min = mask_s[-1] + 2
            df = df.iloc[:loc_min].reset_index(drop=True)
            
        else:
            loc_max = mask_h[0] + 1
            df = df.iloc[:loc_max].reset_index(drop=True)
            mask_s = np.flatnonzero(df[column_names['GroundSpeed']] >= min_speed)
            loc_min = mask_s[0] - look_back_speed

            df = df.iloc[loc_min:].reset_index(drop=True)
        

        self.Radar = df

        pass

    def correct_altitude(self, column_names, Eapt=-9):
        B = 0.003566 #°F/ft atmospheric lapse
        T0 = 518.67 #°R temperature at std mean sea level
        g = 32.17 #ft/s^2 gravity
        B = 0.003566 #°F/ft atmospheric lapse
        R = 1716.59 #ft*lb/(sl*°R)
        P0 = 29.92 #inHg pressure at std mean sea level
        gBR = np.round(g/B/R, 4)
        
        df = self.Radar_raw

        h = df[column_names['Altitude']].to_numpy() #pressure altitude
        

        Papt = self.Meteo[column_names['Pressure']].iloc[0]
        Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)

        delta = (-h[h!=0] * (B/T0) + 1) ** gBR

        h_msl = (-delta ** (1/ gBR) + (Papt / P0) ** (1/gBR)) * T0 / B 
        
        h_agb = h_msl - Eapt

        h[h!=0] = h_agb
        h_agb = h

        self.Radar_raw[column_names['Altitude']] = h_agb #in ft


    #TODO Change this function name 
    #     Make this not an internal method
    def get_mass_w_ANP_simple(self, op_type):
        Aero_coeff = self.Aerodynamic_coefficients
        
        if op_type == 'D':
            dep_steps = self.Departure_steps[['Step Number', 'Step Type',
                'Thrust Rating', 'Flap_ID']]
            TO_Flap_ID = dep_steps[dep_steps['Step Type']==\
                    'Takeoff']['Flap_ID'].unique() #climb_Flap_ID[0] 
     
            dep_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
            C = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==TO_Flap_ID[0]]['C'].iloc[0]

            #lbf2kg = 0.4536
            def mid_values(vec):
                return (vec[1:] + vec[:-1]) / 2

            
            #This will have to be fixed for landings
            V_CTO = mid_values(self.cas_seg)[1]

            weight = (V_CTO / C) ** 2 #in lbs
        else: 
            arr_steps = self.Approach_steps[['Step Number', 'Step Type',
                 'Flap_ID']]
            LD_Flap_ID = arr_steps[arr_steps['Step Type']==\
                    'Land']['Flap_ID'].unique() #climb_Flap_ID[0] 
     
            arr_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
            D = arr_Aero_coeff[arr_Aero_coeff['Flap_ID']==LD_Flap_ID[0]]['D'].iloc[0]           
            V_CA = self.cas[self.h==0.][-1]

            weight = (V_CA / D) ** 2 #in lbs 
            




        self.weight_ANP_simple =  weight
    
    #def get_lift_coeff(self, column_names):
    #    g = 9.81 #m/s 
    #    p0 = 101325 #Pa
    #    k = 1.4 #Adiabatic index of air

    #    Altitude = self.Radar[column_names['Altitude']].to_numpy()
    #    Papt = self.Meteo[column_names['Pressure']].iloc[0]
    #    Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)
    #    pressure_ratio = pressure_ratio_(Altitude, Papt)


    #    #Mach number
    #    kts2mpers = .5144
    #    R = 287.05287 #m^2/(Ks^2) Real gas constant for air
    #    Tapt = self.Meteo[column_names['Temperature']].iloc[0]
    #    temperatures = temperature_profile(Altitude, Tapt) #celsius
    #    celsius2kelvin = 273.15
    #    temperatures += celsius2kelvin
    #    TAS = np.array(self.Radar['TAS (kts)'].iloc[:]) * kts2mpers
    #    

    #    M = TAS / np.sqrt(k * R * temperatures)
    #    
    #    lbm2kgs = 0.45359237
    #    lbs2kgs = 1 * lbm2kgs # 1/g * lbm2kgs

    #    m = self.wei * lbs2kgs
    #    C_L = 2 * m * g / (pressure_ratio * p0 * k * self.surf_area * M**2)
    #    

    #    return C_L, M
    
    def get_lift_coeff_segmented(self, column_names, up_sample=True):
        g = 9.81 #m/s 
        p0 = 101325 #Pa
        k = 1.4 #Adiabatic index of air

        Altitude = self.h_seg
        Papt = self.Meteo[column_names['Pressure']].iloc[0]
        Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)
        pressure_ratio = pressure_ratio_(Altitude, Papt, self.Eapt)
        self.pressure_ratios = pressure_ratio_(self.h, Papt, self.Eapt)

        #Mach number
        kts2mpers = .5144
        R = 287.05287 #m^2/(Ks^2) Real gas constant for air
        Tapt = self.Meteo[column_names['Temperature']].iloc[0]
        temperatures = temperature_profile(Altitude, Tapt) #celsius
        celsius2kelvin = 273.15
        temperatures += celsius2kelvin
        CAS = self.cas_seg * kts2mpers
        
        
        sigmas = air_density_ratio(Altitude, Tapt, Papt, self.Eapt) 
        TAS = CAS / np.sqrt(sigmas)
        
        M = TAS / np.sqrt(k * R * temperatures)

        
        lbm2kgs = 0.45359237
        lbs2kgs = 1 * lbm2kgs # 1/g * lbm2kgs

        m = self.weight_ANP_simple * lbs2kgs
        C_L = 2 * m * g / (pressure_ratio * p0 * k * self.surf_area * M**2)
        
        h = self.h

        if up_sample:
            d_seg = self.d_seg[self.h_seg>0]
            #M     = M[self.h_seg>0]
            C_L   = C_L[self.h_seg>0] 
            M     = np.interp(self.d, self.d_seg, M)
            C_L   = np.interp(self.d[h>0], d_seg, C_L)

        ##############-----ONGROUND----################
        mu    = 0.02
        e     = 1 / (1.05 + 0.007 * np.pi * self.aspect_ratio)
        K     = 1 / (np.pi * self.aspect_ratio * e)
        C_L_  = np.zeros(len(self.d))
        C_L_g = mu / (2 * K)

        C_L_[h>0]  = C_L
        C_L_[h==0] = C_L_g

        return C_L_, M

    #def get_R_with_BADA(self, column_names):
    #    
    #    #Obtain drag over lift coefficient

    #    lg_state = 'LGUP'
    #    
    #    cas = self.cas

    #    C_L, Mach = self.get_lift_coeff(column_names)

    #    R = np.zeros((len(cas)))
    #    
    #    for row, speed in enumerate(cas):
    #        if speed < self.flap_scheduling[1]:
    #            drag_polys_coeff = np.zeros(3)
    #            drag_elems = list(list(self.current_config(speed).findall(lg_state)[0])[0])[0].findall('d')
    #            for i, d in enumerate(drag_elems):
    #                drag_polys_coeff[i] = np.float64(d.text)
    #            
    #                d1, d2, d3 = drag_polys_coeff
    #            
    #            R[row] = d1 / C_L[row] + d2 + d3 * C_L[row] 
    #        else:
    #            drag_polys_coeff = np.zeros(15)
    #            drag_elems = (self.current_config(speed).findall(lg_state)[0].getchildren()[0]).getchildren()[2].findall('d')
    #            scalar = np.float64((self.current_config(speed).findall(lg_state)[0].getchildren()[0]).getchildren()[1].text)

    #            M = Mach[row]
    #            for i, d in enumerate(drag_elems):
    #                drag_polys_coeff[i] = np.float64(d.text)
    #            
    #                d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, \
    #                        d14, d15 = drag_polys_coeff

    #                C0 = d1 + d2 / (1 - M**2)**.5 + d3 / (1 - M**2) +\
    #                        d4 / (1 - M**2)**1.5 + d5 / (1 - M**2)**2  

    #                C2 = d6 + d7 / (1 - M**2)**1.5 + d8 / (1 - M**2)**3 +\
    #                        d9 / (1 - M**2)**4.5 + d10 / (1 - M**2)**6  
    #                
    #                C6 = d11 + d12 / (1 - M**2)**7 + d13 / (1 - M**2)**7.5 +\
    #                        d14 / (1 - M**2)**8 + d15 / (1 - M**2)**8.5  

    #            R[row] = scalar * (C0 / C_L[row] + C2 * C_L[row] + C6 * C_L[row]**5)


    #    self.R_Bada = R
    def get_R_with_BADA_segmented(self, column_names, up_sample=True):
        
        #Obtain drag over lift coefficient

        lg_state = 'LGUP'
        
        cas = self.cas_seg
       
        if up_sample:
            cas = np.interp(self.d, self.d_seg, cas)

        C_L, Mach = self.get_lift_coeff_segmented(column_names)
        self.Mach = Mach
        R   = np.zeros(len(cas))
        C_D = np.zeros(len(cas))
        
        for row, speed in enumerate(cas):
            if speed < self.flap_scheduling[1]:
                drag_polys_coeff = np.zeros(3)
                drag_elems = list(list(self.current_config(speed).findall(lg_state)[0])[0])[0].findall('d')
                for i, d in enumerate(drag_elems):
                    drag_polys_coeff[i] = np.float64(d.text)
                        
                d1, d2, d3 = drag_polys_coeff
                
                R[row] = d1 / C_L[row] + d2 + d3 * C_L[row]
                C_D[row]    = R[row] * C_L[row]
            else:
                drag_polys_coeff = np.zeros(15)
                drag_elems = (self.current_config(speed).findall(lg_state)[0].getchildren()[0]).getchildren()[2].findall('d')
                scalar = np.float64((self.current_config(speed).findall(lg_state)[0].getchildren()[0]).getchildren()[1].text)

                M = Mach[row]
                #We actually don't need to do this everytime
                for i, d in enumerate(drag_elems):
                    drag_polys_coeff[i] = np.float64(d.text)
                
                d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, \
                            d14, d15 = drag_polys_coeff

                C0 = d1 + d2 / (1 - M**2)**.5 + d3 / (1 - M**2) +\
                    d4 / (1 - M**2)**1.5 + d5 / (1 - M**2)**2  

                C2 = d6 + d7 / (1 - M**2)**1.5 + d8 / (1 - M**2)**3 +\
                    d9 / (1 - M**2)**4.5 + d10 / (1 - M**2)**6  
                
                C6 = d11 + d12 / (1 - M**2)**7 + d13 / (1 - M**2)**7.5 +\
                    d14 / (1 - M**2)**8 + d15 / (1 - M**2)**8.5  

                R[row] = scalar * (C0 / C_L[row] + C2 * C_L[row] + C6 * C_L[row]**5)

        self.C_D_Bada = C_D
        self.C_L_Bada = C_L
        
        self.R_Bada = R
        


    def calculate_CAS(self, column_names, op_type='D'):
        if self._Meteo_loaded   == False:
            print('Load meteorological data to calculate CAS')
        elif self._Radar_loaded == False:
            print('Load radar/ADSB data to calculate CAS')
        else:
            #Solution for one windspeed/direction
            winddir    = self.Meteo[column_names['WindDir']].iloc[0]
            windspeed = self.Meteo[column_names['WindSpeed']].iloc[0]
            Tapt = self.Meteo[column_names['Temperature']].iloc[0]
            Tapt = (Tapt * 9/5) + 32.0 #Conversion to Farenheit from C (not ideal)
            Papt = self.Meteo[column_names['Pressure']].iloc[0]
            Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)

            self.Tapt_given = Tapt
            self.Papt_given = Papt
            gamma = np.pi - np.deg2rad(winddir) +\
                    np.deg2rad(self.Radar[column_names['Heading']])
            headwind = -windspeed * np.cos(gamma)
            TAS=(self.Radar[column_names['GroundSpeed']]+headwind) #ground TAS!!
            
            flight = self.Radar
            d = flight['distance (ft)'].to_numpy()
            h = flight[column_names['Altitude']].to_numpy()

            time = flight[column_names['Time']].to_numpy()

            deltaT = np.diff(time)
            deltaD = np.diff(d)
            deltaH = np.diff(h)

            vz = deltaH / deltaT #ft/s
            
            cos_climb_angle = np.multiply(deltaD, 
                    1/(np.sqrt(deltaD**2 + deltaH**2)))
            if op_type == 'D':
                vz = np.r_[0, vz] # This is done since one value of vz is lost due to np.diff()
                cos_climb_angle = np.r_[1, cos_climb_angle]
            else:
                vz = np.r_[vz, 0]
                cos_climb_angle = np.r_[cos_climb_angle, 1]
            TAS = np.sqrt(TAS**2 + vz**2)


            self.climb_angle = np.arccos(cos_climb_angle)

            #TODO check formula for TAS
            #cos_climb_angle = np.ones((len(TAS),))

            
            sigma = air_density_ratio(self.Radar[column_names['Altitude']], 
                                      Tapt, Papt, self.Eapt) 
            
            self.Radar['CAS (kts)'] = TAS * np.sqrt(sigma)
            self.Radar['TAS (kts)'] = TAS
            
            #self.Radar['CAS (kts)'] = np.multiply(np.multiply(TAS, 
            #    np.sqrt(sigma)), 1/(cos_climb_angle))
            #self.Radar['TAS (kts)'] = np.multiply(TAS, 1/(cos_climb_angle))

        pass

    def lat_lon_to_m(self, origin, column_names, proj_type='inm'):
        if self._Radar_loaded == False:
            print('Load radar/ADSB data to calculate CAS')
        elif proj_type == 'inm':
            parameters = inm_map_projection.parameters(origin)
            project = inm_map_projection.project
            self.Radar['x (m)'] , self.Radar['y (m)'] =\
                    zip(*self.Radar[[column_names['Lat'], column_names['Lon']]]\
                    .apply(project, axis = 1, parameters = parameters))

        pass

    def calculate_distance(self, op_type):
        mt2ft = 3.281
        data = self.Radar[['x (m)', 'y (m)']].copy()
        data['distance (ft)'] = np.zeros((len(data), 1))

        for row in np.arange(1, len(data)):
            dist_x = (data['x (m)'].iloc[row-1] - data['x (m)'].iloc[row])**2
            dist_y = (data['y (m)'].iloc[row-1] - data['y (m)'].iloc[row])**2
            dist = np.sqrt(dist_x + dist_y) * mt2ft
            data['distance (ft)'].iloc[row] = data['distance (ft)'].iloc[row-1] +\
                    dist

        if op_type == 'D':
            self.Radar['distance (ft)'] = data['distance (ft)']
        else:
            self.Radar['distance (ft)'] = data['distance (ft)']
            #self.Radar['distance (ft)'] = (data['distance (ft)'].to_numpy())[::-1]*-1

        pass



    def load_data(self, column_names):
        flight = self.Radar
        d = flight['distance (ft)'].to_numpy()
        h = flight[column_names['Altitude']].to_numpy()
        cas = flight['CAS (kts)'].to_numpy()
        time = flight[column_names['Time']].to_numpy()
        time -= time[0]
        #cas_o = cas[ydata>0]
        #xdata_o = xdata[ydata>0]
        #ydata_o = ydata[ydata>0]
        #plt.subplot(221)
        #plt.plot(d, h, '.')
        #plt.xlabel('distance (ft)')
        #plt.ylabel('altitude (ft)')
        #plt.subplot(223)
        #plt.plot(d, cas, '.')
        #plt.xlabel('distance (ft)')
        #plt.ylabel('cas (kts)')
        #plt.subplot(222)
        #plt.plot(time, h, '.')
        #plt.xlabel('time (s)')
        #plt.ylabel('altitude (ft)')
        #plt.subplot(224)
        #plt.plot(time, cas, '.')
        #plt.xlabel('time (s)')
        #plt.ylabel('cas (kts)')
        #plt.show()
        ## Save info from radar prolly should be broken down
        self.h   = np.float64(h)
        self.cas = cas
        self.d   = d
        self.time = time
        pass
    
    
    #def thrust_cutback(self, wrt_to_time
    
    
    
    def segment(self, mincount, maxcount, wrt_to_time=False, after_TO=False,
            normalize=False, cas_weight = 1000, first_climb_weight=2, 
            thrust_cutback = False, no_penalty=True, op_type='D'):
        

        if wrt_to_time:
            X = np.copy(self.time)
        else:
            X = np.copy(self.d)
        
        Y = np.copy(self.h)
        cas = np.copy(self.cas)
        
        first_accel = True
        
        ##############################
        if thrust_cutback:
            for i, step in enumerate(self.steps):
                if step == 'Accelerate' and first_accel and wrt_to_time:
                    x_start = self.time_seg[i]
                    x_end   = self.time_seg[i+2]
                    Y = Y[(X>=x_start) & (X<=x_end)]
                    cas = cas[(X>=x_start) & (X<=x_end)]
                    X = X[(X>=x_start) & (X<=x_end)]
                    first_accel = False
                    break
                elif step == 'Accelerate' and first_accel and not wrt_to_time:
                    x_start = self.d_seg[i]
                    x_end   = self.d_seg[i+2]
                    X = X[(X>=x_start) & (X<=x_end)]
                    Y = Y[(X>=x_start) & (X<=x_end)]
                    cas = cas[(X>=x_start) & (X<=x_end)]
                    first_accel = False
                    break
            mincount = 2
            maxcount = 2
        ############################## ALL this is nothing             

        if normalize:
            norm_x = X.max()
            norm_y = Y.max()
            norm_cas = cas.max()

            X = np.divide(X, norm_x)
            Y =  np.divide(Y, norm_y)
            cas = np.divide(cas, norm_cas)
            
            cas_weight = 1.0

        else:
            norm_x = 1.0
            norm_y = 1.0
            norm_cas = 1.0

        if after_TO:
            X = X[Y>0]
            cas = cas[Y>0]
            Y = Y[Y>0]


        
        xmin = X.min()
        xmax = X.max()
        n = len(X)
        
        AIC = []
        r   = []
        regions = []
        i = 0

        for count in range(mincount, maxcount + 1): 
            seg = np.full(count - 1, (xmax - xmin) / count)

            px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            mask = [[np.abs(X - x) < (xmax - xmin) * 0.1] for x in px_init]


            py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean()\
                    for x in px_init])
            pcas_init = np.array([cas[np.abs(X - x) < (xmax - xmin) * 0.1].mean()\
                    for x in px_init])


            def func(p, count):
                seg = p[:count - 1]
                py = p[count - 1:2*count]
                pcas = p[2*count:]
                px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
                return px, py, pcas

            def err(p):
                px, py, pcas = func(p, count)
                Y2 = np.interp(X, px, py)
                CAS2 = np.interp(X, px, pcas)


                if after_TO:
                    penalty_FC = max(0, np.diff(pcas)[0])**2
                    penalty_TO = 0
                else:
                    penalty_FC = max(0, np.diff(pcas)[1])**2
                    penalty_TO = max(0, np.diff(py)[0])**2
                
                #penalty_CAS = max(0, np.mean(np.diff(pcas)))**2
                #penalty_y   = max(0, np.mean(np.diff(pcas)))**2 
                penalty_CAS = 0.
                penalty_y   = 0.

                #if not np.all(np.diff(pcas)>=0):
                #    penalty_CAS = 1e10
                #else:
                #    penalty_CAS = 0
                #if not np.all(np.diff(py) >= 0):
                #    penalty_y = 10e3
                #else: 
                #    penalty_y = 0
                
                #First climb precision
               
                if thrust_cutback:
                    cost = np.mean((Y-Y2)**2)+cas_weight*np.mean((cas-CAS2)**2)+\
                        penalty_CAS + penalty_y
                elif no_penalty:
                    cost =  np.mean((Y - Y2)**2) + cas_weight*np.mean((cas - CAS2)**2)
                else:
                    cost = np.mean((Y - Y2)**2) + \
                           first_climb_weight*\
                           np.mean((Y[Y<1000./norm_y] - Y2[Y<1000./norm_y])**2) + \
                           cas_weight*np.mean((cas - CAS2)**2) + penalty_CAS + \
                                   penalty_FC + penalty_TO + penalty_y
                return cost

            x0=np.r_[seg, py_init, pcas_init]
            r.append(optimize.minimize(err, x0=x0, method='Nelder-Mead',
                    #options={'adaptive': False, 'fatol':1e-5},
                    )
                 )
            AIC.append(n * np.log10(err(r[i].x)) + 4 * count)
            #BIC = n * np.log10(err(r.x)) + 2 * count * np.log(n)
            
            regions.append(count)

            i += 1


        AIC_min = min(AIC)
        min_index = AIC.index(AIC_min)
        r_min = r[min_index]
        reg_min = regions[min_index]
        print('AICs:' + str(AIC))
        print('Regions:' + str(regions))
        print('Region: ' + str(reg_min))
        
        if thrust_cutback:
            pass
        if wrt_to_time:
            ptimes, py, pcas = func(r_min.x, reg_min)
            if normalize:
                ptimes = np.multiply(ptimes, norm_x)
            px = np.interp(ptimes, self.time, self.d)
        else:        
            px, py, pcas = func(r_min.x, reg_min)
            if normalize:
                px = np.multiply(px, norm_x)
            ptimes = np.interp(px, self.d, self.time)

        if normalize:
            py = np.multiply(py, norm_y)
            pcas = np.multiply(pcas, norm_cas)
        
        
        def mid_values(vec):
            return (vec[1:] + vec[:-1]) / 2

        def clean_cas(cas, op_type):
            cas_avg = mid_values(cas)
            if op_type == 'D':
                for i, _ in enumerate(np.diff(cas)):
                    if np.diff(cas)[i] < 0:
                        cas[i] = cas_avg[i]
                        cas[i+1] = cas_avg[i]
            else:
                for i, _ in enumerate(np.diff(cas)):
                    if np.diff(cas)[i] > 0:
                        cas[i] = cas_avg[i]
                        cas[i+1] = cas_avg[i]

            return cas
        
        pcas = clean_cas(pcas, op_type=op_type)

        py[py<1e-1] = 0
        
        self.d_seg = px
        self.h_seg = py
        self.cas_seg = pcas
        self.time_seg = ptimes
        self.optimizer_result = r_min
        print('Segmentation finished')
        pass

    def plot_segmented(self):
        px = self.d_seg
        py = self.h_seg
        pcas = self.cas_seg
        ptimes = self.time_seg
        X = self.d
        Y = self.h
        CAS = self.cas
        times = self.time
        
        plt.subplot(221)
        plt.plot(X, Y, '.b', label='Radar')
        plt.plot(px, py, '-or', label='Seg')
        #plt.xlabel('Dist [ft]')
        plt.ylabel('Alt [ft]')
        plt.legend()
        
        plt.subplot(223)
        plt.plot(X, CAS, '.b', label='Radar')
        plt.plot(px, pcas, '-or', label='Model')
        plt.xlabel('Dist [ft]')
        plt.ylabel('CAS [kts]')
        #plt.legend()

        plt.subplot(222)
        plt.plot(times, Y, '.b', label='Radar')
        plt.plot(ptimes, py, '-or', label='Model')
        #plt.xlabel('Time [s]')
        #plt.ylabel('Alt [ft]')
        #plt.legend()

        plt.subplot(224)
        plt.plot(times, CAS, '.b', label='Radar')
        plt.plot(ptimes, pcas, '-or', label='Model')
        plt.xlabel('Time [s]')
        #plt.ylabel('CAS [kts]')
        #plt.legend()
        
        plt.show()

    def recognize_steps(self, treshold_CAS=10, after_TO=True):
        cas = self.cas_seg
        cas_derivative = np.diff(self.cas_seg) <= treshold_CAS
        steps = ['Climb' if der else 'Accelerate' for der in cas_derivative]
        self.steps = steps
        pass

    def extrapolate_TO_distance(self):
        dist = self.d_seg
        alt  = self.h_seg
        cas  = self.cas_seg
        steps = self.steps
        time = self.time_seg
        f = interpolate.interp1d(alt[:2], dist[:2], fill_value = 'extrapolate')
        steps_ = np.copy(steps)
        steps_ = np.r_[['TakeOff'], steps_]
        dist_ = np.copy(dist)
        alt_  = np.copy(alt)
        cas_  = np.copy(cas)
        time_  = np.copy(time)
        dist_[0] = f(0)
        alt_[0] = 0.
        time_ = np.r_[0., time_]
        dist_ = np.r_[0., dist_]
        alt_ = np.r_[0., alt_]
        cas_ = np.r_[0, cas_]

        self.d_seg   = dist_
        self.h_seg   = alt_
        self.cas_seg = cas_
        self.steps   = steps_ 
        self.time_seg    = time_
        
        pass
    def _loss(self, x, column_names, thrust_cfg, model=False, op_type = 'D'):
        weight = x[0]
        Tflex  = x[1]

        #weight = 127496.7607

        dist   = self.d_seg
        alt    = self.h_seg
        cas    = self.cas_seg
        times  = self.time_seg
        steps = self.steps
        #print('Laying down new profile...')
        
        def mid_values(vec):
            return (vec[1:] + vec[:-1]) / 2
        def sins_gamma_estimation(dist, alt, cas):
            climbs_deltaH = np.diff(alt)
            climbs_deltaS = np.diff(dist)
            sins_gamma = climbs_deltaH /\
                    np.sqrt(climbs_deltaH**2 + climbs_deltaS**2)
            return sins_gamma
        
        Tapt = self.Meteo[column_names['Temperature']].iloc[0]
        Papt = self.Meteo[column_names['Pressure']].iloc[0]
        wind_speed = self.Meteo[column_names['WindSpeed']].iloc[0]

        Tflex = (Tflex * 9/5) + 32.0 #Conversion to Farenheit from C (not ideal)

        
        Aero_coeff = self.Aerodynamic_coefficients
        dep_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
        Tapt = (Tapt * 9/5) + 32.0 #Conversion to Farenheit from C (not ideal)
        Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)
        

        sigmas       = air_density_ratio(alt, Tapt, Papt, self.Eapt)
        if model==True:
            self.Sigmas = sigmas
        tas          = np.multiply(cas, 1/(np.sqrt(sigmas)))
        mean_sigmas  = mid_values(sigmas)
        mean_sigmas_ = air_density_ratio(mid_values(alt), Tapt, Papt, self.Eapt)

        tas_geom_mean = np.sqrt(mid_values(np.power(tas, 2))) #Impact eq-17
        tas_diff      = np.diff(np.power(tas, 2))
        deltas        = pressure_ratio_(alt, Papt, self.Eapt)
        mean_deltas   = mid_values(deltas)
        #print(deltas)
        W_delta       = weight * 1/(mean_deltas)

        if model==True:  
            self.Tapt    = Tapt
            self.Papt    = Papt
            self.TAS     = tas
            self.THR_SET = []
        
        first_climb = True
        first_accel = True


        #Obtain flaps by type of step: Accel, TO, Climb for now
        dep_steps = self.Departure_steps[['Step Number', 'Step Type',
            'Thrust Rating', 'Flap_ID']]
        climb_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Climb']['Flap_ID'].unique()
        TO_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Takeoff']['Flap_ID'].unique() #climb_Flap_ID[0] 
        accel_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Accelerate']['Flap_ID'].unique()

        N = self.Aircraft['Number Of Engines'].iloc[0]

        Jet_eng_coeff = self.Jet_engine_coefficients
        thrust_rating = {'Reg'   : {'C': 'MaxClimb', 'TO' : 'MaxTakeoff'},
                         'HiTemp': {'C': 'MaxClimbHiTemp', 'TO' : 'MaxTkoffHiTemp'}
                         }
        thrust_type = 'HiTemp'

        
        if thrust_type == 'HiTemp':
            midpoint_Temps = temperature_profile(mid_values(alt), Tflex)
        else:
            midpoint_Temps = temperature_profile(mid_values(alt), Tapt)

        n_steps = len(steps)
        
        sins_gamma = np.zeros(n_steps)
        segment_accel = np.zeros(n_steps)
        
        ROCs = np.multiply(np.diff(alt), 1/(np.diff(times))) * 60.
        self.ROCs = ROCs #ft/min

        

        for i, step in enumerate(steps):
            if step == 'TakeOff':
                

                B8 = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        TO_Flap_ID[0]]['B'].iloc[0]
                theta = temperature_ratio_(Alt=0, Tapt=Tapt)
                
                W_delta_TO = weight / pressure_ratio_(Alt=0, Papt=Papt, Eapt=self.Eapt)
                
                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type]['TO']].reset_index(drop=True)
                thrust_coeff =\
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type][thrust_cfg[i]]].\
                        reset_index(drop=True)
                
                if thrust_type == 'HiTemp':
                    T_thrust = Tflex
                else:
                    T_thrust = Tapt

                #Should be CAS of first point
                Fn_delta = corrected_net_thrust(thrust_coeff,(cas)[i+1],
                    temperature_profile(Alt=0, Tapt=T_thrust), Alt=0, Papt=Papt)
                est_TO8 = TO_ground_roll_distance(B8, theta, W_delta_TO, N, 
                        Fn_delta)

                if model:
                    Vc_0 = 0
                    Fn_delta_0 = corrected_net_thrust(thrust_coeff, Vc_0,
                        temperature_profile(Alt=0, Tapt=T_thrust), Alt=0, Papt=Papt)
                    self.THR_SET.append(Fn_delta_0)
                    self.THR_SET.append(Fn_delta)
            
            elif step == 'Climb' and first_climb:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        TO_Flap_ID[0]]['R'].iloc[0]

                thrust_coeff =\
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type][thrust_cfg[i]]].\
                        reset_index(drop=True)

                Fn_delta = corrected_net_thrust(thrust_coeff, (cas)[i+1],
                        midpoint_Temps[i], mid_values(alt)[i], Papt)
                #print(Fn_delta)
                sins_gamma[i] = \
                        sin_of_climb_angle(N, Fn_delta, W_delta[i], R,
                        mid_values(cas)[i])

                correction = (cas[i+1] - 8) /\
                             (cas[i+1] - wind_speed)
                gamma = np.arcsin(sins_gamma[i]) * correction

                sins_gamma[i] = np.sin(gamma)

                #FIRST CAS

                C = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        TO_Flap_ID[0]]['C'].iloc[0]

                est_first_climb_CAS = first_climb_CAS(C, weight)
                #TODO recheck this following line!
                radar_first_CAS = mid_values(cas)[i]


                first_climb = False

                if model:
                    self.THR_SET.append(Fn_delta)

            elif step == 'Climb' and first_climb==False:
                
                R  = x[i]
                #R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                #        climb_Flap_ID[1]]['R'].iloc[0]
                thrust_coeff = \
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type][thrust_cfg[i]]].\
                        reset_index(drop=True)

                Fn_delta = corrected_net_thrust(thrust_coeff, (cas)[i+1],
                        midpoint_Temps[i], mid_values(alt)[i], Papt)

                sins_gamma[i] =\
                        sin_of_climb_angle(N, Fn_delta, W_delta[i], R,
                        mid_values(cas)[i])
                
                correction = (cas[i+1] - 8) /\
                             (cas[i+1] - wind_speed)
                gamma = np.arcsin(sins_gamma[i]) * correction

                sins_gamma[i] = np.sin(gamma)

                if model:
                    self.THR_SET.append(Fn_delta)
            elif step == 'Accelerate' and first_accel:
                
                R  = x[i]

                #R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                #        accel_Flap_ID[0]]['R'].iloc[0]

                thrust_coeff =\
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type][thrust_cfg[i]]].\
                    reset_index(drop=True)

                Fn_delta = corrected_net_thrust(thrust_coeff, (cas)[i+1],
                    midpoint_Temps[i], mid_values(alt)[i], Papt)
                segment_accel[i] =\
                        distance_accelerate(tas_diff[i],
                    tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])

                correction = (tas_geom_mean[i] - wind_speed)/\
                             (tas_geom_mean[i] -     8     )
                segment_accel[i] *= correction

                first_accel = False
                second_accel = True

                if model:
                    self.THR_SET.append(Fn_delta)
            elif step == 'Accelerate' and first_accel == False and \
                    second_accel==True:

                R  = x[i]
                #R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                #        accel_Flap_ID[1]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type]['TO']].reset_index(drop=True)
                
                thrust_coeff =\
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type][thrust_cfg[i]]].\
                    reset_index(drop=True)

                Fn_delta = corrected_net_thrust(thrust_coeff, (cas)[i+1],
                    midpoint_Temps[i], mid_values(alt)[i], Papt)
                segment_accel[i] =\
                        distance_accelerate(tas_diff[i],
                    tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])
                
                correction = (tas_geom_mean[i] - wind_speed) / \
                             (tas_geom_mean[i] -     8     )
                segment_accel[i] *= correction
             
                second_accel = False

                if model:
                    self.THR_SET.append(Fn_delta)
            elif step == 'Accelerate' and first_accel == False and \
                    second_accel==False:

                R  = x[i]
                #R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                #        accel_Flap_ID[2]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type]['C']].reset_index(drop=True)
                
                thrust_coeff =\
                        Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type][thrust_cfg[i]]].\
                    reset_index(drop=True)

                Fn_delta = corrected_net_thrust(thrust_coeff, (cas)[i+1],
                    midpoint_Temps[i], mid_values(alt)[i], Papt)
                
                segment_accel[i] = distance_accelerate(tas_diff[i],
                    tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])
                
                correction = (tas_geom_mean[i] - wind_speed) / \
                             (tas_geom_mean[i] -     8     )
                segment_accel[i] *= correction

                if model:
                    self.THR_SET.append(Fn_delta)
                
    
        sins_gamma_RADAR = sins_gamma_estimation(dist, alt, cas)
        
        
        
        #Cost calculation
        sins_gamma_RADAR_o = sins_gamma_RADAR[steps=='Climb']            
        seg_accel_RADAR_o  = np.diff(dist)[steps=='Accelerate']
        
        sins_gamma_o = sins_gamma[steps=='Climb']
        seg_accel_o = segment_accel[steps=='Accelerate']
        
        cost_climb = np.mean((sins_gamma_o - \
               sins_gamma_RADAR_o)**2)
        
        cost_accel = np.mean((seg_accel_o - \
               seg_accel_RADAR_o)**2)
        
        cost_first_CAS = (est_first_climb_CAS - radar_first_CAS)**2

        cost_TO8 = (est_TO8 - np.diff(dist)[steps=='TakeOff'])**2
        
        #cost = cost_climb + cost_accel + 2*cost_first_CAS/200 + cost_TO8[0]/1e8
        cost = cost_first_CAS * cost_climb *  cost_accel **2 
            
        
        if not model:
            return cost
        else:
            return sins_gamma, segment_accel, est_first_climb_CAS, est_TO8,\
                cas

    def new_vert_profile(self, column_names, op_type = 'D'):
        
        print('Laying down new profile...')
        costs = []
        
        Aero_coeff = self.Aerodynamic_coefficients
        dep_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
        
        dep_steps = self.Departure_steps[['Step Number', 'Step Type',
            'Thrust Rating', 'Flap_ID']]
        climb_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Climb']['Flap_ID'].unique()
        TO_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Takeoff']['Flap_ID'].unique() #climb_Flap_ID[0] 
        accel_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Accelerate']['Flap_ID'].unique()
                

        #TODO improve guess value selection for flaps
        
        first_climb = True
        first_accel = True
        flaps = np.r_[[]]
        for step in self.steps:
            if step == 'Climb' and first_climb:
                first_climb = False
            elif step == 'Climb' and not first_climb:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        climb_Flap_ID[1]]['R'].iloc[0]
                flaps = np.r_[flaps, R]
            elif step == 'Accelerate' and first_accel:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[0]]['R'].iloc[0]
                flaps = np.r_[flaps, R]
                first_accel = False
                second_accel = True
            elif step == 'Accelerate' and second_accel and not first_accel:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[1]]['R'].iloc[0]
                flaps = np.r_[flaps, R]
                second_accel = False
            elif step == 'Accelerate' and not second_accel and not first_accel:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[2]]['R'].iloc[0]
                flaps = np.r_[flaps, R]

        
        # Different thrust_cutback_points 
        n_steps = len(self.steps)
        n_configs = 3
        thrust_modes = np.empty((n_configs, n_steps), dtype='S2')
        for i, n in enumerate([3, 4, 5]):
            to_thrust = np.array(['TO']*n)
            c_thrust  = np.array(['C']*(n_steps - n))
            thrust_modes[i,:] = np.concatenate((to_thrust, c_thrust), axis=0)
        thrust_modes = thrust_modes.astype(str)

        
        ########################################################################
        
        def estimate_stage_length(column_names):
            if self.Radar_raw[column_names['Altitude']].iloc[-1] == 0:
                origin = self.Radar_raw[[column_names['Lon'], column_names['Lat']]]\
                        .iloc[0]
                end = self.Radar_raw[[column_names['Lon'], column_names['Lat']]]\
                        .iloc[-1]

                dist_ft = haversine(end.to_numpy(), origin.to_numpy())

                ft2nm = 0.000164579 

                dist_nm = dist_ft * ft2nm

                if dist_nm <= 500.:
                    stage_length = 1
                elif (dist_nm > 500.) and (dist_nm <= 1000.):
                    stage_length = 2
                elif (dist_nm > 1000.) and (dist_nm <= 1500.):
                    stage_length = 3 
                elif (dist_nm > 1500.) and (dist_nm <= 2500.):
                    stage_length = 4
                elif (dist_nm > 2500.) and (dist_nm <= 3500.):
                    stage_length = 5
                elif (dist_nm > 3500.) and (dist_nm <= 4500.):
                    stage_length = 6
                elif (dist_nm > 4500.) and (dist_nm <= 5500.):
                    stage_length = 7
                elif (dist_nm > 5500.) and (dist_nm <= 6500.):
                    stage_length = 8 
                else:
                    stage_length = 9
            else:
                sl = []
                while sl not in [1,2,3,4,5,6,7,8,9]:
                    try:
                        sl=int(input('Flight is not complete, enter the stage length: '))
                    except:
                        print('Enter a valid stage length')

                stage_length = sl
            return stage_length
        
        stage_length = estimate_stage_length(column_names)
        weight = self.Default_weights[self.Default_weights['Stage Length']==\
        stage_length]['Weight (lb)'].iloc[0]

        self.weight_default = weight
        Tflex = 60. #Celsius 

        
        x0 = [weight, Tflex]
        x0 = np.r_[x0, flaps]

        flap_bounds = (0.056281, 0.071403)
        flap_bounds = flap_bounds * len(flaps)
        flap_bounds = np.reshape(flap_bounds, (len(flaps), 2))
        flap_bounds = list(map(tuple, flap_bounds))
        bounds =  [(weight - .5*weight, weight + .5*weight),
                   (Tflex -  .5*Tflex , Tflex  + .5*Tflex)]
        for flap_bound in flap_bounds:
            bounds.append(flap_bound)
        #new_profile = optimize.differential_evolution(loss, bounds)
        
        new_profile2 = optimize.minimize(self._loss, x0, bounds=bounds, args=(
            column_names, thrust_modes[0, :]))
        cost = new_profile2.fun
        n_min = 0
        for n in np.arange(n_configs)[1:]:
            new_profile2_ = optimize.minimize(self._loss, x0, bounds=bounds, args=(
                column_names, thrust_modes[n, :]))
            cost_ = new_profile2_.fun
            if cost_ < cost:
                cost = cost_
                new_profile2 = new_profile2_
                n_min = n
        
        self.thrust_cfg = thrust_modes[n_min, :]
        #self.weight_GA = new_profile.x[0]
        #self.Tflex_GA  = new_profile.x[1]
        #self.flaps_GA  = new_profile.x[2:]
        #self.x_GA      = new_profile.x
        self.weight_grad = new_profile2.x[0]
        self.Tflex_grad  = new_profile2.x[1]
        self.flaps_grad  = new_profile2.x[2:]
        self.x_grad      = new_profile2.x
        pass

    
    def plot_ANP_profile(self, column_names, id_, op_type = 'D'):
        sins_g, accels_g, first_cas_g, to8_g, cas_g =\
                self._loss(self.x_grad, column_names, self.thrust_cfg, model=True,
                        )
        
        #sins_p, accels_p, first_cas_p, to8_p, cas_p =\
        #        model([self.weight_pinv, self.Tflex_pinv])
        #
        #
        #weight = ((self.cas_seg[1:] + self.cas_seg[:-1])[1]/2 / 0.394884)**2
       #Tflex  = self.Tapt
       #weight = self.weight_default
       #x_n = [weight,  Tflex, 0.071403, 0.056281, 0.056281, 0.056281, 0.056281,
       #        0.056281, 0.056281, 0.056281, 0.056281, 0.056281]
       #sins_n, accels_n, first_cas_n, to8_n, cas_n =\
       #        self._loss(x_n, column_names, )
        
        def generate_profile_points(first_cas, sins, accels, to8 = None, after_TO = True):
            end_point_altitude = np.empty(len(self.steps))
            end_point_CAS      = np.empty(len(self.steps))
            end_point_altitude[:] = np.NaN
            end_point_CAS[:]      = np.NaN
            for i, step in enumerate(self.steps):
                if step == 'TakeOff':
                    if after_TO:
                        d_anp = np.array([self.d_seg[1]])
                        h_anp = np.array([0])
                        cas_anp = np.array([first_cas])
                    else:
                        d_anp = np.r_[0, to8]
                        h_anp = np.r_[0, 0]
                        cas_anp = np.r_[0, first_cas]
                elif step == 'Climb':
                    h_anp = np.r_[h_anp, (self.h_seg)[i+1]]
                    d_anp = np.r_[d_anp, d_anp[-1] + (np.diff(self.h_seg)[i] \
                            /np.tan(np.arcsin(sins[i])))]
                    cas_anp = np.r_[cas_anp, cas_anp[-1]]
                    end_point_altitude[i] = (h_anp[-1])
                elif step == 'Accelerate':
                    h_anp   = np.r_[h_anp, (self.h_seg)[i+1]]
                    d_anp   = np.r_[d_anp, d_anp[-1] + accels[i]]
                    cas_anp = np.r_[cas_anp, (self.cas_seg)[i+1]]
                    end_point_CAS[i] = (cas_anp[-1])

            self.End_Point_Cas = end_point_CAS
            self.End_Point_Alt = end_point_altitude
            return h_anp, d_anp, cas_anp
        
        #self.h_anp, self.d_anp, self.cas_anp = generate_profile_points(first_cas,
        #    sins, accels)

        self.h_anp_g, self.d_anp_g, self.cas_anp_g = \
                generate_profile_points(first_cas_g, sins_g, accels_g, to8=None,
                        after_TO=True)

        self.to8_g = to8_g


        def plot_anp():
            px = self.d_seg
            py = self.h_seg
            pcas = self.cas_seg
            X = self.d
            Y = self.h
            CAS = self.cas
            
            plt.subplot(211)
            plt.plot(X, Y, '.b', label='Radar')
            plt.plot(px, py, '-or', label='Seg')
            #plt.plot(self.d_anp, self.h_anp, '-og', label='GA')
            #plt.plot(self.d_anp_p, self.h_anp_p, '-oy', label='pinv')
            plt.plot(self.d_anp_g, self.h_anp_g, '-og', label='ANP')
            #plt.xlabel('Dist [ft]')
            plt.ylabel('Alt [ft]')
            plt.legend()
            
            plt.subplot(212)
            plt.plot(X, CAS, '.b', label='Radar')
            plt.plot(px, pcas, '-or', label='Seg')
            #plt.plot(self.d_anp, self.cas_anp, '-og', label='GA')
            #plt.plot(self.d_anp_p, self.cas_anp_p, '-oy', label='pinv')
            plt.plot(self.d_anp_g, self.cas_anp_g, '-og', label='ANP')
            plt.xlabel('Dist [ft]')
            plt.ylabel('CAS [kts]')
            #plt.legend()

            
            #plt.show()
            plt.savefig(os.path.join('output', 'vert_profile_'+id_+'.png'))
            plt.close()

        plot_anp()

    def map_flaps(self, op_type='D'):
        
        Aero_coeff = self.Aerodynamic_coefficients
        dep_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
        dep_steps = self.Departure_steps[['Step Number', 'Step Type',
            'Thrust Rating', 'Flap_ID']]
        
        TO_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Takeoff']['Flap_ID'].unique() #climb_Flap_ID[0] 



        df_flaps = \
                self.Aerodynamic_coefficients[self.Aerodynamic_coefficients['Op Type']\
                == op_type][['R', 'Flap_ID']].reset_index(drop=True)
        
        df_flaps = df_flaps.sort_values(by='R').reset_index(drop=True)

        n_flaps = len(df_flaps)
        flaps_cont = self.flaps_grad
        
        flap_Rs = df_flaps['R'].to_numpy()
        domain  = max(flap_Rs) - min(flap_Rs)
        
        tresholds = {}
        for count, flap_nm in enumerate(df_flaps['Flap_ID'].to_numpy()):
            treshold  = min(flap_Rs) + (count+1) * domain / n_flaps
            tresholds[flap_nm] = treshold 
        
        epsilon = 1e-7
        
        Flap_ID = np.empty(len(flaps_cont), dtype='S10')
        for flap_nm, treshold in tresholds.items():
            mask = (flaps_cont > treshold - domain/n_flaps - epsilon)  & \
                   (flaps_cont <= treshold)
            Flap_ID[mask] = flap_nm
        
        Flap_ID       = Flap_ID.astype(str)
        Flap_ID       = np.r_[[TO_Flap_ID[0], TO_Flap_ID[0]], Flap_ID]
        self.Flap_ID  = Flap_ID

    #def thrust_force_balance(self):
    #    
    #    mask_h = self.h > 0
    #    TAS  = self.Radar['TAS (kts)'].iloc[:]
    #    TAS = TAS[mask_h]
    #    kts2ftps = 1.68781
    #    TAS *= kts2ftps

    #   # def clean_speed(v):
    #    delta_TAS = np.diff(TAS)
    #    #mask      = delta_TAS < 0
    #    #delta_TAS[mask] = 0 

    #    time = self.time[mask_h]

    #    accel = delta_TAS / np.diff(time) #ft/s^2
    #    op_type = 'D'
    #    df_flaps = \
    #            self.Aerodynamic_coefficients[self.Aerodynamic_coefficients['Op Type']\
    #            == op_type][['R', 'Flap_ID']].reset_index(drop=True)

    #    lbs2slug = 0.031081
    #    mass     = self.weight_grad * lbs2slug
    #    g        = 32.1740 #ft/s^2
    #    gamma    = self.climb_angle[mask_h]
    #    
    #    Rs = np.zeros((len(time),))
    #    for n_step in range(len(self.steps)):
    #        mask = np.logical_and((self.time_seg[n_step] < time),
    #            (time <= self.time_seg[n_step+1]))
    #        flap_id = self.Flap_ID[n_step]
    #        Rs[mask] = df_flaps[df_flaps['Flap_ID']==flap_id]['R'].iloc[0] 
    #    
    #    Rs = Rs[1:]
    #    gamma = gamma[1:]
    #    T = mass * (accel + g * (Rs * np.cos(gamma) + np.sin(gamma)))

    #    self.thrust_FB = T / 2


    #    #####################################################################
    #    # With BADA 
    #    Rs_BADA    = np.squeeze(self.R_Bada)[mask_h] #Removing first since break-
    #    #release was manually added
    #    Rs_BADA    = Rs_BADA[1:]
    #    kgs2slugs = 0.06852

    #    mass  = self.get_mass_w_first_climb() * kgs2slugs
    #    
    #    T = mass * (accel + g * (Rs_BADA * np.cos(gamma) + np.sin(gamma)))

    #    self.thrust_FB_BADA = T / 2
    #    

    #    pass

    def thrust_FB_segmented(self, op_type, up_sample=True):
        def mid_values(vec):
            return (vec[1:] + vec[:-1]) / 2

        def clean_cas(cas):
            cas_avg = mid_values(cas)
            for i, _ in enumerate(np.diff(cas)):
                if np.diff(cas)[i] < 0:
                    cas[i] = cas_avg[i]
                    cas[i+1] = cas_avg[i]
            return cas
        
        
        
        times = self.time_seg
        #times = times[1:] #get rid of 1st point on the ground
        
        Tapt = self.Tapt_given
        Papt = self.Papt_given

        cas = np.copy(self.cas_seg)
        #cas = clean_cas(cas)

        h = self.h_seg #get rid of 1st points on ground
        #d = self.d_seg
        
        
        #h = self.h_seg[1:] #get rid of 1st points on ground
        #d = self.d_seg[1:]
        #cas = cas[1:]

        h[h<0] = 0

        #if up_sample:
        #    d = self.d[self.h>0]
        #    times = np.interp(d, self.d_seg[1:], times)
        #    h     = np.interp(d, self.d_seg[1:], h)
        #    cas   = np.interp(d, self.d_seg[1:], cas)
        if up_sample:
            #d = self.d[self.h>0]
            d  = self.d
            times = np.interp(d, self.d_seg, times)
            h     = np.interp(d, self.d_seg, h)
            cas   = np.interp(d, self.d_seg, cas)
           

        
        sigmas = air_density_ratio(h, Tapt, Papt, self.Eapt)
        tas    = np.multiply(cas, 1/(np.sqrt(sigmas)))
        

        
        P_ratio = pressure_ratio_(mid_values(h), Papt, self.Eapt)
        delta_h = np.diff(h)
        delta_d = np.diff(d)
        tan_gamma = delta_h / delta_d
        gamma = np.arctan(tan_gamma)

        if not up_sample:
            tas = tas[1:] # get rid of first point in the ground


        kts2ftps = 1.68781 
        tas *= kts2ftps

        delta_tas = np.diff(tas)

        #delta_tas[delta_tas<0] = 0
        
        delta_time = np.diff(times)

        accel = delta_tas / delta_time #ft/s^2

        lbs2slug = 0.031081
        g = 32.1740 #ft/s^2

        fig, ax = plt.subplots()
        ax.plot(d, tas, 'o')
        ax2 = ax.twinx()
        ax2.plot(mid_values(d), accel, 'ro')
        plt.show()
        
        
        #This is the method with ANP optimized flaps 
        #Rs = Rs[1:]
        #gamma = gamma[1:]
        #accel = accel[1:]
        #T = mass * (accel + g * (Rs * np.cos(gamma) + np.sin(gamma)))

        #self.thrust_FB_seg = T / 2

        
        #def sector_average(x, w):
        #    #this is not the best approach 
        #    remainder = x.size % w
        #    x = x[remainder:]
        #    return np.mean(x.reshape((w, int(x.size / w) )))


        #####################################################################
        # With BADA 
        Rs_BADA  = np.squeeze(self.R_Bada) 
        if not up_sample:
            Rs_BADA  = Rs_BADA[1:]  #get rid of 1st point @ ground

        if up_sample:
            pass
            #Rs_BADA = Rs_BADA[self.h>0]
        Rs_BADA = mid_values(Rs_BADA) #why am I averaging this?

        #window =  accel.size
        #Rs_BADA = sector_average(Rs_BADA, window)
        
        #kgs2slugs = 0.06852

        weight  = self.weight_ANP_simple
        #weight  = 150000
        mass    = weight / g  #Conversion to slugs

        #gamma[1:] = gamma[1]
        #Rs_BADA[0:1] = .075
        #Rs_BADA[:] = .075
        p0 = 101325 #Pa
        k  = 1.4    #Adiabatic index of air
        mu = .02 #Kinetic friction coefficient

        T_ = np.zeros(len(mid_values(self.d)))
        #Tg = np.zeros(len(self.d[self.h==0]))
        P_ratios = self.pressure_ratios[self.h==0]
        Machs    = self.Mach[self.h==0]
        C_D_g    = self.C_D_Bada[self.h==0]
        C_L_g    = self.C_L_Bada[self.h==0]

        newton2lbs = 0.224809
        
        D = .5 * P_ratios * p0 * k * self.surf_area * Machs**2 *\
                C_D_g * newton2lbs
        L = .5 * P_ratios * p0 * k * self.surf_area * Machs**2 *\
                C_L_g * newton2lbs

        h_mid = mid_values(self.h)

        if op_type == 'D':
            T = mass * (accel + g * (Rs_BADA * np.cos(gamma) + np.sin(gamma)))
            Tg = mass * accel[h_mid==0] + mid_values(D) + mu * (weight - mid_values(L))
            
            T_[mid_values(self.h)==0] = Tg 
            T_[mid_values(self.h)>0]  = T[mid_values(self.h)>0]

        else:
            T = mass * (accel + g * (Rs_BADA * np.cos(gamma) - np.sin(gamma)))

        self.thrust_FB_BADA_seg = T_ / 2 / mid_values(self.pressure_ratios)

        #Get rid of first value since equation is not valid in the ground


    def plot_FB(self, op_type, up_sample=True):
        def mid_point(vec):
            return (vec[1:] - vec[:-1]) / 2 + vec[:-1]
        
        fig, ax = plt.subplots()
        
        if not up_sample:
            dist = mid_point(self.d_seg[1:])
        else:
            dist = self.d
        
        ax.plot(mid_point(dist), self.thrust_FB_BADA_seg, '-ko',
               label='Thrust_BADA')
        
        ax.set_xlabel('Dist [ft]')
        ax.set_ylabel('Thrust [lbs]')
        plt.ylim([0, 30000])


        if op_type == 'D':
            dist_def = [0    , 19000, 20000, 39000, 58000, 76000, 99000]
            thru_def = [25000, 20000, 15000, 15500, 16000, 16500, 17000]
            ax.plot(dist_def, thru_def, '-ro', label='ANP')
        else:
            dist_def = np.array([-275800.6, -128651.9,  -79602.3,  -62791.3,  -57243.4,  -49878.1,
            -37055.6,  -35784.9,  -34784.9,    -954.1,    -477. ,       0. ,
             303.5,    2731.6])
            dist_def = dist_def + dist_def[0]*-1
            thru_def = np.array([1.00000e+00, 1.00000e+00, 1.00000e+00, 1.24590e+02, 4.03040e+02,
            5.05710e+02, 9.43380e+02, 1.00688e+03, 4.44768e+03, 4.16992e+03,
            4.16615e+03, 4.16239e+03, 1.00000e+04, 2.50000e+03])
            h_def = np.array([15000. ,  6000. ,  3000. ,  3000. ,  3000. ,  2614. ,  1942. ,
        1875.4,  1823. ,    50. ,    25. ,     0. ,     0. ,     0. ])

            #ax.plot(dist_def[3:], thru_def[3:], '-ro', label='ANP')

        #    pass

        ax2 = ax.twinx()
        ax2.plot(self.d_seg, self.h_seg, '-bo', label='Altitude')
        ax2.set_ylabel('Alt [ft]')
        #ax2.plot(dist_def, h_def, '-ro', label='Default')

        
        fig.legend()
        plt.show()





        




             

    def generate_ANP_user_steps(self, id_):
        
        ROCs_filtered = np.empty(len(self.steps))
        ROCs_filtered[:] = np.NaN
        ROCs_filtered[self.steps=='Accelerate'] = self.ROCs[self.steps==\
                'Accelerate']
        dicti = {
                    'Step Type': self.steps,
                    'Thrust Rating':
                    ['MaxTakeoff' if thrust=='TO' else 'MaxClimb'\
                            for thrust in self.thrust_cfg],
                    'Flap_ID' : self.Flap_ID,
                    'End Point Altitude (ft)': self.End_Point_Alt,
                    'End Point CAS (kt)'     : self.End_Point_Cas,
                    'Rate Of Climb (ft/min)' : ROCs_filtered,
                    'Weight (lbs)' : self.weight_grad,
                    'Tflex (C)'    : self.Tflex_grad
                    }

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dicti.items()]))
        
        df.to_csv(os.path.join('output','Procedural_Steps_'+id_+'.csv'), index=False)

    def gen_fixed_point_profiles(self, id_, plot=True):
        
        d = np.r_[0, self.d_anp_g],
        h = np.r_[0, self.h_anp_g]
        d=np.reshape(d,(max(np.shape(d)),))
        h=np.reshape(h,(max(np.shape(h)),))
        self.h_seg = h
        self.d_seg = d
        cas    = np.r_[0, self.cas_anp_g]
        sigmas = self.Sigmas
        tas    = np.multiply(cas, 1/(np.sqrt(sigmas)))
        self.tas_segmented = tas
        dicti = {
                'Distance': d,
                'Altitude': h,
                'Speed'   : tas,
                'THR_SET' : self.THR_SET
                }

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dicti.items()]))
        df.to_csv(os.path.join('output', 'Fixed_Point_Profiles_'+id_+'.csv'), index=False)
        if plot:
            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'valid') / w
            w = 4

            mask = self.h > 0

            thrust_FB = self.thrust_FB
            
            T_smooth = moving_average(self.thrust_FB, w=w)
            
            plt.plot(d, self.THR_SET, 
                    '-ok', label='CNT')
            
            dist = self.d[mask]
            
            plt.plot(dist[1:], self.thrust_FB,
                '.b', label='Balance')
            plt.plot(dist[1:], self.thrust_FB_BADA,
                '.r', label='BADA')
            #plt.plot(dist[w:], T_smooth,
            #    '-r', label='w='+str(w))
            #w = 2
            #plt.plot(dist[w:], moving_average(self.thrust_FB, w=w),
            #    '-', label='w='+str(w)) 

            self.thrust_FB_segmented()
            T_seg = self.thrust_FB_seg

            
            plt.plot(self.d_seg[2:], self.thrust_FB_BADA_seg, 
                    '-oy', label='BADA_seg')

            
            plt.plot(self.d_seg[2:], T_seg, 
                    '-og', label='Segmented')
            plt.xlabel('Dist [ft]')
            plt.ylabel('Thrust [lbs]')
            plt.legend()
            #plt.show()
            plt.savefig(os.path.join('output', 'thrust_'+id_+'.png'))
            plt.close()


    
    def new_vert_profile_pinv(self, column_names, op_type = 'D'):
        
        print('Laying down new profile...')
        costs = []
        
        #weight = 127496.7607

        dist   = self.d_seg
        alt    = self.h_seg
        cas    = self.cas_seg
        times  = self.time_seg
        steps = self.steps
        #print('Laying down new profile...')
        
        def mid_values(vec):
            return (vec[1:] + vec[:-1]) / 2
        def sins_gamma_estimation(dist, alt, cas):
            climbs_deltaH = np.diff(alt)
            climbs_deltaS = np.diff(dist)
            sins_gamma = climbs_deltaH /\
                    np.sqrt(climbs_deltaH**2 + climbs_deltaS**2)
            return sins_gamma
        
        Tapt = self.Meteo[column_names['Temperature']].iloc[0]
        Papt = self.Meteo[column_names['Pressure']].iloc[0]

        #Tflex = (Tflex * 9/5) + 32.0 #Conversion to Farenheit from C (not ideal)

        
        Aero_coeff = self.Aerodynamic_coefficients
        dep_Aero_coeff = Aero_coeff[Aero_coeff["Op Type"] == op_type]
        Tapt = (Tapt * 9/5) + 32.0 #Conversion to Farenheit from C (not ideal)
        Papt = Papt / 33.864 #Conversion mbar to inHg (not ideal)
        

        sigmas       = air_density_ratio(alt, Tapt, Papt, self.Eapt)
        tas          = np.multiply(cas, 1/(np.sqrt(sigmas)))
        mean_sigmas  = mid_values(sigmas)
        mean_sigmas_ = air_density_ratio(mid_values(alt), Tapt, Papt, self.Eapt)

        tas_geom_mean = np.sqrt(mid_values(np.power(tas, 2))) #Impact eq-17
        tas_diff      = np.diff(np.power(tas, 2))
        deltas        = pressure_ratio_(alt, Papt, self.Eapt)
        mean_deltas   = mid_values(deltas)
        #print(deltas)
        #W_delta       = weight * np.reciprocal(mean_deltas)

        
        
        first_climb = True
        first_accel = True


        #Obtain flaps by type of step: Accel, TO, Climb for now
        dep_steps = self.Departure_steps[['Step Number', 'Step Type',
            'Thrust Rating', 'Flap_ID']]
        climb_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Climb']['Flap_ID'].unique()
        TO_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Takeoff']['Flap_ID'].unique() #climb_Flap_ID[0] 
        accel_Flap_ID = dep_steps[dep_steps['Step Type']==\
                'Accelerate']['Flap_ID'].unique()

        N = self.Aircraft['Number Of Engines'].iloc[0]

        Jet_eng_coeff = self.Jet_engine_coefficients
        thrust_rating = {'Reg'   : {'C': 'MaxClimb', 'TO' : 'MaxTakeoff'},
                         'HiTemp': {'C': 'MaxClimbHiTemp', 'TO' : 'MaxTkoffHiTemp'}
                         }
        thrust_type = 'HiTemp'

        
        #if thrust_type == 'HiTemp':
        #    midpoint_Temps = temperature_profile(mid_values(alt), Tflex)
        #else:
        #    midpoint_Temps = temperature_profile(mid_values(alt), Tapt)

        sins_gamma = np.zeros(np.shape(steps))
        segment_accel = np.zeros(np.shape(steps))
        ROCs = np.multiply(np.diff(alt), 1/(np.diff(times))) * 60.
        self.ROCs = ROCs #ft/min
        
        sins_gamma_RADAR = sins_gamma_estimation(dist, alt, cas)
        accel_seg_RADAR  = np.diff(dist)
        
        A = np.zeros((len(steps) - 1 + 1, 2)) #-1 no takeoff +1 first CAS
        b = np.zeros((len(steps) - 1 + 1, 1)) #-1 no takeoff +1 first CAS

        for i, step in enumerate(steps):
            if step == 'TakeOff':
                
                pass
                #B8 = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                #        TO_Flap_ID[0]]['B'].iloc[0]
                #theta = temperature_ratio_(Alt=0, Tapt=Tapt)
                #
                #W_delta_TO = weight / pressure_ratio_(Alt=0, Papt=Papt)
                #
                #thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                #        thrust_rating[thrust_type]['TO']].reset_index(drop=True)
                #
                #if thrust_type == 'HiTemp':
                #    T_thrust = Tflex
                #else:
                #    T_thrust = Tapt

                ##Should be CAS of first point
                #Fn_delta = corrected_net_thrust(thrust_coeff,mid_values(cas)[i+1],
                #    temperature_profile(Alt=0, Tapt=T_thrust), Alt=0, Papt=Papt)
                #est_TO8 = TO_ground_roll_distance(B8, theta, W_delta_TO, N, 
                #        Fn_delta)
            
            elif step == 'Climb' and first_climb:
                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        TO_Flap_ID[0]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type]['TO']].reset_index(drop=True)

                #Fn_delta = corrected_net_thrust(thrust_coeff, mid_values(cas)[i],
                #        midpoint_Temps[i], mid_values(alt)[i], Papt)
                ##print(Fn_delta)
                #sins_gamma[i] = sin_of_climb_angle(N, Fn_delta, W_delta[i], R,
                #        mid_values(cas)[i])

                alpha = sin_of_climb_angle_pinv(N, sins_gamma_RADAR[i], R,
                        (cas)[i+1])

                beta, H = corrected_net_thrust_pinv(thrust_coeff, 
                        (cas)[i+1], mid_values(alt)[i], Papt)

                A[i-1, :] = [alpha / mean_deltas[i], -H]
                b[i-1]      = beta

                #FIRST CAS

                C = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        TO_Flap_ID[0]]['C'].iloc[0]

                #est_first_climb_CAS = first_climb_CAS(C, weight)
                radar_first_CAS = mid_values(cas)[i]
                
                ksi = first_climb_CAS_pinv(C, radar_first_CAS)
                
                A[-1, :] = [1, 0]
                b[-1]      = ksi
                
                first_climb = False


            elif step == 'Climb' and first_climb==False:

                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        climb_Flap_ID[1]]['R'].iloc[0]
                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                        thrust_rating[thrust_type]['C']].reset_index(drop=True)

               # Fn_delta = corrected_net_thrust(thrust_coeff, mid_values(cas)[i],
               #         midpoint_Temps[i], mid_values(alt)[i], Papt)

               # sins_gamma[i] = sin_of_climb_angle(N, Fn_delta, W_delta[i], R,
               #         mid_values(cas)[i])
                
                alpha = sin_of_climb_angle_pinv(N, sins_gamma_RADAR[i], R,
                        (cas)[i+1])

                beta, H = corrected_net_thrust_pinv(thrust_coeff, 
                        (cas)[i+1], mid_values(alt)[i], Papt)

                A[i-1, :] = [alpha / mean_deltas[i], -H]
                b[i-1]      = beta
            
            elif step == 'Accelerate' and first_accel:
                

                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[1]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type]['C']].reset_index(drop=True)

               # Fn_delta = corrected_net_thrust(thrust_coeff, mid_values(cas)[i],
               #     midpoint_Temps[i], mid_values(alt)[i], Papt)
               # segment_accel[i] = distance_accelerate(tas_diff[i],
               #     tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])

                phi = distance_accel_pinv(tas_diff[i], tas_geom_mean[i], N, 
                        accel_seg_RADAR[i], R, ROCs[i])

                beta, H = corrected_net_thrust_pinv(thrust_coeff, 
                        cas[i+1], mid_values(alt)[i], Papt)

                A[i-1, :] = [phi / mean_deltas[i], -H]
                b[i-1]      = beta
                
                
                first_accel = False
                second_accel = True
            elif step == 'Accelerate' and first_accel == False and \
                    second_accel==True:

                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[1]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type]['C']].reset_index(drop=True)

                #Fn_delta = corrected_net_thrust(thrust_coeff, mid_values(cas)[i],
                #    midpoint_Temps[i], mid_values(alt)[i], Papt)
                #segment_accel[i] = distance_accelerate(tas_diff[i],
                #    tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])
                
                phi = distance_accel_pinv(tas_diff[i], tas_geom_mean[i], N, 
                        accel_seg_RADAR[i], R, ROCs[i])

                beta, H = corrected_net_thrust_pinv(thrust_coeff, 
                        (cas)[i+1], mid_values(alt)[i], Papt)

                A[i-1, :] = [phi / mean_deltas[i], -H]
                b[i-1]      = beta
             
                second_accel = False
            elif step == 'Accelerate' and first_accel == False and \
                    second_accel==False:

                R = dep_Aero_coeff[dep_Aero_coeff['Flap_ID']==\
                        accel_Flap_ID[2]]['R'].iloc[0]

                thrust_coeff = Jet_eng_coeff[Jet_eng_coeff['Thrust Rating']==\
                    thrust_rating[thrust_type]['C']].reset_index(drop=True)

                #Fn_delta = corrected_net_thrust(thrust_coeff, mid_values(cas)[i],
                #    midpoint_Temps[i], mid_values(alt)[i], Papt)
                #segment_accel[i] = distance_accelerate(tas_diff[i],
                #    tas_geom_mean[i], N, Fn_delta, W_delta[i], R, ROCs[i])
                    
                phi = distance_accel_pinv(tas_diff[i], tas_geom_mean[i], N, 
                        accel_seg_RADAR[i], R, ROCs[i])

                beta, H = corrected_net_thrust_pinv(thrust_coeff, 
                        (cas)[i+1], mid_values(alt)[i], Papt)

                A[i-1, :] = [phi / mean_deltas[i], -H]
                b[i-1]      = beta
        
            
        x = np.dot(np.linalg.pinv(A), b)       
        self.weight_pinv = x[0]
        self.Tflex_pinv  = x[1]
