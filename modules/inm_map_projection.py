import numpy as np
def parameters(origin):
    #Calculates constants needed for coordinate transformation
    
    A = 6378137.000 #m
    B = 6356752.314 #m
    origin *= np.pi / 180
    lat_o = origin[0] 
    lon_o = origin[1] 

    cos_o = np.cos(lat_o)
    sin_o = np.sin(lat_o)
    
    Rp =                A**2 /                         \
        np.sqrt(A**2 * cos_o**2 + B**2 * sin_o**2)
    Rm = Rp**3 * B**2 / A**4

    Eo = .5 * np.tan(lat_o) / Rp

    parameters = {'cos_o' : cos_o,
                  'sin_o' : sin_o,
                  'Rp'    : Rp,
                  'Rm'    : Rm,
                  'Eo'    : Eo,
                  'origin': origin,
                 }
    return parameters

def project(point, parameters):
    cos_o = parameters['cos_o']
    sin_o = parameters['sin_o']
    Rp = parameters['Rp']
    Rm = parameters['Rm']
    Eo = parameters['Eo']
    origin = parameters['origin']
    
    point *= np.pi/180

    lat_o = origin[0]
    lon_o = origin[1]
    lat_p = point[0]
    lon_p = point[1]
    
    y_o = Rm * (lat_p - lat_o)

    if np.abs(lon_o - lon_p) < 1e-10:
        x = 0
        y = y_o
    else:
        x = (Rp * cos_o - y_o * sin_o) * (lon_p - lon_o)
        y = y_o + Eo * x**2

    return [x, y]

