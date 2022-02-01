import numpy as np
def aircraft_motion(y, t, T, m, alpha, hd, rd):
    v     = y[0]
    gamma = y[1]
    h     = y[2]
    r     = y[3]
    
    T = controller(h, r, hd(t), rd(t))

    #TODO big assumptions here improve with BADA
    C_D = .068547 
    S   = 30 #m
    rho = 1 #kg/m^3 
    C_L = 1
    g   = 9.81 #m/s^2

    v_dot = (T*np.cos(alpha) - C_D*rho*v**2*S/2) / m - g*np.sin(gamma)
    gamma_dot = (T*np.sin(alpha) + C_L*rho*v**2*S/2) / (m*v) - g/v*np.cos(gamma)
    h_dot = v*np.sin(gamma)
    r_dot = v*np.cos(gamma)

    y_dot = np.zeros((len(y),1))
    
    y_dot[0] = v_dot
    y_dot[1] = gamma_dot
    y_dot[2] = h_dot
    y_dot[3] = r_dot
    
    return y_dot

def controller(h, r, hd, rd):
    eh = h - hd
    er = r - rd
    kph = 1 
    kpr = 1
    T = kph*eh + kpr*er
    return T
