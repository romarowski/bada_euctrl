import numpy as np
from modules.NoiseCraft import NoiseCraft
def thrust(m, TAS, R_f, gamma):
    kts2ftps = 1.68781
    lbs2slug = 0.031081
    g = 32.1740 #ft/s^2
    m = m * lbs2slug
    #TAS kts
    TAS *= kts2ftps
    return m * (TAS + g * (R_f * np.cos(gamma) - np.sin(gamma)))

