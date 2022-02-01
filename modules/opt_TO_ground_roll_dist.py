# W=x[0]; alpha=x[1]
import numpy as np
def cost(x, B8, theta, delta, E, F, C, Ga, Gb, h, T, H, d_r):
    d = B8*theta*(x[0]/delta)**2/(x[1]*(E+F*C*np.sqrt(x[1])+Ga*h+Gb*h**2+H*T))
    return (d - d_r)**2
