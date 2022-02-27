import xml.etree.ElementTree as ET
import numpy as np
import pdb

tree = ET.parse('data/A320-212/A320-212.xml')
root = tree.getroot()

AFCM = root.findall('./AFCM/')

speed = 200 

S = AFCM[0]
configurations = AFCM[1:] #First one is always the Surface area S

surf_area = np.float64(S.text)

speeds = np.array(
        [int(config.findall('vfe')[0].text) for config in configurations])
#List of speeds per configuration arranged from higher to lower speed

configs_indexes = np.arange(speeds.size) #

current_config = lambda speed: configurations[max(configs_indexes[speeds>speed])]


drag_polys_coeff = np.zeros(3)

lg_state = 'LGUP'

drag_elems = list(list(current_config(speed).findall(lg_state)[0])[0])[0].findall('d')

for i, d in enumerate(drag_elems):
    print(d.text)    
    drag_polys_coeff[i] = np.float64(d.text)
