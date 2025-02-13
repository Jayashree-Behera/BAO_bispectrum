# To get camb data for a given list of H0 and omega_c * h^2

import numpy as np
import os, sys
from pathlib import Path
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from my_utils.call_data import CAMB

camb = CAMB()

list_H0 = np.linspace(67.77-2.5,67.77+2.5,11)
list_Om = np.linspace(0.307-0.03,0.307+0.03,11)
list_z = [0.5]

k = 0
for h0 in list_H0:
    for om in list_Om: 
        ombh = 0.04825*(h0/100)**2
        omch = (om-0.04825)*(h0/100)**2
        results = camb.results(H0 = h0, ombh2 = ombh, omch2 = omch, omk = -0.0, all_z = list_z);
        
        filename = f"camb_{int(np.round(h0,2)*100)}{int(om*10000)}_matterpower_z{''.join(map(str, list_z))}_{k:04}.dat"
        parent_folder = "/global/u1/s/shreeb/Project1/nobao_project/data/camb_new/camb_python/"
        infoname = f"infocamb_{int(np.round(h0,2)*100)}{int(om*10000)}_matterpower_z{''.join(map(str, list_z))}_{k:04}.npy"
        info_folder = "/global/u1/s/shreeb/Project1/nobao_project/data/camb_new/camb_python_info/"
        print(k)
        np.save(info_folder + infoname, results.get_derived_params())
        kh,z,pkh = results.get_matter_power_spectrum(maxkh = 0.3/0.7, npoints = 200)
        pk_kh = np.vstack((kh,pkh)).T
        np.savetxt(parent_folder + filename, pk_kh)
        k=k+1

# list_H0 = np.linspace(64,72,10)
# list_omch2 = np.linspace(0.112,0.142,10)
# list_z = [0.5]

# k = 0
# for h0 in list_H0:
#     for omch in list_omch2: 
#         results = camb.results(H0 = h0, ombh2 = 0.04825*(h0/100)**2, omch2 = omch, omk = -0.0, all_z = list_z);
#         kh,z,pkh = results.get_matter_power_spectrum(maxkh = 0.3/0.7, npoints = 200)
#         pk_kh = np.vstack((kh,pkh)).T
        
#         filename = f"camb_{int(h0*100)}{int(omch*10000)}_matterpower_z{''.join(map(str, list_z))}_{k:04}.dat"
#         parent_folder = "/global/u1/s/shreeb/Project1/nobao_project/data/camb_python_info/"
#         print(filename)
#         np.savetxt(parent_folder + filename, pk_kh)
#         k+=1