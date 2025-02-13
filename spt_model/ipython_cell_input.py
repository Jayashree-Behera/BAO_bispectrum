
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from my_utils.mcmc.mcmc import RunMCMC
from my_utils.utils import cutslice

import scipy.interpolate as interpolate
from scipy import integrate
import scipy.stats
import math

from tqdm import tqdm
import time
import emcee                    # for MCMC part
import warnings                 # for ignoring the warnings (not recommended)
warnings.filterwarnings('ignore')

def Bisp(var, parc, fpk, pars=(2000**3,1000**3)):
    '''
    Bispectrum as a function of 5 variables - k1,k2,k3,mu1,phi12 and cosmological
    parameters parc. This is the scoccimaro equation for bispectrum
    '''
    k1, k2, k3, mu1, phi12 = var
    #apar, aper, f, b1, b2 = parc
    f, b1, b2 = parc
    navg, Vs = pars

    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu2 = mu1*mu12 - np.sqrt((1 - mu1**2)*(1 - mu12**2))*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3

    Z1k1 = b1 + f*mu1**2
    Z1k2 = b1 + f*mu2**2
    Z1k3 = b1 + f*mu3**2

    F12 = 5./7. + mu12/2*(k1/k2 + k2/k1) + 2./7.*mu12**2
    F23 = 5./7. + mu23/2*(k2/k3 + k3/k2) + 2./7.*mu23**2
    F31 = 5./7. + mu31/2*(k3/k1 + k1/k3) + 2./7.*mu31**2

    G12 = 3./7. + mu12/2*(k1/k2 + k2/k1) + 4./7.*mu12**2
    G23 = 3./7. + mu23/2*(k2/k3 + k3/k2) + 4./7.*mu23**2
    G31 = 3./7. + mu31/2*(k3/k1 + k1/k3) + 4./7.*mu31**2

    Z2k12 = b2/2. + b1*F12 + f*mu3**2*G12
    Z2k12 -= f*mu3*k3/2.*(mu1/k1*Z1k2 + mu2/k2*Z1k1)
    Z2k23 = b2/2. + b1*F23 + f*mu1**2*G23
    Z2k23 -= f*mu1*k1/2.*(mu2/k2*Z1k3 + mu3/k3*Z1k2)
    Z2k31 = b2/2. + b1*F31 + f*mu2**2*G31
    Z2k31 -= f*mu2*k2/2.*(mu3/k3*Z1k1 + mu1/k1*Z1k3)

    Bi = 2*Z2k12*Z1k1*Z1k2*fpk(k1)*fpk(k2)
    Bi += 2*Z2k23*Z1k2*Z1k3*fpk(k2)*fpk(k3)
    Bi += 2*Z2k31*Z1k3*Z1k1*fpk(k3)*fpk(k1)

    return Bi

def Bi0(kk, fpk, f, b1, b2):
    '''
    Integrate Bisp to get the monopole
    '''
    Y00=np.sqrt(1/np.pi)/2
    k1,k2,k3=kk[:,0],kk[:,1],kk[:,2]
    parc=(f,b1,b2)
    out=[]
    for i in range(len(k1)):
        f = lambda phi12,mu1: Bisp((k1[i],k2[i],k3[i],mu1,phi12), parc, fpk, pars=(2000**3,1000**3))*Y00
        ans,error = integrate.dblquad(f, -1, 1, lambda x: 0, lambda x: 2*np.pi)
        out.append(ans)
    return out


Bk=np.load("data/glam/bk_z0.50.npz")
Pk=np.load("data/glam/pk_z0.50.npz")
k,pkm,pknm,pk,pkn=Pk['k'], Pk['pkm'].T[0],Pk['pknm'].T[0],Pk['pk'],Pk['pkn']
kk,bkm,bknm,bk,bkn=Bk['k'], Bk['bkm'].T[0], Bk['bknm'].T[0],Bk['bk'],Bk['bkn']

fpk=interpolate.interp1d(k,pkm)
fpk_nobao=interpolate.interp1d(k,pknm)

#getting glam full_bispectrum data and it's covariance 
kg,bg,cov,_=cutslice(0.005,0.05,kk,bkm,bk)

B_mono = Bi0(kg,fpk,1,1,1)

# for i in tqdm(range(1)):
#     #B_full = Bisp((kg[:,0],kg[:,1],kg[:,2],1,0),(1,1,1),fpk,pars=(2000**3,1000**3))
#     B_mono = Bi0(kg,fpk,1,1,1)

#     #print(B_full)
#     print(B_mono)
