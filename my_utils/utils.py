# This python script contains the basic formulas for power spectrum and bispectrum.

import numpy as np
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
from cosmoprimo import *

from scipy import integrate
from numba import jit

def smoothPk(k,A0,keq,a0,a2,a4,n):
    '''
    Eisenstein smooth equation
    '''
    k = np.array(k)
    q = k/keq
    L = np.log(2*np.exp(1)+1.8*q)
    C = 14.2+731/(1+62.5*q)
    T = L/(L+C*(q**2))
    return A0*(a0+(T**2)*(k**n)) #+a2*(k**2)+a4*(k**4))

def fit_smoothPk(Pkcamb,k,method = "cosmoprimo"):
    if len(Pkcamb.shape) == 2:
        Pcamb = Pkcamb[:,1]
        kcamb = Pkcamb[:,0]
    else:
        Pcamb = Pkcamb
        kcamb = k
    kcamb[0]=0.001
    Pcamb[0]=0.001
    if method == "cosmoprimo":
        pk_interpolator = PowerSpectrumInterpolator1D(kcamb,Pcamb)
        fpk = pk_interpolator(k)
        pk_smooth_interpolator = PowerSpectrumBAOFilter(pk_interpolator, engine='wallish2018').smooth_pk_interpolator()
        ps = pk_smooth_interpolator(k)
    else:
        fpk = np.interp(k,kcamb,Pcamb)
        Ps = lambda k,A0,keq,a0,a2,a4,n : smoothPk(k,A0,keq,a0,a2,a4,n)
        p0 = [30000,0.05,0,0,0,0]
        popt, _ = curve_fit(Ps, k ,fpk, p0, (1/k)**2, maxfev=1000000)
        ps = smoothPk(k,*popt)
    return fpk,ps

def less_baoPk(Pkcamb,k,E,method = "cosmoprimo"):
    fpk,ps = fit_smoothPk(Pkcamb,k,method)
    less_bao = (fpk/ps - 1)*np.exp(-(E*k)**2)
    return fpk, ps, ps*(1+less_bao)

def cutslice(kmin,kmax,kk,bkm,bk,bknm,bkn,bk_mol,bao):
    is_good = np.ones(kk.shape[0], '?')
    for i in range(3):is_good &= (kk[:, i] > kmin) & (kk[:, i] <= kmax)
    kg = kk[is_good, :]
    bg = bkm[is_good]
    bgn = bknm[is_good]
    nbins, nmocks = bk[is_good, :].shape
    hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)

    glam_cov = np.cov(bk[is_good, :], rowvar=True)#/ nmocks
    glam_std = np.std(bk[is_good], axis=1)   
    bao_cov = np.cov(bao[is_good, :], rowvar=True)

    mol_cov_ = np.cov(bk_mol[is_good,:], rowvar=True)
    mol_std_ = np.diagonal(mol_cov_)**0.5
    
    red_cov = mol_cov_/np.outer(mol_std_, mol_std_)
    scaled_cov = red_cov*np.outer(glam_std, glam_std)

    print(f'kmax={kmax}, kmin={kmin}, nbins={nbins}, nmocks={nmocks}, hf = {hartlapf}')
    return kg,bg,bgn,glam_cov,scaled_cov,hartlapf,bao_cov

@jit
def Bisp(var, parc, pk1, pk2, pk3, linear = True, model = "SPT"):
    '''
    Bispectrum using GEOFPT or SPT as a function of 5 variables - k1,k2,k3,mu1,phi12 and cosmological
    parameters parc.
    
    model = GEO or SPT 
    
    '''
    k1, k2, k3, mu1, phi12 = var
    f, b1, b2 = parc

    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3
    
    # #Can also be evaluated using the following equations.
    # mu12 = -(k1**2 + k2**2 - k3**2)/(2*k1*k2) 
    # mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    # mu3 = -(mu1*k1 + mu2*k2)/k3
    # mu31 = -(k3**2 + k1**2 - k2**2)/(2*k3*k1)
    # mu23 = -(k2**2 + k3**2 - k1**2)/(2*k2*k3)

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
    
    if model == "GEO":
        multi_factor = Z2_geofpt(var, z = 0.5)
        Z2k12 = Z2k12 * multi_factor
        Z2k23 = Z2k23 * multi_factor
        Z2k31 = Z2k31 * multi_factor
        linear = False

    if not linear:
        I = 2*(b1**2+f**2/5+2*b1*f/3)
        pk1 = pk1/I
        pk2 = pk2/I
        pk3 = pk3/I
    
    Bi = Z2k12*Z1k1*Z1k2*pk1*pk2
    Bi += Z2k23*Z1k2*Z1k3*pk2*pk3
    Bi += Z2k31*Z1k3*Z1k1*pk3*pk1

    return 2.*Bi

@jit
def Z2_geofpt(var, z = 0.5):
    '''
    Calculates the constant geometry dependent multiplicative factor of Z2 in GEOFPT model
    '''

    k1, k2, k3, mu1, phi12 = var
    
    bf_fs = {0.5: (1.0033, -0.0040, 0.0240, -0.0568, 0.01325), 1: (1.0180, -0.0041, 0.0149, -0.0547, 0.01144), 2 : (1.037, 0.0020, 0.0056, -0.048, 0.0081)}
    
    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3
    
    ext_angles = np.sort([np.arccos(mu12), np.arccos(mu31), np.arccos(mu23)])
    theta_min, theta_med, theta_max = ext_angles[0], ext_angles[1], ext_angles[2]
    f1,f2,f3,f4,f5 = bf_fs[z]
    
    # calculate the semi-perimeter of k-triangle 
    s = (k1 + k2 + k3) / 2  
  
    # calculate the area  
    A = (s*(s-k1)*(s-k2)*(s-k3)) ** 0.5 
    A_norm = 0.001
    
    multi_factor = f1 + (f2*theta_med/theta_max) + (f3*theta_min/theta_max) + (f4*A/A_norm) + (f5*(A/A_norm)**2)
    
    return multi_factor

def Bi0(kk, pk1, pk2, pk3, f, b1, b2, S0, S1, linear = True, model = "SPT"):
    '''
    Integrate Bisp to get the monopole
    '''
    Y00=np.sqrt(1/np.pi)/2
    k1,k2,k3=kk[:,0],kk[:,1],kk[:,2]
    parc=(f,b1,b2)
    out=[]
    for i in range(len(k1)):
        func = lambda phi12,mu1: Bisp((k1[i],k2[i],k3[i],mu1,phi12), parc,pk1[i],pk2[i],pk3[i],linear = linear, model = model)
        ans,_ = integrate.dblquad(func, -1, 1, lambda x: 0, lambda x: np.pi)
        ans = ans + S0 + S1 * (pk1[i] + pk2[i] + pk3[i]) # Add extra nuissance params. This term is not a part of scoccimaro equation. Skip this step if you want to use only scoccimaro eq.
        out.append(2*Y00*ans)
    return np.array(out)

def Bi_wiggle(kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1, linear = True, model = "SPT"):
    '''
    pkm - full mean power spectrum
    pknm - nobao/smooth mean power spectrum
    k - respective k-values at which power spectrum is measured
    kk - k-triplets where bispectrum is measured
    f,b1,b2 - free parameters
    '''
    B_full = Bi0(kk,pk1,pk2,pk3,f,b1,b2,S0,S1, linear = linear, model = model)
    B_nbao = Bi0(kk,pkn1,pkn2,pkn3,f,b1,b2,S0,S1, linear = linear, model = model)
    B_wig = np.array(B_full)/np.array(B_nbao)
    return B_wig
    

if __name__ == '__main__':
    Bisp(var,parc,pk1,pk2,pk3,pars=(2000**3,1000**3))
    Bi0(kk,pk1,pk2,pk3,f,b1,b2)
    Bi_wiggle(kk, pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2)
    Bi_meas(kk, pk1,pk2,pk3,pkn1,pkn2,pkn3, alpha,f,b1,b2,f1,b11,b21)