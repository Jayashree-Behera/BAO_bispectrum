# this script will have functions to specifically download the matter power spectrum 
# using different simulations for a given set of cosmological parameters.

from fileinput import filename
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import sys, platform, os

#camb imports
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

class CAMB:
    def __init__(self):
        self.pars = camb.CAMBparams(WantCls = False,DoLensing = False, NonLinear = False)

    def cosmology(self, H0 = 70, ombh2 = 0.0226, omch2 = 0.112, omk = -0.0):
        self.pars.set_cosmology(H0 = H0, ombh2 = ombh2, omch2 = omch2, omk = omk);

    def initial_power(self, pars_initpower = (2.46e-9, 0.96, 0.005, 0.005, 0)):
        As, ns, ps, pt, r = pars_initpower
        self.pars.InitPower.set_params(As = As, ns = ns, pivot_scalar = ps, pivot_tensor = pt, r = r);

    def set_z_k(self, all_z, kmax = 0.5 , linear_only = True):
        self.pars.set_matter_power(redshifts = all_z, kmax = kmax);
        if linear_only:
            self.pars.NonLinear = model.NonLinear_none
        else:
            # Non-Linear spectra (Halofit)
            self.pars.NonLinear = model.NonLinear_both

    def results(self,H0, ombh2, omch2, omk, all_z):
        '''
        all_z : list of redshifts (should be a list even if it's a single value)
        output : an instance of the results which can be used to get the power spectrum
        '''
        self.cosmology(H0, ombh2, omch2, omk)
        self.initial_power()
        self.set_z_k(all_z)
        results = camb.get_results(self.pars)
        #kh, z, pkh = results.get_matter_power_spectrum(maxkh = 0.3/0.7, npoints = 200)

        return results
    


        





