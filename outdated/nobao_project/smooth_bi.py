import numpy as np
import argparse
import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from my_utils.mcmc.mcmc import RunMCMC
from my_utils.utils import *

import math
import warnings          # for ignoring the warnings (not recommended)
warnings.filterwarnings('ignore')

Bk_molino=np.load("../Molino_mocks/bk_molino_z0.npz")
bk_mol = Bk_molino["bk0"]

redshifts = ["BGS","ELG","LRG","QSO"]
i=0

Bk=np.load("data/Abacus/"+ redshifts[i] +"/bk_"+redshifts[i]+".npz")
Pk=np.load("data/Abacus/"+ redshifts[i] +"/pk_"+redshifts[i]+".npz")
k,pkm,pk=Pk['k'], Pk['pkm'].T[0],Pk['pk']
kk,bkm,bk=Bk['kk'], Bk['bkm'].T[0],Bk['bk']

kg,bg,bgn,cov,scaled_cov,_ = cutslice(0.015,0.3,kk,bkm,bk,bknm,bkn,bk_mol)

icov = np.linalg.inv(scaled_cov)

def prepare_pk(E = None, use_glam = False):
    if not use_glam:
        pkcamb = np.loadtxt(pkm)
        _,pknmf,pkmf = less_baoPk(pkm,k,E)
        pknmf = pknmf/np.mean(pknmf) * np.mean(pkm)
        pkmf = pkmf/np.mean(pkmf) * np.mean(pknm)
    else:
        pkmf = pkm
        pknmf = pknm
    
    return pkmf,pknmf

def model(kk,*args):
    f,b1,b2,E,S0,S1 = args

    pkm,pknm = prepare_pk(E, use_glam = False)

    pk1=np.interp(kk[:,0],k,pkm)
    pk2=np.interp(kk[:,1],k,pkm)
    pk3=np.interp(kk[:,2],k,pkm)

    pkn1=np.interp(kk[:,0],k,pknm)
    pkn2=np.interp(kk[:,1],k,pknm)
    pkn3=np.interp(kk[:,2],k,pknm)

    pkn11=np.interp(kk[:,0],k,pknm)
    pkn12=np.interp(kk[:,1],k,pknm)
    pkn13=np.interp(kk[:,2],k,pknm)

    res = Bi_wiggle(kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1)*Bi0(kk,pkn11,pkn12,pkn13,f,b1,b2,S0,S1)
    return res

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default=os.path.join("../my_utils","mcmc_params.yaml"))
    # args.add_argument("--output_path",default = None)
    parsed_args = args.parse_args()

    filepath = parsed_args.filepath
    output_path = "mcmc_abacus/mcmc_" + redshifts[i] + ".h5"
        
    mcmc_chain = RunMCMC(model,kg,bg,icov,config_path = parsed_args.config)
    
    mcmc_chain.mcmc_run(output_path = output_path)
