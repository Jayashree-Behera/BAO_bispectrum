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

# Bk=np.load("data/glam/bk_z0.50_mehdi.npz")
# Pk=np.load("data/glam/pk_z0.50_mehdi.npz")
# k,pkm,pknm,pk,pkn=Pk['k'], Pk['pkm'].T[0],Pk['pknm'].T[0],Pk['pk'],Pk['pwithbao_kn']
# kk,bkm,bknm,bk,bkn=Bk['k'], Bk['bkm'].T[0], Bk['bknm'].T[0],Bk['bk'],Bk['bkn']

Bk=np.load("data/withbao_glam/bk_z0.50_03.npz")
Pk=np.load("data/withbao_glam/pk_z0.50_03.npz")
k,pkm,pknm,pk,pkn=Pk['k'], Pk['pkm'],Pk['pknm'],Pk['pk'],Pk['pkn']
kk,bkm,bknm,bk,bkn,bao=Bk['kk'], Bk['bkm'], Bk['bknm'],Bk['bk'],Bk['bkn'],Bk['bk_bao']

kg,bg,bgn,cov,scaled_cov,_, bao_cov = cutslice(0,0.2,kk,bkm,bk,bknm,bkn,bk_mol,bao)
icov = np.linalg.inv(bao_cov)

def prepare_pk(k,E = None, use_glam = True):
    if use_glam:
        pkmf = pkm
        pknmf = pknm
        
    else:
        pkcamb = np.loadtxt(filepath)
        pkcamb[:,1] = pkcamb[:,1]/np.max(pkcamb[:,1]) * np.max(pkm)
        _,pknmf,pkmf = less_baoPk(pkcamb,k,E)
        # pknmf = pknmf/np.mean(pknmf) * np.mean(pkm)
        # pkmf = pkmf/np.mean(pkmf) * np.mean(pknm) 
    
    return pkmf,pknmf

def model_fullBk(kk,*args):
    alpha,f,b1,b2,E,S0,S1= args
    
    pkmf,pknmf = prepare_pk(k,E,use_glam = False)

    pk1=np.interp(alpha*kk[:,0],k,pkmf)
    pk2=np.interp(alpha*kk[:,1],k,pkmf)
    pk3=np.interp(alpha*kk[:,2],k,pkmf)
    res = Bi0(alpha*kk,pk1,pk2,pk3,f,b1,b2,S0,S1) #Bi_wiggle(alpha*kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1)
    return res

def curvefit_fullBk(kk,bkm):
    k1,k2,k3=kk[:,0],kk[:,1],kk[:,2]
    r=np.array(k1*k2*k3)**2
    model = lambda kk,alpha,f,b1,b2,E,S0,S1 : model_fullBk(kk,alpha,f,b1,b2,E,S0,S1)
    b0 = [  1,   0.09231214,   1.77502499,  -0.93034206, 4.19969726, -18.3910402 ,  23.34651357]
    bopt, bcov = curve_fit(model_fullBk,kk,bkm,b0,1/r,maxfev=1000000)
    print(bopt)
    return bopt

def model(kk,*args):
    alpha,f,b1,b2,E,S0,S1 = args
    # f,b1,b2,S0,S1 = 0.09226504,   1.77488279,  -0.93236249, -18.54630872,  23.46928206 #k<=0.2
    # f,b1,b2,S0,S1 =   0.03416564,  1.80925269, -0.68822191, 2.89850395, 11.94816641 #full k

    pkm,pknm = prepare_pk(k,E,use_glam = False)

    pk1=np.interp(alpha*kk[:,0],k,pkm)
    pk2=np.interp(alpha*kk[:,1],k,pkm)
    pk3=np.interp(alpha*kk[:,2],k,pkm)

    pkn1=np.interp(alpha*kk[:,0],k,pknm)
    pkn2=np.interp(alpha*kk[:,1],k,pknm)
    pkn3=np.interp(alpha*kk[:,2],k,pknm)

    # pkn11=np.interp(kk[:,0],k,pknm)
    # pkn12=np.interp(kk[:,1],k,pknm)
    # pkn13=np.interp(kk[:,2],k,pknm)

    res = Bi_wiggle(alpha*kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1, model = "SPT")      #*Bi0(kk,pkn11,pkn12,pkn13,f,b1,b2,S0,S1)
    return res

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default=os.path.join("../my_utils","mcmc_params.yaml"))
    args.add_argument("--filepath",default = "data/camb_new/camb_python/camb_67773070_matterpower_z0.5_0060.dat")
    #args.add_argument("--output_path",default = None)
    parsed_args = args.parse_args()
    filepath = parsed_args.filepath
    print(filepath)
    # _,f,b1,b2,_,S0,S1 = curvefit_fullBk(kg,bg)
        
    output_path = "/global/homes/s/shreeb/BAO_bispectrum/spt_model/fid_fits/"+ Path(parsed_args.filepath).stem + ".h5"
    mcmc_chain = RunMCMC(model,kg,bg/bgn,icov,config_path = parsed_args.config)
    mcmc_chain.mcmc_run(output_path = output_path)
