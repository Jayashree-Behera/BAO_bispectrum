import numpy as np
import argparse
import pathlib
import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
sys.path.append("/global/homes/s/shreeb/BAO_bispectrum/nobao_project/slurm")
from my_utils.mcmc.mcmc import RunMCMC
from my_utils.utils import *

import math
import warnings          # for ignoring the warnings (not recommended)
warnings.filterwarnings('ignore')

Bk_molino=np.load("../Molino_mocks/bk_molino_z0.npz")
bk_mol = Bk_molino["bk0"]

def prepare_pk(k,E = None, use_glam = True):
    if use_glam:
        pkmf = pkm
        pknmf = pknm     
    else:
        pkcamb = np.loadtxt(filepath)
        # pkcamb[:,1] = pkcamb[:,1]/np.max(pkcamb[:,1]) * np.max(pkm)
        _,pknmf,pkmf = less_baoPk(pkcamb,k,h,E)
    # plt.plot(pkmf/pknmf)
    return pkmf,pknmf

def model_fullBk(kk,*args):
    alpha,f,b1,b2,E,S0,S1= args
    
    pkmf,pknmf = prepare_pk(k,E,use_glam = False)

    pk1=np.interp(alpha*kk[:,0],k,pkmf)
    pk2=np.interp(alpha*kk[:,1],k,pkmf)
    pk3=np.interp(alpha*kk[:,2],k,pkmf)
    
    pkn1=np.interp(alpha*kk[:,0],k,pknm)
    pkn2=np.interp(alpha*kk[:,1],k,pknm)
    pkn3=np.interp(alpha*kk[:,2],k,pknm)
    
    res = Bi0(alpha*kk,pk1,pk2,pk3,f,b1,b2,S0,S1) #Bi_wiggle(alpha*kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1) #
    return res

def curvefit_fullBk(kk,bkm):
    k1,k2,k3=kk[:,0],kk[:,1],kk[:,2]
    r=np.array(k1*k2*k3)**2
    model = lambda kk,alpha,f,b1,b2,E,S0,S1 : model_fullBk(kk,alpha,f,b1,b2,E,S0,S1)
    b0 = [  1,   0.09231214,   1.77502499,  -0.93034206, 4.19969726, 0 ,  10]
    bopt, bcov = curve_fit(model_fullBk,kk,bkm,b0,1/r,maxfev=100000)
    print(bopt)
    pkmf,pknmf = prepare_pk(k,bopt[-3],use_glam = False)
    # plt.plot(pkmf/pknmf)
    return bopt

def model(kk,*args):
    # alpha,f,b1,b2,E,S0,S1 = args
    alpha,E = args

    pkm,pknm = prepare_pk(k,E,use_glam = False)

    pk1=np.interp(alpha*kk[:,0],k,pkm)
    pk2=np.interp(alpha*kk[:,1],k,pkm)
    pk3=np.interp(alpha*kk[:,2],k,pkm)

    pkn1=np.interp(alpha*kk[:,0],k,pknm)
    pkn2=np.interp(alpha*kk[:,1],k,pknm)
    pkn3=np.interp(alpha*kk[:,2],k,pknm)

    res = Bi_wiggle(alpha*kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1)      #*Bi0(kk,pkn11,pkn12,pkn13,f,b1,b2,S0,S1)
    return res


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default = os.path.join("../my_utils","mcmc_params.yaml"))
    args.add_argument("--filepaths",default = os.path.join("/pscratch/sd/s/shreeb/shreeb/BAO_Bispectrum_with_data/nobao_project/data","camb_new/camb_python"))
    args.add_argument("--index",  type=int, default = None)

    parsed_args = args.parse_args()
    filepaths = parsed_args.filepaths
   
    # CAMB filepaths for single z and different pk
    Bk=np.load("data/withbao_glam/bk_z0.50_03.npz")
    Pk=np.load("data/withbao_glam/pk_z0.50_03.npz")
    k,pkm,pknm,pk,pkn=Pk['k'], Pk['pkm'],Pk['pknm'],Pk['pk'],Pk['pkn']
    kk,bkm,bknm,bk,bkn,bao=Bk['kk'], Bk['bkm'], Bk['bknm'],Bk['bk'],Bk['bkn'],Bk['bk_bao']

    kg,bg,bgn,cov,scaled_cov,_, bao_cov = cutslice(0,0.3,kk,bkm,bk,bknm,bkn,bk_mol,bao)
    icov = np.linalg.inv(bao_cov)
    for file in os.listdir(filepaths):
        if file[-3:] == "dat" and file.endswith(f"{parsed_args.index:04}.dat"):
            filepath = os.path.join(filepaths, file)
            # h = int(pathlib.PurePath(filepath).name[5:9])/10000 
            h = 1
            print(h,filepath)
            bopt = curvefit_fullBk(kg,bg)
            
            output_path = "curvefit_results/cf_" + Path(filepath).stem + ".dat"

            np.savetxt(output_path, bopt)
        
