import numpy as np
import matplotlib.pyplot as plt
import argparse


np.random.seed(123)

# Choose the "true" parameters.

# Generate some synthetic data from the model.
N = 30
x = np.sort(100 * np.random.rand(N))
yerr = 0.1 + 0.05 * np.random.rand(N)
y = 10 * x + 15
y = y + yerr

# A = np.vander(x, 2)
# C = np.diag(yerr * yerr)
# ATA = np.dot(A.T, A / (yerr**2)[:, None])
# cov = np.linalg.inv(ATA)

cov = 0.5 - np.random.rand(N**2).reshape((N, N))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

# model = lambda x,m,c: m * x + c

def model(x,m,c):
    return m*x + c

import os, sys
# dir2 = os.path.abspath('')
# dir1 = os.path.dirname(dir2)
# if not dir1 in sys.path: sys.path.append(dir1)
from mcmc.mcmc import RunMCMC

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default=os.path.join("my_utils","mcmc_params.yaml"))
    parsed_args = args.parse_args()
    
    mcmc_chain = RunMCMC(model,config_path = parsed_args.config)
    
    mcmc_chain.mcmc_run(x,y,cov)
    
    # module = config['model_path'][' module']
    # function = config['model_path']['function']
    # model = class_for_name(module,function)
    print(mcmc_chain.config['params']['params_range'])