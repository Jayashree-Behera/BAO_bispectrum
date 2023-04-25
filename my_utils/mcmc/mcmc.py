import emcee
import numpy as np
import importlib
from multiprocessing import cpu_count, Pool
import yaml
import math

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def magnitude(lst):
    return np.array([10**int(math.floor(math.log10(np.abs(x)))) for x in lst])

class RunMCMC:
    def __init__(self,model,x,y,icov,config_path):
        self.model = model
        self.config = read_params(config_path)
        self.x = x
        self.y = y
        self.icov = icov

    def ln_like(self,params):
        '''
            x - parameter input
            y - mean absolute output
            params - free parameters
            cov - covariance of y
            input_path - input path of the model
        '''
        model_y = self.model(self.x, *params)
        diff = self.y - model_y
        #chisq = res.dot(icov.dot(res))
        return -0.5 * diff.dot(self.icov.dot(diff))

    def ln_prior(self,params):
        '''
            params - free parameters
            params_range - 2d array that contains min and max of each parameter
        '''
        params_range = self.config['params']['params_range']
        for i in range(len(params)):
            if params_range[i][0]<=params[i]<=params_range[i][1]:
                continue
            else:
                return -np.inf
        return 0.0

    def ln_prob(self,params):
        lp = self.ln_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_like(params)

    def mcmc_run(self,output_path):
        '''
            ndim - number of free parameters
            nwalkers - controls the number of starting parameter values
            num_steps - number of chains for mcmc
            guess - initial guess/fiducial value of each parameter
        '''
        # params_range = self.config['params']['params_range']

        ndim = self.config['base']['ndim']
        nwalkers = self.config['base']['nwalkers']
        num_steps = self.config['base']['num_steps']
        guess = self.config['params']['guess']    
        output_path = output_path

        np.random.seed(self.config['base']['random_state'])

        starting_params = guess + 1e-4 * np.random.randn(nwalkers,ndim)

        with Pool() as pool:
            filename = output_path
            backend = emcee.backends.HDFBackend(filename)
            #print("Initial size: {0}".format(backend.iteration))

            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob,
                                            pool=pool,backend=backend)

            # pos, prob, state = sampler.run_mcmc(starting_params, num_steps,progress=True)
            sampler.run_mcmc(starting_params , num_steps,store=True,progress=True)
            #print("Final size: {0}".format(backend.iteration))

        

