from typing import Any, Callable
import emcee
import numpy as np
from scipy.stats import norm
import inspect
from erebus.utility.bayesian_parameter import Parameter
from uncertainties import ufloat

import multiprocessing as mp

import dill
import emcee

# Make it use dill instead of pickle so it can handle more complex objects (priors)
# We run separat chains in parallel
# We cannot use the built-in pool parameter of emcee because serialization is the bottleneck
# (running a single chain on multiple cores is slower than doing it normally)
from multiprocessing.reduction import ForkingPickler
ForkingPickler.dumps = dill.dumps
ForkingPickler.loads = dill.loads

# TODO: Save/load using emcee.backends.HDFBackend + H5Serializable
class WrappedMCMC:
    '''
    Wrapper class for emcee
    User can define a method that is to be fit and parameters with Bayesian priors
    The model will always fit for a Gaussian noise parameter "y_err"
    '''
    def __init__(self):
        self.params : dict = {}
        '''Dictionary of parameters. Each is a bayesian_parameter instance.'''
        self.model_function : Callable[[Any], float] = None
        '''The function to be fit'''
        
        # Values set once it is done running
        self.sampler : emcee.EnsembleSampler = None
        '''The emcee EnsembleSampler which this class wrapped'''
        self.results : dict = {}
        '''Dictionary of results after fitting'''
        self.auto_correlation = 0
        '''The integrated autocorrelation time after the MCMC has finished running'''
        self.iterations = 0
        '''How many iterations this MCMC ran for before stopping.'''

    def add_parameter(self, name : str, param : Parameter):
        '''
        Adds bayesian parameters to our method, must be called before set_method
        '''
        self.params[name] = param

    def set_method(self, method : Callable[[Any], float]):
        '''
        Updates the method which will be fit for, and verifies that the method signature matches the parameters specified
        First parameter must be 'x'
        After that the order which parameters were added in must match their positions in the method signature
        '''
        # We verify that the parameters of our method (plus y_err) matches our defined bayesian parameters 
        # (excluding the method taking 'x' as its first parameter)
        method_params = inspect.getfullargspec(method).args + ["y_err"]
        # Similarly we 
        defined_params = ["x"] + [p for p in self.params]
        if method_params != defined_params:
            raise Exception("Defined parameters don't match given method! Method:", method_params, "Defined", defined_params)
        self.model_function = method

    def get_free_params(self) -> list[Parameter]:
        '''
        Returns the keys of any parameter that isn't fixed i.e., will be fitted for
        '''
        return [p for p in self.params if self.params[p].type != "fixed"]

    def evaluate_model(self, x : np.ndarray, *params : list[Parameter]) -> float:
        '''
        Evaluates the model with the given x input and parameters
        '''
        # Params is a list of free parameters, last one is always the error
        free_params = self.get_free_params()
        if len(free_params) != len(params):
            raise Exception(f"Number of free parameters ({len(free_params)}) doesn't match number of inputs ({len(params)})")
        for i, param in enumerate(free_params):
            self.params[param].set_value(params[i])
        all_params = [self.params[p].value for p in self.params]
        # Excluding the error from the function call
        return self.model_function(x, *all_params[:-1])

    def log_likelihood(self, theta : list[float], x : float, y : float):
        '''Given the parameters theta and the x and y values, calculates the Bayesian log likelihood.'''
        # y_err is a gaussian noise parameter
        model = self.evaluate_model(x, *theta)
        y_err = self.params["y_err"].value
        
        # Function taken from https://colab.research.google.com/drive/15EsEFbbLiU2NFaNrfiCTlF_i65ShDlmS?usp=sharingw#scrollTo=Qkwg2fcNL-26
        return np.sum(norm.logpdf(y, loc=model, scale=y_err))
        
    def __log_prior(self, theta : list[float]):
        free_params = self.get_free_params()
        log_prior = 0.0
        for (i, param) in enumerate(free_params):
            log_prior += self.params[param].log_prior_fn(theta[i])
        return log_prior
    
    def __log_probability(self, theta : list[float], x : float, y : float):
        lp = self.__log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        # Value should always be negative
        return lp + self.log_likelihood(theta, x, y)

    def run(self, x, y, max_steps = 200000, walkers = 64, cache_file = None, force_clear_cache = False) -> tuple[np.ndarray, emcee.EnsembleSampler, float, int]:         
        '''
        Runs the MCMC, gets the results (with errors), ensemble sampler instance, autocorrelation time, and interation count
        '''   
        
        # The initial guess will be whatever the free parameters were initially set to
        initial_guess = [self.params[p].value for p in self.get_free_params()]
        
        initial_guess_var = [self.params[p].initial_guess_variation for p in self.get_free_params()]
        ndim = len(initial_guess_var)
        nchains = 2
        
        backends = [None] * nchains        
        sampler = [None] * nchains
        for i in range(0, nchains):
            if cache_file is not None:
                # Must be separate files to prevent saving race conditions
                backends[i] = emcee.backends.HDFBackend(cache_file.replace(".h5", "_chain%d.h5" % i))
                if force_clear_cache:
                    backends[i].reset(walkers, ndim)
            sampler[i] = emcee.EnsembleSampler(walkers, ndim, self.__log_probability, 
                                                args=(x, y), backend=backends[i])
    
        # Let walkers get away from starting positions
        pos = [None] * nchains
        
        burn_in = 1000
        
        def start_chain(i):
            pos = np.array(initial_guess) + (np.array(initial_guess_var) * (2 * np.random.rand(walkers, len(initial_guess)) - 1))
            pos, _, _ = sampler[i].run_mcmc(pos, burn_in, skip_initial_state_check=True, 
                                            store=True, progress=True)
            sampler[i].reset()
            print("Moved away from starting positions for chain #", i)
            return pos
        
        # Unclear how to determine if backend has data except for checking for an exception
        has_backend = True
        if cache_file is not None:
            try:
                backends[0].get_last_sample()
            except:
                print("No currently saved data")
                has_backend = False
        else:
            has_backend = False
        
        if not has_backend:
            print("Initial guesses:", initial_guess, "variation:", initial_guess_var)
            print("Initial likelihood:", self.log_likelihood(initial_guess, x, y))
            print(f"Fitting for {len(initial_guess)} parameters")
            
            with mp.Pool(processes=nchains) as pool:
                pos = pool.map(start_chain, np.arange(0, nchains))
            
            pos = np.array(pos)
            print("Initial guesses shape:", pos.shape)
        
            initial_guess_likelihoods = np.array([self.log_likelihood(pos[i,j,:], x, y) for j in np.arange(0, walkers) for i in np.arange(0, nchains)])
            mean_initial_guess_likelihood = np.mean(initial_guess_likelihoods)

            print("Mean likelihood at start:", mean_initial_guess_likelihood)

            if not np.isfinite(mean_initial_guess_likelihood):
                raise Exception("Impossible starting positions")
        
        # https://mystatisticsblog.blogspot.com/2019/04/gelman-rubin-convergence-criteria-for.html
        
        def gelman_rubin_convergence(withinchainvar, meanchain, chain_length, N):    
            meanall = np.mean(meanchain, axis=0)
            mean_wcv = np.mean(withinchainvar, axis=0)
            vom = np.zeros(ndim, dtype=np.float64)
            for jj in range(0, N):
                vom += (meanall - meanchain[jj])**2/(N-1.)
            B = vom - mean_wcv/chain_length
            return np.sqrt(1. + B/mean_wcv)
        
        rstate = [None] * nchains
        for i in range(0, nchains):
            rstate[i] = np.random.get_state()
            
        if has_backend:
            for i in range(0, nchains):
                pos[i] = backends[i].get_last_sample()[0]
                rstate[i] = backends[i].get_last_sample()[2]
        
        withinchainvar = np.zeros((nchains, ndim), dtype=np.float64)
        meanchain = np.zeros((nchains, ndim), dtype=np.float64)

        minlength = 10000
        ichaincheck = 10000
        chainstep = minlength
        iteration_counter = burn_in if backends[0] is None else np.min([backends[0].iteration, backends[1].iteration])
        loopcriteria = True
        epsilon = 0.04
        
        def run_chain(jj):
            print("Processing chain #%d" % jj)
            for result in sampler[jj].sample(pos[jj], iterations=chainstep, rstate0=rstate[jj],
                                             progress=True, store=True, skip_initial_state_check=True):
                result_pos = result[0]
                result_rstate = result[2]
            chain_length_per_walker = int(sampler[jj].get_chain().shape[1])
            chainsamples = sampler[jj].chain[:, int(chain_length_per_walker/2):, :]\
                                    .reshape((-1, ndim))
            return result_pos, result_rstate, chainsamples

        # Run chain until the chain has converged
        while loopcriteria:
            with mp.Pool(processes=nchains) as pool:
                run_chain_res = pool.map(run_chain, np.arange(0, nchains))
            for jj in range(0, nchains):
                pos[jj], rstate[jj], chainsamples = run_chain_res[jj]
                chain_length = len(chainsamples)
                # Variance for each parameter within one chain
                withinchainvar[jj] = np.var(chainsamples, axis=0)
                # Mean for each parameter within one chain
                meanchain[jj] = np.mean(chainsamples, axis=0)
            
            R = gelman_rubin_convergence(withinchainvar, meanchain, chain_length, nchains)
            try:
                auto_correlation_time = np.mean(sampler[0].get_autocorr_time())
            except Exception as e:
                print(e)
                auto_correlation_time = np.inf
            all_within_epsilon = all(np.abs(1 - R) < epsilon)
            all_converged = np.isfinite(auto_correlation_time)
            loopcriteria = (not all_within_epsilon or not all_converged) and iteration_counter < max_steps
            
            print("Rubin gelman convergence:", R, "converged?", all_within_epsilon)
            print("Autocorr time:", auto_correlation_time, "converged?", all_converged)
            print("Iterations:", iteration_counter, "Max steps:", max_steps)
            print("Continue looping?", loopcriteria)
            
            chainstep = ichaincheck
            iteration_counter += chainstep
        
        try:
            auto_correlation_time = np.mean(sampler[0].get_autocorr_time())
            print("Autocorr time:", auto_correlation_time)
            discard = int(auto_correlation_time) * 3 if np.isfinite(auto_correlation_time) else 0
        except:
            print("Autocorr time was really bad")
            auto_correlation_time = np.inf
            discard = 0
            
        flat_samples = sampler[0].get_chain(discard=discard, thin=15, flat=True)
        
        # Takes the median value of each fitted parameter and the 68% confidence interval as errors
        res = []
        for i in range(ndim):
            percentiles = np.percentile(flat_samples[:, i], [16, 50, 84])
            diffs = np.diff(percentiles)
            res.append([percentiles[1], diffs[0], diffs[1]])
            
        # Including all parameters including fixed values
        res_index = 0
        for key in self.params:
            if self.params[key].type == "fixed":
                self.results[key] = ufloat(self.params[key].value, 0)
            else:
                self.results[key] = ufloat(res[res_index][0], np.mean([res[res_index][1], res[res_index][2]])) 
                res_index += 1
        self.sampler = sampler[0]
        self.auto_correlation = auto_correlation_time
        self.iterations = iteration_counter
    