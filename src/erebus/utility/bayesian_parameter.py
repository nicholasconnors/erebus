import numpy as np
from scipy.special import gamma
from uncertainties.core import Variable as UFloat
from uncertainties import ufloat

class Parameter:
    '''
    Represents a parameter with Bayesian priors
    To be used with mcmc_model.py
    '''

    def __init__(self):
        self.type = "undefined"
        # Log prior function varies depending on if its Gaussian, uniform, etc
        self.log_prior_fn = lambda _: None
        # Evaluated value of the log prior function at the current value. Is updated when the value is.
        self.log_prior = None
        self.value = None
        # When we start our MCMC, the initial values will be modified by this amount
        self.initial_guess_variation = 1e-2

    def set_value(self, value: float):
        '''
        Set the current value of this parameter, will update the log_prior accordingly
        '''
        self.value = float(value)
        self.log_prior = self.log_prior_fn(value)
        
    def set_initial_value(self, value: float):
        '''
        Set the initial value of this parameter. Will print an error if this value is impossible according to our priors
        '''
        if np.isneginf(self.log_prior_fn(value)):
            print("Tried to set initial value to something impossible")
        else:
            self.set_value(value)
            if self.type == 'uniform':
                # Update the initial guess so that the walkers don't get stuck in impossible areas
                self.initial_guess_variation = np.min([value - self.minimum, self.maximum - value]) / 4
                
    def __str__(self):
        return f"{self.type} - Initial value: {self.value}"
    
    @classmethod
    def prior_from_ufloat(cls, parameter : float | UFloat, force_fixed : bool = False):
        if isinstance(parameter, float):
            return Parameter.fixed(parameter)
        elif force_fixed:
            return Parameter.fixed(parameter.nominal_value)
        else:
            return Parameter.gaussian_prior(parameter.nominal_value, parameter.std_dev)
        
    @classmethod
    def uniform_prior(cls, initial_guess: float, minimum: float, maximum: float) -> "Parameter":
        '''
        Creates a uniform prior, constant between min and max
        '''
        def prior(minimum, maximum, value):
            if value < minimum or value > maximum:
                return -np.inf
            return 0.0
        param = Parameter()
        param.minimum = float(minimum)
        param.maximum = float(maximum)
        param.log_prior_fn = lambda value: prior(minimum, maximum, value)
        param.type = "uniform"
        param.set_value(initial_guess)
        # Initial guess variation set up to cover a quarter of the allowed parameter space
        param.initial_guess_variation = np.min([initial_guess - minimum, maximum - initial_guess]) / 4
        return param
    
    @classmethod
    def gaussian_prior(cls, mu : float, sigma : float) -> "Parameter":
        '''
        Creates a gaussian prior
        '''
        def prior(mu, sigma, value):
            return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (value - mu)**2/sigma**2
        param = Parameter()
        param.log_prior_fn = lambda value: prior(mu, sigma, value)
        param.type = "gaussian"
        param.set_value(mu)
        param.initial_guess_variation = sigma / 2
        return param

    @classmethod
    def fixed(cls, value: float) -> "Parameter":
        '''
        Creates a fixed parameter
        '''
        param = Parameter()
        param.type = "fixed"
        param.log_prior_fn = lambda _: 0.0
        param.set_value(value)
        return param
    
    @classmethod
    def gamma_prior(cls, alpha, beta) -> "Parameter":
        '''
        Creates a gamma prior
        '''
        def prior(alpha, beta, value):
            return alpha * np.log(beta) - np.log(gamma(alpha)) + (alpha - 1) * value - (beta * value)
        param = Parameter()
        param.type = "gamma"
        param.log_prior_fn = lambda value: prior(alpha, beta, value)
        param.set_value(0.5)
        param.initial_guess_variation = 0.4
        return param
    
    @classmethod
    def positive_gaussian_prior(cls, mu : float, sigma : float) -> "Parameter":
        '''
        Creates a positive gaussian prior
        '''
        def prior(mu, sigma, value):
            if value < 0:
                return -np.inf
            return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (value - mu)**2/sigma**2
        param = Parameter()
        param.log_prior_fn = lambda value: prior(mu, sigma, value)
        param.type = "gaussian"
        # just to stop the initial values going negative
        initial_value = mu if mu != 0 else sigma / 2
        param.set_value(initial_value)
        param.initial_guess_variation = np.min([sigma / 2, initial_value / 2]) # Stop initial guess going negative
        return param

