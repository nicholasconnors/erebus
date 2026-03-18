import copy
import os
from typing import List

import batman
import numpy as np
import uncertainties.umath as umath

from erebus.systematics.frame_normalized_pca import perform_fn_pca_on_aperture
from erebus.mcmc_model import WrappedMCMC
from erebus.photometry_data import PhotometryData
from erebus.utility.bayesian_parameter import Parameter
from erebus.utility.h5_serializable_file import H5Serializable
from erebus.utility.planet import Planet
from erebus.utility.run_cfg import ErebusRunConfig
from erebus.utility.utils import create_method_signature
import inspect

EREBUS_CACHE_DIR = "erebus_cache"

class BaseFit(H5Serializable):
    #
    # Virtual methods
    #
    def _get_predicted_t_sec_from_x(self, x : List[float]):
        raise Exception("Unimplemented method")
    
    def _get_transit_model_from_x(self, x : List[float], params : batman.TransitParams):
        raise Exception("Unimplemented method")
    
    def _is_joint_fit(self):
        raise Exception("Unimplemented method")
    
    def _get_visit_index_from_x(self, x : List[float]):
        raise Exception("Unimplemented method")
    
    def _get_visit_starting_time_from_x(self, x : List[float]):
        raise Exception("Unimplemented method")
    
    def _get_eigenvalues_from_x(self, x : List[float]):
        raise Exception("Unimplemented method")
    
    def fit_method(self, x : List[float], *args) -> List[float]:
        raise Exception("Unimplemented method")
    
    #
    # Init
    #
    def __init__(self):
        self.config : ErebusRunConfig = None
        self.mcmc : WrappedMCMC = None
        self.planet : Planet = None
        self.time = List[float]
        self.raw_flux = List[float]
    
    def _add_shared_physical_params(self, mcmc : WrappedMCMC, config : ErebusRunConfig, planet : Planet):
        fixed_timing_params = config.fix_eclipse_timing or config.fit_no_eclipse
        fixed_planet_params = config.fit_no_eclipse
        mcmc.add_parameter("rp_rstar", Parameter.prior_from_ufloat(planet.rp_rstar, positive_only=True, force_fixed=fixed_planet_params))
        mcmc.add_parameter("a_rstar", Parameter.prior_from_ufloat(planet.a_rstar, positive_only=True, force_fixed=fixed_planet_params))
        mcmc.add_parameter("p", Parameter.prior_from_ufloat(planet.p, positive_only=True, force_fixed=fixed_timing_params))
        mcmc.add_parameter("inc", Parameter.prior_from_ufloat(planet.inc, positive_only=True, force_fixed=fixed_planet_params))
        
        # using ecosw and esinw as parameters instead of using e and w directly
        # since w is circular it causes degeneracies (eg, 10 degrees and 370 degrees)
        if planet.w is not None and planet.ecc is not None:
            ecosw = planet.ecc * umath.cos(planet.w * np.pi / 180)
            esinw = planet.ecc * umath.sin(planet.w * np.pi / 180)
            mcmc.add_parameter("esinw", Parameter.prior_from_ufloat(esinw, force_fixed=fixed_timing_params))
            mcmc.add_parameter("ecosw", Parameter.prior_from_ufloat(ecosw, force_fixed=fixed_timing_params))
        else:
            if fixed_timing_params:
                mcmc.add_parameter("esinw", Parameter.fixed(0))
                mcmc.add_parameter("ecosw", Parameter.fixed(0))
            else:
                if planet.ecc is not None:
                    # Uniform for cos/sin omega from -1 to 1
                    e = (planet.ecc.nominal_value + planet.ecc.std_dev)
                    mcmc.add_parameter("esinw", Parameter.uniform_prior(0, -e, e))
                    mcmc.add_parameter("ecosw", Parameter.uniform_prior(0, -e, e))
                else:
                    # Uniform prior for esinw/ecosw from -1 to 1
                    mcmc.add_parameter("esinw", Parameter.uniform_prior(0, -1, 1))
                    mcmc.add_parameter("ecosw", Parameter.uniform_prior(0, -1, 1))
    
    def _try_add_eclipse_timing_parameter(self, name : str, mcmc : WrappedMCMC, config : ErebusRunConfig):
        if config.fit_uniform_eclipse_timing_offset is not None:
            start = config.fit_uniform_eclipse_timing_offset[0]
            end = config.fit_uniform_eclipse_timing_offset[1]
            center = (start + end)/2
            mcmc.add_parameter(name, Parameter.uniform_prior(center, start, end))
            return True
        if config.fit_gaussian_eclipse_timing_offset is not None:
            center = config.fit_gaussian_eclipse_timing_offset[0]
            std_dev = config.fit_gaussian_eclipse_timing_offset[1]
            mcmc.add_parameter(name, Parameter.gaussian_prior(center, std_dev))
            return True
        return False
    
    def _add_systematic_parameters(self, suffix : str, visit_index : int, mcmc : WrappedMCMC, config : ErebusRunConfig):
        if config.fit_fnpca:
            for i in range(0, 5):
                mcmc.add_parameter(f"pc{(i+1)}{suffix}", Parameter.uniform_prior(0.1, -10, 10))
        else:
            for i in range(0, 5):
                mcmc.add_parameter(f"pc{(i+1)}{suffix}", Parameter.fixed(0))
        
        if config.fit_exponential:
            mcmc.add_parameter(f"exp1{suffix}", Parameter.uniform_prior(0.01, -0.1, 0.1))
            mcmc.add_parameter(f"exp2{suffix}", Parameter.uniform_prior(-60.0, -200.0, -1.0))
        else:
            mcmc.add_parameter(f"exp1{suffix}", Parameter.fixed(0))
            mcmc.add_parameter(f"exp2{suffix}", Parameter.fixed(0))

        if config.fit_linear:
            mcmc.add_parameter(f"a{suffix}", Parameter.uniform_prior(1e-3, -2, 2))
        else:
            mcmc.add_parameter(f"a{suffix}", Parameter.fixed(0))
            
        mcmc.add_parameter(f"b{suffix}", Parameter.uniform_prior(1e-6, -0.01, 0.01))
        
        if config._custom_parameters is not None:
            for key in config._custom_parameters:
                param = config._custom_parameters[key]
                if visit_index is not None and visit_index in config._custom_parameters_override:
                    param = config._custom_parameters_override[visit_index][key]
                mcmc.add_parameter(f"{key}{suffix}", copy.deepcopy(param))
    
    #
    # Modeling
    #
    def physical_model(
        self, 
        x : List[float], 
        fp : float, 
        rp_rstar : float,
        a_rstar : float,
        p : float,
        inc : float,
        esinw : float,
        ecosw : float,
        t_sec_offset : float
    ) -> List[float]:
        '''
        Model for the lightcurve using batman
        fp is expected written in ppm
        x is time in BJD-2,400,000.5
        '''
        if self.params is None:
            params = batman.TransitParams()
            params.limb_dark = "quadratic"
            params.u = [0.3, 0.3]
        
        predicted_t_sec = self._get_predicted_t_sec_from_x(x)
        predicted_t_sec = predicted_t_sec.nominal_value if hasattr(predicted_t_sec, 'nominal_value') else predicted_t_sec
        
        t0 = self.planet.get_closest_t0(x[0]) 
        params.t0 = t0.nominal_value if hasattr(t0, 'nominal_value') else t0
        params.t_secondary = predicted_t_sec + (2 * p * ecosw / np.pi) + t_sec_offset
        params.fp = fp
        params.rp = rp_rstar
        params.inc = inc
        params.per = p
        params.a = a_rstar  
        
        ecc = umath.sqrt(ecosw ** 2 + esinw **2)
        w = (umath.atan2(esinw, ecosw) % (2 * np.pi)) * 180 / np.pi

        params.ecc = ecc
        params.w = w % 360
        
        transit_model = self._get_transit_model_from_x(x, params)

        flux_model = transit_model.light_curve(params)
        
        # Todo: Implement lightcurve phase
        # if self.config.fit_lightcurve_phase:
        #     flux_model = ((flux_model - 1) * ((1 / 2.0) * np.cos(2 * np.pi * x / params.per + np.pi) + (1/2.0))) + 1
        
        return flux_model
    
    def systematic_model(
        self, 
        x : List[float], 
        pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float,
        exp1 : float, exp2 : 
        float, a : float, b : float, 
        *extra_params
    ) -> List[float]:
        '''
        x is time in BJD-2,400,000.5
        Assumes all x are from the same visit
        '''
        
        # t is time from start of visit
        t = x - self._get_visit_starting_time_from_x(x)
        
        systematic = np.ones_like(t)
        if self.config.fit_fnpca:
            coeffs = np.array([pc1, pc2, pc3, pc4, pc5])
            eigenvalues = self._get_eigenvalues_from_x(x)
            pca = np.zeros_like(eigenvalues[0])
            for i in range(0, 5):
                pca += coeffs[i] * eigenvalues[i]
            systematic += pca
        if self.config.fit_exponential:
            systematic += (exp1 * np.exp(exp2 * t))
        if self.config.fit_linear:
            systematic += (a * t)
        if self.config._custom_systematic_model is not None:
            # Custom systematic must always have x, visit_index, and joint_fit bool
            systematic += self.config._custom_systematic_model(t, self._get_visit_index_from_x(x), self._is_joint_fit(), *extra_params)
        
        systematic += b
        
        return systematic
    
    def get_number_of_systematic_args(self):
        # Excluding self and x
        number_of_systematic_args = len(inspect.getfullargspec(self.systematic_model).args) - 2
        if self.config._custom_parameters is not None:
            number_of_systematic_args += len(self.config._custom_parameters)
        return number_of_systematic_args
        
    def get_number_of_physical_args(self):
        # Excluding self and x
        number_of_physical_args = len(inspect.getfullargspec(self.physical_model).args) - 2
        return number_of_physical_args
    
    def run(self):
        '''
        Performs the fit via MCMC. Caches the results to the disk.
        '''
        self.mcmc.run(self.time, self.raw_flux,
                      force_clear_cache=self._force_clear_cache,
                      max_steps = self.config.max_steps if self.config.max_steps is not None else 2000000)
        
        self.results = self.mcmc.results
        print(self.mcmc.results)
        
        self.auto_correlation = self.mcmc.auto_correlation
        self.iterations = self.mcmc.iterations
        self.final_log_likelihood = self.mcmc.final_log_likelihood
        self.BIC = self.mcmc.BIC
        
        self.save_to_path(self._cache_file)
    
    def has_converged(self):
        return ( 
            hasattr(self, "auto_correlation") 
            and self.auto_correlation is not None
            and np.isfinite(self.auto_correlation)
        )