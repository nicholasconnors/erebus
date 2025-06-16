import os
import hashlib
from typing import List
import numpy as np
from erebus.photometry_data import PhotometryData
from erebus.utility.planet import Planet
from erebus.mcmc_model import WrappedMCMC
from erebus.utility.bayesian_parameter import Parameter
from erebus.utility.run_cfg import ErebusRunConfig
from erebus.frame_normalized_pca import perform_fn_pca_on_aperture
from erebus.utility.h5_serializable_file import H5Serializable
import batman
import json
import matplotlib.pyplot as plt
from uncertainties import ufloat
from erebus.utility.utils import create_method_signature
import inspect
from erebus.utility.utils import bin_data
import copy
from uncertainties import ufloat

EREBUS_CACHE_DIR = "erebus_cache"

class JointFit(H5Serializable):    
    '''
    A joint fit takes multiple eclipse observations and fits for all of them at once with a shared eclipse depth value.
    Orbital parameters are also shared, but systematics are per visit.
    '''
    def _exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['config', 'photometry_data_list', 'time', 'raw_flux', 'params',
                'transit_models', 'mcmc', "starting_times", "_force_clear_cache"]
    
    def get_predicted_t_sec_of_visit(self, index : int):
        '''
        Predicted t_sec given a perfectly circular orbit, for a given visit
        '''
        if index in self.__predicted_t_secs:
            return self.__predicted_t_secs[index]
        
        planet = self.planet
        nominal_period = planet.p if isinstance(planet.p, float) else planet.p.nominal_value
        start_time = self.starting_times[index]
        t0 = planet.get_closest_t0(start_time)
        predicted_t_sec = (t0 - start_time + planet.p / 2.0) % nominal_period
        number_of_periods = np.abs(t0.nominal_value - start_time + planet.p.nominal_value / 2.0) / planet.p.nominal_value
        std_dev = np.sqrt(t0.std_dev**2 + (number_of_periods * planet.p.std_dev)**2)
        predicted_t_sec = ufloat(predicted_t_sec.nominal_value, std_dev)
        
        self.__predicted_t_secs[index] = predicted_t_sec
        return predicted_t_sec
    
    def get_visit_index_from_time(self, time : float):
        '''
        Information on each visit that is part of this joint fit is stored in various arrays
        This method takes a given time and determines which visit index it corresponds to
        '''
        # Could memoize this but unsure of the memory vs time tradeoff
        # Starting times are in descending order
        for i in range(0, len(self.starting_times)):
            if time >= self.starting_times[i] and (i == len(self.starting_times) - 1 or time < self.starting_times[i + 1]):
                return i
        raise Exception(f"Time {time} was outside of the range of possible times ({self.starting_times})")
    
    def __init__(self, photometry_data_list : List[PhotometryData], planet : Planet, config : ErebusRunConfig,
                 force_clear_cache : bool = False, override_cache_path : str = None):
        self.source_folder = photometry_data_list[0].source_folder
        source_folder_hash = hashlib.md5(self.source_folder.encode()).hexdigest()
        self.config_hash = hashlib.md5(json.dumps(config.model_dump()).encode()).hexdigest()
        self.planet_name = planet.name
        
        self._cache_file = f"{EREBUS_CACHE_DIR}/{source_folder_hash}_{self.config_hash}_joint_fit.h5"
        
        self.results = {}
        
        self.__predicted_t_secs = {}
        '''Memoize predicted t_sec to save time'''
        self.__closest_t0 = {}
        '''Memoize t0 for each visit index to save time'''
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path
        
        if os.path.isfile(self._cache_file) and not force_clear_cache:
            self.load_from_path(self._cache_file)
            
        self._force_clear_cache = force_clear_cache
        
        self.planet = planet
        self.photometry_data_list = photometry_data_list
        
        self.start_trim = 0 if config.trim_integrations is None else config.trim_integrations[0]
        self.end_trim = None if config.trim_integrations is None else -np.abs(config.trim_integrations[1])
        
        # For the joint fit we bin the data to speed up convergence
        self.bin_size = 4
        self.time = np.concatenate([bin_data(data.time[self.start_trim:self.end_trim], self.bin_size)[0] for data in photometry_data_list])
        self.starting_times = np.sort(np.array([np.min(data.time) for data in photometry_data_list]))
        self.raw_flux = np.concatenate([bin_data(data.raw_flux[self.start_trim:self.end_trim], self.bin_size)[0] for data in photometry_data_list])
        
        # Orders might be wrong, assumes each visit was in order
        sort = np.argsort(self.time)
        self.time = self.time[sort]
        self.raw_flux = self.raw_flux[sort]
        
        self.config = config
        
        self.chain = None
        
        self.params = None
        self.transit_models = {}
        
        self.joint_eigenvalues = []
        self.joint_eigenvectors = [] 
        self.pca_variance_ratios = []
        for i, data in enumerate(photometry_data_list):
            eigenvalues, eigenvectors, variance_ratios = perform_fn_pca_on_aperture(data.normalized_frames[self.start_trim:self.end_trim])
            binned_eigenvalues = np.array([bin_data(ev, self.bin_size)[0] for ev in eigenvalues])
            self.joint_eigenvalues.append(binned_eigenvalues)
            self.joint_eigenvectors.append(eigenvectors)
            self.pca_variance_ratios.append(variance_ratios)
                
        mcmc = WrappedMCMC()
        
        # For circular orbit predict the eclipse time, else use a uniform prior
        # Joint fit represents t_sec as an offset from the predicted time for a circular orbit
        if isinstance(planet.ecc, float) and planet.ecc == 0:
            print("Circular orbit: using gaussian prior for t_sec_offset")
            predicted_t_sec = self.get_predicted_t_sec_of_visit(0)
            mcmc.add_parameter("t_sec_offset", Parameter.gaussian_prior(0, predicted_t_sec.std_dev))
        else:
            print("Eccentric orbit: using uniform prior for t_sec_offset")
            duration = np.max(photometry_data_list[0].time - np.min(photometry_data_list[0].time))
            mcmc.add_parameter("t_sec_offset", Parameter.uniform_prior(0, -duration / 3.0, duration / 3.0))
        
        mcmc.add_parameter("fp", Parameter.uniform_prior(200e-6, -1500e-6, 1500e-6))
        # For the joint fit we fix the orbital parameters
        mcmc.add_parameter("rp_rstar", Parameter.prior_from_ufloat(planet.rp_rstar, True))
        mcmc.add_parameter("a_rstar", Parameter.prior_from_ufloat(planet.a_rstar, True))
        mcmc.add_parameter("p", Parameter.prior_from_ufloat(planet.p, True))
        mcmc.add_parameter("inc", Parameter.prior_from_ufloat(planet.inc, True))
        mcmc.add_parameter("ecc", Parameter.prior_from_ufloat(planet.ecc, True))
        mcmc.add_parameter("w", Parameter.prior_from_ufloat(planet.w, True))
        
        for visit_index in range(0, len(photometry_data_list)):
            if self.config.fit_fnpca:
                for i in range(0, 5):
                    mcmc.add_parameter(f"pc{(i+1)}_{visit_index}", Parameter.uniform_prior(0.1, -10, 10))
            else:
                for i in range(0, 5):
                    mcmc.add_parameter(f"pc{(i+1)}_{visit_index}", Parameter.fixed(0))
            
            if self.config.fit_exponential:
                mcmc.add_parameter(f"exp1_{visit_index}", Parameter.uniform_prior(0.01, -0.1, 0.1))
                mcmc.add_parameter(f"exp2_{visit_index}", Parameter.uniform_prior(-60.0, -200.0, -1.0))
            else:
                mcmc.add_parameter(f"exp1_{visit_index}", Parameter.fixed(0))
                mcmc.add_parameter(f"exp2_{visit_index}", Parameter.fixed(0))

            if self.config.fit_linear:
                mcmc.add_parameter(f"a_{visit_index}", Parameter.uniform_prior(1e-3, -2, 2))
            else:
                mcmc.add_parameter(f"a_{visit_index}", Parameter.fixed(0))
                
            mcmc.add_parameter(f"b_{visit_index}", Parameter.uniform_prior(1e-6, -0.01, 0.01))
            
            if self.config._custom_parameters is not None:
                for key in self.config._custom_parameters:
                    mcmc.add_parameter(f"{key}_{visit_index}", copy.deepcopy(self.config._custom_parameters[key]))
            
        mcmc.add_parameter("y_err", Parameter.uniform_prior(400e-6, 0, 2000e-6))
        
        args = ["x"] + [key for key in mcmc.params][:-1]
        fit_method = create_method_signature(self.fit_method, args)

        mcmc.set_method(fit_method)
        
        self.mcmc = mcmc
        
        self.save_to_path(self._cache_file)
    
    def physical_model(self, x : List[float], t_sec_offset : float, fp : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float) -> List[float]:
        '''
        Model for the lightcurve using batman
        fp is expected written in ppm
        Assumes all x values are from the same visit
        '''
        visit_index = self.get_visit_index_from_time(x[0])
        # t_sec is relative to the start of the visit
        predicted_t_sec = self.get_predicted_t_sec_of_visit(visit_index).nominal_value
        t_sec = predicted_t_sec + t_sec_offset + self.starting_times[visit_index]
        
        if self.params is None:
            params = batman.TransitParams()
            params.limb_dark = "quadratic"
            params.u = [0.3, 0.3]
        
        if visit_index not in self.__closest_t0:
            self.__closest_t0[visit_index] = self.planet.get_closest_t0(x[0]).nominal_value
        
        params.t0 = self.__closest_t0[visit_index]
        params.t_secondary = t_sec
        params.fp = fp
        params.rp = rp_rstar
        params.inc = inc
        params.per = p
        params.a = a_rstar  
        params.ecc = ecc
        params.w = w
                
        # TODO: breaks when x changes (x does change for final result plotting so commentd out for now)
        #if visit_index not in self.transit_models or self.transit_models[visit_index] is None:
        self.transit_models[visit_index] = batman.TransitModel(params, x, transittype="secondary")

        flux_model = self.transit_models[visit_index].light_curve(params)
        return flux_model
    
    def systematic_model(self, x : List[float], pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float, 
                         exp1 : float, exp2 : float, a : float, b : float, *extra_params) -> List[float]:
        '''
        Assumes all x are from the same visit
        '''
        visit_index = self.get_visit_index_from_time(x[0])
        starting_time = self.starting_times[visit_index]
        time = x - starting_time

        systematic = np.ones_like(x)
        if self.config.fit_fnpca:
            coeffs = np.array([pc1, pc2, pc3, pc4, pc5])
            pca = np.ones_like(self.joint_eigenvalues[visit_index][0])
            for i in range(0, 5):
                pca += coeffs[i] * self.joint_eigenvalues[visit_index][i]
            systematic *= pca
        if self.config.fit_exponential:
            systematic *= (exp1 * np.exp(exp2 * time)) + 1
        if self.config.fit_linear:
            systematic *= (a * time) + 1
        if self.config._custom_systematic_model is not None:
            flat_args = np.array(extra_params).flatten()
            systematic *= self.config._custom_systematic_model(x, *flat_args)
        
        systematic += b
        
        return systematic
    
    def get_number_of_systematic_args(self):
        number_of_systematic_args = len(inspect.getfullargspec(self.systematic_model).args) - 2
        if self.config._custom_parameters is not None:
            number_of_systematic_args += len(self.config._custom_parameters)
        return number_of_systematic_args
        
    def get_number_of_physical_args(self):
        # Excluding self and x
        number_of_physical_args = len(inspect.getfullargspec(self.physical_model).args) - 2
        return number_of_physical_args
        
    def fit_method(self, *args) -> List[float]:
        '''
        Fits for the output lightcurve given the list of arguments
        '''
        x = np.array(args[0])

        number_of_physical_args = self.get_number_of_physical_args()
        number_of_systematic_args = self.get_number_of_systematic_args()
        
        physical_args = args[1:number_of_physical_args + 1]
                
        # Systematic arguments we're actually using will depend on the x value
        # x is a list of times
        visit_indices = np.array([self.get_visit_index_from_time(xi) for xi in x])
        results = np.zeros_like(x)
        for visit_index in range(0, len(self.photometry_data_list)):
            filt = visit_indices == visit_index
            time = x[filt]
                        
            systematic_index_start = (number_of_physical_args + 1) + (visit_index * number_of_systematic_args)
            systematic_args = args[systematic_index_start:systematic_index_start + number_of_systematic_args]
        
            systematic = self.systematic_model(time, *systematic_args)
            physical = self.physical_model(time, *physical_args)
            results[filt] = systematic * physical
            
        return results

    def run(self):
        '''
        Performs the joint fit via MCMC. Caches the results to the disk.
        '''
        self.mcmc.run(self.time, self.raw_flux, walkers = 80, 
                      cache_file = None if self.config.skip_emcee_backend_cache else self._cache_file.replace(".h5", "_mcmc.h5"),
                      force_clear_cache=self._force_clear_cache)
        
        self.results = self.mcmc.results
        self.chain = self.mcmc.sampler.get_chain(discard=200, thin=15, flat=True)
        print(self.mcmc.results)
        
        self.auto_correlation = self.mcmc.auto_correlation
        self.iterations = self.mcmc.iterations
        
        self.save_to_path(self._cache_file)