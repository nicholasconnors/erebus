import copy
import inspect
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
from erebus.utility.utils import bin_data, create_method_signature

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
                'transit_models', 'mcmc', "starting_times", "_force_clear_cache",
                'predicted_t_secs', 'time_per_visit', 'all_eigenvalues', 'start_trim',
                'end_trim']
    
    def get_predicted_t_sec_of_visit(self, index : int):
        '''
        Predicted t_sec given a perfectly circular orbit, for a given visit
        '''
        if index in self.predicted_t_secs:
            return self.predicted_t_secs[index]
        
        start_time = self.starting_times[index]
        predicted_t_sec = self.planet.get_predicted_tsec(start_time)

        self.predicted_t_secs[index] = predicted_t_sec
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
        self.config_hash = config.get_hash()
        self.planet_name = planet.name
        
        self._cache_file = f"{EREBUS_CACHE_DIR}/{self.config_hash}_joint_fit.h5"
        
        self.results = {}
        
        self.predicted_t_secs = {}
        '''Memoize predicted t_sec to save time'''
        self.__closest_t0 = {}
        '''Memoize t0 for each visit index to save time'''
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path
        
        if os.path.isfile(self._cache_file) and not force_clear_cache:
            self.load_from_path(self._cache_file)
            
        self._force_clear_cache = force_clear_cache
        
        self.planet = planet
        
        # Maps the saved photometry data index to the visit index
        self.visit_indices = np.arange(0, len(photometry_data_list))
        self.photometry_data_list = photometry_data_list
        
        self.config = config
        
        # Remove visits we're skipping
        if self.config.skip_visits is not None and len(self.config.skip_visits) > 0:
            self.visit_indices = np.delete(self.visit_indices, self.config.skip_visits)
            self.photometry_data_list = np.delete(self.photometry_data_list, self.config.skip_visits)
        
        self.start_trim = [config.get_trim_integrations(i)[0] for i in np.arange(len(photometry_data_list))]
        self.end_trim = [config.get_trim_integrations(i)[1] for i in np.arange(len(photometry_data_list))]
        
        # For the joint fit we bin the data to speed up convergence
        self.bin_size = config.joint_fit_bin_size
                
        self.chain = None
        
        self.params = None
        self.transit_models = {}
        
        self.joint_eigenvalues = []
        self.joint_eigenvectors = []
        self.pca_variance_ratios = []
        self.time = []
        self.raw_flux = []
        for i, data in enumerate(photometry_data_list):
            eigenvalues, eigenvectors, variance_ratios = perform_fn_pca_on_aperture(data.normalized_frames[self.start_trim[i]:self.end_trim[i]])
            binned_eigenvalues = np.array([bin_data(ev, self.bin_size)[0] for ev in eigenvalues])
            self.joint_eigenvalues.append(binned_eigenvalues)
            self.joint_eigenvectors.append(eigenvectors)
            self.pca_variance_ratios.append(variance_ratios)
            if config.skip_visits is not None and i in config.skip_visits:
                continue
            binned_time = bin_data(data.time[self.start_trim[i]:self.end_trim[i]], self.bin_size)[0]
            self.time.append(binned_time)
            binned_flux = bin_data(data.raw_flux[self.start_trim[i]:self.end_trim[i]], self.bin_size)[0]
            self.raw_flux.append(binned_flux)
            print(np.array(binned_eigenvalues).shape)
            
        # time per visit used to interpolate FNPCA systematic
        self.time = np.concatenate(self.time)
        self.all_eigenvalues = np.concatenate(self.joint_eigenvalues, axis=1)
        self.starting_times = np.sort(np.array([np.min(data.time) for data in photometry_data_list]))
        self.raw_flux = np.concatenate(self.raw_flux)
        
        # Orders might be wrong, assumes each visit was in order
        sort = np.argsort(self.time)
        self.time = self.time[sort]
        self.all_eigenvalues = self.all_eigenvalues[:,sort]
        self.raw_flux = self.raw_flux[sort]
        
        self.eigenvalue_lookup = dict(zip(self.time, self.all_eigenvalues.T))
            
        # If visits have different lengths (number of integrations) then these arrays can't be saved (inhomogenous)
        # Pad with NaN in this case
        n_visits = len(self.joint_eigenvalues)
        n_eigenvalues = len(self.joint_eigenvalues[0])
        max_length = max([len(eigenvalues[0]) for eigenvalues in self.joint_eigenvalues])
        padded_joint_eigenvalues = np.full((n_visits, n_eigenvalues, max_length), np.nan)
        for i in range(n_visits):
            for j in range(n_eigenvalues):
                v = self.joint_eigenvalues[i][j]
                padded_joint_eigenvalues[i, j, :len(v)] = v
        
        self.joint_eigenvalues = np.array(padded_joint_eigenvalues)
        self.joint_eigenvectors = np.array(self.joint_eigenvectors)
        print(f"Eigenvalues shape: {self.joint_eigenvalues.shape}")
        print(f"Eigenvectors shape: {self.joint_eigenvectors.shape}")
                
        # Get the predicted eclipse times in advance
        for n in range(0, len(photometry_data_list)):
            self.get_predicted_t_sec_of_visit(n)
                
        mcmc = WrappedMCMC(self._cache_file.replace(".h5", "_mcmc.h5"))
        
        lower_limit = 0 if config.prevent_negative_eclipse_depth else -2000e-6
        
        if config.fit_no_eclipse:
            mcmc.add_parameter("fp", Parameter.fixed(0))
        else:
            mcmc.add_parameter("fp", Parameter.uniform_prior(400e-6, lower_limit, 2000e-6))
             
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
        
        flag_t_sec_set = False
        if not config.fit_no_eclipse:
            if self.config.fit_uniform_eclipse_timing_offset is not None:
                center = self.config.fit_uniform_eclipse_timing_offset[0]
                half_width = self.config.fit_uniform_eclipse_timing_offset[1]
                mcmc.add_parameter("t_sec_offset", Parameter.uniform_prior(center, -half_width, half_width))
                flag_t_sec_set = True
            elif self.config.fit_gaussian_eclipse_timing_offset is not None:
                center = self.config.fit_gaussian_eclipse_timing_offset[0]
                std_dev = self.config.fit_gaussian_eclipse_timing_offset[1]
                mcmc.add_parameter("t_sec_offset", Parameter.gaussian_prior(center, std_dev))
                flag_t_sec_set = True
        if not flag_t_sec_set:
            mcmc.add_parameter("t_sec_offset", Parameter.fixed(0))
        
        for visit_index in self.visit_indices:
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
                    param = self.config._custom_parameters[key]
                    if visit_index in self.config._custom_parameters_override:
                        param = self.config._custom_parameters_override[visit_index][key]
                    mcmc.add_parameter(f"{key}_{visit_index}", copy.deepcopy(param))
            
        mcmc.add_parameter("y_err", Parameter.uniform_prior(400e-6, 0, 2000e-6))
        
        args = ["x"] + [key for key in mcmc.params][:-1]
        fit_method = create_method_signature(self.fit_method, args)

        mcmc.set_method(fit_method)
        
        self.mcmc = mcmc
        
        self.save_to_path(self._cache_file)
    
    def physical_model(self, x : List[float], fp : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, esinw : float, ecosw : float, t_sec_offset : float) -> List[float]:
        '''
        Model for the lightcurve using batman
        fp is expected written in ppm
        Assumes all x values are from the same visit
        '''
        visit_index = self.get_visit_index_from_time(x[0])
        # t_sec is relative to the start of the visit
        predicted_t_sec = self.get_predicted_t_sec_of_visit(visit_index).nominal_value
        t_sec = predicted_t_sec + self.starting_times[visit_index] + (2 * p * ecosw / np.pi) + t_sec_offset

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
        
        ecc = umath.sqrt(ecosw ** 2 + esinw **2)
        w = (umath.atan2(esinw, ecosw) % (2 * np.pi)) * 180 / np.pi
        
        params.ecc = ecc
        params.w = w
                
        # TODO: breaks when x changes (x does change for final result plotting so commentd out for now)
        #if visit_index not in self.transit_models or self.transit_models[visit_index] is None:
        self.transit_models[visit_index] = batman.TransitModel(params, x, transittype="secondary")

        flux_model = self.transit_models[visit_index].light_curve(params)
        
        if self.config.fit_lightcurve_phase:
            flux_model = ((flux_model - 1) * ((1 / 2.0) * np.cos(2 * np.pi * x / params.per + np.pi) + (1/2.0))) + 1
        
        return flux_model
    
    def systematic_model(self, x : List[float], pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float, 
                         exp1 : float, exp2 : float, a : float, b : float, *extra_params) -> List[float]:
        '''
        Assumes all x are from the same visit
        '''
        visit_index = self.get_visit_index_from_time(x[0])
        time = np.array(x) - self.starting_times[visit_index]
        
        systematic = np.ones_like(x)
        if self.config.fit_fnpca:
            coeffs = np.array([pc1, pc2, pc3, pc4, pc5])
            eigenvalues = np.array([self.eigenvalue_lookup[xi] for xi in x]).T
            
            # remove NaNs to account for jagged numpy array
            eigenvalues = np.array([row[~np.isnan(row)] for row in eigenvalues])

            pca = np.ones_like(eigenvalues[0])
            for i in range(0, 5):
                pca += coeffs[i] * eigenvalues[i]

            systematic *= pca
            
        if self.config.fit_exponential:
            systematic *= (exp1 * np.exp(exp2 * time)) + 1
        if self.config.fit_linear:
            systematic *= (a * time) + 1
        if self.config._custom_systematic_model is not None:
            flat_args = np.array(extra_params).flatten()
            systematic *= self.config._custom_systematic_model(x, visit_index, True, *flat_args)
        
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
        for i, visit_index in enumerate(self.visit_indices):
            filt = visit_indices == visit_index
            time = x[filt]
                        
            systematic_index_start = (number_of_physical_args + 1) + (i * number_of_systematic_args)
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
                      force_clear_cache=self._force_clear_cache,
                      max_steps = self.config.max_steps if self.config.max_steps is not None else 2000000)
        
        self.results = self.mcmc.results
        self.chain = self.mcmc.sampler.get_chain(discard=200, thin=15, flat=True)
        print(self.mcmc.results)
        
        self.auto_correlation = self.mcmc.auto_correlation
        self.iterations = self.mcmc.iterations
        self.final_log_likelihood = self.mcmc.final_log_likelihood
        self.BIC = self.mcmc.BIC
        
        self.save_to_path(self._cache_file)
    
    def has_converged(self):
        return hasattr(self, "auto_correlation") and self.auto_correlation is not None \
            and np.isfinite(self.auto_correlation)