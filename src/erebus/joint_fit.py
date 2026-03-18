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
from erebus.base_fit import BaseFit

EREBUS_CACHE_DIR = "erebus_cache"

class JointFit(BaseFit):    
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
                'end_trim', 'eigenvalue_map', 'visit_index_filter', 'planet']
    
    def get_predicted_t_sec_of_visit(self, index : int) -> float:
        '''
        Predicted t_sec given a perfectly circular orbit, for a given visit
        In BJD-2,400,000.5
        '''
        if index in self.predicted_t_secs:
            return self.predicted_t_secs[index]
        
        start_time = self.starting_times[index]
        predicted_t_sec = self.planet.get_predicted_tsec(start_time).nominal_value + start_time

        self.predicted_t_secs[index] = predicted_t_sec
        return predicted_t_sec
    
    def get_visit_index_from_time(self, time : float):
        '''
        Information on each visit that is part of this joint fit is stored in various arrays
        This method takes a given time and determines which visit index it corresponds to
        '''
        if time in self.__visit_index_lookup:
            return self.__visit_index_lookup[time]
        
        # Starting times are in acending order
        for i in range(0, len(self.starting_times)):
            if time >= self.starting_times[i] and (i == len(self.starting_times) - 1 or time < self.starting_times[i + 1]):
                self.__visit_index_lookup[time] = i
                return i
        raise Exception(f"Time {time} was outside of the range of possible times ({self.starting_times})")
    
    def __init__(self, photometry_data_list : List[PhotometryData], planet : Planet, config : ErebusRunConfig,
                 force_clear_cache : bool = False, override_cache_path : str = None):
        super().__init__()

        
        self.config_hash = config.get_hash()
        self.planet_name = planet.name
        
        self._cache_file = f"{EREBUS_CACHE_DIR}/{self.config_hash}_joint_fit.h5"
        
        self.results = {}
        
        self.predicted_t_secs = {}
        '''Memoize predicted t_sec to save time'''
        self.__closest_t0 = {}
        '''Memoize t0 for each visit index to save time'''
        
        self.__visit_index_lookup = {}
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path
        
        if os.path.isfile(self._cache_file) and not force_clear_cache:
            self.load_from_path(self._cache_file)
            
        self._force_clear_cache = force_clear_cache
        
        self.planet = planet
        
        # Make sure visits are in order
        photometry_data_list = sorted(photometry_data_list, key=lambda data: np.min(data.time))      
        
        # Maps the saved photometry data index to the visit index
        self.visit_indices = np.arange(0, len(photometry_data_list))
        self.photometry_data_list = photometry_data_list
        
        self.config = config
        
        self.starting_times = np.array([np.min(data.time) for data in photometry_data_list])
        
        # Track visits we're skipping
        if self.config.skip_visits is not None and len(self.config.skip_visits) > 0:
            self.visit_indices = np.delete(self.visit_indices, self.config.skip_visits)
            # self.photometry_data_list = np.delete(self.photometry_data_list, self.config.skip_visits)
        
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

            # If we are not fitting on this visit do not add it to the combined lists
            if i not in self.visit_indices:
                continue

            binned_time = bin_data(data.time[self.start_trim[i]:self.end_trim[i]], self.bin_size)[0]
            self.time.append(binned_time)
            binned_flux = bin_data(data.raw_flux[self.start_trim[i]:self.end_trim[i]], self.bin_size)[0]
            self.raw_flux.append(binned_flux)
            print(np.array(binned_eigenvalues).shape)
            
        # time per visit used to interpolate FNPCA systematic
        self.time = np.concatenate(self.time)
        self.all_eigenvalues = np.concatenate(self.joint_eigenvalues, axis=1)
        self.raw_flux = np.concatenate(self.raw_flux)              
        
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
        # Calling early to memoize them already
        for n in range(0, len(photometry_data_list)):
            self.get_predicted_t_sec_of_visit(n)
            
        # Map all times to a visit
        visit_index_map = np.array([self.get_visit_index_from_time(xi) for xi in self.time])
        self.visit_index_filter = {}
        for i in range(0, len(self.photometry_data_list)):
            self.visit_index_filter[i] = visit_index_map == i
            
        # Get eigenvalues per visit
        self.eigenvalue_map = {}
        for i in range(0, len(self.photometry_data_list)):
            self.eigenvalue_map[i] = self.all_eigenvalues.T[i].T
                
        # 
        # MCMC setup
        # 
        
        mcmc = WrappedMCMC(self._cache_file.replace(".h5", "_mcmc.h5"))
        
        fp_lower_limit = 0 if config.prevent_negative_eclipse_depth else -2000e-6
        fp_upper_limit = 2000e-6
        
        if config.fit_no_eclipse or config.fit_eclipse_depth_per_visit:
            mcmc.add_parameter("fp", Parameter.fixed(0))
        else:
            mcmc.add_parameter("fp", Parameter.uniform_prior(400e-6, fp_lower_limit, fp_upper_limit))
             
        self._add_shared_physical_params(mcmc, config, planet)
        
        # If not fitting eclipse, don't fit timing
        # If fitting a separate timing per eclipse, don't set those global offset
        should_fit_individual_t_sec_offset = not config.fit_no_eclipse and config.fit_eclipse_depth_per_visit
        should_fit_global_t_sec_offset = not config.fit_no_eclipse and not config.fit_eclipse_depth_per_visit
        should_fit_individual_eclipse_depths = not config.fit_no_eclipse and config.fit_eclipse_depth_per_visit
        
        flag_global_t_sec_set = False
        if should_fit_global_t_sec_offset:
            flag_global_t_sec_set = self._try_add_eclipse_timing_parameter("t_sec_offset", mcmc, config)
        if not flag_global_t_sec_set:
            mcmc.add_parameter("t_sec_offset", Parameter.fixed(0))
        
        for visit_index in self.visit_indices:
            flag_visit_t_sec_set = False
            name = f"t_sec_offset_{(visit_index)}"
            if should_fit_individual_t_sec_offset:
                flag_visit_t_sec_set = self._try_add_eclipse_timing_parameter(name, mcmc, config)
            if not flag_visit_t_sec_set:
                mcmc.add_parameter(name, Parameter.fixed(0))
        
        for visit_index in self.visit_indices:
            name = f"fp_{(visit_index)}"
            if should_fit_individual_eclipse_depths:
                mcmc.add_parameter(name, Parameter.uniform_prior(400e-6, fp_lower_limit, fp_upper_limit))
            else:
                mcmc.add_parameter(name, Parameter.fixed(0))

        for visit_index in self.visit_indices:
            self._add_systematic_parameters(f"_{visit_index}", visit_index, mcmc, config)

        mcmc.add_parameter("y_err", Parameter.uniform_prior(400e-6, 0, 2000e-6))
        
        # All args except for y_err
        args = ["x"] + [key for key in mcmc.params][:-1]
        fit_method = create_method_signature(self.fit_method, args)

        mcmc.set_method(fit_method)
        
        self.mcmc = mcmc
        
        number_of_physical_args = self.get_number_of_physical_args()
        number_of_systematic_args = self.get_number_of_systematic_args()
        print(f"Number of arguments: Phys {number_of_physical_args} Sys {number_of_systematic_args}")
        
        self.save_to_path(self._cache_file)
        
    #override
    def _get_predicted_t_sec_from_x(self, x : List[float]):
        visit_index = self.get_visit_index_from_time(x[0])
        return self.get_predicted_t_sec_of_visit(visit_index)
    
    #override
    def _get_transit_model_from_x(self, x : List[float], params : batman.TransitParams):
        #visit_index = self.get_visit_index_from_time(x[0])
        #if visit_index not in self.transit_models or self.transit_models[visit_index] is None:
        #    self.transit_models[visit_index] = batman.TransitModel(params, x, transittype="secondary")
        #return self.transit_models[visit_index]
        return batman.TransitModel(params, x, transittype="secondary")
    
    #override
    def _is_joint_fit(self):
        return True
    
    #override
    def _get_visit_index_from_x(self, x : List[float]):
        return self.get_visit_index_from_time(x[0])
    
    #override
    def _get_visit_starting_time_from_x(self, x : List[float]):
        i = self._get_visit_index_from_x(x)
        return self.starting_times[i]
    
    #override
    def _get_eigenvalues_from_x(self, x : List[float]):
        '''
        Assumes all x are from same visit
        '''
        i = self.get_visit_index_from_time(x[0])
        return self.eigenvalue_map[i]
        
    def get_systematic_index_start(self, i):
        num_visits = len(self.visit_indices)
        number_of_physical_args = self.get_number_of_physical_args()
        number_of_systematic_args = self.get_number_of_systematic_args()

        # Skip physical args, then 2 * num_visits for individual t_sec and fp, then skip systematic args for other visits
        # Does not skip x
        return (number_of_physical_args) + (2 * num_visits) + (i * number_of_systematic_args)
    
    #override
    def fit_method(self, *args) -> List[float]:
        '''
        Fits for the output lightcurve given the list of arguments
        '''
        x = np.array(args[0])

        number_of_physical_args = self.get_number_of_physical_args()
        number_of_systematic_args = self.get_number_of_systematic_args()
        
        physical_args = args[1:number_of_physical_args + 1]
        num_visits = len(self.visit_indices)
        
        #for i, (name, val) in enumerate(zip(self.mcmc.params.keys(), args[1:])):
        #    print(f"{i}, {name}, {val}")
        
        #print(list(self.mcmc.params.keys())[1:number_of_physical_args + 1])
                
        # Systematic arguments we're actually using will depend on the x value
        # x is a list of times
        results = np.zeros_like(x)
        for i, visit_index in enumerate(self.visit_indices):
            # Use visit_index to get filter, i elsewhere
            filt = self.visit_index_filter[visit_index]
            time = x[filt]
                        
            # +1 to skip x
            systematic_index_start = self.get_systematic_index_start(i) + 1
            systematic_args = args[systematic_index_start:systematic_index_start + number_of_systematic_args]
            #print(list(self.mcmc.params.keys())[systematic_index_start:systematic_index_start + number_of_systematic_args])

            systematic = self.systematic_model(time, *systematic_args)
            
            # Relies on argument positions of t_sec_offset and depth, not ideal
            if self.config.fit_eclipse_timing_offset_per_visit:
                visit_specific_t_sec_offset = args[number_of_physical_args + 1 + i]
                physical_args = list(physical_args)
                physical_args[-1] = visit_specific_t_sec_offset
                physical_args = tuple(physical_args)
            
            if self.config.fit_eclipse_depth_per_visit:
                visit_specific_fp = args[number_of_physical_args + 1 + i + num_visits]
                physical_args = list(physical_args)
                physical_args[0] = visit_specific_fp
                physical_args = tuple(physical_args)
            
            physical = self.physical_model(time, *physical_args)
            #print(physical_args)
            #print(systematic_args)
            results[filt] = systematic * physical
                        
        return results