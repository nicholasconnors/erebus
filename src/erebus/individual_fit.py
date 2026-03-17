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
from erebus.base_fit import BaseFit

EREBUS_CACHE_DIR = "erebus_cache"

class IndividualFit(BaseFit):
    __instance = None
    
    def _exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['config', 'time', 'raw_flux', 'params', 'transit_model', 'mcmc', '__instance', 
                'photometry_data', '_force_clear_cache', 'predicted_t_sec', 'start_trim', 'end_trim', 'planet']
    
    def __init__(self, photometry_data : PhotometryData, planet : Planet, config : ErebusRunConfig,
                 force_clear_cache : bool = False, override_cache_path : str = None, index = None):
        super().__init__()
        
        self.visit_name = photometry_data.visit_name
        self.config_hash = config.get_hash()
        self.planet_name = planet.name
        self.planet = planet
        self.order_label = 'X'
        self.index = index
        self.photometry_data = photometry_data

        self._cache_file = f"{EREBUS_CACHE_DIR}/{self.visit_name}_{self.config_hash}_individual_fit.h5"
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path
        
        trim = config.get_trim_integrations(index)
        self.start_trim = trim[0]
        self.end_trim = trim[1]
        
        self.start_time = np.min(photometry_data.time)
        self.time = photometry_data.time[self.start_trim:self.end_trim]
        self.raw_flux = photometry_data.raw_flux[self.start_trim:self.end_trim]
        self.config = config
        
        self.results = {}
        self.chain = None
        
        self.params = None
        self.transit_model = None
        
        self.eigenvalues, self.eigenvectors, self.pca_variance_ratios = perform_fn_pca_on_aperture(photometry_data.normalized_frames[self.start_trim:self.end_trim])
                
        mcmc = WrappedMCMC(self._cache_file.replace(".h5", "_mcmc.h5"))
        
        start_time = np.min(photometry_data.time)
        self.predicted_t_sec = planet.get_predicted_tsec(start_time)
        
        lower_limit = 0 if config.prevent_negative_eclipse_depth else -2000e-6
        
        if config.fit_no_eclipse:
            mcmc.add_parameter("fp", Parameter.fixed(0))
        else:
            mcmc.add_parameter("fp", Parameter.uniform_prior(400e-6, lower_limit, 2000e-6))
             
        self._add_shared_physical_params(mcmc, config, planet)
        
        flag_t_sec_set = False
        if not config.fit_no_eclipse:
            flag_t_sec_set = self._try_add_eclipse_timing_parameter("t_sec_offset", mcmc, config)
        if not flag_t_sec_set:
            mcmc.add_parameter("t_sec_offset", Parameter.fixed(0))
        
        self._add_systematic_parameters("", self.index, mcmc, config)

        # y_err always goes last
        mcmc.add_parameter("y_err", Parameter.uniform_prior(400e-6, 0, 2000e-6))      
                  
        args = ["x"] + [key for key in mcmc.params][:-1]
                
        fit_method = create_method_signature(self.fit_method, args)
                  
        mcmc.set_method(fit_method)
                  
        self.mcmc = mcmc
                                
        if os.path.isfile(self._cache_file) and not force_clear_cache:
            self.load_from_path(self._cache_file)
        else:
            self.save_to_path(self._cache_file)
        
        self._force_clear_cache = force_clear_cache
    
    #override
    def _get_predicted_t_sec_from_x(self, x : List[float]):
        return self.predicted_t_sec.nominal_value
    
    #override
    def _get_transit_model_from_x(self, x : List[float], params : batman.TransitParams):
        if self.transit_model is None:
            self.transit_model = batman.TransitModel(params, x, transittype="secondary")
        return self.transit_model
    
    #override
    def _is_joint_fit(self):
        return False
    
    #override
    def _get_visit_index_from_x(self, x : List[float]):
        return self.index
    
    #override
    def _get_visit_starting_time_from_x(self, x : List[float]):
        return self.start_time
    
    #override
    def _get_eigenvalues_from_x(self, x : List[float]):
        return self.eigenvalues
        
    @staticmethod
    def __fit_method(x : List[float], fp : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, esinw : float, ecosw : float, t_sec_offset : float,
                       pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float,
                       exp1 : float, exp2 : float, a : float, b : float, *extra_params) -> List[float]:
        systematic = IndividualFit.__instance.systematic_model(x, pc1, pc2, pc3, pc4, pc5, exp1, exp2, a, b, *extra_params)
        physical = IndividualFit.__instance.physical_model(x, fp, rp_rstar, a_rstar, p, inc, esinw, ecosw, t_sec_offset)
        return physical * systematic 
    
    #override
    def fit_method(self, x : List[float], *args) -> List[float]:
        '''
        For external use, calls the method used for fitting (*args is a list of the parameters)
        '''
        IndividualFit.__instance = self
        return IndividualFit.__fit_method(x, *args)
