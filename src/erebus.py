import os
import hashlib
from typing import List
import numpy as np
from src.utility.h5_serializable_file import H5Serializable
from src.utility.run_cfg import ErebusRunConfig
from src.photometry_data import PhotometryData
from src.wrapped_fits import WrappedFits
from src.utility.planet import Planet
from src.mcmc_model import WrappedMCMC
import src.utility.fits_file_utils as f_util 
from src.utility.bayesian_parameter import Parameter
import batman
from src.individual_lightcurve_fit import IndividualLightcurveFit

EREBUS_CACHE_DIR = "erebus_cache"

class Erebus(H5Serializable):   
    '''
    Object instance for running the full pipeline
    ''' 
    
    def __init__(self, run_cfg : ErebusRunConfig, force_clear_cache : bool = False):        
        # TODO: Allow defining a matrix of possible run inputs
        # Config file should be split into one for a set of runs and one for each individual run
        self.config = run_cfg
        
        self.results = []
        
        self.visit_names = f_util.get_fits_files_visits_in_folder(run_cfg.calints_path)
        if run_cfg.skip_visits is not None:
            filt = np.array([i not in run_cfg.skip_visits for i in range(0, len(self.visit_names))])
            self.visit_names = self.visit_names[filt]
            
        self.fits = [WrappedFits(run_cfg.calints_path, visit_name, force_clear_cache) for visit_name in self.visit_names]
        self.photometry = [PhotometryData(fit, run_cfg.aperture_radius, (run_cfg.annulus_start, run_cfg.annulus_end), force_clear_cache) for fit in self.fits]
        self.planet = Planet(run_cfg.planet_path)
        
        if self.config.perform_individual_fits:
            for i in range(0, len(self.visit_names)):
                individual_fit = IndividualLightcurveFit(self.photometry[i], self.fits[i], self.planet, self.config)
                self.results.append(individual_fit)
        # TODO: Joint fitting
    
    def run(self):
        for fit in self.results:
            fit.run()
        
        