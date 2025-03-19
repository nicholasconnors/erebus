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
        if self.visit_names is None or len(self.visit_names) == 0:
            print("No visits found, aborting")
            return
        
        if run_cfg.skip_visits is not None:
            filt = np.array([i not in run_cfg.skip_visits for i in range(0, len(self.visit_names))])
            self.visit_names = self.visit_names[filt]
        
        self.photometry = []
        for i in range(0, len(self.visit_names)):
            star_pos = None if run_cfg.star_position is None else (tuple)(run_cfg.star_position)
            fit = WrappedFits(run_cfg.calints_path, self.visit_names[i], 
                              force_clear_cache=force_clear_cache,
                              star_pixel_position=star_pos)
            self.photometry.append(PhotometryData(fit, run_cfg.aperture_radius,
                                                  (run_cfg.annulus_start, run_cfg.annulus_end),
                                                  force_clear_cache))
            # Improve memory usage
            del fit
        self.planet = Planet(run_cfg.planet_path)
        
        if self.config.perform_individual_fits:
            for i in range(0, len(self.visit_names)):
                individual_fit = IndividualLightcurveFit(self.photometry[i], 
                                                         self.planet, self.config)
                self.results.append(individual_fit)
                print(f"Visit {self.visit_names[i]} " + ("was already run" if 'fp' in individual_fit.results else "was not yet run"))
        # TODO: Joint fitting
    
    def run(self, force_clear_cache : bool = False):
        for fit in self.results:
            has_run = 'fp' in fit.results
            if not has_run or force_clear_cache:
                fit.run()
            else:
                print(fit.visit_name + " already ran")
        
        