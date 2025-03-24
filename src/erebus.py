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
from src.individual_fit import IndividualFit
from src.joint_fit import JointFit
import json

EREBUS_CACHE_DIR = "erebus_cache"

class Erebus(H5Serializable):   
    '''
    Object instance for running the full pipeline
    ''' 
    
    def exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['config', 'individual_fits', 'joint_fit', 'photometry', 'planet']
    
    
    def __init__(self, run_cfg : ErebusRunConfig, force_clear_cache : bool = False):    
        config_hash = hashlib.md5(json.dumps(run_cfg.model_dump()).encode()).hexdigest()   
        self.cache_file = f"{EREBUS_CACHE_DIR}/{config_hash}_erebus.h5"
    
        self.config = run_cfg
        
        self.photometry = []
        self.individual_fits = []
        
        if force_clear_cache or not os.path.isfile(self.cache_file):
            self.visit_names = f_util.get_fits_files_visits_in_folder(run_cfg.calints_path)
            if self.visit_names is None or len(self.visit_names) == 0:
                print("No visits found, aborting")
                return
        
            if run_cfg.skip_visits is not None:
                filt = np.array([i not in run_cfg.skip_visits for i in range(0, len(self.visit_names))])
                self.visit_names = self.visit_names[filt]
        
        else:
            self.load_from_path(self.cache_file)
            
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
                individual_fit = IndividualFit(self.photometry[i], 
                                                         self.planet, self.config,
                                                         force_clear_cache)
                self.individual_fits.append(individual_fit)
                print(f"Visit {self.visit_names[i]} " + ("was already run" if 'fp' in individual_fit.results else "was not yet run"))
        if self.config.perform_joint_fit:
            self.joint_fit = JointFit(self.photometry, self.planet, self.config, force_clear_cache)
        
        if force_clear_cache:
            self.save_to_path(self.cache_file)
    
    def run(self, force_clear_cache : bool = False):
        if self.config.perform_individual_fits:
            for fit in self.individual_fits:
                has_run = 'fp' in fit.results
                if not has_run or force_clear_cache:
                    fit.run()
                else:
                    print(fit.visit_name + " already ran")
        if self.config.perform_joint_fit:
            has_run = 'fp' in self.joint_fit.results
            if not has_run or force_clear_cache:
                self.joint_fit.run()
            else:
                print("Joint fit already ran")
        