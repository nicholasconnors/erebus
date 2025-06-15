import os
import hashlib
import numpy as np
from erebus.utility.h5_serializable_file import H5Serializable
from erebus.utility.run_cfg import ErebusRunConfig
from erebus.utility import utils as utils
from erebus.photometry_data import PhotometryData
from erebus.wrapped_fits import WrappedFits
from erebus.utility.planet import Planet
import erebus.utility.fits_file_utils as f_util 
from erebus.individual_fit import IndividualFit
from erebus.joint_fit import JointFit
import json
from datetime import datetime
from erebus.joint_fit_results import JointFitResults
from erebus.individual_fit_results import IndividualFitResults
import erebus.plotting as plotting

EREBUS_CACHE_DIR = "erebus_cache"

class Erebus(H5Serializable):   
    '''
    Object instance for running the full pipeline, starting from calints files.
    ''' 
    
    def _exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['individual_fits', 'joint_fit', 'photometry', 'planet']
    
    @staticmethod
    def load(path : str):
        '''Helper method to directly load an Erebus instance cache file.'''
        return Erebus(None, override_cache_path=path)
    
    def __init__(self, run_cfg : ErebusRunConfig | str, force_clear_cache : bool = False,
                 override_cache_path : str = None):    
        
        if isinstance(run_cfg, str):
            run_cfg = ErebusRunConfig.load(run_cfg) 
        
        config_hash = hashlib.md5(json.dumps(run_cfg.model_dump()).encode()).hexdigest()   
        self._cache_file = f"{EREBUS_CACHE_DIR}/{config_hash}_erebus.h5"
    
        if override_cache_path is not None:
            self._cache_file = override_cache_path
    
        self.config : ErebusRunConfig = run_cfg
        '''The configuration file used for this instance of the Erebus pipeline.'''
        
        self.photometry : list[PhotometryData] = []
        '''The photometry data of each visit.'''
        
        self.individual_fits : list[IndividualFit] = []
        '''The individual fit instances of each visit.'''
        
        self.joint_fit : JointFit = None
        '''The joint fit instance.'''
        
        # Record absolute path so that a run file can be moved elsewhere and still work
        root_folder = os.path.dirname(os.path.abspath(run_cfg.path))
        if os.path.isabs(run_cfg.calints_path):
            self._calints_abs_path = run_cfg.calints_path
        else:
            self._calints_abs_path = root_folder + os.sep + run_cfg.calints_path
        
        self.visit_names : list[str] = []
        '''The unique names of each visit.'''
        
        # Load from file if needed
        if force_clear_cache or not os.path.isfile(self._cache_file):
            self.visit_names = f_util.get_fits_files_visits_in_folder(self._calints_abs_path)
            if self.visit_names is None or len(self.visit_names) == 0:
                print("No visits found, aborting")
                return
        
            if run_cfg.skip_visits is not None:
                filt = np.array([i not in run_cfg.skip_visits for i in range(0, len(self.visit_names))])
                self.visit_names = self.visit_names[filt]
        
        else:
            self.load_from_path(self._cache_file)
            
        for i in range(0, len(self.visit_names)):
            star_pos = None if run_cfg.star_position is None else (tuple)(run_cfg.star_position)
            fit = WrappedFits(self._calints_abs_path, self.visit_names[i], 
                              force_clear_cache=force_clear_cache,
                              star_pixel_position=star_pos)
            self.photometry.append(PhotometryData(fit, run_cfg.aperture_radius,
                                                  (run_cfg.annulus_start, run_cfg.annulus_end),
                                                  force_clear_cache))
            # Improve memory usage
            del fit
            
        # Planet path is relative to the config file
        planet_path = run_cfg.planet_path
        if not os.path.isabs(planet_path): 
            planet_path = os.path.join(os.path.dirname(run_cfg.path), planet_path)
        self.planet = Planet(planet_path)
        '''The planet configuration file used for this instance of the pipeline'''
        
        if self.config.perform_individual_fits:
            for i in range(0, len(self.visit_names)):
                individual_fit = IndividualFit(self.photometry[i], 
                                                         self.planet, self.config,
                                                         force_clear_cache)
                self.individual_fits.append(individual_fit)
                print(f"Visit {self.visit_names[i]} " + ("already ran" if 'fp' in individual_fit.results else "wasn't run yet"))
            
            # Label the visits by the order they were observed
            individual_fit_order = np.argsort([fit.start_time for fit in self.individual_fits]) + 1           
            for i, fit in enumerate(self.individual_fits):
                # If you are skipping visits this won't be done since it will be inaccurate
                if self.config.skip_visits is None:
                    fit.order = individual_fit_order[i]
                else:
                    fit.order = "X"

        if self.config.perform_joint_fit:
            self.joint_fit = JointFit(self.photometry, self.planet, self.config, force_clear_cache)
            print(f"Joint fit " + ("already ran" if 'fp' in self.joint_fit.results else "wasn't run yet"))
        
        self.save_to_path(self._cache_file)
    
    def run(self, force_clear_cache : bool = False, output_folder="./output_{DATE}/"):
        '''
        Performs all individual and joint fits. Results and plots are saved to the given folder.
        Output folder can optionally include the current time by writing {DATE}
        '''
        time = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        output_folder = output_folder.replace("{DATE}", time)
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            
        eigenvec_folder = output_folder + "eigenvec"
        if not os.path.isdir(eigenvec_folder):
            os.makedirs(eigenvec_folder)
        
        figure_folder = output_folder + "figures"
        if not os.path.isdir(figure_folder):
            os.makedirs(figure_folder)
            
        # Save inputs that were used
        self.config.save(output_folder + self.planet.name + "_run_config.yaml")
        self.planet.save(output_folder + self.planet.name + "_planet_config.yaml")
        
        if self.config.perform_individual_fits:
            for fit in self.individual_fits:
                has_run = 'fp' in fit.results
                if not has_run or force_clear_cache:
                    fit.run()
                else:
                    print("Skipping " + fit.visit_name + ": already ran")
                plotting.plot_fnpca_individual_fit(fit, figure_folder)
                plotting.plot_eigenvectors(fit, eigenvec_folder)        
                plotting.corner_plot(fit.mcmc, f"{figure_folder}/{fit.planet_name}_{fit.visit_name}_{fit.config_hash}_corner.pdf")
                plotting.chain_plot(fit.mcmc, f"{figure_folder}/{fit.planet_name}_{fit.visit_name}_{fit.config_hash}_chain.pdf")
    
                path = output_folder + self.planet.name + "_visit_" + str(fit.order) + "_" + fit.visit_name
                IndividualFitResults(fit).save_to_path(path + ".h5")
                
                dict = fit.results.copy()
                dict['auto_corr'] = fit.auto_correlation
                utils.save_dict_to_json(dict, path + ".json")

        if self.config.perform_joint_fit:
            has_run = 'fp' in self.joint_fit.results
            if not has_run or force_clear_cache:
                self.joint_fit.run()
            else:
                print("Skipping joint fit: already ran")
            plotting.plot_joint_fit(self.joint_fit, figure_folder)
            plotting.corner_plot(self.joint_fit.mcmc, f"{figure_folder}/{self.joint_fit.planet_name}_joint_{self.joint_fit.config_hash}_corner.pdf")
            plotting.chain_plot(self.joint_fit.mcmc, f"{figure_folder}/{self.joint_fit.planet_name}_joint_{self.joint_fit.config_hash}_chain.pdf")
    
            path = output_folder + self.planet.name + "_joint_fit"
            JointFitResults(self.joint_fit).save_to_path(path + ".h5")
            
            dict = self.joint_fit.results.copy()
            dict['auto_corr'] = self.joint_fit.auto_correlation
            utils.save_dict_to_json(dict, path + ".json")
        
        