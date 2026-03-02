import json
import hashlib
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

class ErebusRunConfig(BaseModel):
    '''
    Settings for running through the entire Erebus pipeline.
    Serializable to/from YAML. One of perform_joint_fit or perform_individual_fits must
    be true else the run will not do anything.
    
    Attributes:
        fit_fnpca (bool): Optional bool to use FN-PCA in the systematic model.
        fit_exponential (bool): Optional bool to use an exponential curve in the systematic model.
        fit_linear (bool): Optional bool to use a linear slope in the systematic model.
        perform_joint_fit (bool): Optional bool to fit all visits together with a shared eclipse depth.
        perform_individual_fits (bool): Optional bool to fit each visit with their own eclipse depth.
        calints_path (str): Relative path from the folder containing this file to where the calints.fits files are.
        uncal_path (str): Relative path from the folder containing this file to where the uncal.fits files are. One of uncal_path or calints_path must be set.
        planet_path (str): Relative path from the folder containing this file to where the planet config is.
        aperture_radius (int): Pixel radius for aperture photometry.
        annulus_start (int): Inner pixel radius of disk used for background subtraction.
        annulus_end (int): Outer pixel radius of disk used for background subtraction.
        skip_visits (list[int]): Optional list of indices to skip when doing individual fits. Index based on visit ID.
        trim_integrations (list[int]): Length-two list with the number of integrations to clip from the start and end. Optional.
        star_position (list[int]): X and y pixel coordinates of the star. Optional (will search for the star or assume its centered).
        prevent_negative_eclipse_depth (bool): Optional bool to force eclipse depth to be positive.
        fix_eclipse_timing (bool): Optional bool to force t0, period, ecosw to be fixed
        fit_uniform_eclipse_timing_offset (float): Optional float (days) to fit t_sec offset as a uniform prior. First value is time from 0.5 phase, second value is half width
        fit_gaussian_eclipse_timing_offset (float): Optional float (days) to fit t_sec offset as a gaussian prior. First value is time from 0.5 phase, second value is std dev
        joint_fit_bin_size (int): Optional bin size for joint fit, if performed. Defaults to 4.
        fit_no_eclipse (bool): Optionally fit only systematic (as in, eclipse depth is 0).
    '''    
    fit_fnpca : Optional[bool] = False
    fit_exponential : Optional[bool] = False
    fit_linear : Optional[bool] = False
    perform_joint_fit : Optional[bool] = False
    perform_individual_fits : bool
    calints_path : Optional[str] = None
    uncal_path : Optional[str] = None
    planet_path : str
    aperture_radius : int
    annulus_start : int
    annulus_end : int
    skip_visits : Optional[List[int]] = None
    trim_integrations : Optional[Union[List[int], List[List[int]]]] = None
    star_position : Annotated[Optional[List[int]], Field(max_length=2, min_length=2)] = None
    path : Optional[str] = Field(None, exclude=True)
    prevent_negative_eclipse_depth: Optional[bool] = False
    fix_eclipse_timing: Optional[bool] = False
    fit_uniform_eclipse_timing_offset : Annotated[Optional[List[float]], Field(max_length=2, min_length=2)] = None
    fit_gaussian_eclipse_timing_offset : Annotated[Optional[List[float]], Field(max_length=2, min_length=2)] = None
    max_steps : Optional[int] = None
    fit_lightcurve_phase : Optional[bool] = False
    joint_fit_bin_size : Optional[int] = 4
    fit_no_eclipse : Optional[bool] = False
    
    _custom_systematic_model = None
    _custom_parameters : dict = None
    _custom_parameters_override : dict = {}
    
    def get_trim_integrations(self, visit_index):
        # Assumes that trim_integrations is properly formatted as either None, [start, end], [[visit 1 start, end], [visit 2 start, end], [etc]]
        result = [0, None]
        if self.trim_integrations is not None:
            # List of two
            if isinstance(self.trim_integrations[0], int):
                result = self.trim_integrations
            else:
                result = self.trim_integrations[visit_index]
        if result[1] is not None:
            if result[1] <= 0:
                result[1] = None
            else:
                result[1] = -abs(result[1])
        return result
    
    def set_custom_systematic_model(self, model, params):
        '''
        Optionally provide a callable function and dictionary of Parameter objects for bayesian priors.
        
        Order of parameters must match their order in the model method signature.
        
        Params are given as a dictionary of their names (matching the method signature) to a Parameter object.
        
        Model method signature must start with x.
        
        When used in conjunction with built-in fitting model provided by Erebus this model will be multiplied 
        by those fitting models and a best-fit y-offset applied.
        '''
        self._custom_systematic_model = model
        self._custom_parameters = params
        print("Registered custom systematic model")
        
    def set_custom_systematic_model_prior_for_visit_index(self, index, params):
        '''
        Optionally override custom systematic model priors for specific visits
        
        Params are given as a dictionary of their names (matching the method signature) to a Parameter object.
        '''
        self._custom_parameters_override[index] = params
        
    def get_hash(self):
        '''
        Returns a unique hash representing this run config
        '''
        return hashlib.md5((json.dumps(self.model_dump()) + json.dumps(list(self._custom_parameters.keys()))).encode()).hexdigest()
    
    @staticmethod
    def load(path : str):
        '''
        Loads EreusRunConfig from a yaml file
        '''
        config = parse_yaml_file_as(ErebusRunConfig, path)
        config.path = path
        return config
    
    def save(self, path : str):
        '''
        Saves this ErebusRunConfig instance to the path as a yaml file
        '''
        to_yaml_file(path, self)
    
    @staticmethod
    def _save_schema(path : str):
        '''
        Saves the json schema for validating ErebusRunConfig instances
        '''
        run_schema = ErebusRunConfig.model_json_schema()
        run_schema_json = json.dumps(run_schema, indent=2)
        with open(path, "w") as f:
            f.write(run_schema_json)