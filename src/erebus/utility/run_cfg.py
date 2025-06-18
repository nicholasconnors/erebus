import json

from typing import Annotated, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from uncertainties import ufloat
from uncertainties.core import Variable as UFloat

from pydantic_yaml import parse_yaml_file_as
from pydantic_yaml import to_yaml_file

import numpy as np

from erebus.utility.bayesian_parameter import Parameter

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
        calints_path (str): Relative path from the folder containing this file to where the .fits files are.
        planet_path (str): Relative path from the folder containing this file to where the planet config is.
        aperture_radius (int): Pixel radius for aperture photometry.
        annulus_start (int): Inner pixel radius of disk used for background subtraction.
        annulus_end (int): Outer pixel radius of disk used for background subtraction.
        skip_visits (list[int]): Optional list of indices to skip when doing individual fits. Index based on visit ID.
        trim_integrations (list[int]): Length-two list with the number of integrations to clip from the start and end. Optional.
        star_position (list[int]): X and y pixel coordinates of the star. Optional (will search for the star or assume its centered).
        skip_emcee_backend_cache (bool): Optional bool to not save emcee backend. Speeds up run time but can lose progress during a run if stopped early.
    '''    
    fit_fnpca : Optional[bool] = False
    fit_exponential : Optional[bool] = False
    fit_linear : Optional[bool] = False
    perform_joint_fit : Optional[bool] = False
    perform_individual_fits : bool
    calints_path : str = None
    planet_path : str
    aperture_radius : int
    annulus_start : int
    annulus_end : int
    skip_visits : Optional[List[int]] = None
    trim_integrations : Annotated[Optional[List[int]], Field(max_length=2, min_length=2)] = None
    star_position : Annotated[Optional[List[int]], Field(max_length=2, min_length=2)] = None
    path : Optional[str] = Field(None, exclude=True)
    skip_emcee_backend_cache: Optional[bool] = False
    
    _custom_systematic_model = None
    _custom_parameters : dict = None
    
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
    
    def load(path : str):
        config = parse_yaml_file_as(ErebusRunConfig, path)
        config.path = path
        return config
    
    def save(self, path : str):
        to_yaml_file(path, self)
    
    def _save_schema(path : str):
        run_schema = ErebusRunConfig.model_json_schema()
        run_schema_json = json.dumps(run_schema, indent=2)
        with open(path, "w") as f:
            f.write(run_schema_json)