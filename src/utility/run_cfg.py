import json

from typing import Annotated, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from uncertainties import ufloat
from uncertainties.core import Variable as UFloat

from pydantic_yaml import parse_yaml_file_as

import numpy as np

class ErebusRunConfig(BaseModel):
    '''
    Settings for running through the entire pipeline
    Serializable to YAML
    
    One of calints_path or uncalints_path must be set
    '''    
    fit_fnpca : Optional[bool] = False
    fit_exponential : Optional[bool] = False
    fit_linear : Optional[bool] = False
    perform_joint_fit : Optional[bool] = False
    perform_individual_fits : bool
    calints_path : Optional[str] = None
    uncalints_path : Optional[str] = None
    planet_path : str
    aperture_radius : int
    annulus_start : int
    annulus_end : int
    skip_visits : Optional[List[int]] = None
    trim_integrations : Annotated[Optional[List[int]], Field(max_length=2, min_length=2)] = None
    star_position : Annotated[Optional[List[int]], Field(max_length=2, min_length=2)] = None
    path : Optional[str] = Field(None, exclude=True)
    
    def load(path : str):
        config = parse_yaml_file_as(ErebusRunConfig, path)
        config.path = path
        return config
    
    def save_schema(path : str):
        run_schema = ErebusRunConfig.model_json_schema()
        run_schema_json = json.dumps(run_schema, indent=2)
        with open(path, "w") as f:
            f.write(run_schema_json)