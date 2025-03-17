import json

from typing import Annotated, List, Optional

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
    '''    
    fit_fnpca : Optional[bool] = False
    fit_exponential : Optional[bool] = False
    fit_linear : Optional[bool] = False
    perform_joint_fit : Optional[bool] = False
    perform_individual_fits : bool
    calints_path : str
    planet_path : str
    aperture_radius : int
    annulus_start : int
    annulus_end : int
    skip_visits : Optional[List[int]] = None
    
    def load(path : str):
        return parse_yaml_file_as(ErebusRunConfig, path)
    
    def save_schema(path : str):
        run_schema = ErebusRunConfig.model_json_schema()
        run_schema_json = json.dumps(run_schema, indent=2)
        with open(path, "w") as f:
            f.write(run_schema_json)