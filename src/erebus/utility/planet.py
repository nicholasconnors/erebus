import json

from typing import Annotated, List

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from uncertainties import ufloat
from uncertainties.core import Variable as UFloat

from pydantic_yaml import parse_yaml_file_as
from pydantic_yaml import to_yaml_file

import numpy as np

class Planet:
    '''
    Class which represents known planet values with uncertainties.
    Loaded from a yaml file and optionally validated against a schema. Uncertainties are
    expressed by writing the values into the yaml file as a list of floats, where the first value is
    the absolute value of the measurement. The second value is taken as the symmetric error, and if
    a third value is provided it is taken a the upper error.
    
    Attributes:
        name (str): The name of the planet.
        t0 (UFloat | float): The last known time of conjunction of the planet.
        a_rstar (UFloat | float): The ratio of semi-major axis to star radius.
        p (UFloat | float): The period of the planet in days.
        rp_rstar (UFloat | float): The ratio of the planet's radius to the star's radius.
        inc (UFloat | float): The inclination of the planet in degrees.
        ecc (UFloat | float): The eccentricity of the planet (0 inclusive to 1 exclusive).
        w (UFloat | float): The argument of periastron of the planet (degrees).
    '''
    name : str
    t0 : UFloat | float
    a_rstar : UFloat | float
    p : UFloat | float
    rp_rstar : UFloat | float
    inc : UFloat | float
    ecc : UFloat | float
    w : UFloat | float
    
    class __PlanetYAML(BaseModel):
        '''
        Serialized YAML representation of a Planet for the Erebus pipeline.
        
        Planet parameters with optional uncertainties are represented as lists of up to 3 floats
        1 float = no uncertainty, 2 floats = symmetric error, 3 floats = asymmetric error.
        
        Attributes:
            name        Name of the planet
            t0          Midpoint time of reference transit
            a_rstar     Semi-major axis in units of stellar radii
            p           Orbital period in days
            rp_rstar    Radius of the exoplanet in units of stellar radii
            inc         Inclination in degrees
            ecc         Eccentricity
            w           Argument of periastron in degrees   
        '''
        def __make_title(field_name: str, _: FieldInfo) -> str:
            return field_name
        
        name : str
        t0 : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        a_rstar : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        p : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        rp_rstar : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        inc : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        ecc : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        w : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]

    def __ufloat_from_list(self, l : List[float]) -> UFloat | float:
        if len(l) == 1:
            return l[0]
        elif len(l) == 2:
            return ufloat(l[0], np.abs(l[1]))
        elif len(l) == 3:
            return ufloat(l[0], np.max(np.abs(l[1:])))

    def __load_from_yaml(self, yaml : __PlanetYAML):
        self.name = yaml.name
        self.t0 = self.__ufloat_from_list(yaml.t0)
        self.a_rstar = self.__ufloat_from_list(yaml.a_rstar)
        self.p = self.__ufloat_from_list(yaml.p)
        self.rp_rstar = self.__ufloat_from_list(yaml.rp_rstar)
        self.inc = self.__ufloat_from_list(yaml.inc)
        self.ecc = self.__ufloat_from_list(yaml.ecc)
        self.w = self.__ufloat_from_list(yaml.w)
        self.__yaml = yaml
    
    def save(self, path : str):
        to_yaml_file(path, self.__yaml)
    
    def __init__(self, yaml_path : str):
        self.__load_from_yaml(parse_yaml_file_as(Planet.__PlanetYAML, yaml_path))
    
    def _save_schema(path : str):
        planet_schema = Planet.__PlanetYAML.model_json_schema()
        planet_schema_json = json.dumps(planet_schema, indent=2)
        with open(path, "w") as f:
            f.write(planet_schema_json)
        