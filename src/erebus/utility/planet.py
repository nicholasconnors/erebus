import json

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from uncertainties import ufloat
from uncertainties.core import Variable as UFloat

from pydantic_yaml import parse_yaml_file_as
from pydantic_yaml import to_yaml_file

import numpy as np
import os

class Planet:
    '''
    Class which represents known planet values with uncertainties.
    Loaded from a yaml file and optionally validated against a schema. Uncertainties are
    expressed by writing the values into the yaml file as a list of floats, where the first value is
    the absolute value of the measurement. The second value is taken as the symmetric error, and if
    a third value is provided it is taken a the upper error.
    
    Attributes:
        name (str): The name of the planet.
        t0 (UFloat | float): Midpoint time of reference transit in BJD
        t0_lookup_path (str): Optional replacement for t0, relative path to csv file with calculated TTVs (Two columns: t0 in BJD-2,450,000, error)
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
    
    # Name mangling breaks pickle
    class _PlanetYAML(BaseModel):
        '''
        Serialized YAML representation of a Planet for the Erebus pipeline.
        
        Planet parameters with optional uncertainties are represented as lists of up to 3 floats
        1 float = no uncertainty, 2 floats = symmetric error, 3 floats = asymmetric error.
        
        Attributes:
            name            Name of the planet
            t0              Midpoint time of reference transit in BJD
            t0_lookup_path  Optional replacement for t0, relative path to file with calculated TTVs (Two columns: t0, error)
            a_rstar         Semi-major axis in units of stellar radii
            p               Orbital period in days
            rp_rstar        Radius of the exoplanet in units of stellar radii
            inc             Inclination in degrees
            ecc             Eccentricity
            w               Argument of periastron in degrees   
        '''
        def __make_title(field_name: str, _: FieldInfo) -> str:
            return field_name
        
        name : str
        t0 : Optional[Annotated[List[float], Field(default=None,max_length=3, field_title_generator=__make_title)]] = None
        t0_lookup_path: Optional[str] = None
        a_rstar : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        p : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        rp_rstar : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        inc : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        ecc : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        w : Annotated[List[float], Field(max_length=3, field_title_generator=__make_title)]
        cache : Optional[dict] = Field(include_in_schema=False, default=None)

    def __ufloat_from_list(self, l : List[float]) -> UFloat | float:
        if l is None:
            return None
        elif len(l) == 1:
            return l[0]
        elif len(l) == 2:
            return ufloat(l[0], np.abs(l[1]))
        elif len(l) == 3:
            return ufloat(l[0], np.max(np.abs(l[1:])))

    def __load_from_yaml(self, yaml : _PlanetYAML):
        self.name = yaml.name
        self.t0 = self.__ufloat_from_list(yaml.t0)
        self.t0_lookup_path = yaml.t0_lookup_path
        self.a_rstar = self.__ufloat_from_list(yaml.a_rstar)
        self.p = self.__ufloat_from_list(yaml.p)
        self.rp_rstar = self.__ufloat_from_list(yaml.rp_rstar)
        self.inc = self.__ufloat_from_list(yaml.inc)
        self.ecc = self.__ufloat_from_list(yaml.ecc)
        self.w = self.__ufloat_from_list(yaml.w)
        self._yaml = yaml
        
        if not hasattr(yaml, 'cache'):
            yaml.cache = None

        # Saving extra data to a cache for when files are reloaded elsewhere
        if yaml.cache is None:
            yaml.cache = {}
        if self.__path is None:
            self.__path = yaml.cache["path"]
        if self.t0_lookup_path is not None and "t0_lookup" not in yaml.cache:
            path = self.t0_lookup_path
            if not os.path.isabs(path):
                folder = os.path.dirname(self.__path)
                if not os.path.isabs(folder):
                    folder = os.getcwd() + "/" + folder
                path = folder + "/" + path
            # Can't serialize np array
            yaml.cache['t0_lookup'] = np.loadtxt(path, delimiter=',').tolist()
    
    def save(self, path : str):
        to_yaml_file(path, self._yaml)
    
    def __init__(self, yaml_path : str):
        self.__path = yaml_path
        self.__load_from_yaml(parse_yaml_file_as(Planet._PlanetYAML, yaml_path))
    
    def _save_schema(path : str):
        planet_schema = Planet._PlanetYAML.model_json_schema()
        planet_schema_json = json.dumps(planet_schema, indent=2)
        with open(path, "w") as f:
            f.write(planet_schema_json)
            
    def get_closest_t0(self, obs_start):
        '''Given a start time in BJD-2,400,000.5, use the lookup file to get the closest t0'''
        if self.t0_lookup_path is None:
            return self.t0 - 2400000.5
        else:            
            table = np.array(self._yaml.cache['t0_lookup'])
            t0s = table[:,0] + 2450000 - 2400000.5
            lt_target = np.argwhere(t0s < obs_start)
            if len(lt_target) == 0:
                ind = 0
            else:
                ind = np.max(lt_target)
            return ufloat(table[ind,0] + 2450000 - 2400000.5, table[ind,1])
        