from typing import List
import h5py
import numpy as np
import os
import inspect
import json
from uncertainties.core import Variable as UFloat
from uncertainties import ufloat
import types
from pydantic import BaseModel
from pydantic_core import from_json
from erebus.utility.utils import _JSONDecoder
from erebus.utility.utils import _JSONEncoder
from erebus.utility.planet import Planet

class H5Serializable:   
    '''
    A class that can be serialized to/from an h5 file. Does not support groups.
    '''
    def _exclude_keys(self) -> List[str]:
        '''
        Excluded from serialization
        '''
        return []
    
    def load_from_path(self, file_path : str):
        '''
        Loads a serializable file and returns itself
        '''
        try:
            hf = h5py.File(file_path, 'r')

            # TODO: Support recursion through groups
            for name, value in hf.attrs.items():
                if name in self._exclude_keys():
                    continue
                # Dictionaries and ufloats have custom serialization to strings
                if isinstance(value, str):
                    if value.startswith("JSON"):
                        value = json.loads(value[len("JSON"):], cls=_JSONDecoder)
                    elif value.startswith("UFLOAT"):
                        nominal_value, std_dev = value[len("UFLOAT"):].split("+/-")
                        value = ufloat(float(nominal_value), float(std_dev))
                    elif value.startswith("PYDANTIC"):
                        value = from_json(value[len("PYDANTIC"):])
                    elif isinstance(value, Planet):
                        value = from_json(value[len("PLANET"):])    
                        value = Planet.__load_from_yaml(value)                
                    
                self.__setattr__(name, value)
            for name, value in hf.items():
                if name in self._exclude_keys():
                    continue
                if isinstance(value, h5py.Dataset):
                    v = value[()]
                    # string arrays get serialized to bytes
                    if len(v) != 0 and isinstance(v[0], bytes):
                        v = [b.decode("utf-8") for b in v]
                    self.__setattr__(name, np.array(v))
        except Exception as e:
            print(f"Failed to load h5 data: {e}")
            raise
        finally:
            hf.close()
        return self
    
    def save_to_path(self, file_path : str):
        folder = os.path.dirname(os.path.abspath(file_path))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        try:
            hf = h5py.File(file_path, 'w')
            # Filter out "private" attributes
            names = [key for key in dir(self) if not key.startswith('_') and not key in self._exclude_keys()]
            for name in names:
                # Don't try to save functions
                if isinstance(getattr(self.__class__, name, None), types.FunctionType):
                    continue
                try:
                    value = self.__getattribute__(name)
                    if value is None:
                        continue
                    # numpy strings don't serialize properly
                    # frankly I don't even know what a np string is but they crop up sometimes
                    if isinstance(value, np.str_):
                        value = str(value)
                    # dictionaries don't serialize either so we go through json
                    elif isinstance(value, dict):
                        value = "JSON" + json.dumps(value, cls=_JSONEncoder)
                    elif isinstance(value, UFloat):
                        value = f"UFLOAT{value.nominal_value}+/-{value.std_dev}"

                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        value = [str(v) if isinstance(v, np.str_) else v for v in value]
                        hf.create_dataset(name, data = value)
                    elif isinstance(value, BaseModel):
                        value = "PYDANTIC" + json.dumps(value.model_dump(mode='json'), cls=_JSONEncoder)
                    elif isinstance(value, Planet):
                        value = "PLANET" + json.dumps(value._yaml.model_dump(mode='json'), cls=_JSONEncoder)
                    elif not inspect.ismethod(value):
                        hf.attrs[name] = value
                except Exception as e:
                    print(f"Couldn't save [{name}]")
                    raise
        except Exception as e:
            print(f"Failed to save h5 data: {e}")
            raise
        finally:
            hf.close()