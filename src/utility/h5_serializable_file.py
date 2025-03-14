import h5py
import numpy as np
import os
import inspect

class H5Serializable:
    '''
    A class that can be serialized to/from an h5 file. Does not support groups.
    '''
    def load_from_path(self, file_path : str):
        try:
            hf = h5py.File(file_path, 'r')

            # TODO: Support recursion through groups
            for name, value in hf.attrs.items():
                self.__setattr__(name, value)
            for name, value in hf.items():
                if isinstance(value, h5py.Dataset):
                    self.__setattr__(name, np.array(value[()]))
        except Exception as e:
            print(f"Failed to load h5 data: {e}")
            raise
        finally:
            hf.close()
    
    def save_to_path(self, file_path : str):
        folder = os.path.dirname(os.path.abspath(file_path))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        try:
            hf = h5py.File(file_path, 'w')
            # Filter out "private" attributes
            names = [key for key in dir(self) if not key.startswith('__')]
            for name in names:
                try:
                    value = self.__getattribute__(name)
                    # numpy strings don't serialize properly
                    # frankly I don't even know what a np string is but they crop up sometimes
                    if isinstance(value, np.str_):
                        value = str(value)
                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        hf.create_dataset(name, data = value)
                    elif not inspect.ismethod(value):
                        hf.attrs[name] = value
                except Exception as e:
                    print(f"Couldn't save {name}")
                    raise
        except Exception as e:
            print(f"Failed to save h5 data: {e}")
            raise
        finally:
            hf.close()