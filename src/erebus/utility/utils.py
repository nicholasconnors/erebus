from typing import Callable, List
import numpy as np
import json as json
from uncertainties.core import Variable as UFloat
from uncertainties import ufloat

def gaussian_2D(xy, a : float, mu_x : float, mu_y : float, sigma : float, offset : float) -> list[float]:
    '''
    Gaussian with a background level "offset", assuming same sigma in x and y
    Returns the values flattened into a 1d array
    '''
    x, y = xy
    z = a * np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)) + offset
    return z.ravel()

def subarray_2D(array : np.ndarray, x : int, y : int, width : int) -> np.ndarray:
    '''
    Takes a subarray from the given 2d array centered on x and y
    '''
    # Width must be odd else the center won't be at the center
    if width % 2 == 0:
        print("subarray_2D width must be an odd number")
        width = width - 1

    # For the SUB64 MIRI subarray we might need to pad it
    # or if the star is near the edge of a frame (not sure why that would happen but better safe than sorry?)
    padding_up = np.max([width//2 - y, 0])
    padding_down = np.max([y + width//2 + 1 - array.shape[0], 0])
    padding_left = np.max([width//2 - x, 0])
    padding_right = np.max([x + width//2 + 1 - array.shape[1], 0])
    padded_array = np.pad(array, [(padding_up, padding_down), (padding_left, padding_right)])
    padded_y = y + padding_up
    padded_x = x + padding_left

    # slice only the desired subarray
    return padded_array[padded_y-width//2:padded_y+1+width//2, padded_x-width//2:padded_x+1+width//2]

def bin_data(array: np.ndarray, bin_size : int):
    '''
    Returns the means and standard error of each bin
    '''
    if len(array) < bin_size:
        return array, np.zeros_like(array)
    
    # Get length which is divisible by bin_size
    length = (len(array) // bin_size) * bin_size
    means = []
    errs = []
    for i in np.arange(0, length, bin_size):
        array_slice = array[i:i+bin_size]
        means.append(np.mean(array_slice))
        errs.append(np.std(array_slice) / np.sqrt(bin_size))
    return np.array(means), np.array(errs)

def create_method_signature(method : Callable, args : List[str]) -> Callable:
    '''
    Takes a method and redefines it to use a list of arguments
    '''
    args_str = ", ".join(args)
    function_def = f"def func({args_str}):\n\treturn original_function({args_str})\n"
    function_code = compile(function_def, "", "exec")
    function_globals = {}
    eval(function_code, {"original_function": method}, function_globals)
    method_with_signature = function_globals["func"]
    return method_with_signature

def get_eclipse_duration(inc : float, a_rstar : float, rp_rstar : float, per : float) -> float:
    '''
    Length of the eclipse in the same units as the period
    Requires inclination in degrees
    '''
    b = a_rstar * np.cos(inc * np.pi / 180)
    l = np.sqrt((1 + rp_rstar) ** 2 - b**2)
    eclipse_phase_length = np.arcsin(l / a_rstar) / np.pi
    l = eclipse_phase_length * per
    return l

def get_predicted_t_sec(planet, photometry_data) -> float:
    '''
    Predicted t_sec given a perfectly circular orbit, given a planet and photometry data
    '''
    start_time = np.min(photometry_data.time)
    t0 = planet.get_closest_t0(start_time)
    nominal_period = planet.p if isinstance(planet.p, float) else planet.p.nominal_value
    predicted_t_sec = (t0 - start_time + planet.p / 2.0) % nominal_period
    return predicted_t_sec 

def save_dict_to_json(dict, path):
    with open(path, "w") as file:
        json.dump(dict, file, indent=4, cls=_JSONEncoder)
        
class _JSONEncoder(json.JSONEncoder):
    '''
    JSON encoder that supports ufloats
    '''
    def default(self, obj):
        if isinstance(obj, UFloat):
            return {'__ufloat__': True, 'nominal_value': obj.nominal_value, 'std_dev': obj.std_dev}
        return super().default(obj)

class _JSONDecoder(json.JSONDecoder):
    '''
    JSON decoder that supports ufloats
    '''
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, d):
        if "__ufloat__" in d:
            return ufloat(float(d['nominal_value']), float(d['std_dev']))
        return d