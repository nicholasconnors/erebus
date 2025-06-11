from typing import Callable, List
import numpy as np
from scipy.stats import binned_statistic
import json as json

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

def merge_functions(func1, func2):
    sig1 = inspect.signature(func1)
    sig2 = inspect.signature(func2)

    merged_params = OrderedDict()

    # Add parameters from func1
    for name, param in sig1.parameters.items():
        merged_params[name] = param

    # Add parameters from func2 (if not already in)
    for name, param in sig2.parameters.items():
        if name not in merged_params:
            merged_params[name] = param
        else:
            if param != merged_params[name]:
                raise ValueError(f"Conflict in parameter '{name}'")

    # Create the merged signature
    merged_signature = inspect.Signature(parameters=merged_params.values())

    # Define the actual callable function
    def merged_func_template(*args, **kwargs):
        bound = merged_signature.bind(*args, **kwargs)
        bound.apply_defaults()

        # Call original functions with their specific arguments
        func1_args = {
            name: bound.arguments[name]
            for name in sig1.parameters if name in bound.arguments
        }
        func2_args = {
            name: bound.arguments[name]
            for name in sig2.parameters if name in bound.arguments
        }

        print("Calling func1...")
        result1 = func1(**func1_args)
        print("Calling func2...")
        result2 = func2(**func2_args)

        return result1, result2  # Or do something custom here

    # Assign the signature to the new function
    merged_func = FunctionType(
        merged_func_template.__code__,
        globals(),
        name='merged_func',
        argdefs=merged_func_template.__defaults__,
        closure=merged_func_template.__closure__
    )
    merged_func.__signature__ = merged_signature
    merged_func.__name__ = f"merged_{func1.__name__}_{func2.__name__}"
    merged_func.__doc__ = f"Merged function of `{func1.__name__}` and `{func2.__name__}`."

    return merged_func

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
    nominal_period = planet.p if isinstance(planet.p, float) else planet.p.nominal_value
    predicted_t_sec = (planet.t0 - np.min(photometry_data.time) - 2400000.5 + planet.p / 2.0) % nominal_period
    return predicted_t_sec 

def save_dict_to_json(dict, path):
    with open(path, "w") as file:
        json.dump(dict, file, indent=4)