import numpy as np
from scipy.optimize import curve_fit
from src import utils

def fit_star_position(frame, xy):
    '''
    Frames is a 2d array
    '''
    
    x, y = xy
    max_y, max_x = np.unravel_index(np.nanargmax(frame), frame.shape)
    initial_guess = [np.nanmax(frame), max_x, max_y, 10, np.median(frame)]
    lower_bounds = [0, np.min(x), np.min(y), 3, 0]
    upper_bounds = [np.inf, np.max(x), np.max(y), np.inf, np.inf]
    params, _ = curve_fit(utils.gaussian_2D, xy, frame.ravel(), p0=initial_guess, bounds=(lower_bounds, upper_bounds))

    # Return just the mean x and y
    return params[1], params[2]