import numpy as np

def gaussian_2D(xy, a : float, mu_x : float, mu_y : float, sigma : float, offset : float):
    '''
    Gaussian with a background level "offset", assuming same sigma in x and y
    Returns the values flattened into a 1d array
    '''
    x, y = xy
    z = a * np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)) + offset
    return z.ravel()