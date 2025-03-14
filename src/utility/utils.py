import numpy as np

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
