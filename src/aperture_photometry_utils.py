import numpy as np
from scipy.optimize import curve_fit
from src import utils
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

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
    star_x = int(round(params[1]))
    star_y = int(round(params[2]))
    print(f"Found star at: {star_x}, {star_y}")
    return star_x, star_y

def clean_frames(raw_frames : np.ndarray, outlier_threshold : float) -> np.ndarray:
    '''
    Interpolates nans and outliers in per-pixel light curves
    Interpolates bad pixels (nan for > 3/4th of the observation) based on surrounding pixels
    Does not overwrite values in the original array
    '''
    frames = np.zeros_like(raw_frames)
    
    print(f"Cleaning {len(frames)} frames")

    # Evaluating per-pixel light curves
    bad_pixel_count = 0
    interpolated_pixel_count = 0
    outlier_count = 0
    bad_pixels = []
    good_pixels = []
    for i in range(0, frames.shape[1]):
        for j in range(0, frames.shape[2]):
            pixel_light_curve = raw_frames[:,i,j]
            frames[:,i,j] = pixel_light_curve

            # First reject nan, 0
            nan = np.isnan(pixel_light_curve)
            zero = pixel_light_curve <= 0
            mask = np.logical_or(nan, zero)

            # If more than 75% of the observation was bad just ditch the pixel
            if mask.sum() > frames.shape[0] * 3 / 4:
                bad_pixel_count+=1
                bad_pixels.append((i,j))
                continue		
            else:
                good_pixels.append((i,j))

            if not (~mask).all():
                frames[:,i,j][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pixel_light_curve[~mask])
                interpolated_pixel_count+=1

            pixel_light_curve = frames[:,i,j]
            # Now that nans are removed, do outlier detection
            outlier = np.abs(pixel_light_curve - np.median(pixel_light_curve)) > outlier_threshold * np.std(pixel_light_curve)
            if not (~outlier).all():
                outlier_count+=1
                frames[:,i,j][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pixel_light_curve[~mask])

    bad_pixel_mask = np.zeros_like(frames[0], dtype=bool)
    for (x, y) in bad_pixels:
        bad_pixel_mask[x, y] = True

    for ind, frame in enumerate(frames):
        # Now interpolate bad pixels based on the surrounding pixels
        # LinearNDInterpolator will not be able to fill in pixels outside of its convex hull
        # For those we use the neartest value in the grid: should only be happening in the background anyway where its very uniform
        linear_interp = LinearNDInterpolator(good_pixels, frame[~bad_pixel_mask])
        nearest_interp = NearestNDInterpolator(good_pixels, frame[~bad_pixel_mask])
        
        for i, j in bad_pixels:
            linear_interp_value = linear_interp(i, j)
            frames[ind, i, j] = linear_interp_value if not np.isnan(linear_interp_value) else nearest_interp(i, j)

    pixels_per_frame = frames.shape[1] * frames.shape[2]
    total_pixels = frames.shape[0] * frames.shape[1] * frames.shape[2]
    print(f"{bad_pixel_count} pixels were bad out of {pixels_per_frame}")
    print(f"{interpolated_pixel_count} values were interpolated out of {total_pixels}")
    print(f"{outlier_count} values were outliers out of {total_pixels}")
    
    return frames