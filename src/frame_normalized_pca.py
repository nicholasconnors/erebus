from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA as NormalPCA
import src.utility.aperture_photometry_utils as ap_utils
import numpy as np

from src.photometry_data import PhotometryData
from src.wrapped_fits import WrappedFits

def perform_fnpca(data : PhotometryData, fits : WrappedFits):
    '''
    Performs Frame-Normalized PCA on a photometric time series data set
    
    Returns the eigenvalue eigenimage pairs
    '''
    center_x = fits.frames[0].shape[0]//2+1
    center_y = fits.frames[0].shape[1]//2+1
    points_in_aperture = ap_utils.get_points_in_disk(center_x, center_y, 0, data.radius)
    average_in_annulus = ap_utils.average_values_over_disk(center_x, center_y, data.annulus_start, data.annulus_end, fits.frames)
    # For each frame, take square surrounding aperture, take points only in aperture, do bg-subtraction, normalize
    size = data.radius * 2 + 1
    frames = np.zeros((len(fits.frames), size, size))
    for i in range(0, len(fits.frames)):
        for point in points_in_aperture:
            x = point[0]
            y = point[1]
            j = x - center_x + data.radius
            k = y - center_y + data.radius
            frames[i, j, k] = fits.frames[i, x, y]
        frames[i] -= average_in_annulus[i]
        frames[i] /= np.sum(frames[i])
    
    # Perform PCA on normalized frames
    nint, nrow, ncol = frames.shape
    flatcube = frames.reshape(frames.shape[0],np.product(frames.shape[1:]))
    ipca = NormalPCA()
    ipca.fit(flatcube)
    eigenvalues = ipca.fit_transform(flatcube).T
    eigenimages = np.array([image.reshape((size, size)) for image in ipca.components_])
    
    return eigenvalues, eigenimages

    
    