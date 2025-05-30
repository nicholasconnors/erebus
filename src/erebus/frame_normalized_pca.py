from typing import Tuple
from sklearn.decomposition import PCA as NormalPCA
import erebus.utility.aperture_photometry_utils as ap_utils
import numpy as np

def perform_fnpca_on_full_frame(frames : np.ndarray, radius : int,
                                annulus_start : int, annulus_end : int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs Frame-Normalized PCA on a photometric time series data set. Expects the star to be centered
    on each frame.
    
    Returns the eigenvalue-eigenimage pairs and ratios of explained variance.
    '''
    center_x = frames[0].shape[0]//2+1
    center_y = frames[0].shape[1]//2+1
    points_in_aperture = ap_utils.get_points_in_disk(center_x, center_y, 0, radius)
    average_in_annulus = ap_utils.average_values_over_disk(center_x, center_y, annulus_start, annulus_end, frames)
    # For each frame, take square surrounding aperture, take points only in aperture, do bg-subtraction, normalize
    size = radius * 2 + 1
    normalized_frames = np.zeros((len(frames), size, size))
    for i in range(0, len(frames)):
        for point in points_in_aperture:
            x = point[0]
            y = point[1]
            j = x - center_x + radius
            k = y - center_y + radius
            normalized_frames[i, j, k] = frames[i, x, y]
        normalized_frames[i] -= average_in_annulus[i]
        normalized_frames[i] /= np.sum(frames[i])
    
    return perform_fn_pca_on_aperture(normalized_frames)

def perform_fn_pca_on_aperture(aperture_frames : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs Frame-Normalized PCA on a photometric time series data set. Expects the star to be centered
    on each frame. Expects each frame to already be normalized and background subtracted, with pixels outside
    the aperture set to 0.
    
    Returns the eigenvalue-eigenimage pairs and ratios of explained variance.
    '''
    length, width, height = aperture_frames.shape
    flat_frames = aperture_frames.reshape(length, width * height)
    pca = NormalPCA()
    pca.fit(flat_frames)
    eigenvalues = pca.fit_transform(flat_frames).T
    eigenvectors = np.array([image.reshape((width, height)) for image in pca.components_])
    
    return eigenvalues, eigenvectors, pca.explained_variance_ratio_