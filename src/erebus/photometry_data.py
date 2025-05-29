import os
import hashlib
import numpy as np
import erebus.utility.aperture_photometry_utils as ap_utils
from erebus.wrapped_fits import WrappedFits
from erebus.utility.h5_serializable_file import H5Serializable
from sklearn.decomposition import PCA as NormalPCA

EREBUS_CACHE_DIR = "erebus_cache"

class PhotometryData(H5Serializable):
    '''
    A class representing the photometric data from a single visit loaded from a calints fits file
    with a specific aperture and annulus
    
    Also prepares frames for FN PCA
    
    Acts as Stage 3 of the Erebus pipeline
    '''
    @staticmethod
    def load(path : str):
        return PhotometryData(None, None, None, override_cache_path=path)
    
    def __init__(self, fits_file : WrappedFits, radius : int, annulus : tuple[int, int],
                 force_clear_cache : bool = False, override_cache_path : str = None):  
        
        if override_cache_path is not None:
            self.cache_file = override_cache_path
        else:
            # The cache folder name is based on a hash of the source folder
            self.source_folder = fits_file.source_folder
            self.visit_name = fits_file.visit_name
            
            source_folder_hash = hashlib.md5(self.source_folder.encode()).hexdigest()
            file_prefix = f"{self.visit_name}_{radius}_{annulus[0]}_{annulus[1]}_{source_folder_hash}"
            self.cache_file = f"{EREBUS_CACHE_DIR}/{file_prefix}_photometry_data.h5"
        
        if not force_clear_cache and os.path.isfile(self.cache_file):
            self.load_from_path(self.cache_file)
        else:
            # Defining all attributes
            self.raw_flux = []
            self.time = fits_file.t
            
            self.normalized_frames = []
            
            self.radius = radius
            self.annulus_start = annulus[0]
            self.annulus_end = annulus[1]
            
            self.fits_file_location = os.path.abspath(fits_file.cache_file)
            
            self.__do_aperture_photometry(fits_file)
            self.__get_normalized_frames(fits_file)
            self.save_to_path(self.cache_file)
    
    def __do_aperture_photometry(self, fits_file : WrappedFits):
        center_x = fits_file.frames[0].shape[0]//2
        center_y = fits_file.frames[0].shape[1]//2
        average_in_aperture = ap_utils.average_values_over_disk(center_x, center_y, 0, self.radius, fits_file.frames)
        average_in_annulus = ap_utils.average_values_over_disk(center_x, center_y, self.annulus_start, self.annulus_end, fits_file.frames)
        flux = average_in_aperture - average_in_annulus
        flux = flux / np.median(flux)
        self.raw_flux = flux
     
    def __get_normalized_frames(self, fits_file : WrappedFits):
        center_x = fits_file.frames[0].shape[0]//2
        center_y = fits_file.frames[0].shape[1]//2
        points_in_aperture = ap_utils.get_points_in_disk(center_x, center_y, 0, self.radius)
        average_in_annulus = ap_utils.average_values_over_disk(center_x, center_y, self.annulus_start, self.annulus_end, fits_file.frames)
        # For each frame, take square surrounding aperture, take points only in aperture, do bg-subtraction, normalize
        size = self.radius * 2 + 1
        self.normalized_frames = np.zeros((len(fits_file.frames), size, size))
        for i in range(0, len(fits_file.frames)):
            for point in points_in_aperture:
                x = point[0]
                y = point[1]
                j = x - center_x + self.radius
                k = y - center_y + self.radius
                # Background subtracted
                self.normalized_frames[i, j, k] = np.max([0.0, fits_file.frames[i, x, y] - average_in_annulus[i]])
            # Normalize
            self.normalized_frames[i] /= np.sum(self.normalized_frames[i])
