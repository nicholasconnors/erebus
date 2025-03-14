import os
import hashlib
import numpy as np
import src.aperture_photometry_utils as ap_utils
from src.wrapped_fits import WrappedFits
from src.h5_serializable_file import H5Serializable

EREBUS_CACHE_DIR = "erebus_cache"

class PhotometryData(H5Serializable):
    '''
    A class representing the photometric data from a single visit loaded from a calints fits file
    with a specific aperture and annulus
    '''
    def __init__(self, fits_file : WrappedFits, radius : int, annulus : tuple[int, int],
                 force_clear_cache : bool = False):  
        # The cache folder name is based on a hash of the source folder
        self.source_folder = fits_file.source_folder
        self.visit_name = fits_file.visit_name
        
        source_folder_hash = hashlib.md5(self.source_folder.encode()).hexdigest()
        file_prefix = f"{self.visit_name}_{radius}_{annulus[0]}_{annulus[1]}_{source_folder_hash}"
        self.cache_file = f"{EREBUS_CACHE_DIR}/_{file_prefix}_photometry_data.h5"

        # Defining all attributes
        self.light_curve = []
        self.t = fits_file.t
        
        self.radius = radius
        self.annulus_start = annulus[0]
        self.annulus_end = annulus[1]
        
        if not force_clear_cache and os.path.isfile(self.cache_file):
            self.load_from_path(self.cache_file)
        else:
            self.__do_aperture_photometry(fits_file)
            self.save_to_path(self.cache_file)
    
    def __do_aperture_photometry(self, fits_file : WrappedFits):
        center_x = fits_file.frames[0].shape[0]//2+1
        center_y = fits_file.frames[0].shape[1]//2+1
        average_in_aperture = ap_utils.average_values_over_disk(center_x, center_y, 0, self.radius, fits_file.frames)
        average_in_annulus = ap_utils.average_values_over_disk(center_x, center_y, self.annulus_start, self.annulus_end, fits_file.frames)
        flux = average_in_aperture - average_in_annulus
        flux = flux / np.median(flux)
        self.light_curve = flux
     
