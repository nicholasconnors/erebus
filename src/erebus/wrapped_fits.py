import numpy as np
import os
import hashlib
import erebus.utility.fits_file_utils as f_utils
import erebus.utility.aperture_photometry_utils as ap_utils
import erebus.utility.utils as utils
from erebus.utility.h5_serializable_file import H5Serializable

EREBUS_CACHE_DIR = "erebus_cache"

class WrappedFits(H5Serializable):
    '''
    A class wrapping the flux and time data from the calints fits files of a single visit
    If the star pixel position is not provided we will fit for it
    Contains the flux in a 127x127 pixel region centered around the star
    Performs outerlier rejection and interpolation of bad pixels
    
    Acts as the Stage 3 input to the Erebus pipeline
    '''
    @staticmethod
    def load(path : str):
        return WrappedFits(None, None, override_cache_path=path)
    
    def __init__(self, source_folder : str, visit_name : str, force_clear_cache : bool = False,
                 override_cache_path : str = None, star_pixel_position : tuple[int, int] = None):      
        if override_cache_path is not None:
            self.cache_file = override_cache_path 
        else:
            # Since extracting photometric data takes a long time, we cache it
            # The cache folder name is based on a hash of the source folder
            source_folder_hash = hashlib.md5(source_folder.encode()).hexdigest()
            
            self.cache_file = f"{EREBUS_CACHE_DIR}/{visit_name}_{source_folder_hash}_wrapped_fits.h5"
        
        if not force_clear_cache and os.path.isfile(self.cache_file):
            self.load_from_path(self.cache_file)
        else:
            self.visit_name = visit_name
            self.source_folder = source_folder
            
            # Defining all attributes
            self.t = []
            self.star_x = None if star_pixel_position is None else star_pixel_position[0]
            self.star_y = None if star_pixel_position is None else star_pixel_position[1]
            self.frames = []
            self.raw_frames = []
            self.first_frame = []
            
            self.__load_from_calints_file()
            self.save_to_path(self.cache_file)
    
    def __load_from_calints_file(self):
        '''
        Loads a calints file and wraps it taking only 127x127 pixels around the star and the time series
        '''
        frames, time = f_utils.load_all_calints_for_visit(self.source_folder, self.visit_name)
        
        self.t = time

        if self.star_x is None or self.star_y is None:
            x = range(0, frames[0].shape[0])
            y = range(0, frames[0].shape[1])
            x, y = np.meshgrid(x, y)
            self.star_x, self.star_y = ap_utils.fit_star_position(frames[0], (x, y))
        
        self.first_frame = frames[0]
        
        # Get 127x127 frame
        self.raw_frames = np.array([utils.subarray_2D(frame, self.star_x, self.star_y, 127) for frame in frames])
        
        # Remove nan and 5 sigma outliers
        self.frames = ap_utils.clean_frames(self.raw_frames, 5)