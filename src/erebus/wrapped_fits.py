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
    Performs outerlier rejection and interpolation of bad pixels.
    
    Acts as the equivalent to a JWST Stage 3 input to the Erebus pipeline.
    '''
    @staticmethod
    def load(path : str):
        '''Helper method to directly load a WrappedFits instance cache file.'''
        return WrappedFits(None, None, override_cache_path=path)
    
    def __init__(self, source_folder : str, visit_name : str, force_clear_cache : bool = False,
                 override_cache_path : str = None, star_pixel_position : tuple[int, int] = None):      
        self.frames = []
        '''3D array respresenting time series images making up this observation.'''
        self.raw_frames = []
        '''3D array respresenting time series images making up this observation, before performing outlier and nan rejection.'''
        self.time : list[float] = []
        '''The time values loaded from the corresponding fits files'''
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path 
        else:
            # Since extracting photometric data takes a long time, we cache it
            # The cache folder name is based on a hash of the source folder
            source_folder_hash = hashlib.md5(source_folder.encode()).hexdigest()
            
            self._cache_file = f"{EREBUS_CACHE_DIR}/{visit_name}_{source_folder_hash}_wrapped_fits.h5"
        
        if not force_clear_cache and os.path.isfile(self._cache_file):
            self.load_from_path(self._cache_file)
        else:
            self.visit_name : str = visit_name
            '''The unique name of the visit being observed.'''
            self.source_folder : str = source_folder
            '''The directory containing the files this WrappedFits is based on'''
            
            # Defining all attributes            
            self._star_x = None if star_pixel_position is None else star_pixel_position[0]
            self._star_y = None if star_pixel_position is None else star_pixel_position[1]
            self.frames = []
            self.raw_frames = []
            
            self.__load_from_calints_file()
            self.save_to_path(self._cache_file)
    
    def __load_from_calints_file(self):
        '''
        Loads a calints file and wraps it taking only 127x127 pixels around the star and the time series
        '''
        frames, time = f_utils.load_all_calints_for_visit(self.source_folder, self.visit_name)
        
        self.time = time

        if self._star_x is None or self._star_y is None:
            x = range(0, frames[0].shape[0])
            y = range(0, frames[0].shape[1])
            x, y = np.meshgrid(x, y)
            self._star_x, self._star_y = ap_utils.fit_star_position(frames[0], (x, y))
                
        # Get 127x127 frame
        self.raw_frames = np.array([utils.subarray_2D(frame, self._star_x, self._star_y, 127) for frame in frames])
        
        # Remove nan and 5 sigma outliers
        self.frames = ap_utils.clean_frames(self.raw_frames, 5)