import hashlib
import os

import numpy as np

import erebus.utility.aperture_photometry_utils as ap_utils
from erebus.utility.h5_serializable_file import H5Serializable
from erebus.spectroscopy.spectroscopy_wrapped_fits import SpectroscopyWrappedFits

EREBUS_CACHE_DIR = "erebus_cache"

class SpectroscopyData(H5Serializable):
    '''
    A class representing the spectroscopic light curve data of a single visit within a range of wavelengths.
    
    Acts as Stage 3 of the Erebus pipeline
    '''
    @staticmethod
    def load(path : str):
        '''Helper method to directly load a SpectroscopyData instance cache file.'''
        return SpectroscopyData(None, None, None, None, override_cache_path=path)
    
    @staticmethod
    def load_eureka(path : str):
        '''Directly loads a Eureka light curve'''
        raise Exception("Unimplemented")
    
    @staticmethod
    def load_exotedrf(path : str):
        '''Directly loads an Exotedrf light curve'''
        raise Exception("Unimplemented")
    
    def __init__(self, fits_file : SpectroscopyWrappedFits, do_optimal_extraction : bool, wl_start : float, wl_end : float,
                 force_clear_cache : bool = False, override_cache_path : str = None):
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path
        else:
            # The cache folder name is based on a hash of the source folder
            self.visit_name : str = fits_file.visit_name
            '''The unique name of the visit being observed.'''
            self.source_folder : str = fits_file.source_folder
            '''The directory containing the files this WrappedFits is based on'''
            
            source_folder_hash = hashlib.md5(self.source_folder.encode()).hexdigest()
            wl_start_str = int(wl_start * 100)
            wl_end_str = int(wl_end * 100)
            file_prefix = f"{self.visit_name}_{('opt' if do_optimal_extraction else 'apt')}_{wl_start_str}_{wl_end_str}_{source_folder_hash}"
            self._cache_file = f"{EREBUS_CACHE_DIR}/{file_prefix}_spectroscopy_data.h5"
        
        if not force_clear_cache and os.path.isfile(self._cache_file):
            self.load_from_path(self._cache_file)
            # back compat check
            self.__extract_spatial_profiles(fits_file)
        else:
            # Defining all attributes
            self.raw_flux = []
            '''Raw flux from the star after performing background subtraction.'''
            self.time = fits_file.time
            '''The time values loaded from the corresponding fits files.'''
            self.fits_file_location = os.path.abspath(fits_file._cache_file)
            '''Absolute path of the cache file for the fits file that aperture photometry was performed on.'''
            self.spatial_profiles = []
            '''Spatial profile of each frame (collapsed along wavelength axis)'''
            
            self.wl_start = wl_start
            self.wl_end = wl_end
            
            if do_optimal_extraction:
                self.__do_optimal_extraction(fits_file)
            else:
                self.__do_aperture_photometry(fits_file)
            
            # Get the spatial profiles for PCA
            self.__extract_spatial_profiles(fits_file)
            
            # Outlier removal after photometry
            valid_inds = np.array([np.abs(f - np.median(self.raw_flux)) < 5 * np.std(self.raw_flux) for f in self.raw_flux])
            self.raw_flux = self.raw_flux[valid_inds]
            self.time = self.time[valid_inds]
            self.spatial_profiles = self.spatial_profiles[valid_inds]
            
            self.save_to_path(self._cache_file)
    
    def __extract_spatial_profiles(self, fits_file : SpectroscopyWrappedFits):
        mask_wl = (fits_file.wavelengths >= self.wl_start) & (fits_file.wavelengths <= self.wl_end)
        profile = np.sum(fits_file.frames[:, mask_wl, :], axis=1)
        print("Profile shape:", profile.shape)
        self.spatial_profiles = profile
    
    def __do_aperture_photometry(self, fits_file : SpectroscopyWrappedFits):
        '''
        Performs aperture photometry with a boxcar filter
        '''
        mask_wl = (fits_file.wavelengths >= self.wl_start) & (fits_file.wavelengths <= self.wl_end)
        boxcar = np.mean(fits_file.frames[:, mask_wl, slice(32-4, 32+4)], axis=(1,2))
        self.raw_flux = boxcar / np.median(boxcar)

    
    def __do_optimal_extraction(self, fits_file : SpectroscopyWrappedFits):
        '''
        Performs optimal extraction following Horne et al
        '''
        mask_wl = (fits_file.wavelengths >= self.wl_start) & (fits_file.wavelengths <= self.wl_end)
        if not np.any(mask_wl):
            raise ValueError(f"Wavelength range {self.wl_start} to {self.wl_end} has no values")

        frames = fits_file.frames[:, mask_wl, :]
        errors = fits_file.errors[:, mask_wl, :]
        
        # Optimal extraction (Horne et al, https://iopscience.iop.org/article/10.1086/131801/pdf)
        # Normalized spatial profile from median frame
        median_frame = np.nanmedian(frames, axis = 0)
        median_frame = np.clip(median_frame, 0, None)
        column_sum = np.sum(median_frame, axis=1, keepdims=True)
        column_sum[column_sum <= 0] = np.nan # avoid div by zero
        P = median_frame / column_sum # normalizalized spatial profile
        P = np.nanmedian(P, axis = 0)
        P = np.nan_to_num(P) # Replaces nans with 0 and infty with large finite numbers
        
        # We perform optimal extraction with a 10 pixel radius aperture
        peak = int(np.argmax(P))
        half_width = 10
        spatial_window = slice(max(peak - half_width, 0), min(peak + half_width, len(P)))
        P = P[spatial_window]
        # Renormalize
        P = np.clip(P, 0, None)
        if np.sum(P) <= 0:
            raise ValueError("Spatial profile is empty")
        P = P / np.sum(P)
        
        # D is the data of the frames within our window (already background subtracted)
        D = frames[:, :, spatial_window] # fit optimal extraction on a 20 pixel width window
        # V is the variance of the data within our window
        V = errors[:, :, spatial_window] ** 2 # error squared is variance

        # Sometimes the errors are nans or 0 so remove those
        good_pixels = np.isfinite(D) & np.isfinite(V) & (V > 0)
        print(f"Invalid errors count: {np.sum(~good_pixels)} / {len(good_pixels)}")
        D = np.where(good_pixels, D, 0.0) # bad pixels set to 0
        V = np.where(good_pixels, V, 1.0) # bad pixels have variance 1
        M = good_pixels.astype(float) # 1 for good 0 for ba
        
        # For debugging
        self.spatial_profile = P
        self.variance = V
        self.data = D
        self.mask = M
        
        # Gives very weird results otherwise, revisit later
        V = np.ones_like(D) 
        
        # According to Table I, Equation 8
        optimal_flux = np.sum(M * P * D / V, axis=(1, 2)) / np.sum(M * P**2 / V, axis=(1, 2))
        
        # Report as normalized flux
        self.raw_flux = optimal_flux / np.nanmedian(optimal_flux)
