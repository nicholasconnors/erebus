import hashlib
import os

import numpy as np

from erebus.utility.h5_serializable_file import H5Serializable
from glob import glob
import os
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from erebus.utility.planet import Planet
import erebus.utility.fits_file_utils as f_util
from jwst.datamodels import dqflags
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel, interpolate_replace_nans
from astropy.stats import sigma_clip

EREBUS_CACHE_DIR = "erebus_cache"

class SpectroscopyWrappedFits(H5Serializable):
    '''
    Holds data from a spectroscopy fits file
    '''

    def _exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['planet']
    

    def __fix_outliers(arr, sigma=3):
        arr_copy = arr.copy()
        median = np.median(arr_copy)
        std = np.std(arr_copy)
        outlier_mask = np.abs(arr_copy - median) > sigma * std
        good_data = np.where(~outlier_mask)[0]
        bad_data = np.where(outlier_mask)[0]
        arr_copy[bad_data] = np.interp(bad_data, good_data, arr[good_data])
        print(f"Removed {len(bad_data)} outliers")
        return arr_copy

    def __load_files(self, fits_files):
        if len(fits_files) == 0:
            print("No files were given to load?")
            return
        
        raw_frames = []
        frames = []
        times = []
        errors = []
        wavelengths = None

        do_not_use = dqflags.interpret_bit_flags('DO_NOT_USE', mnemonic_map=dqflags.pixel)
        kernel = Gaussian2DKernel(x_stddev=1.0)
        kernel1d = Gaussian1DKernel(1.0)
        background_x = np.concatenate((np.arange(10, 25), np.arange(72-25, 72-10)))

        for i in tqdm(range(len(fits_files))):
            with fits.open(fits_files[i]) as hdul:
                if wavelengths is None:
                    wl = hdul['WAVELENGTH'].data
                    wl = np.nan_to_num(wl, nan=0)
            
                    wl_cols = np.mean(wl, axis=0)
                    col_mask = np.where(wl_cols == 0)[0]
        
                    wl_rows = np.mean(wl, axis=1)
                    row_mask = np.where(wl_rows == 0)[0]
            
                    trimmed_wl = np.delete(wl, col_mask, axis=1)
                    trimmed_wl = np.delete(trimmed_wl, row_mask, axis=0)
            
                    wavelengths = trimmed_wl.mean(axis=1)
        
                data = hdul['SCI'].data
        
                dq_array = hdul['DQ'].data
                do_not_use_index = np.where((dq_array & do_not_use) != 0)
                data[do_not_use_index] = np.nan
                
                err = hdul['ERR'].data
        
                # Index 5 is BJD midpoint
                time = np.array([row[5] for row in hdul['INT_TIMES'].data])
        
                for j in tqdm(range(len(data))):
                    frame = data[j]
                    raw_frames.append(frame)
        
                    while np.isnan(frame).any():
                        frame = interpolate_replace_nans(frame, kernel)
                    
                    # Mask out edges
                    frame[row_mask, :] = 0
                    frame[:, col_mask] = 0
        
                    for k in range(len(frame)):
                        background_left = frame[k, 10:25]
                        background_right = frame[k, -25:-10]
                        background_y = np.concatenate((background_left, background_right))

                        # outliers
                        clipped = sigma_clip(background_y, sigma = 3, maxiters = 5, masked = True)
                        good_indices = ~clipped.mask

                        # fallback to not clipping if we dont have two points but this shouldnt happen
                        if np.sum(good_indices) < 2:
                            good_indices = np.ones_like(background_y, dtype=bool)
                        
                        a, b = np.polyfit(background_x[good_indices], background_y[good_indices], 1)
                        
                        background = a * np.arange(len(frame[k])) + b
            
                        frame[k] -= background
        
                    trimmed_frame = np.delete(frame, col_mask, axis=1)
                    trimmed_frame = np.delete(trimmed_frame, row_mask, axis=0)
                    
                    trimmed_err = np.delete(err[j], col_mask, axis=1)
                    trimmed_err = np.delete(trimmed_err, row_mask, axis=0)
                    
                    errors.append(trimmed_err)
                    frames.append(trimmed_frame)
                    times.append(time[j])
        
        raw_frames = np.array(raw_frames)
        frames = np.array(frames)
        errors = np.array(errors)
        times = np.array(times)

        # Sort BEFORE anything else
        s = np.argsort(times)
        raw_frames = raw_frames[s]
        frames = frames[s]
        times = times[s]
        errors = errors[s]
        
        # For each pixel light curve remove 5 sigma outliers
        print("Interpolating outliers in per-pixel light curves")
        sigma_clipped = sigma_clip(frames, sigma=5, axis=0, masked=True)
        frames = sigma_clipped.filled(np.nan)
        kernel = Gaussian1DKernel(stddev=1)
        for x in range(frames.shape[1]):
            for y in range(frames.shape[2]):
                while np.isnan(frames[:, x, y]).any():
                    frames[:, x, y] = interpolate_replace_nans(frames[:, x, y], kernel)
        print("Done")
        
        self.raw_frames = raw_frames
        self.frames = frames
        self.time = times
        self.errors = errors
        self.wavelengths = wavelengths
        
    @staticmethod
    def get_files_for_eclipse_index(source_folder : str, planet : Planet, eclipse_index : int):
        print("FOLDER", source_folder)
        fits_files = glob(f"{source_folder}/*/*/*/*.fits")
        if len(fits_files) == 0:
            fits_files = glob(f"{source_folder}/*/*/*.fits")
        if len(fits_files) == 0:
            fits_files = glob(f"{source_folder}/*/*.fits")
        
        print("FITS FILES:", fits_files)
        results = []
        for f in fits_files:
            with fits.open(f) as hdul:
                start = np.min(np.array([row[5] for row in hdul['INT_TIMES'].data]))
                results.append((f, start))
        
        sorted_results = sorted(results, key=lambda x: x[1])
        starts = [r[1] for r in sorted_results]
        
        t0 = planet.t0.nominal_value - 2400000.5
        p = planet.p.nominal_value

        tecl = t0 + p/2.0
        while tecl < np.min(starts):
            tecl += p
        
        start = tecl + (p * eclipse_index) - p/4.0
        end = tecl + (p * eclipse_index) + p/4.0
        print(f"Picking out eclipse files for eclipse index {eclipse_index}")
        relevant_eclipse_files = [f for f, t in results if start <= t <= end]
        return relevant_eclipse_files

    def __load_folder(self):
        visit_names = f_util.get_fits_files_visits_in_folder(f"{self.source_folder}/")
        print("Visits: ", visit_names)
        print("Is phase curve? ", self.is_phase_curve)

        if self.is_phase_curve:
            relevant_eclipse_files = SpectroscopyWrappedFits.get_files_for_eclipse_index(self.source_folder, self.planet, self.eclipse_index)
            print("Loading files: ", relevant_eclipse_files)
            self.visit_name = f"{visit_names[0]}_eclipse_{self.eclipse_index}"
            self.__load_files(relevant_eclipse_files)
        else:
            fits_files = glob(f"{self.source_folder}/*/*{visit_names[self.eclipse_index]}*.fits")
            if len(fits_files) == 0:
                fits_files = glob(f"{self.source_folder}/*/*/*{visit_names[self.eclipse_index]}*.fits")
            self.visit_name = visit_names[self.eclipse_index]
            print("Loading files: ", fits_files)
            self.__load_files(fits_files)
    
    @staticmethod
    def load_from_file(planet : Planet, cache_file : str):
        '''
        Loads in spectroscopy data from an Erebus cache file
        '''
        return SpectroscopyWrappedFits(planet, None, None, None, override_cache_path=cache_file)
    
    def __init__(self, planet : Planet, source_folder: str, eclipse_index : int, is_phase_curve : bool, force_clear_cache : bool = False, override_cache_path : str = None):
        print(f"Initializing spectroscopy wrapped fits from folder [{source_folder}]")
        
        self.planet = planet
        '''The planet being observed'''
        
        self.raw_frames = None
        
        self.frames = None
        
        self.time = None
        
        self.errors = None
        
        self.wavelengths = None
        
        if override_cache_path is not None:
            self._cache_file = override_cache_path 
        elif source_folder is not None:
            # Since extracting data takes a long time, we cache it
            # The cache folder name is based on a hash of the source folder
            source_folder_hash = hashlib.md5(source_folder.encode()).hexdigest()
            
            self._cache_file = f"{EREBUS_CACHE_DIR}/{planet.name}_{eclipse_index}_{source_folder_hash}_spectroscopy_fits.h5"
        else:
            self._cache_file = None
            
        if not force_clear_cache and self._cache_file is not None and os.path.isfile(self._cache_file):
            self.load_from_path(self._cache_file)
        else:
            self.eclipse_index : str = eclipse_index
            '''The index of the eclipse being observed. Either alphanumerical from visit name, or sequential for phase curves.'''
            self.source_folder : str = source_folder
            '''The directory containing the files this object is based on'''
            self.is_phase_curve = is_phase_curve
            '''If the data is from a phase curve (eclipses are extracted individually)'''
            
            self.__load_folder()
            self.save_to_path(self._cache_file)

            