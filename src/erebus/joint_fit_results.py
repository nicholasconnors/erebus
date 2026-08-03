
import numpy as np

from erebus.joint_fit import JointFit
from erebus.utility.h5_serializable_file import H5Serializable


class JointFitResults(H5Serializable):
    '''
    Class containing the results of an individual fit run
    '''
    
    def __init__(self, fit : JointFit):
        if fit is not None:
            self.time = fit.time
            '''A list of time values for each visit.'''
            self.raw_flux = fit.raw_flux
            '''A list containing the raw (not yet detrended) lightcurves of each visit.'''
            self.joint_eigenvalues = fit.joint_eigenvalues
            '''A list of eigenvalue lists per visit. Index first by vist number than by principal component number.'''
            self.joint_eigenvectors = fit.joint_eigenvectors
            '''A list of eigenvector lists per visit. Index first by vist number than by principal component number.'''
            self.pca_variance_ratios = fit.pca_variance_ratios
            '''A list of PCA explained variance ratio lists per visit. Index first by vist number than by principal component number.'''
            self.results = fit.results
            '''A dictionary of results for the fit parameters (e.g., eclipse depth)'''
            self.planet_name = fit.planet_name
            '''The name of the planet observed'''
            self.config = fit.config
            '''The config file used to create this run'''
            self.planet = fit.planet
            '''The planet config file used to create this run'''
            self.config_hash = fit.config_hash
            '''The unique hash of the config file. Used for naming cache files.'''
            self.predicted_t_secs = fit.predicted_t_secs
            '''Predicted 0.5 eclipse time for each visit'''
            self.bin_size = fit.bin_size
            '''Joint fits are binned to speed up convergence'''

            # Time given relative to the predicted t_sec for that visit
            self.detrended_flux_per_visit = []
            '''A list containing the detrended lightcurves of each visit.'''
            self.time_per_visit = []
            '''A list containing the time values corresponding to detrended_flux_per_visit'''
            self.systematic_per_visit = []
            '''A list containing the systematic model corresponding to detrended_flux_per_visit'''
            self.raw_flux_per_visit = []
            '''A list containing the raw flux of each visit'''
            
            self.starting_times = fit.starting_times
            
            # Time relative to predicted t_sec and used to run the physical model
            self.model_flux_per_visit = []
            '''The best fit detrended lightcurves per visit.'''

            args = [x.nominal_value for x in list(fit.results.values())]

            number_of_physical_args = fit.get_number_of_physical_args()
            number_of_systematic_args = fit.get_number_of_systematic_args()

            physical_args = args[0:number_of_physical_args]
            for i, visit_index in enumerate(fit.visit_indices):
                filt = fit.visit_index_filter[visit_index]
                time = fit.time[filt]
                flux = fit.raw_flux[filt]
                
                # Only add visits that were included
                systematic_index_start = fit.get_systematic_index_start(i)
                systematic_args = args[systematic_index_start:systematic_index_start + number_of_systematic_args]
            
                systematic = fit.systematic_model(time, *systematic_args)
                physical = fit.physical_model(time, *physical_args)
                
                self.detrended_flux_per_visit.append(flux / systematic)
                self.systematic_per_visit.append(systematic)
                self.raw_flux_per_visit.append(flux)
                time_offset = fit.get_predicted_t_sec_of_visit(visit_index)
                self.time_per_visit.append((time - time_offset) * 24)
                self.model_flux_per_visit.append(physical)
                        
            # To be serialized 2D arrays cannot be jagged
            self.detrended_flux_per_visit = JointFitResults.__pad_2d_array(self.detrended_flux_per_visit)
            self.systematic_per_visit = JointFitResults.__pad_2d_array(self.systematic_per_visit)
            self.raw_flux_per_visit = JointFitResults.__pad_2d_array(self.systematic_per_visit)
            self.time_per_visit = JointFitResults.__pad_2d_array(self.time_per_visit)
            self.model_flux_per_visit = JointFitResults.__pad_2d_array(self.model_flux_per_visit)
            
            self.final_log_likelihood = fit.final_log_likelihood
            self.BIC = fit.BIC
    
    @staticmethod
    def __pad_2d_array(array_2d):
        max_length = max([len(array) for array in array_2d])
        n_arrays = len(array_2d)
        padded_array_2d = np.full((n_arrays, max_length), np.nan)
        for i in range(n_arrays):
            array = array_2d[i]
            padded_array_2d[i, :len(array)] = array
        return np.array(padded_array_2d)

    @staticmethod
    def load(path : str):
        '''After running an Erebus instance, the results file can be loaded later using this method.'''
        return JointFitResults(None).load_from_path(path)