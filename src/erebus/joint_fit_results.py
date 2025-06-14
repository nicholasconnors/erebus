from erebus.utility.h5_serializable_file import H5Serializable
from erebus.joint_fit import JointFit
import inspect
import numpy as np

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

            # Time given relative to the predicted t_sec for that visit
            self.detrended_flux_per_visit = []
            '''A list containing the detrended lightcurves of each visit.'''
            self.relative_time_per_visit = []
            '''A list containing the time values corresponding to detrended_flux_per_visit'''
            
            # Time relative to predicted t_sec and used to run the physical model
            self.model_flux_per_visit = []
            '''The best fit detrended lightcurves per visit.'''
            self.model_time_per_visit = []
            '''A list containing the time values corresponding to model_flux_per_visit'''

            args = [x.nominal_value for x in list(fit.results.values())]

            number_of_physical_args = fit.get_number_of_physical_args()
            number_of_systematic_args = fit.get_number_of_systematic_args()

            physical_args = args[0:number_of_physical_args]
            visit_indices = np.array([fit.get_visit_index_from_time(xi) for xi in fit.time])
            for visit_index in range(0, len(fit.photometry_data_list)):
                filt = visit_indices == visit_index
                time = fit.time[filt]
                flux = fit.raw_flux[filt]
                            
                systematic_index_start = (number_of_physical_args) + (visit_index * number_of_systematic_args)
                systematic_args = args[systematic_index_start:systematic_index_start + number_of_systematic_args]
            
                systematic = fit.systematic_model(time, *systematic_args)
                physical_time = np.linspace(np.min(time), np.max(time), 1000)
                physical = fit.physical_model(physical_time, *physical_args)
                
                self.detrended_flux_per_visit.append(flux / systematic)
                time_offset = fit.get_predicted_t_sec_of_visit(visit_index).nominal_value + fit.starting_times[visit_index]
                self.relative_time_per_visit.append((time - time_offset) * 24)
                self.model_time_per_visit.append((physical_time - time_offset) * 24)
                self.model_flux_per_visit.append(physical)
    
    @staticmethod
    def load(path : str):
        '''After running an Erebus instance, the results file can be loaded later using this method.'''
        return JointFitResults(None).load_from_path(path)