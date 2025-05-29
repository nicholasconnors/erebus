from src.erebus.utility.h5_serializable_file import H5Serializable
from src.erebus.joint_fit import JointFit
import inspect
import numpy as np

class JointFitResults(H5Serializable):
    '''
    Class containing the results of an individual fit run
    '''
    
    def __init__(self, fit : JointFit):
        if fit is not None:
            self.time = fit.time
            self.raw_flux = fit.raw_flux
            self.joint_eigenvalues = fit.joint_eigenvalues
            self.joint_eigenvectors = fit.joint_eigenvectors
            self.pca_variance_ratios = fit.pca_variance_ratios
            self.results = fit.results
            self.planet_name = fit.planet_name
            self.config = fit.config
            self.config_hash = fit.config_hash
            
            # Time given relative to the predicted t_sec for that visit
            self.detrended_flux_per_visit = []
            self.relative_time_per_visit = []
            
            # Time relative to predicted t_sec and used to run the physical model
            self.model_time_per_visit = []
            self.model_flux_per_visit = []
            
            args = [x.nominal_value for x in list(fit.results.values())]
            number_of_physical_args = len(inspect.getfullargspec(fit.physical_model).args) - 2
            physical_args = args[0:number_of_physical_args]
            number_of_systematic_args = len(inspect.getfullargspec(fit.systematic_model).args) - 2
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
        return JointFitResults(None).load_from_path(path)