from src.erebus.utility.h5_serializable_file import H5Serializable
from src.erebus.individual_fit import IndividualFit

class IndividualFitResults(H5Serializable):
    '''
    Class containing the results of an individual fit run
    '''
    
    def __init__(self, fit : IndividualFit):
        if fit is not None:
            self.time = fit.time
            self.start_time = fit.start_time
            self.raw_flux = fit.raw_flux
            self.eigenvalues = fit.eigenvalues
            self.eigenvectors = fit.eigenvectors
            self.pca_variance_ratios = fit.pca_variance_ratios
            self.order = fit.order
            self.results = fit.results
            self.planet_name = fit.planet_name
            self.visit_name = fit.visit_name
            self.config = fit.config
            self.config_hash = fit.config_hash
            
            self.first_frame = fit.first_frame
            self.frames = fit.photometry_data.normalized_frames
            
            res_nominal_values = [fit.results[k].nominal_value for k in fit.results][:-1]
            systematic_params = res_nominal_values[9:]
            self.flux_model = fit.fit_method(fit.time, *res_nominal_values)
            self.systematic_factor = fit.systematic_model(fit.time, *systematic_params)
    
    @staticmethod
    def load(path : str):
        return IndividualFitResults(None).load_from_path(path)