from src.utility.h5_serializable_file import H5Serializable
from src.individual_fit import IndividualFit

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
            self.results = fit.results
    
    @staticmethod
    def load(path : str):
        return IndividualFitResults().load_from_path(path)