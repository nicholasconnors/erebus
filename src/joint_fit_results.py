from src.utility.h5_serializable_file import H5Serializable
from src.joint_fit import JointFit

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
    
    @staticmethod
    def load(path : str):
        return JointFitResults().load_from_path(path)