from erebus.utility.h5_serializable_file import H5Serializable
from erebus.individual_fit import IndividualFit

class IndividualFitResults(H5Serializable):
    '''
    Class containing the results of an individual fit run
    '''
    
    def __init__(self, fit : IndividualFit):
        if fit is not None:
            self.time = fit.time
            '''The time values that the model was fit on. Starts at 0.'''
            self.start_time = fit.start_time
            '''The BJD date the observation started on.'''
            self.raw_flux = fit.raw_flux
            '''Light curve that is not yet detrended.'''
            self.eigenvalues = fit.eigenvalues
            '''For each principal component, the eigenvalue. The first 5 are used as the FN-PCA systematic model.'''
            self.eigenvectors = fit.eigenvectors
            '''For each principal component, this is its eigenimage.'''
            self.pca_variance_ratios = fit.pca_variance_ratios
            '''For each principal component, how much of the variance does it explain.'''
            self.order = fit.order
            '''If there were other visits observed, this is the numerical ordering they occuring in.'''
            self.results = fit.results
            '''A dictionary of results for the fit parameters (e.g., eclipse depth)'''
            self.planet_name = fit.planet_name
            '''The name of the planet observed'''
            self.visit_name = fit.visit_name
            '''The unique visit name as read on MAST.'''
            self.config = fit.config
            '''The config file used to create this run'''
            self.planet = fit.planet
            '''The planet config file used to create this run'''
            self.config_hash = fit.config_hash
            '''The unique hash of the config file. Used for naming cache files.'''
            self.frames = fit.photometry_data.normalized_frames
            '''The frames which aperture photometry was performed on for the fit.'''
            self.predicted_t_sec = fit.predicted_t_sec
            '''The predicted 0.5 phase eclipse time'''
            
            res_nominal_values = [fit.results[k].nominal_value for k in fit.results][:-1]
            systematic_params = res_nominal_values[9:]
            
            self.flux_model = fit.fit_method(fit.time, *res_nominal_values)
            '''The best fit detrended lightcurve.'''
            self.systematic_factor = fit.systematic_model(fit.time, *systematic_params)
            '''The systematic factor which was divided out of the raw lightcurve to get the detrended one.'''
    
    @staticmethod
    def load(path : str):
        '''After running an Erebus instance, the results file can be loaded later using this method.'''
        return IndividualFitResults(None).load_from_path(path)