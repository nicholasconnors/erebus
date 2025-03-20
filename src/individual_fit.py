import os
import hashlib
from typing import List
import numpy as np
from src.photometry_data import PhotometryData
from src.utility.planet import Planet
from src.mcmc_model import WrappedMCMC
from src.utility.bayesian_parameter import Parameter
from src.utility.run_cfg import ErebusRunConfig
from src.frame_normalized_pca import perform_fn_pca_on_aperture
from src.utility.h5_serializable_file import H5Serializable
import batman
import json
import matplotlib.pyplot as plt

EREBUS_CACHE_DIR = "erebus_cache"

class IndividualFit(H5Serializable):
    instance = None
    
    def exclude_keys(self):
        '''
        Excluded from serialization
        '''
        return ['config', 'time', 'raw_flux', 'params', 'transit_model', 'mcmc', 'instance']
    
    def __init__(self, photometry_data : PhotometryData, planet : Planet, config : ErebusRunConfig,
                 force_clear_cache : bool = False):
        self.source_folder = photometry_data.source_folder
        self.visit_name = photometry_data.visit_name
        source_folder_hash = hashlib.md5(self.source_folder.encode()).hexdigest()
        config_hash = hashlib.md5(json.dumps(config.model_dump()).encode()).hexdigest()
        
        self.cache_file = f"{EREBUS_CACHE_DIR}/{self.visit_name}_{source_folder_hash}_{config_hash}_individual_fit.h5"
        
        self.start_trim = 0 if config.trim_integrations is None else config.trim_integrations[0]
        self.end_trim = None if config.trim_integrations is None else -np.abs(config.trim_integrations[1])
        
        self.time = photometry_data.time[self.start_trim:self.end_trim] - np.min(photometry_data.time)
        self.raw_flux = photometry_data.raw_flux[self.start_trim:self.end_trim]
        self.config = config
        
        self.results = {}
        self.chain = None
        
        self.params = None
        self.transit_model = None
        
        self.eigenvalues, self.eigenvectors = perform_fn_pca_on_aperture(photometry_data.normalized_frames[self.start_trim:self.end_trim])
                
        mcmc = WrappedMCMC()
        
        # For circular orbit predict the eclipse time, else use a uniform prior
        if isinstance(planet.ecc, float) and planet.ecc == 0:
            print("Circular orbit: using gaussian prior for t_sec")
            nominal_period = planet.p if isinstance(planet.p, float) else planet.p.nominal_value
            predicted_t_sec = (planet.t0 - np.min(photometry_data.time) - 2400000.5 + planet.p / 2.0) % nominal_period
            mcmc.add_parameter("t_sec", Parameter.prior_from_ufloat(predicted_t_sec))
        else:
            print("Eccentric orbit: using uniform prior for t_sec")
            duration = np.max(photometry_data.time - np.min(photometry_data.time))
            mcmc.add_parameter("t_sec", Parameter.uniform_prior(duration / 2.0, duration / 6.0, duration * 5.0 / 6.0))
        
        mcmc.add_parameter("fp", Parameter.uniform_prior(200e-6, -1500e-6, 1500e-6))
        mcmc.add_parameter("t0", Parameter.prior_from_ufloat(planet.t0))
        mcmc.add_parameter("rp_rstar", Parameter.prior_from_ufloat(planet.rp_rstar))
        mcmc.add_parameter("a_rstar", Parameter.prior_from_ufloat(planet.a_rstar))
        mcmc.add_parameter("p", Parameter.prior_from_ufloat(planet.p))
        mcmc.add_parameter("inc", Parameter.prior_from_ufloat(planet.inc))
        mcmc.add_parameter("ecc", Parameter.prior_from_ufloat(planet.ecc))
        mcmc.add_parameter("w", Parameter.prior_from_ufloat(planet.w))
        
        if self.config.fit_fnpca:
            for i in range(0, 5):
                mcmc.add_parameter(f"pc{(i+1)}", Parameter.uniform_prior(0.1, -10, 10))
        else:
            for i in range(0, 5):
                mcmc.add_parameter(f"pc{(i+1)}", Parameter.fixed(0))
        
        if self.config.fit_exponential:
            mcmc.add_parameter("exp1", Parameter.uniform_prior(0.01, -0.1, 0.1))
            mcmc.add_parameter("exp2", Parameter.uniform_prior(-60.0, -200.0, -1.0))
        else:
            mcmc.add_parameter("exp1", Parameter.fixed(0))
            mcmc.add_parameter("exp2", Parameter.fixed(0))

        if self.config.fit_linear:
            mcmc.add_parameter("a", Parameter.uniform_prior(1e-3, -2, 2))
        else:
            mcmc.add_parameter("a", Parameter.fixed(0))
            
        mcmc.add_parameter("b", Parameter.uniform_prior(1e-6, -0.01, 0.01))
            
        mcmc.add_parameter("y_err", Parameter.uniform_prior(400e-6, 0, 2000e-6))
        
        mcmc.set_method(IndividualFit.__mcmc_fit_method)
        
        self.mcmc = mcmc
        
        self.first_frame = photometry_data.normalized_frames[0]
        
        if os.path.isfile(self.cache_file) and not force_clear_cache:
            self.load_from_path(self.cache_file)
        else:
            self.save_to_path(self.cache_file)
    
    def physical_model(self, x : List[float], t_sec : float, fp : float, t0 : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float) -> List[float]:
        '''
        Model for the lightcurve using batman
        fp is expected written in ppm
        '''
        if self.params is None:
            params = batman.TransitParams()
            params.limb_dark = "quadratic"
            params.u = [0.3, 0.3]
            
        params.t0 = t0
        params.t_secondary = t_sec
        params.fp = fp
        params.rp = rp_rstar
        params.inc = inc
        params.per = p
        params.a = a_rstar  
        params.ecc = ecc
        params.w = w
        
        if self.transit_model is None:
            transit_model = batman.TransitModel(params, x, transittype="secondary")

        flux_model = transit_model.light_curve(params)
        return flux_model
    
    def systematic_model(self, x : List[float], pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float, 
                         exp1 : float, exp2 : float, a : float, b : float) -> List[float]:
        systematic = np.ones_like(x)
        if self.config.fit_fnpca:
            coeffs = np.array([pc1, pc2, pc3, pc4, pc5])
            pca = np.ones_like(self.eigenvalues[0])
            for i in range(0, 5):
                pca += coeffs[i] * self.eigenvalues[i]
            systematic *= pca
        if self.config.fit_exponential:
            systematic *= (exp1 * np.exp(exp2 * x)) + 1
        if self.config.fit_linear:
            systematic *= (a * x) + 1
        
        systematic += b
        
        return systematic
        
    # TODO: would be nice to generate this dynamically
    def fit_method(self, x : List[float], t_sec : float, fp : float, t0 : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float, 
                       pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float,
                       exp1 : float, exp2 : float, a : float, b : float) -> List[float]:
        systematic = self.systematic_model(x, pc1, pc2, pc3, pc4, pc5, exp1, exp2, a, b)
        physical = self.physical_model(x, t_sec, fp, t0, rp_rstar, a_rstar, p, inc, ecc, w)
        return physical * systematic 
    
    def __mcmc_fit_method(x : List[float], t_sec : float, fp : float, t0 : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float, 
                       pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float,
                       exp1 : float, exp2 : float, a : float, b : float) -> List[float]:
        '''
        MCMC method input must be static
        '''
        return IndividualFit.instance.fit_method(x, t_sec, fp, t0, rp_rstar,
                                                           a_rstar, p, inc, ecc, w,
                                                           pc1, pc2, pc3, pc4, pc5,
                                                           exp1, exp2, a, b)

    def run(self):
        # Since the MCMC runs off a static method set the static instance to this object first
        IndividualFit.instance = self
        self.mcmc.run(self.time, self.raw_flux)
        self.results = self.mcmc.results
        self.chain = self.mcmc.sampler.get_chain(discard=200, thin=15, flat=True)
        print(self.mcmc.results)
        
        self.save_to_path(self.cache_file)
        
        self.mcmc.corner_plot()
        self.mcmc.chain_plot()
    
    def plot_fn_pca(self):
        pass
    
    def plot_initial_guess(self):
        # TODO: Initial guess and first frame
        pass