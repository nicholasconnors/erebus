import os
import hashlib
from typing import List
import numpy as np
from src.photometry_data import PhotometryData
from src.utility.planet import Planet
from src.mcmc_model import WrappedMCMC
from src.utility.bayesian_parameter import Parameter
from src.utility.run_cfg import ErebusRunConfig
import batman

# TODO: Make this serializable with its results
# Base hash off of the photometry hash, planet name, config to json string hash
class IndividualLightcurveFit:
    instance = None
    
    def __init__(self, photometry_data : PhotometryData, planet : Planet, config : ErebusRunConfig):
        self.time = photometry_data.t
        self.raw_flux = photometry_data.light_curve
        self.config = config
        
        self.params = None
        self.transit_model = None
        
        IndividualLightcurveFit.instance = self
        
        nominal_period = planet.p if isinstance(planet.p, float) else planet.p.nominal_value
        predicted_t_sec = (planet.t0 - np.min(photometry_data.t) - 2400000.5 + planet.p / 2.0) % nominal_period
        
        mcmc = WrappedMCMC()
        mcmc.add_parameter("t_sec", Parameter.prior_from_ufloat(predicted_t_sec))
        mcmc.add_parameter("fp", Parameter.uniform_prior(0, -2000, 2000))
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
            mcmc.add_parameter("a", Parameter.uniform_prior(0.01, -0.1, 0.1))
        else:
            mcmc.add_parameter("a", Parameter.fixed(0))

        # All models involve an offset
        if self.config.fit_fnpca or self.config.fit_exponential or self.config.fit_linear:
            mcmc.add_parameter("b", Parameter.uniform_prior(0, -0.01, 0.01))
        else:
            mcmc.add_parameter("b", Parameter.fixed(0))
            
        mcmc.add_parameter("y_err", Parameter.uniform_prior(400, 0, 2000))
        
        mcmc.set_method(IndividualLightcurveFit.fit_method)
        
        self.mcmc = mcmc
    
    def physical_model(self, x : List[float], t_sec : float, fp : float, t0 : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float) -> List[float]:
        '''
        Model for the lightcurve using batman
        fp is expected written in ppm
        output is normalized such that the out-of-eclipse baseline is 1
        '''
        if self.params is None:
            params = batman.TransitParams()
            params.t0 = t0
            params.t_secondary = t_sec
            params.fp = fp * 1e-6
            params.rp = rp_rstar
            params.inc = inc
            params.per = p
            params.a = a_rstar  
            params.ecc = ecc
            params.w = w
            params.limb_dark = "quadratic"
            params.u = [0.3, 0.3]
            transit_model = batman.TransitModel(params, x, transittype="secondary")
        params.t_secondary = t_sec
        params.fp = fp * 1e-6
        flux_model = transit_model.light_curve(params)
        return (flux_model - params.fp) * 1e6 # ppm
    
    def systematic_model(self, x : List[float], pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float, 
                         exp1 : float, exp2 : float, a : float, b : float) -> List[float]:
        systematic = np.ones_like(x)
        if self.config.fit_fnpca:
            systematic *= 1 # TODO
        elif self.config.fit_exponential:
            systematic *= exp1 * np.exp(exp2 * x)
        elif self.config.fit_linear:
            systematic *= a * x
        
        # All models involve an offset
        if self.config.fit_fnpca or self.config.fit_exponential or self.config.fit_linear:
            systematic += b
        
        return systematic
        
    # TODO: would be nice to generate this dynamically
    def fit_method(x : List[float], t_sec : float, fp : float, t0 : float, rp_rstar : float,
                       a_rstar : float, p : float, inc : float, ecc : float, w : float, 
                       pc1 : float, pc2 : float, pc3 : float, pc4 : float, pc5 : float,
                       exp1 : float, exp2 : float, a : float, b : float) -> List[float]:
        systematic = IndividualLightcurveFit.instance.systematic_model(x, pc1, pc2, pc3, pc4, pc5, exp1, exp2, a, b)
        physical = IndividualLightcurveFit.instance.physical_model(x, t_sec, fp, t0, rp_rstar, a_rstar, p, inc, ecc, w)
        return physical * systematic 

    def run(self):
        self.mcmc.run(self.time, self.raw_flux)
        self.mcmc.corner_plot()
        self.mcmc.chain_plot()
        print(self.mcmc.results)