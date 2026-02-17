import os

# For Eureka, has to happen before imports
os.environ['CRDS_PATH'] = '/home/nconnors/crds_cache'
os.environ['CRDS_SERVER_URL']= "https://jwst-crds.stsci.edu"

os.environ['CRDS_CONTEXT'] = 'jwst-latest'


print("CRDS server at", os.environ['CRDS_SERVER_URL'])

import matplotlib.pyplot as plt

from erebus import Erebus
from erebus.utility.run_cfg import ErebusRunConfig
import sys
import numpy as np
from erebus.utility.bayesian_parameter import Parameter
from erebus.systematics.frame_normalized_pca import perform_fn_pca_on_aperture

import matplotlib.pyplot as plt

global obs1_eigenvalues
global obs2_eigenvalues
global cutoff

obs1_eigenvalues = []
obs2_eigenvalues = []
cutoff = 2980

def split_fnpca(x, obs1_pc1, obs1_pc2, obs1_pc3, obs1_pc4, obs1_pc5, 
                obs2_pc1, obs2_pc2, obs2_pc3, obs2_pc4, obs2_pc5,
                obs2_exp1, obs2_exp2, obs2_b):
    systematic = np.ones_like(x)
    
    obs1_coeffs = np.array([obs1_pc1, obs1_pc2, obs1_pc3, obs1_pc4, obs1_pc5])
    obs2_coeffs = np.array([obs2_pc1, obs2_pc2, obs2_pc3, obs2_pc4, obs2_pc5])
    
    obs1_pca = np.ones_like(obs1_eigenvalues[0])
    obs2_pca = np.ones_like(obs2_eigenvalues[0])
    for i in range(0, 5):
        obs1_pca += obs1_coeffs[i] * obs1_eigenvalues[i]
        obs2_pca += obs2_coeffs[i] * obs2_eigenvalues[i]
    
    systematic[:cutoff] *= obs1_pca
    systematic[cutoff:] *= obs2_pca
    
    systematic[cutoff:] *= (obs2_exp1 * np.exp(obs2_exp2 * (x[cutoff:] - x[cutoff]))) + 1 + obs2_b

    return systematic

params = {
    "obs1_pc1": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc2": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc3": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc4": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc5": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc1": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc2": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc3": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc4": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc5": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_exp1": Parameter.uniform_prior(0.001, -0.01, 0.01),
    "obs2_exp2": Parameter.uniform_prior(-60.0, -80.0, -40.0),
    "obs2_b": Parameter.uniform_prior(1e-6, -1e-3, 1e-3)
}

if __name__ == "__main__":    
    if len(sys.argv) < 2:
        print("Requires at least one argument (config name)")
        sys.exit(1)

    config = sys.argv[1]
    
    clear_cache = False
    if len(sys.argv) == 3:
        clear_cache = sys.argv[2]
        
    cfg = ErebusRunConfig.load("./run_cfgs/" + config + ".yaml")
    
    # We have to do FNPCA manually for the two observations
    cfg.fit_fnpca = False
    cfg.fit_linear = False
    cfg.perform_individual_fits = True
    cfg.perform_joint_fit = False
    cfg.calints_path = "/home/nconnors/Research/GJ3929b_visit4_combined/"
    cfg.trim_integrations = [1500, None]
    cfg.set_custom_systematic_model(split_fnpca, params)
    
    erebus = Erebus(cfg, force_clear_cache = clear_cache)
    
    # Correct photometry for visit 4
    visit4 = erebus.photometry[-1]
    
    plt.plot(visit4.time, visit4.raw_flux)
    plt.savefig("debug_figures/gj3929b_visit4_uncorrected_flux.png")
    plt.close()

    data = np.column_stack((visit4.time, visit4.raw_flux))
    np.savetxt('debug_figures/gj3929b_visit4_uncorrected_flux.csv', data, delimiter=',', header='time, flux')
    
    cutoff = np.where(visit4.raw_flux < 0.9950)[0][0]
    visit4.raw_flux[:cutoff] = visit4.raw_flux[:cutoff] / np.median(visit4.raw_flux[:cutoff])
    visit4.raw_flux[cutoff:] = visit4.raw_flux[cutoff:] / np.median(visit4.raw_flux[cutoff:])
    
    plt.plot(visit4.time, visit4.raw_flux)
    plt.savefig("debug_figures/gj3929b_visit4_corrected_flux.png")
    plt.close()
    
    data = np.column_stack((visit4.time, visit4.raw_flux))
    np.savetxt('debug_figures/gj3929b_visit4_corrected_flux.csv', data, delimiter=',', header='time, flux')
    
    print("Fixed flux for visit 4")
    
    start_trim = erebus.individual_fits[-1].start_trim
    
    print(f"Trimming {start_trim} from start from end")
    
    cutoff = cutoff - start_trim
    frames = visit4.normalized_frames[start_trim:]
    obs1_eigenvalues, _, _ = perform_fn_pca_on_aperture(frames[:cutoff])
    obs2_eigenvalues, _, _ = perform_fn_pca_on_aperture(frames[cutoff:])

    model_time = visit4.time[start_trim:]
    initial_params = [v.value for v in params.values()]
    test_model = erebus.individual_fits[-1].systematic_model(model_time, 0, 0, 0, 0, 0, 0, 0, 0, 0, *initial_params)

    plt.plot(model_time, visit4.raw_flux[start_trim:])    
    plt.plot(model_time, test_model)
    plt.axvline(model_time[cutoff], color='black', linestyle='--')
    plt.savefig("debug_figures/visit4_initial_parameters.png")
    plt.close()
    
    print("Plotted initial guess of custom systematics")

    erebus.run(force_clear_cache = True)