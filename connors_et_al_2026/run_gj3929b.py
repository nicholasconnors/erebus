import os

# For Eureka, has to happen before imports
os.environ['CRDS_PATH'] = '/home/nconnors/crds_cache'
os.environ['CRDS_SERVER_URL']= "https://jwst-crds.stsci.edu"

os.environ['CRDS_CONTEXT'] = 'jwst-latest'


print("CRDS server at", os.environ['CRDS_SERVER_URL'])

import sys
import numpy as np
import matplotlib.pyplot as plt

from erebus import Erebus
from erebus.utility.run_cfg import ErebusRunConfig
from erebus.utility.bayesian_parameter import Parameter
from erebus.systematics.frame_normalized_pca import perform_fn_pca_on_aperture
from erebus.utility.utils import bin_data

class State:
    def __init__(self):
        visit1_ev = []
        visit2_ev = []
        visit3_ev = []
        visit4_ev = []
        
        visit1_ev_binned = []
        visit2_ev_binned = []
        visit3_ev_binned = []
        visit4_ev_binned = []
        
        visit4_cutoff_index = 0
        visit4_cutoff_index_binned = 0
        
        cutoff_time = 0

state = State()

def custom_systematic(x, visit_index, is_joint_fit, 
                      obs1_pc1, obs1_pc2, obs1_pc3, obs1_pc4, obs1_pc5, 
                      obs2_pc1, obs2_pc2, obs2_pc3, obs2_pc4, obs2_pc5, 
                      obs2_exp1, obs2_exp2, 
                      obs2_b
    ):        
    # Final visit
    if visit_index == 3:
        systematic = np.zeros_like(x)
        
        obs1_coeffs = np.array([obs1_pc1, obs1_pc2, obs1_pc3, obs1_pc4, obs1_pc5])
        obs2_coeffs = np.array([obs2_pc1, obs2_pc2, obs2_pc3, obs2_pc4, obs2_pc5])
        
        ev = (state.visit4_ev_binned if is_joint_fit else state.visit4_ev).T
        cutoff = state.visit4_cutoff_index_binned if is_joint_fit else state.visit4_cutoff_index
        
        pca = np.zeros_like(ev[0])
        for i in range(0, 5):
            pca[cutoff:] += obs1_coeffs[i] * ev[i][cutoff:]
            pca[:cutoff] += obs2_coeffs[i] * ev[i][:cutoff]
        
        systematic += pca
        
        exp_term = (obs2_exp1 * np.exp(obs2_exp2 * (x[cutoff:] - state.cutoff_time))) + obs2_b
        exp_term = np.where(np.isfinite(exp_term), exp_term, 0.0)
        exp_term = np.clip(exp_term, 0, 10)
        
        systematic[cutoff:] += obs2_b
        
        systematic[cutoff:] += exp_term

        return systematic
    else:
        if visit_index == 0:
            ev = state.visit1_ev_binned if is_joint_fit else state.visit1_ev
        elif visit_index == 1:
            ev = state.visit2_ev_binned if is_joint_fit else state.visit2_ev
        elif visit_index == 2:
            ev = state.visit3_ev_binned if is_joint_fit else state.visit3_ev
        else:
            print (f"Invalid visit index??? {visit_index}")
                    
        coeffs = np.array([obs1_pc1, obs1_pc2, obs1_pc3, obs1_pc4, obs1_pc5])

        systematic = np.zeros_like(x)
        for i in range(0, 5):
            systematic += coeffs[i] * ev[i]
                
        return systematic

visit4_params = {
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
    "obs2_exp1": Parameter.uniform_prior(1e-6, 0.01e-6, 100e-6),
    "obs2_exp2": Parameter.uniform_prior(-300.0, -5000.0, -1.0),
    "obs2_b": Parameter.uniform_prior(1e-6, -500e-6, 500e-6)
}

params = {
    "obs1_pc1": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc2": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc3": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc4": Parameter.uniform_prior(0.1, -10, 10),
    "obs1_pc5": Parameter.uniform_prior(0.1, -10, 10),
    "obs2_pc1": Parameter.fixed(0),
    "obs2_pc2": Parameter.fixed(0),
    "obs2_pc3": Parameter.fixed(0),
    "obs2_pc4": Parameter.fixed(0),
    "obs2_pc5": Parameter.fixed(0),
    "obs2_exp1": Parameter.fixed(0),
    "obs2_exp2": Parameter.fixed(0),
    "obs2_b": Parameter.fixed(0),
}

if __name__ == "__main__":    
    config = "gj3929b"
    
    try:
        aperture_size = int(sys.argv[1])
        shared_t_sec = sys.argv[2].lower() == 'true'
        shared_fp = sys.argv[3].lower() == 'true'
        fit_fnpca = sys.argv[4].lower() == 'true'
    except Exception as e:
        print("Input must be: Aperture size (int), shared t_sec (bool), shared fp (bool), fit_fnpca (bool)")
        raise e
    
    if not fit_fnpca:
        print("Fixing PCA priors to 0")
        for i in range(1, 6):
            params[f'obs1_pc{i}'] = Parameter.fixed(0)
            visit4_params[f'obs1_pc{i}'] = Parameter.fixed(0)
            visit4_params[f'obs2_pc{i}'] = Parameter.fixed(0)
    
    joint_fit = shared_fp or shared_t_sec
        
    cfg = ErebusRunConfig.load("./run_cfgs/" + config + ".yaml")   
    
    cfg.set_custom_systematic_model(custom_systematic, params)
    cfg.set_custom_systematic_model_prior_for_visit_index(3, visit4_params)
    
    cfg.perform_individual_fits = not joint_fit
    cfg.perform_joint_fit = joint_fit
    cfg.joint_fit_bin_size = 1
    cfg.aperture_radius = aperture_size
    cfg.fit_eclipse_timing_offset_per_visit = not shared_t_sec
    cfg.fit_eclipse_depth_per_visit = not shared_fp
    cfg.fit_linear = True
    cfg.fit_fnpca = fit_fnpca
    cfg.fit_no_eclipse = False
    
    # Testing
    #cfg.max_steps = 1000
    #cfg.skip_visits = [0, 1, 2]
    
    erebus = Erebus(cfg, force_clear_cache = False)
    
    # Correct photometry for visit 4
    visit4 = erebus.photometry[3]
    
    '''
    plt.plot(visit4.time, visit4.raw_flux)
    plt.savefig("debug_figures/gj3929b_visit4_uncorrected_flux.png")
    plt.close()

    data = np.column_stack((visit4.time, visit4.raw_flux))
    np.savetxt('debug_figures/gj3929b_visit4_uncorrected_flux.csv', data, delimiter=',', header='time, flux')
    
    #cutoff = np.where(visit4.raw_flux < 0.9950)[0][0]
    '''
    
    cutoff = np.where(np.diff(visit4.time) > 0.01)[0][0] + 1

    visit4.raw_flux[:cutoff] = visit4.raw_flux[:cutoff] / np.median(visit4.raw_flux[:cutoff])
    visit4.raw_flux[cutoff:] = visit4.raw_flux[cutoff:] / np.median(visit4.raw_flux[cutoff:])   

    '''    
    plt.plot(visit4.time, visit4.raw_flux)
    plt.savefig("debug_figures/gj3929b_visit4_corrected_flux.png")
    plt.close()
    
    data = np.column_stack((visit4.time, visit4.raw_flux))
    np.savetxt('debug_figures/gj3929b_visit4_corrected_flux.csv', data, delimiter=',', header='time, flux')
    '''
    
    print("Fixed flux for visit 4")
    
    # Cut first 2000 integrations
    s = np.argsort(visit4.time)
    visit4.raw_flux = visit4.raw_flux[s]
    visit4.time = visit4.time[s]
    visit4.normalized_frames = visit4.normalized_frames[s]
    
    visit4.raw_flux = visit4.raw_flux[1500:]
    visit4.time = visit4.time[1500:]
    visit4.normalized_frames = visit4.normalized_frames[1500:]
    cutoff -= 1500
    
    cutoff_time = visit4.time[cutoff]
    
    # Correct again because of the tilt event
    visit4.raw_flux[:cutoff] = visit4.raw_flux[:cutoff] / np.median(visit4.raw_flux[:cutoff])
    visit4.raw_flux[cutoff:] = visit4.raw_flux[cutoff:] / np.median(visit4.raw_flux[cutoff:])   
    
    # Cut first 500 from visit 1
    visit1 = erebus.photometry[0]
    
    #s = np.argsort(visit1.time)
    #visit1.raw_flux = visit1.raw_flux[s]
    #visit1.time = visit1.time[s]
    #visit1.normalized_frames = visit1.normalized_frames[s]
    
    #visit1.raw_flux = visit1.raw_flux[500:]
    #visit1.time = visit1.time[500:]
    #visit1.normalized_frames = visit1.normalized_frames[500:]
    
    plt.plot(visit4.time, visit4.raw_flux)
    plt.axvline(cutoff_time)
    #plt.savefig("./debug_figures/gj3929b_visit4_cutoff.png")
    plt.close()
    
    start_trim = 500
    
    print(f"Trimming {start_trim} from start")
    
    bin_size = cfg.joint_fit_bin_size
    
    # end trim is broken on joint fit, should fix that
    cutoff = cutoff - start_trim
    
    # Trim a bit more from the start so that the cutoff point falls at the start of a bin when doing joint fit
    trim = cutoff % bin_size
    visit4.raw_flux = visit4.raw_flux[trim:]
    visit4.time = visit4.time[trim:]
    visit4.normalized_frames = visit4.normalized_frames[trim:]
    cutoff = cutoff - trim
    
    # Do PCA
    
    visit4_frames = visit4.normalized_frames[start_trim:]
    
    obs1_ev = perform_fn_pca_on_aperture(visit4_frames[:cutoff])[0].T
    obs2_ev = perform_fn_pca_on_aperture(visit4_frames[cutoff:])[0].T
    obs1_ev_binned = np.array([bin_data(ev, bin_size)[0] for ev in obs1_ev.T]).T
    obs2_ev_binned = np.array([bin_data(ev, bin_size)[0] for ev in obs2_ev.T]).T
    
    visit1_ev = perform_fn_pca_on_aperture(erebus.photometry[0].normalized_frames[start_trim:])[0]
    visit2_ev = perform_fn_pca_on_aperture(erebus.photometry[1].normalized_frames[start_trim:])[0]
    visit3_ev = perform_fn_pca_on_aperture(erebus.photometry[2].normalized_frames[start_trim:])[0]
    visit4_ev = np.concatenate((obs1_ev, obs2_ev))
    
    visit1_ev_jf = np.array([bin_data(ev, bin_size)[0] for ev in visit1_ev])
    visit2_ev_jf = np.array([bin_data(ev, bin_size)[0] for ev in visit2_ev])
    visit3_ev_jf = np.array([bin_data(ev, bin_size)[0] for ev in visit3_ev])
    visit4_ev_jf = np.concatenate((obs1_ev_binned, obs2_ev_binned))
    
    visit4_time = erebus.photometry[-1].time[start_trim:]
    visit4_time_jf = bin_data(erebus.photometry[-1].time[start_trim:], bin_size)[0]
    
    data = np.column_stack((visit4_time_jf, visit4_ev_jf.T[0], visit4_ev_jf.T[1], visit4_ev_jf.T[2], visit4_ev_jf.T[3], visit4_ev_jf.T[4]))
    np.savetxt(f'debug_figures/gj3929b_visit4_{aperture_size}px_eigenvalues.csv', data, delimiter=',', header='time, pc1, pc2, pc3, pc4, pc5')
    data = np.column_stack((visit4_time, visit4_ev.T[0], visit4_ev.T[1], visit4_ev.T[2], visit4_ev.T[3], visit4_ev.T[4]))
    np.savetxt(f'debug_figures/gj3929b_visit4_{aperture_size}px_eigenvalues_binned.csv', data, delimiter=',', header='time, pc1, pc2, pc3, pc4, pc5')
    
    state.visit1_ev = visit1_ev
    state.visit2_ev = visit2_ev
    state.visit3_ev = visit3_ev
    state.visit4_ev = visit4_ev
    
    state.visit1_ev_binned = visit1_ev_jf
    state.visit2_ev_binned = visit2_ev_jf
    state.visit3_ev_binned = visit3_ev_jf
    state.visit4_ev_binned = visit4_ev_jf

    state.visit4_cutoff_index = cutoff
    state.visit4_cutoff_index_binned = int(cutoff / bin_size)
    state.cutoff_time = visit4_time[cutoff]
    
    print(f"CUTOFF: {cutoff} {state.visit4_cutoff_index_binned}")

    jf_str = "joint" if joint_fit else "individual"
    if shared_fp and not shared_t_sec:
        jf_str = "variable_t_sec"
    elif not shared_fp and shared_t_sec:
        jf_str = "variable_fp"
    sys_str = "fnpca" if fit_fnpca else "linear"
    folder_name = f"./output_final_{{NAME}}_{aperture_size}_{sys_str}_{jf_str}_{{DATE}}/"
    
    erebus._Erebus__setup_fits()

    '''
    if not joint_fit:
        visit4_fit = erebus.individual_fits[3]
        values = [visit4_fit.mcmc.params[v].value for v in visit4_fit.mcmc.get_free_params()]
        print(values)
        
        print(cutoff_time)
        print(visit4_fit.time[0])
        
        cutoff_time = visit4_fit.time[cutoff]
        
        args = ["x"] + [key for key in visit4_fit.mcmc.params][:-1]
        fit_method = create_method_signature(IndividualFit._IndividualFit__fit_method, args)
        visit4_fit.mcmc.set_method(fit_method)
        IndividualFit._IndividualFit__instance = visit4_fit
        
        obs1_indices = np.where(visit4_fit.time < cutoff_time)[0]
        obs2_indices = np.where(visit4_fit.time >= cutoff_time)[0]
        
        time = visit4_fit.time
        starting_model = visit4_fit.mcmc.evaluate_model(time, *values)
        plt.plot(time, visit4_fit.raw_flux)
        plt.plot(time, starting_model)
        plt.axvline(cutoff_time, color='black')
        for i, f in enumerate(erebus.individual_fits):
            if i != 3:
                continue
            plt.plot(f.time, f.raw_flux, label=f"{i}")
        plt.legend()
        plt.savefig("./debug_figures/visit_4_start.png")
        plt.close()
    else:
        time = erebus.joint_fit.time
        flux = erebus.joint_fit.raw_flux
        plt.plot(time, flux)
        plt.axvline(cutoff_time, color='black')
        plt.xlim([erebus.joint_fit.starting_times[-1], np.max(erebus.joint_fit.time)])
        plt.plot(visit4.time, visit4.raw_flux)
        plt.savefig("./debug_figures/visit_4_start_joint_fit.png")
        plt.close()
        '''
        
    erebus.run(force_clear_cache = True, output_folder=folder_name)
    