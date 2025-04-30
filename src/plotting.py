from src.utility.utils import bin_data
from src.utility.utils import get_eclipse_duration
import matplotlib.colors as colors
from scipy.ndimage.filters import uniform_filter1d
from pathlib import Path
from src.individual_fit import IndividualFit
from src.joint_fit import JointFit
import matplotlib.pyplot as plt
import numpy as np
import inspect

def plot_fnpca_individual_fit(individual_fit : IndividualFit, save_to_directory : str = None):
    '''
    Will save both a png and a pdf
    Expects the individual fit to be done with fnpca and a linear component
    '''
    yerr = individual_fit.results['y_err'].nominal_value
    t_sec = individual_fit.results['t_sec'].nominal_value
    rp = individual_fit.results['rp_rstar'].nominal_value
    inc = individual_fit.results['inc'].nominal_value
    a = individual_fit.results['a_rstar'].nominal_value
    per = individual_fit.results['p'].nominal_value
    fp = individual_fit.results['fp'].nominal_value
    fp_err = individual_fit.results['fp'].std_dev

    raw_time = individual_fit.time
    time = (raw_time - t_sec) * 24 # hours
    flux = individual_fit.raw_flux
    
    bin_size = len(time) // 30
    bin_time, _ = bin_data(time, bin_size)
    bin_flux, _ = bin_data(flux, bin_size)
    bin_yerr = yerr / np.sqrt(bin_size)
    duration = get_eclipse_duration(inc, a, rp, per) * 24
    res_nominal_values = [individual_fit.results[k].nominal_value for k in individual_fit.results][:-1]
    systematic_params = res_nominal_values[9:]
    flux_model = individual_fit.fit_method(raw_time, *res_nominal_values)
    systematic_factor = individual_fit.systematic_model(raw_time, *systematic_params)
    
    fig = plt.figure(figsize=(9, 5.5))
    grid = fig.add_gridspec(4, 2)
    
    ############################################################################## Layout
    flux_subfigure = fig.add_subfigure(grid[0:3,0])
    allan_deviation_subfigure = fig.add_subfigure(grid[3,0])
    pca_subfig = fig.add_subfigure(grid[:,1:])
    
    flux_gridspec = flux_subfigure.add_gridspec(3, 1, wspace=0, hspace=0.1)
    flux_gridspec.update(top=1.0, right=0.85)
    flux_axs = flux_gridspec.subplots(sharex=True, sharey=True)
    flux_axs[0].set_title("Eclipse depth fitting")
    flux_axs[2].set_xlabel("Time from 0.5 phase (hours)")
    
    allan_gs = allan_deviation_subfigure.add_gridspec(1, 1)
    allan_gs.update(bottom=0.0, top=0.8, right=0.85)
    allan_ax = allan_gs.subplots()
    # This is NOT an Allan deviation plot (Kipping 2025)
    #allan_ax.set_title("Allan deviation plot")
    
    eigenvalue_gs = pca_subfig.add_gridspec(6,1, hspace=0.0, wspace=0.1)
    eigenvalue_axs = eigenvalue_gs.subplots(sharex=True, sharey=False)
    eigenvalue_gs.update(left=0.1, right=0.74, top=1.0, bottom=0)
    eigenvalue_axs[0].set_title("FN-PCA decomposition of lightcurve")
    eigenvalue_axs[-1].set_xlabel("Time from 0.5 phase (hours)")
    
    eigenimage_gs = pca_subfig.add_gridspec(6,1, hspace=0.0, wspace=0)
    eigenimage_gs.update(left=0.74, right=0.96, top=1.0, bottom=0)
    eigenimage_axs = eigenimage_gs.subplots(sharex=True, sharey=False)

    ############################################################################## Fluxes

    # Raw Flux
    flux_axs[0].errorbar(time, flux, yerr, linestyle='', marker='.', alpha = 0.2, color='gray')
    flux_axs[0].errorbar(bin_time, bin_flux, bin_yerr, linestyle='', marker='.', color='black', zorder=3)
    flux_axs[0].axvspan(- duration / 2, duration / 2, color='red', alpha=0.2)
    flux_axs[0].plot(time, flux_model, color='red')
    flux_axs[0].set_ylabel("Raw flux\n(ppm)")

    # Detrended flux
    detrended_flux = flux / systematic_factor
    bin_detrended_flux, _ = bin_data(detrended_flux, bin_size)
    flux_axs[1].errorbar(time, detrended_flux, yerr, linestyle='', marker='.', alpha = 0.2, color='gray')
    flux_axs[1].errorbar(bin_time, bin_detrended_flux, bin_yerr, linestyle='', marker='.', color='black', zorder=3)
    flux_axs[1].axvspan(- duration / 2, duration / 2, color='red', alpha=0.2)
    flux_axs[1].plot(time, flux_model / systematic_factor, color='red')
    flux_axs[1].set_ylabel("Detrended flux\n(ppm)")
    flux_axs[1].text(0.5, 0.95, f"Eclipse depth: {fp*1e6:0.0f}+/-{fp_err*1e6:0.0f}ppm", horizontalalignment='center', verticalalignment='top', transform=flux_axs[1].transAxes)

    # Systematic factor
    linear_component = individual_fit.results['a'].nominal_value * raw_time + individual_fit.results['b'].nominal_value + 1
    flux_axs[2].plot(time, systematic_factor, color='red')
    flux_axs[2].plot(time, linear_component, color='black', linestyle='--', label='Linear component')
    flux_axs[2].axvspan(- duration / 2, duration / 2, color='red', alpha=0.2)
    flux_axs[2].legend()
    flux_axs[2].set_ylabel("Systematc factor\n(ppm)")

    # Allan deviation plot
    def get_res(x, y, bin_size):
        model = flux_model
        if bin_size > 1:
            x, _ = bin_data(x, bin_size)
            y, _ = bin_data(y, bin_size)
            model, _ = bin_data(flux_model, bin_size)
        res = (y - model) * 1000000 # To ppm
        rms = np.sqrt(np.mean(res**2))
        return rms
    
    n = np.arange(1, 40)
    r = np.array([get_res(time, flux, i) for i in n])
    r_ideal = r[0]/np.sqrt(n)
    allan_ax.plot(n, r, color='red')
    allan_ax.plot(n, r_ideal, linestyle='--', color='black')
    allan_ax.set_xscale('log')
    allan_ax.set_yscale('log')
    allan_ax.set_xlabel("Bin size")
    allan_ax.set_xticks([1, 10])
    allan_ax.set_xticklabels(["1", "10"])
    allan_ax.set_yticks([100, 1000])
    allan_ax.set_yticklabels(["100", "1000"])
    allan_ax.set_ylabel("RMS of residuals")
    plt.setp(allan_ax.get_xminorticklabels(), visible=False)
    plt.setp(allan_ax.get_yminorticklabels(), visible=False)
    allan_ax.text(0.5, 0.95, f"Scatter: {yerr*1e6:0.0f}ppm", horizontalalignment='center', verticalalignment='top', transform=allan_ax.transAxes)

    ############################################################################## Eigenvalues
    # First row is the raw lightcurve and a single frame image
    eigenvalue_axs[0].plot(time, flux, marker='.',linestyle='', color='grey', alpha=0.3)
    eigenvalue_axs[0].plot(bin_time, bin_flux, marker='.', linestyle='', color='black')
    eigenvalue_axs[0].set_ylabel("Raw flux\n(ppm)")
    
    eigenimage_axs[0].imshow(individual_fit.first_frame)

    for i in range(0, 5):
        eigenvalue = individual_fit.eigenvalues[i]
        eigenvalue = eigenvalue / np.max(np.abs(eigenvalue))
        eigenvalue_ax = eigenvalue_axs[i+1]
        eigenvalue_ax.plot(time, eigenvalue, marker='.', linestyle='', alpha=0.3, color='cornflowerblue')
        eigenvalue_ax.plot(time, uniform_filter1d(eigenvalue, size=30), color='blue')
        eigenvalue_ax.axhline(0, color='black', linestyle='--')
        eigenvalue_ax.set_ylabel(f"PC{(i+1)}")
        eigenvalue_ax.set_yticks([])
        eigenvalue_ax.set_ylim([-1, 1])
    
        eigenimage = individual_fit.eigenvectors[i]
        eigenimage /= np.max(eigenimage)
        im = eigenimage_axs[i+1].imshow(eigenimage, cmap='bwr', interpolation='nearest', norm = colors.SymLogNorm(0.5, vmin=-1, vmax=1))

    for ax in eigenvalue_axs:
        ax.axvspan(- duration / 2, duration / 2, color='red', alpha=0.2)
    
    for ax in eigenimage_axs:
        ax.set_yticks([])
        ax.set_xticks([])

    cbar_gs = pca_subfig.add_gridspec(1,1, hspace=0.0, wspace=0.1)
    cbar_ax = cbar_gs.subplots(sharex=True, sharey=False)
    cbar_gs.update(left=0.96, right=0.98, top=1.0-1*(1.0/(6)), bottom=0)
    fig.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar_ax.set_yticklabels([-1, 0, 1])
    cbar_ax.set_ylabel("Scale (symlog)")

    if save_to_directory is not None:
        path = f"{save_to_directory}/{individual_fit.config.fit_fnpca}_{individual_fit.planet_name}_{individual_fit.visit_name}_{individual_fit.config_hash}"
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".pdf", bbox_inches='tight')
    plt.close()
    
def plot_eigenvectors(individual_fit : IndividualFit, save_to_directory : str = None):
    '''
    Plots the 5 highest ranked eigenimages to a given folder (if provided)
    '''
    for i in range(0, 5):
        eigenimage = individual_fit.eigenvectors[i]
        eigenimage /= np.max(np.abs(eigenimage))
        plt.imshow(eigenimage, cmap='bwr', interpolation='nearest', norm = colors.SymLogNorm(0.5, vmin=-1, vmax=1))
        plt.yticks([])
        plt.xticks([])
        if save_to_directory is not None:
            path = f"{save_to_directory}/{individual_fit.config.fit_fnpca}_{individual_fit.planet_name}_{individual_fit.visit_name}_eigenimage_{(i+1)}"
            plt.savefig(path + ".png", bbox_inches='tight')
            plt.savefig(path + ".pdf", bbox_inches='tight')
        plt.close()

def plot_joint_fit(joint_fit : JointFit, save_to_directory : str = None):
    fp = joint_fit.results['fp'].nominal_value
    fp_err = joint_fit.results['fp'].std_dev
    inc = joint_fit.results["inc"].nominal_value
    a = joint_fit.results["a_rstar"].nominal_value
    rp = joint_fit.results["rp_rstar"].nominal_value
    per = joint_fit.results["p"].nominal_value
    offset = joint_fit.results["t_sec_offset"].nominal_value * 24
    duration = get_eclipse_duration(inc, a, rp, per) * 24
    print("Offset: ", offset, "hours")
    eclipse_start = offset - duration/2
    eclipse_end = offset + duration/2
    
    args = [x.nominal_value for x in list(joint_fit.results.values())]
    number_of_physical_args = len(inspect.getfullargspec(joint_fit.physical_model).args) - 2
    physical_args = args[0:number_of_physical_args]
    number_of_systematic_args = len(inspect.getfullargspec(joint_fit.systematic_model).args) - 2
    visit_indices = np.array([joint_fit.get_visit_index_from_time(xi) for xi in joint_fit.time])
    #for visit_index in range(0, len(joint_fit.photometry_data_list)):
    detrended_visit = []
    time_visit = []
    physical_time_visit = []
    physical_visit = []
    for visit_index in range(0, len(joint_fit.photometry_data_list)):
        filt = visit_indices == visit_index
        time = joint_fit.time[filt]
        flux = joint_fit.raw_flux[filt]
                    
        systematic_index_start = (number_of_physical_args) + (visit_index * number_of_systematic_args)
        systematic_args = args[systematic_index_start:systematic_index_start + number_of_systematic_args]
    
        systematic = joint_fit.systematic_model(time, *systematic_args)
        physical_time = np.linspace(np.min(time), np.max(time), 1000)
        physical = joint_fit.physical_model(physical_time, *physical_args)
        
        detrended_visit.append(flux / systematic)
        time_offset = joint_fit.get_predicted_t_sec_of_visit(visit_index).nominal_value + joint_fit.starting_times[visit_index]
        time_visit.append((time - time_offset) * 24)
        physical_time_visit.append((physical_time - time_offset) * 24)
        physical_visit.append(physical)
    
    for i in range(0, len(joint_fit.photometry_data_list)):
        plt.plot(time_visit[i], detrended_visit[i], linestyle='', marker='.', color='grey', alpha=0.2)
    
    combined_times = np.concatenate(time_visit)
    combined_flux = np.concatenate(detrended_visit)
    sort = np.argsort(combined_times)
    combined_times = combined_times[sort]
    combined_flux = combined_flux[sort]
    
    bin_size = len(combined_times) // 30
    bin_time, _ = bin_data(combined_times, bin_size)
    bin_flux, _ = bin_data(combined_flux, bin_size)
    yerr = joint_fit.results['y_err'].nominal_value
    
    plt.errorbar(bin_time, bin_flux, yerr/np.sqrt(bin_size), color='black', linestyle='', marker='.')
    
    plt.plot(physical_time_visit[0], physical_visit[0], color='red')
    plt.axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    plt.ylabel("Normalized flux")
    plt.xlabel("Time from 0.5 phase (hours)")
    plt.title("Phase folded light curve")
    
    plt.gca().text(0.5, 0.95, f"Eclipse depth: {fp*1e6:0.0f}+/-{fp_err*1e6:0.0f}ppm", horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
    
    if save_to_directory is not None:
        path = f"{save_to_directory}/{joint_fit.config.fit_fnpca}_{joint_fit.planet_name}_joint_fit_{joint_fit.config_hash}"
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".pdf", bbox_inches='tight')
    plt.close()