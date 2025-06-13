from erebus.utility.utils import bin_data
from erebus.utility.utils import get_eclipse_duration
import matplotlib.colors as colors
from scipy.ndimage.filters import uniform_filter1d
from pathlib import Path
from erebus.individual_fit import IndividualFit
from erebus.joint_fit import JointFit
from erebus.joint_fit_results import JointFitResults
from erebus.individual_fit_results import IndividualFitResults
import matplotlib.pyplot as plt
import numpy as np
import inspect
from erebus.mcmc_model import WrappedMCMC
import corner

def plot_fnpca_individual_fit(individual_fit : IndividualFit | IndividualFitResults, save_to_directory : str = None, show : bool = False):
    '''
    Creates an informative plot for an individual fit. Will save both a png and a pdf.
    Expects the individual fit to be done with fnpca/exponential and a linear component.
    '''
    if isinstance(individual_fit, IndividualFit):
        individual_fit = IndividualFitResults(individual_fit)
    
    yerr = individual_fit.results['y_err'].nominal_value
    t_sec = individual_fit.predicted_t_sec
    t_sec_offset = individual_fit.results['t_sec'].nominal_value - t_sec
    rp = individual_fit.results['rp_rstar'].nominal_value
    inc = individual_fit.results['inc'].nominal_value
    a = individual_fit.results['a_rstar'].nominal_value
    per = individual_fit.results['p'].nominal_value
    fp = individual_fit.results['fp'].nominal_value
    fp_err = individual_fit.results['fp'].std_dev
        
    flux_model = individual_fit.flux_model
    systematic_factor = individual_fit.systematic_factor

    raw_time = individual_fit.time
    time = (raw_time - t_sec) * 24 # hours
    flux = individual_fit.raw_flux
    
    bin_size = len(time) // 30
    bin_time, _ = bin_data(time, bin_size)
    bin_flux, _ = bin_data(flux, bin_size)
    bin_yerr = yerr / np.sqrt(bin_size)
    duration = get_eclipse_duration(inc, a, rp, per) * 24
    eclipse_start = t_sec_offset * 24 - duration / 2
    eclipse_end = t_sec_offset * 24 + duration / 2
    
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
    flux_axs[0].axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    flux_axs[0].plot(time, flux_model, color='red')
    flux_axs[0].set_ylabel("Raw flux\n(ppm)")

    # Detrended flux
    detrended_flux = flux / systematic_factor
    bin_detrended_flux, _ = bin_data(detrended_flux, bin_size)
    flux_axs[1].errorbar(time, detrended_flux, yerr, linestyle='', marker='.', alpha = 0.2, color='gray')
    flux_axs[1].errorbar(bin_time, bin_detrended_flux, bin_yerr, linestyle='', marker='.', color='black', zorder=3)
    flux_axs[1].axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    flux_axs[1].plot(time, flux_model / systematic_factor, color='red')
    flux_axs[1].set_ylabel("Detrended flux\n(ppm)")
    flux_axs[1].text(0.5, 0.95, f"Eclipse depth: {fp*1e6:0.0f}+/-{fp_err*1e6:0.0f}ppm", horizontalalignment='center', verticalalignment='top', transform=flux_axs[1].transAxes)

    # Systematic factor
    linear_component = individual_fit.results['a'].nominal_value * raw_time + individual_fit.results['b'].nominal_value + 1
    flux_axs[2].plot(time, systematic_factor, color='red')
    flux_axs[2].plot(time, linear_component, color='black', linestyle='--', label='Linear component')
    flux_axs[2].axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    flux_axs[2].legend()
    flux_axs[2].set_ylabel("Systematc factor\n(ppm)")

    # This is not an Allan deviation plot https://arxiv.org/abs/2504.13238
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
    
    eigenimage_axs[0].imshow(individual_fit.frames[0])

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
        
        eigenvalue_ax.text(0.9, 0.95, f"{individual_fit.pca_variance_ratios[i]*100:0.1f}%", horizontalalignment='center', verticalalignment='top', transform=eigenvalue_ax.transAxes)

        eigenimage = individual_fit.eigenvectors[i]
        eigenimage /= np.max(eigenimage)
        im = eigenimage_axs[i+1].imshow(eigenimage, cmap='bwr', interpolation='nearest', norm = colors.SymLogNorm(0.5, vmin=-1, vmax=1))

    for ax in eigenvalue_axs:
        ax.axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    
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
    
    if show:
        plt.show()    
    
    plt.close()
    
def plot_eigenvectors(individual_fit : IndividualFit, save_to_directory : str = None):
    '''
    Plots the 5 highest ranked eigenimages to a given folder (if provided).
    '''
    for i in range(0, 5):
        eigenimage = individual_fit.eigenvectors[i]
        eigenimage /= np.max(np.abs(eigenimage))
        plt.imshow(eigenimage, cmap='bwr', interpolation='nearest', norm = colors.SymLogNorm(0.5, vmin=-1, vmax=1))
        plt.yticks([])
        plt.xticks([])
        if save_to_directory is not None:
            path = f"{save_to_directory}/{individual_fit.planet_name}_{individual_fit.visit_name}_eigenimage_{(i+1)}"
            plt.savefig(path + ".png", bbox_inches='tight')
            plt.savefig(path + ".pdf", bbox_inches='tight')
        plt.close()

def plot_joint_fit(joint_fit : JointFit | JointFitResults, save_to_directory : str = None, show : bool = False):
    '''
    Creates an informative plot for the joint fit results. Saves as a png and a pdf.
    
    File name starts with planet name, then visit name, then unique hash of config file settings.
    '''
    if isinstance(joint_fit, JointFit):
        joint_fit = JointFitResults(joint_fit)
    
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
    
    detrended_flux_per_visit = joint_fit.detrended_flux_per_visit
    relative_time_per_visit = joint_fit.relative_time_per_visit
    model_time_per_visit = joint_fit.model_time_per_visit
    model_flux_per_visit = joint_fit.model_flux_per_visit
    
    for i in range(0, np.shape(relative_time_per_visit)[0]):
        plt.plot(relative_time_per_visit[i], detrended_flux_per_visit[i], linestyle='', marker='.', color='grey', alpha=0.2)
    
    combined_times = np.concatenate(relative_time_per_visit)
    combined_flux = np.concatenate(detrended_flux_per_visit)
    sort = np.argsort(combined_times)
    combined_times = combined_times[sort]
    combined_flux = combined_flux[sort]
    
    bin_size = len(combined_times) // 30
    bin_time, _ = bin_data(combined_times, bin_size)
    bin_flux, _ = bin_data(combined_flux, bin_size)
    yerr = joint_fit.results['y_err'].nominal_value
    
    plt.errorbar(bin_time, bin_flux, yerr/np.sqrt(bin_size), color='black', linestyle='', marker='.')
    
    plt.plot(model_time_per_visit[0], model_flux_per_visit[0], color='red')
    plt.axvspan(eclipse_start, eclipse_end, color='red', alpha=0.2)
    plt.ylabel("Normalized flux")
    plt.xlabel("Time from 0.5 phase (hours)")
    plt.title("Phase folded light curve")
    
    plt.gca().text(0.5, 0.95, f"Eclipse depth: {fp*1e6:0.0f}+/-{fp_err*1e6:0.0f}ppm", horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
    
    if save_to_directory is not None:
        path = f"{save_to_directory}/{joint_fit.planet_name}_joint_fit_{joint_fit.config_hash}"
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".pdf", bbox_inches='tight')
        
    if show:
        plt.show()
    
    plt.close()

def corner_plot(mcmc : WrappedMCMC, save_to_path : str = None, show : bool = False):
    '''
    Call this on an MCMC model after it has run in order to show and optionally save a corner plot.
    '''
    
    if mcmc.sampler is None:
        print("Cannot make corner plot: MCMC run data isn't cached yet")
        return
    
    labels = mcmc.get_free_params()
    corner.corner(
        mcmc.sampler.get_chain(discard=200, thin=15, flat=True), labels=labels
    )
    
    if save_to_path is not None:
        plt.savefig(save_to_path)
        
    if show:
        plt.show()
        
    plt.close()

def chain_plot(mcmc : WrappedMCMC, save_to_path : str = None, show : bool = False):
    '''
    Call this on an MCMC model after it has run in order to show and optionally save a chain plot.

    '''
    if mcmc.sampler is None:
        print("Cannot make chain plot: MCMC run data isn't cached yet")
        return
    samples = mcmc.sampler.get_chain()
    labels = mcmc.get_free_params()
    ndim = len(labels)
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex = True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step number")

    if save_to_path is not None:
        plt.savefig(save_to_path)
    
    if show:
        plt.show()
        
    plt.close()