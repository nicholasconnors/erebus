from src.utility.utils import bin_data
from src.utility.utils import get_eclipse_duration
import matplotlib.colors as colors
from scipy.ndimage.filters import uniform_filter1d
from pathlib import Path
from src.individual_fit import IndividualFit
import matplotlib.pyplot as plt
import numpy as np

def plot_fnpca_individual_fit(individual_fit : IndividualFit, save_to_path : str = None):
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
    allan_ax.set_title("Allan deviation plot")
    
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
    allan_ax.set_ylabel("RMS")
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

    if save_to_path is not None:
        path = Path(save_to_path).stem
        plt.savefig(f"{path}.png")
        plt.savefig(f"{path}.pdf")
    plt.show()
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
            path = f"{save_to_directory}/{individual_fit.planet.name}_{individual_fit.visit_name}_eigenimage_{(i+1)}"
            plt.savefig(path + ".png")
            plt.savefig(path + ".pdf")
        plt.show()
        plt.close()