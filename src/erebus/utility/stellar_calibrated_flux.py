from erebus.individual_fit_results import IndividualFitResults
from erebus.wrapped_fits import WrappedFits
import numpy as np
import matplotlib.pyplot as plt
from erebus.utility.utils import get_eclipse_duration
from uncertainties import ufloat
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy import units as units

def get_stellar_flux_calibration_F1500W(visit : IndividualFitResults, fits : WrappedFits, plot_dir : str = None):
    '''
    Given a individual visit result and wrapped fits file, returns the absolute calibrated stellar flux
    Assumes that the pipeline was run starting on uncal data or data otherwise calibrated with the correct parameters
    Follows the procedure outlined in Gordon et al 2025
    
    Optionally saves plots to plot_dir if provided
    '''
    aperture_radius = 5.69
    annulus_inner_radius = 8.63
    annulus_outer_radius = 11.45

    inc = visit.results["inc"].nominal_value
    a = visit.results["a_rstar"].nominal_value
    rp = visit.results["rp_rstar"].nominal_value
    per = visit.results["p"].nominal_value
    duration = get_eclipse_duration(inc, a, rp, per) * 24

    # Use Photutils to calculate the flux
    def get_stellar_flux(frames, center):
        aperture = CircularAperture([center], r=aperture_radius)
        annulus = CircularAnnulus([center], r_in=annulus_inner_radius, r_out=annulus_outer_radius)

        results = []
        for frame in frames:
            aperture_flux = aperture_photometry(frame, aperture)['aperture_sum'][0]
            annulus_flux = aperture_photometry(frame, annulus)['aperture_sum'][0]

            background = annulus_flux * aperture.area / annulus.area

            net_flux = aperture_flux - background
            results.append(net_flux)

        return np.array(results)

    # Only take flux when the planet is eclipsed.
    # Duration is in hours, need it in days for this
    eclipse_start = (visit.predicted_t_sec + visit.results['t_sec_offset']).nominal_value - duration/48
    eclipse_end = (visit.predicted_t_sec + visit.results['t_sec_offset']).nominal_value + duration/48


    # Find proper indices that are within the eclipse
    t = np.array(fits.time - np.min(fits.time))
    inds = np.where((eclipse_start < t) & (eclipse_end > t))
    f = get_stellar_flux(fits.frames[inds], (63, 63))
    t_hours = (t[inds] - np.min(t[inds])) * 24

    if plot_dir is not None:
        plt.plot(fits.time, get_stellar_flux(fits.frames, (63, 63)))
        plt.savefig(plot_dir + "/raw_stellar_flux.png")
        plt.savefig(plot_dir + "/raw_stellar_flux.pdf")
        plt.close()

        plt.imshow(fits.frames[0].T)
        plt.plot([fits.frames[0].shape[0]//2], [fits.frames[0].shape[1]//2], marker='x')
        plt.savefig(plot_dir + "/star_position_in_frame.png")
        plt.savefig(plot_dir + "/star_position_in_frame.pdf")
        plt.close()

        plt.axvline(np.min(t_hours), color='black', linestyle='--')
        plt.axvline(np.max(t_hours), color='black', linestyle='--')
        plt.plot(t_hours, f)
        plt.ylabel("Flux within aperture (DN)")
        plt.xlabel("Time since start of eclipse (hours)")
        plt.savefig(plot_dir + "/flux_during_eclipse.png")
        plt.savefig(plot_dir + "/flux_during_eclipse.pdf")
        plt.close()
        
    # Length of each integration (seconds)
    dt = duration * 3600 / len(fits.time[inds])
    # Aperture area in pixels
    aperture_area = np.pi * aperture_radius**2
    print("Eclipse duration: ", duration, "hours")
    print("Aperture area: ", aperture_area, "square pixels")
    print("Length of integrations: ", dt, "seconds")

    # Aperture correction factor from Gordon et al
    A_corr = ufloat(1.497, 0.019)

    # Inidivudal pixel sizes from MIRI docs
    pixel_solid_angle = (((0.11 * units.arcsec).to(units.rad))**2).value

    # Calibration factor depends on time, using coefficients for MIRI F1500W
    def calibration_factor(t):
        A = 0.3703
        B = -0.0107
        tau = 200    #Days
        t0 = 59720   #In MJD
        match visit.subarray if hasattr(visit, "subarray") else "unspecified":
            case "FULL":
                D_sa = 1
            case "BRIGHTSKY":
                D_sa = 1.005
            case "SUB256":
                D_sa = 0.98
            case "SUB128":
                D_sa = 1
            case "SUB64":
                D_sa = 0.966
            case _:
                print(f"\n\nWARNING: Unhandled subarray: {visit.subarray} defaulting to D_sa = 1\n\n")
                D_sa = 1

        # Units of (MJy/sr) / (DN/s*pixel)
        return (A + B * np.exp(-(t - t0)/tau)) / D_sa

    # Calibration factors within the eclipse
    C = np.array([calibration_factor(t) for t in fits.time[inds]])
    # DN / s * pixel within the eclipse
    # f is in DN / s pixel
    N_ap = f

    print("A_corr: ", A_corr)
    print("Omega_pix: ", pixel_solid_angle, "sr")
    print("Aperture area: ", aperture_area, "pixels")

    print("Average calibration factor: ", np.mean(C), "(MJy/sr) / (DN/s/pixel)")
    print("Average DN/s/pixel: ", np.mean(N_ap))

    calibrated_fluxes = np.array([(N_ap_i * A_corr * pixel_solid_angle * Ci) for N_ap_i, Ci in zip(N_ap, C)])

    # Converted from MJy to mJy
    calibrated_flux = np.mean(calibrated_fluxes) * 1e9

    print("Calibrated flux: ", calibrated_flux, "mJy")

    return calibrated_flux