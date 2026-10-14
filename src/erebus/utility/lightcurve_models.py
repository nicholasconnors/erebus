import numpy as np
import batman

'''
Adapted from SPCA https://github.com/lisadang27/SPCA/blob/master/SPCA/astro_models.py
'''


def transit_model(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2):
    ecc = np.sqrt(ecosw ** 2 + esinw ** 2)
    w = (np.arctan2(esinw, ecosw) % (2 * np.pi)) * 180 / np.pi
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc    
    params.ecc = ecc
    params.w = w
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)
    t_sec = m.get_t_secondary(params)
    true_anomaly = m.get_true_anomaly() # radians

    return flux, t_sec, true_anomaly

def eclipse(time, t0, per, rp, a, inc, ecc, w, fp, t_sec):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc    
    params.ecc = ecc
    params.w = w
    params.limb_dark = "uniform"
    params.u = []
    params.fp = fp
    params.t_secondary = t_sec
    
    m = batman.TransitModel(params, time, transittype="secondary")
    flux = m.light_curve(params)

    return flux

def planet_flux(time, true_anomaly, t0, per, rp, a, inc, ecosw, esinw, fp, t_sec, A, B):
    ecc = np.sqrt(ecosw ** 2 + esinw ** 2)
    w = (np.arctan2(esinw, ecosw) % (2 * np.pi)) * 180 / np.pi
    
    if ecc == 0.0:
        t = time - t_sec
        freq = 2.0 * np.pi / per
        phi = (freq * t)
    else:
        phi = true_anomaly + w * np.pi / 180.0 + np.pi / 2.0

    phase_variation = 1.0 + A * (np.cos(phi) - 1.0) + B * np.sin(phi)

    eclipse_flux = eclipse(time, t0, per, rp, a, inc, ecc, w, fp, t_sec)
    f_planet = phase_variation * (eclipse_flux - 1.0)

    return f_planet

def model_lightcurve(time, fp, t_sec_offset, t0, per, rp, a, inc, ecosw, esinw, q1, q2, A, B):
    transit, predicted_t_sec, true_anomaly = transit_model(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2)    
    planet = planet_flux(time, true_anomaly, t0, per, rp, a, inc, ecosw, esinw, fp, predicted_t_sec + t_sec_offset, A, B)
    
    return transit + planet