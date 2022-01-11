import numpy as np
import os

import astropy.units as u
from astropy import constants as const

import pickle as pkl
from scipy.optimize import brent
from scipy.special import legendre as P
from numba import jit

import starry


starry.config.lazy = False
starry.config.quiet = True


HOMEPATH = os.path.abspath(os.path.split(__file__)[0])


def get_smoothing_filter(ydeg, sigma=0.1):
    """
    Returns a smoothing matrix which applies an isotropic Gaussian beam filter
    to a spherical harmonic coefficient vector. This helps suppress ringing
    artefacts around spot like features. The standard deviation of the Gaussian
    filter controls the strength of the smoothing. Features on angular scales
    smaller than ~ 1/sigma are strongly suppressed.

    Args:
        ydeg (int): Degree of the map.
        sigma (float, optional): Standard deviation of the Gaussian filter.
            Defaults to 0.1.

    Returns:
        ndarray: Diagonal matrix of shape (ncoeff, ncoeff) where ncoeff = (l + 1)^2.
    """
    l = np.concatenate([np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)])
    s = np.exp(-0.5 * l * (l + 1) * sigma ** 2)
    S = np.diag(s)
    return S


def load_filter(name="f444w"):
    if name == "f356w":
        path = os.path.join(HOMEPATH, "../../data/filter_files/F356W_ModAB_mean.csv")
        dat = np.loadtxt(path, skiprows=1, delimiter=",")
        return dat.T
    elif name == "f322w2":
        path = os.path.join(
            HOMEPATH, "../../data/filter_files/F322W2_filteronly_ModAB_mean.txt",
        )
        dat = np.loadtxt(path)
        return dat.T
    elif name == "f444w":
        path = os.path.join(HOMEPATH, "../../data/filter_files/F444W_ModAB_mean.csv")
        dat = np.loadtxt(path, skiprows=1, delimiter=",")
        return dat.T
    else:
        raise ValueError("Filter name not recognized.")


def planck(T, lam):
    """
    Planck function.

    Args:
        T (float): Blackbody temperature in Kelvin. 
        lam (float or ndarray): Wavelength in um (microns).

    Returns:
        float or ndarray: Planck function.
    """
    h = const.h
    c = const.c
    kB = const.k_B

    T *= u.K
    lam *= u.um

    return (2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * kB * T)) - 1.0)).to(
        u.W / u.m ** 3
    ) / u.sr


def integrate_planck_over_filter(T, filt):
    """
    Integrate Planck curve over a photometric filter.
    """
    wav_filt = filt[0]
    throughput = filt[1]
    I = planck(T, wav_filt).value
    return np.trapz(I * throughput, x=wav_filt * u.um.to(u.m)) * u.W / u.m ** 2 / u.sr


@jit
def cost_fn_scalar_int(T, target_int, lam, thr, h, c, kB):
    I = 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * kB * T)) - 1.0)
    I_int = np.trapz(I * thr, x=lam)
    return (I_int - target_int) ** 2


def inverse_integrate_planck_over_filter(intensity, filt):
    """
    Inverse transform of `integrate_planck_over_filter`. 

    Args:
        intensity(float): Integral of Planck curve over some bandpass.
        lam (ndarray): Filter wavelengths in um (microns).
        throughput (ndarray): Filter throughput.

    Returns:
        float: Planck temperature.
    """
    h = const.h.value
    c = const.c.value
    kB = const.k_B.value

    lam = filt[0] * u.um.to(u.m)

    if not np.any(np.isnan(intensity)):
        return brent(
            cost_fn_scalar_int,
            args=(intensity, lam, filt[1], h, c, kB),
            brack=(10, 5000),
            tol=1e-04,
            maxiter=400,
        )
    else:
        return np.nan


def inferred_intensity_to_bbtemp(I_planet_raw, filt, params_s, params_p):
    """
    Convert inferred starry intensity map to a BB temperature map.
    """
    wav_filt = filt[0]
    throughput = filt[1]

    # Star spectral radiance integrated over solid angle and bandpass
    I_star = np.pi * integrate_planck_over_filter(params_s["T"].value, filt,)

    # Rescale the intensity of the planet map to physical units
    I_planet = I_planet_raw * I_star * (params_s["r"] / params_p["r"]) ** 2

    # Plot temperature map of the planet
    bbtemp_map_inf = np.copy(I_planet[:, :].value)

    for i in range(I_planet.shape[0]):
        for j in range(I_planet.shape[1]):
            bbtemp_map_inf[i, j] = inverse_integrate_planck_over_filter(
                I_planet[i, j].value, filt
            )
    return bbtemp_map_inf


@jit
def cost_fn_spectral_rad(T, target_int, lam, h, c, kB):
    I = 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * kB * T)) - 1.0)
    return np.sum((I - target_int) ** 2)


def __spectral_radiance_to_bbtemp(intensity, lam):
    """
    Fit a Planck curve to a vector of spectral radiances and return the
    best-fit temperature.

    Args:
        intensity (ndarray): Spectral radiance evaluated at wavelengths `lam`, 
        in units of W/m**3.
        lam (ndarray): Corresponding wavelengths in um (microns).

    Returns:
        float: Temperature of the best-fit Planck curve.
    """
    h = const.h.value
    c = const.c.value
    kB = const.k_B.value

    lam *= u.um
    intensity *= u.W / u.m ** 3

    if not np.any(np.isnan(intensity)):
        return brent(
            cost_fn_spectral_rad,
            args=(intensity, lam.to(u.m).value, h, c, kB),
            brack=(10, 5000),
            tol=1e-04,
            maxiter=400,
        )
    else:
        return np.nan


def starry_intensity_to_bbtemp(
    int_array, map_wavelengths,
):
    """
    Convert the intensity of a starry map rendered in a specific projection 
    viewing angles) to a blackbody temperature map in the same projection.
    """
    bbtemp = np.copy(int_array[0, :, :])

    for i in range(int_array.shape[1]):
        for j in range(int_array.shape[2]):
            if np.all(np.isnan(int_array[:, i, j])):
                bbtemp[i, j] = np.nan
            else:
                bbtemp[i, j] = __spectral_radiance_to_bbtemp(
                    int_array[:, i, j] / np.pi, map_wavelengths
                )
    return bbtemp


def BInv(ydeg=15, npts=1000, eps=1e-9, sigma=15, **kwargs):
    """
    Return the matrix B+. This expands the 
    band profile `b` in Legendre polynomials.
    """
    theta = np.linspace(0, np.pi, npts)
    cost = np.cos(theta)
    B = np.hstack(
        [np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1) for l in range(ydeg + 1)]
    )
    BInv = np.linalg.solve(B.T @ B + eps * np.eye(ydeg + 1), B.T)
    l = np.arange(ydeg + 1)
    i = l * (l + 1)
    S = np.exp(-0.5 * i / sigma ** 2)
    BInv = S[:, None] * BInv
    return BInv


def get_band_ylm(ydeg, nw, amp, lat, sigma):
    """
    Get the Ylm expansion of a Gassian band at fixed latitude.
    """
    # off center Gaussian spot in Polar frame
    gauss = (
        lambda x, mu, sig: 1
        / (sig * np.sqrt(2 * np.pi))
        * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))
    )

    theta = np.linspace(0, np.pi, 1000)
    b = gauss(theta, np.pi / 2 - lat, sigma)

    yband_m0 = BInv(ydeg=ydeg) @ b
    yband_m0 /= yband_m0[0]

    map = starry.Map(ydeg=ydeg, nw=nw)

    if nw is None:
        map[1:, 0] = yband_m0[1:]
    else:
        map[1:, 0, :] = np.repeat(yband_m0[1:, None], nw, axis=1)

    map.rotate([1, 0, 0], -90.0)

    return amp * map._y


def add_band(map, amp, relative=True, sigma=0.1, lat=0.0):
    """
    Add an azimuthally symmetric band to a starry map.
    """
    if amp is not None:
        amp, _ = map._math.vectorize(map._math.cast(amp), np.ones(map.nw))
        # Normalize?
        if not relative:
            amp /= map.amp

    # Parse remaining kwargs
    sigma, lat = map._math.cast(sigma, lat)

    # Get the Ylm expansion of the band
    yband = get_band_ylm(map.ydeg, map.nw, amp, lat * map._angle_factor, sigma)
    y_new = map._y + yband
    amp_new = map._amp * y_new[0]
    y_new /= y_new[0]

    # Update the map and the normalizing amplitude
    map._y = y_new
    map._amp = amp_new

    return map


def load_params_from_pandexo_output(path_to_pandexo_file, planet="hd189"):
    # Open pickle file
    with open(path_to_pandexo_file, "rb") as handle:
        model = pkl.load(handle)

    # Get spectrum if desired
    wave = model["FinalSpectrum"]["wave"]
    spectrum = model["FinalSpectrum"]["spectrum"]
    error = model["FinalSpectrum"]["error_w_floor"]
    randspec = model["FinalSpectrum"]["spectrum_w_rand"]

    SNR = float(np.trapz(spectrum / error, x=wave))
    texp = model["timing"]["Time/Integration incl reset (sec)"] * u.s

    n_eclipses = int(model["timing"]["Number of Transits"])
    nint = model["timing"]["APT: Num Groups per Integration"]
    filter_name = model["PandeiaOutTrans"]["input"]["configuration"]["instrument"][
        "filter"
    ]

    return {"snr": SNR, "texp": texp, "filter_name": filter_name}
