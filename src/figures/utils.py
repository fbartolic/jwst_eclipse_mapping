import numpy as np
import os

from matplotlib import pyplot as plt

import astropy.units as u
from astropy import constants as const

import pickle as pkl
from scipy.optimize import brent
from scipy.special import legendre as P
from numba import jit

import starry

# from starry._plotting import (
#    get_moll_latitude_lines,
#    get_moll_longitude_lines,
# )



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
    if name == "F356W":
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
    #    elif name == "f560w":
    #        miri_filters = jwst.get_miri_filter_wheel()
    #        filt = np.stack([miri_filters[0].wl, miri_filters[0].throughput])
    #        return filt
    #    elif name == "f770w":
    #        miri_filters = jwst.get_miri_filter_wheel()
    #        filt = np.stack([miri_filters[1].wl, miri_filters[1].throughput])
    #        return filt
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


#
# def generate_spotted_map(
#    T_nightside,
#    T_dayside,
#    spot_lons,
#    spot_lats,
#    spot_radii,
#    spot_teffs,
#    map_wavelengths,
#    ydeg=25,
# ):
#    Nw = len(map_wavelengths)
#    map_planet = starry.Map(ydeg=ydeg, nw=Nw)
#    I0 = map_planet.intensity()[0]
#
#    # Put one large spot on nightside at nightside temperature
#    I_nightside = np.pi * planck(T_nightside, map_wavelengths)
#    contrast = 1 - I_nightside.value / I0
#
#    map_planet.spot(
#        contrast=contrast, radius=90, lat=0.0, lon=180.0,
#    )
#
#    # Put one large spot on dayside to simulate insolation
#    I_dayside = np.pi * planck(T_dayside, map_wavelengths)
#    contrast = 1 - I_dayside.value / I0
#
#    map_planet.spot(
#        contrast=contrast, radius=90, lat=0.0, lon=0.0,
#    )
#
#    # Add spots
#    for i in range(len(spot_lons)):
#        spot_int = np.pi * planck(spot_teffs[i], map_wavelengths)
#        # Subtract contribution to intensity from previous spots
#        c_diff = map_planet.intensity(lat=spot_lats[i], lon=spot_lons[i])[0] / I0 - 1
#        contrast = 1.0 - (spot_int.value / I0 - c_diff)
#
#        map_planet.spot(
#            contrast=contrast, radius=spot_radii[i], lat=spot_lats[i], lon=spot_lons[i],
#        )
#    return map_planet
#

# def generate_map_with_ellipsoidal_spot(
#    T_nightside, T_dayside, T_spot, map_wavelengths, a=30, b=15, ydeg=25, force_psd=False,
# ):
#    Nw = len(map_wavelengths)
#    map_planet = starry.Map(ydeg=ydeg, nw=Nw)
#
#    I_nightside = np.pi * planck(T_nightside, map_wavelengths)
#    I_dayside = np.pi * planck(T_dayside, map_wavelengths)
#    I_spot = np.pi * planck(T_spot, map_wavelengths)
#
#    # Add ellipsoidal spot (very hacky)
#    X, Y = np.meshgrid(np.linspace(-180, 180, 200), np.linspace(-90, 90, 200))
#    ellipse = lambda x, y, a, b: x ** 2 / a ** 2 + y ** 2 / b ** 2 < 1
#    mask_el = ellipse(X, Y, a, b)
#
#    x_list = []
#    map_tmp = starry.Map(ydeg)
#
#    for i in range(len(map_wavelengths)):
#        # Dayside and spot
#        Z = I_dayside[i].value * np.ones_like(X)
#        Z[mask_el] = I_spot[i].value
#
#        # Nightside
#        mask_nightside = np.logical_or(X > 90, X < -90)
#        Z[mask_nightside] = I_nightside[i].value
#
#        map_tmp.load(Z, smoothing=1.5 / 25, force_psd=force_psd)
#        x_list.append(map_tmp._y * map_tmp.amp)
#
#    x = np.stack(x_list).T
#    map_planet[1:, :, :] = x[1:] / x[0]
#    map_planet.amp = x[0]
#
#    return map_planet


# def compute_flux(
#    t, params_s, params_p, map_star, map_planet, filt, map_wavelengths, texp=1.02183 * u.s,
# ):
#
#    #    map_planet.obl = params_p["obl"]
#    #    map_planet.inc = params_p["inc"]
#
#    # We need to scale the map ampltidues by R^2 to get fluxes in physical units
#    Rs = params_s["r"].to(u.m).value
#    Rp = params_p["r"].to(u.m).value
#    map_planet.amp *= (Rp / Rs) ** 2
#
#    # Initialize system
#    star = starry.Primary(map_star, r=params_s["r"], m=params_s["m"])
#    #    star.map[1] = params_s["u"][0]
#    #    star.map[2] = params_s["u"][1]
#
#    planet = starry.Secondary(
#        map_planet,
#        r=params_p["r"],
#        porb=params_p["porb"],
#        prot=params_p["prot"],
#        t0=params_p["t0"],
#        inc=params_p["inc"],
#        #        ecc=params_p["ecc"],
#        #        omega=params_p["omega"],
#        #        Omega=params_p["Omega"],
#        theta0=180,
#    )
#
#    sys = starry.System(star, planet, texp=(texp.to(u.d)).value, oversample=9, order=0)
#
#    # Compute flux
#    A = sys.design_matrix(t)
#
#    ftrue = sys.flux(t)
#    x = np.concatenate([map_star.amp * map_star._y, map_planet.amp * map_planet._y,])
#    ftrue = np.tensordot(A, x, axes=1)
#
#    # Compute flux for uniform map
#    x_unif = np.copy(x)
#    x_unif[map_star.Ny + 1 :] = np.zeros_like(map_planet._y[1:])
#    f_dayside = planet.map.flux(theta=0)[0]
#    x_unif[map_star.Ny] = f_dayside
#
#    ftrue_unif = np.tensordot(A, x_unif, axes=1)
#
#    wav_filt = filt[0]
#    throughput = filt[1]
#
#    # Interpolate filter throughput to map wavelength grid
#    throughput_interp = np.interp(map_wavelengths, wav_filt, throughput)
#
#    # Integrate flux over bandpass
#    ftrue_band = np.trapz(ftrue * throughput_interp, axis=1)
#    ftrue_unif_band = np.trapz(ftrue_unif * throughput_interp, axis=1)
#
#    norm = np.max(ftrue_band)
#
#    # Rescale the amplitude of the planet map back to its original value
#    map_planet.amp *= (Rs / Rp) ** 2
#
#    return ftrue_band / norm, ftrue_unif_band / norm, sys
#
#

#
# def generate_lightcurve(t, ftrue, ftrue_unif, snr=1.46):
#    mask_ecl = np.logical_and(t > -0.2, t < 0.2)
#
#    eclipse_depth = (np.max(ftrue[mask_ecl]) - np.min(ftrue[mask_ecl])) / np.max(
#        ftrue[mask_ecl]
#    )
#    sigma = eclipse_depth / snr
#    fobs = ftrue + np.random.normal(0, sigma, size=len(ftrue))
#    ferr = sigma * np.ones_like(fobs)
#
#    tb = Table()
#    tb["t"] = t
#    tb["fobs"] = fobs
#    tb["ferr"] = ferr
#    mask_in = mask_ingress_egress(ftrue_unif[mask_ecl], option="ingress")
#    mask_eg = mask_ingress_egress(ftrue_unif[mask_ecl], option="egress")
#    tecl = t[mask_ecl]
#    tb["mask_in"] = np.logical_and(t > tecl[mask_in][0], t < tecl[mask_in][-1])
#    tb["mask_eg"] = np.logical_and(t > tecl[mask_eg][0], t < tecl[mask_eg][-1])
#
#    return tb
#
#
# def generate_simulated_lightcurve(
#    map_planet, params_s, params_p, filt, map_wavelengths, texp, snr=16
# ):
#    # Filter througput interpolated to map_wavelengths
#    thr_interp = np.interp(map_wavelengths, filt[0], filt[1])
#
#    # Initialize star map
#    map_star = starry.Map(ydeg=1, udeg=2, nw=len(map_wavelengths))
#    Llam = (4 * np.pi) * np.pi * planck(params_s["T"].value, map_wavelengths,)
#    map_star.amp = Llam / 4
#
#    # Generate high cadence lightcurve  excluding transit
#    delta_t = params_p["porb"] / 2 + 0.1 * u.d
#    npts = int((2 * delta_t.to(u.s)) / (texp))  # total number of data points
#    t = np.linspace(-delta_t.value, delta_t.value, npts)
#
#    # Masks for eclipse, transit and phase curves
#    mask_ecl = np.logical_and(t < 0.1, t > -0.1)
#    mask_tran = np.abs(t) > 1.05
#    mask_phase = ~np.logical_or(mask_ecl, mask_tran)
#
#    t_ecl = t[mask_ecl]
#    t_tran = t[mask_tran][::10]  # subsample for performance reasons
#    t_phase = t[mask_phase][::5]
#
#    t_combined = np.sort(np.concatenate([t_ecl, t_phase]))
#
#    # Generate light curve
#    fsim, fsim_unif, sys = compute_flux(
#        t_combined, params_s, params_p, map_star, map_planet, filt, map_wavelengths, texp=texp,
#    )
#
#    lc = generate_lightcurve(t_combined, fsim, fsim_unif, snr=snr,)
#
#    return t_combined, fsim, fsim_unif, sys, lc
#


#def compute_design_matrix(t, params_p, params_s, texp, ydeg):
#    # Star map parameters
#    star = starry.Primary(
#        starry.Map(ydeg=1, udeg=2),
#        r=params_s["r"].value,
#        m=params_s["m"].value,
#        length_unit=u.Rsun,
#        mass_unit=u.Msun,
#    )
#    #    star.map[1] = params_s["u"][0]
#    #    star.map[2] = params_s["u"][1]
#
#    planet = starry.Secondary(
#        starry.Map(ydeg=ydeg, inc=params_p["inc"].value,),
#        #        ecc=params_p["ecc"],
#        #        omega=params_p["omega"].value,
#        r=params_p["r"].value,
#        porb=params_p["porb"].value,
#        prot=params_p["prot"].value,
#        t0=params_p["t0"].value,
#        inc=params_p["inc"].value,
#        theta0=180,
#        length_unit=u.Rsun,
#        angle_unit=u.deg,
#        time_unit=u.d,
#    )
#    sys_fit = starry.System(star, planet, texp=(texp.to(u.d)).value)
#
#    # Design matrix
#    A_full = sys_fit.design_matrix(t)
#    A = A_full[:, 4:]
#
#    return A, A_full
#

# def plot_model(
#    lc,
#    ftrue_unif,
#    samples=None,
#    map_params=None,
#    fig_title=None,
#    inner_pad=2,
#    outer_pad=3,
#    ylim=None,
# ):
#    fig, ax = plt.subplots(
#        2, 2, figsize=(11, 7), gridspec_kw={"wspace": 0.1, "height_ratios": [3, 1]},
#    )
#    for a in ax[0, :]:
#        a.errorbar(
#            lc["t"] * u.d.to(u.min),
#            lc["fobs"],
#            lc["ferr"],
#            fmt="o",
#            color="black",
#            alpha=0.1,
#        )
#
#    # Residuals
#    if map_params is not None:
#        res = lc["fobs"] - map_params["fpred"]
#    else:
#        res = lc["fobs"] - np.median(samples["fpred"], axis=0)
#
#    print("chi-sq: ", np.sum(res ** 2 / np.array(lc["ferr"]) ** 2))
#
#    for a in ax[1, :]:
#        a.errorbar(
#            lc["t"] * u.d.to(u.min),
#            res / lc["ferr"][0],
#            lc["ferr"] / lc["ferr"][0],
#            fmt="o",
#            color="black",
#            alpha=0.1,
#        )
#
#    for a in ax[:, 0]:
#        a.set_xlim(
#            lc["t"][lc["mask_in"]][0] * u.d.to(u.min) - outer_pad,
#            lc["t"][lc["mask_in"]][-1] * u.d.to(u.min) + inner_pad,
#        )
#    for a in ax[:, 1]:
#        a.set_xlim(
#            lc["t"][lc["mask_eg"]][0] * u.d.to(u.min) - inner_pad,
#            lc["t"][lc["mask_eg"]][-1] * u.d.to(u.min) + outer_pad,
#        )
#
#    for a in ax[1, :]:
#        a.set_xlabel("time [minutes]")
#        a.set(ylim=(-4, 4))
#
#    for a in ax[0, :]:
#        a.set_xticklabels([])
#
#    # Make broken axis
#    for a in ax:
#        a[0].spines["right"].set_visible(False)
#        a[1].spines["left"].set_visible(False)
#        a[1].tick_params(axis="y", colors=(0, 0, 0, 0))
#
#        d = 0.01
#        kwargs = dict(transform=a[0].transAxes, color="k", clip_on=False)
#        a[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
#        a[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#        kwargs.update(transform=a[1].transAxes)
#        a[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
#        a[1].plot((-d, +d), (-d, +d), **kwargs)
#
#    if map_params is not None:
#        for a in ax[0, :]:
#            a.plot(lc["t"] * u.d.to(u.min), map_params["fpred"], "C1-", lw=2.0)
#    else:
#        for i in range(20):
#            for a in ax[0, :]:
#                a.plot(
#                    lc["t"] * u.d.to(u.min),
#                    samples["fpred"][i],
#                    "C1-",
#                    lw=2.0,
#                    alpha=0.1,
#                )
#
#    for a in ax[0, :]:
#        a.plot(lc["t"] * u.d.to(u.min), ftrue_unif, "C0-")
#        a.set_ylim(ylim)
#
#    for a in ax.reshape((-1)):
#        a.grid(alpha=0.5)
#
#    ax[0, 0].set(ylabel="Flux")
#    ax[1, 0].set(ylabel="Res")
#
#    fig.suptitle(fig_title, fontsize=16)
#

# def fit_model_pm(
#    model, init_vals=None, nwarmup=1000, nsamples=1000, nchains=1, fit_map=False,
# ):
#    if not fit_map:
#        with model:
#            trace = pm.sample(
#                draws=nsamples,
#                start=init_vals,
#                init="adapt_diag",
#                chains=nchains,
#                cores=nchains,
#                target_accept=0.99,
#            )
#
#            prior = pm.sample_prior_predictive()
#            posterior_predictive = pm.sample_posterior_predictive(trace)
#
#            pm_data = az.from_pymc3(
#                trace=trace, prior=prior, posterior_predictive=posterior_predictive,
#            )
#
#        return trace, pm_data
#
#    else:
#        with model:
#            start = pmx.optimize(start=model.test_point, vars=[model.fs_delta],)
#            map_params = pmx.optimize(start=start)
#
#        return map_params
#

# def fit_numpyro_model(
#    model, init_vals=None, nwarmup=1000, nsamples=1000, nchains=1,
# ):
#    nuts_kernel = NUTS(
#        model,
#        dense_mass=False,
#        init_strategy=init_to_value(values=init_vals),
#        target_accept_prob=0.99,
#    )
#
#    mcmc = MCMC(
#        nuts_kernel, num_warmup=nwarmup, num_samples=nsamples, num_chains=nchains,
#    )
#    rng_key = random.PRNGKey(0)
#    mcmc.run(rng_key)
#    samples = mcmc.get_samples()
#    samples_np = {key: np.array(samples[key]) for key in samples.keys()}
#    samples_az = az.from_numpyro(
#        mcmc, posterior_predictive={"obs": np.array(samples["fpred"])}
#    )
#
#    return samples_np, samples_az
#
#
# def fit_numpyro_laplace(rng_key, model, nsteps=1200, lr=0.02):
#    guide = AutoLaplaceApproximation(model)
#    svi = SVI(model, guide, optim.Adam(lr), Trace_ELBO())
#    rng_key, rng_key_ = random.split(rng_key)
#    svi_state = svi.init(rng_key_)
#    rng_key, rng_key_ = random.split(rng_key)
#    svi_result = svi.run(rng_key_, nsteps)
#    rng_key, rng_key_ = random.split(rng_key)
#    samples = guide.sample_posterior(rng_key, svi_result.params, (50,))
#
#    return samples, svi_result
#


# def max_intensity_position(ydeg_inf, samples, bounds=None, nsamples=300):
#    """
#    For each sample find local maximum to get spot position,
#    then compute emitted power from the spot.
#    """
#    lat_list = []
#    lon_list = []
#
#    map = starry.Map(ydeg_inf)
#
#    for n in np.random.randint(len(samples["x"]), size=nsamples):
#        x = samples["x"][n]
#        map[1:, :] = x[1:] / x[0]
#        map.amp = x[0]
#
#        # Find maximum
#        map[1:, :] = -map[1:, :]
#        lat, lon, _ = map.minimize(oversample=2, ntries=2, bounds=bounds)
#        map[1:, :] = -map[1:, :]
#
#        # Convert to East longitude
#        if lon < 0:
#            lon = 360 - np.abs(lon)
#
#        lat_list.append(lat)
#        lon_list.append(lon)
#
#    return (
#        np.array(lat_list),
#        np.array(lon_list),
#    )
#

# def lon_lat_to_mollweide(lon, lat):
#    lat *= np.pi / 180
#    lon *= np.pi / 180
#
#    f = lambda x: 2 * x + np.sin(2 * x) - np.pi * np.sin(lat)
#    theta = optimize.newton(f, 0.3)
#
#    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
#    y = np.sqrt(2) * np.sin(theta)
#
#    return x, y
#


# def plot_grid_lines(ax, alpha=0.6):
#    """
#    Code from https://github.com/rodluger/starry/blob/0546b4e445f6570b9a1cf6e33068e01a96ecf20f/starry/maps.py.
#    """
#    ax.axis("off")
#
#    borders = []
#    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
#    y = np.sqrt(2) * np.sqrt(1 - (x / (2 * np.sqrt(2))) ** 2)
#    borders += [ax.fill_between(x, 1.1 * y, y, color="w", zorder=-1)]
#    borders += [ax.fill_betweenx(0.5 * x, 2.2 * y, 2 * y, color="w", zorder=-1)]
#    borders += [ax.fill_between(x, -1.1 * y, -y, color="w", zorder=-1)]
#    borders += [ax.fill_betweenx(0.5 * x, -2.2 * y, -2 * y, color="w", zorder=-1)]
#
#    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
#    a = np.sqrt(2)
#    b = 2 * np.sqrt(2)
#    y = a * np.sqrt(1 - (x / b) ** 2)
#    borders = [None, None]
#    (borders[0],) = ax.plot(x, y, "k-", alpha=1, lw=1.5)
#    (borders[1],) = ax.plot(x, -y, "k-", alpha=1, lw=1.5)
#    lats = get_moll_latitude_lines()
#    latlines = [None for n in lats]
#    for n, l in enumerate(lats):
#        (latlines[n],) = ax.plot(l[0], l[1], "k-", lw=0.8, alpha=alpha, zorder=100)
#    lons = get_moll_longitude_lines()
#    lonlines = [None for n in lons]
#    for n, l in enumerate(lons):
#        (lonlines[n],) = ax.plot(l[0], l[1], "k-", lw=0.8, alpha=alpha, zorder=100)
#    ax.fill_between(x, y, y + 10, color="white")
#    ax.fill_between(x, -(y + 10), -y, color="white")


# def plot_pixel_map(ydeg_inf, p, s=30):
#    npix = len(p)
#
#    map = starry.Map(ydeg=ydeg_inf)
#    lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
#    #     lon = (lon + 180 + rotate_ang) % 360  - 180
#
#    x_mol = np.zeros(npix)
#    y_mol = np.zeros(npix)
#
#    for idx, (lo, la) in enumerate(zip(lon, lat)):
#        x_, y_ = lon_lat_to_mollweide(lo, la)
#        x_mol[idx] = x_
#        y_mol[idx] = y_
#
#    fig, ax = plt.subplots(figsize=(6, 4))
#
#    order = np.argsort(p)
#    im1 = ax.scatter(
#        x_mol[order],
#        y_mol[order],
#        s=s,
#        c=p[order],
#        ec="none",
#        cmap="OrRd",
#        marker="o",
#        norm=colors.Normalize(vmin=0),
#    )
#
#    dx = 2.0 / 300
#    extent = (
#        -(1 + dx) * 2 * np.sqrt(2),
#        2 * np.sqrt(2),
#        -(1 + dx) * np.sqrt(2),
#        np.sqrt(2),
#    )
#    ax.axis("off")
#    ax.set_xlim(-2 * np.sqrt(2) - 0.05, 2 * np.sqrt(2) + 0.05)
#    ax.set_ylim(-np.sqrt(2) - 0.05, np.sqrt(2) + 0.05)
#
#    ax.set_aspect("equal")
#
#    cbar_ax = fig.add_axes([0.92, 0.29, 0.02, 0.4])
#    fig.colorbar(im1, cax=cbar_ax)
#
#    # Plot grid lines
#    plot_grid_lines(ax, alpha=0.3)
#


def get_mean_map(
    ydeg,
    samples_ylm,
    projection="Mollweide",
    inc=90,
    theta=0.0,
    nsamples=15,
    resol=300,
    return_std=False,
):
    """
    Given a set of samples from a posterior distribution over the spherical
    harmonic coefficients, the function computes a mean map in pixel space.

    Args:
        ydeg (int): Degree of the map.
        samples_ylm (list): List of (amplitude weighted) Ylm samples.
        projection (str, optional): Map projection. Defaults to "Mollweide".
        inc (int, optional): Map inclination. Defaults to 90.
        theta (float, optional): Map phase. Defaults to 0.0.
        nsamples (int, optional): Number of samples to use to compute the
            mean. Defaults to 15.
        resol (int, optional): Map resolution. Defaults to 300.
        return_std(bool, optional): If true, the function returns both the
        mean map and the standard deviation as a tuple. By default False.

    Returns:
        ndarray: Pixelated map in the requested projection. Shape (resol, resol).
    """
    if len(samples_ylm) < nsamples:
        raise ValueError(
            "Length of Ylm samples list has to be greater than", "nsamples"
        )
    imgs = []
    map = starry.Map(ydeg=ydeg)
    map.inc = inc

    for n in np.random.randint(0, len(samples_ylm), nsamples):
        x = samples_ylm[n]
        map.amp = x[0]
        map[1:, :] = x[1:] / map.amp

        if projection == "Mollweide" or projection == "rect":
            im = map.render(projection=projection, res=resol)
        else:
            im = map.render(theta=theta, res=resol)
        imgs.append(im)

    if return_std:
        return np.nanmean(imgs, axis=0), np.nanstd(imgs, axis=0)
    else:
        return np.nanmean(imgs, axis=0)


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
