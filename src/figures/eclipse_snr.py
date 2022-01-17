"""Scripts taken from https://github.com/rodluger/planetplanet/blob/master/planetplanet/detect/jwst.py"""
import numpy as np
import astropy.units as u

from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from utils import (
    load_filter,
    planck,
)


def jwst_background(wl):
    """
    Calculates the JWST thermal background from (Glasse et al. 2015)

    Args:
        wl (ndarray): Wavelengths [:math:`\mu \mathrm{m}`]

    Returns:
        ndarray: Background flux [:math:`\mathrm{W/m}^2 / \mu \mathrm{m}`]
    """

    # Solid angle
    omega_background = np.pi * (0.42 / 206265.0 * wl / 10.0) ** 2.0

    # Number of background grey body components (see table 1 of Glasse et al.):
    nback = 7
    emiss = [4.2e-14, 4.3e-8, 3.35e-7, 9.7e-5, 1.72e-3, 1.48e-2, 1.31e-4]
    tb = [5500.0, 270.0, 133.8, 71.0, 62.0, 51.7, 86.7]

    # Set up a vector for computing the background flux emissivity:
    Fback = np.zeros_like(wl)

    # Sum over the background components (insert 1 steradian for omega):
    for i in range(nback):
        Fback += emiss[i] * planck(tb[i], wl).value / 1e06 * omega_background

    return Fback


def photon_rate(lam, flux, wl, throughput, atel=25.0):
    """
    Compute the photon count rate registered by the detector.

    Args:
        lam (ndarray): High-res wavelengths [:math:`\mu \mathrm{m}`]
        flux (ndarray): Spectral flux density \
           [:math:`\mathrm{W/m}^2 / \mu \mathrm{m}`]
        wl (ndarray): Filter wavelengths [:math:`\mu \mathrm{m}`]
        throughput (ndarray): Filter throughput.
        atel (float): Aperture area [:math:`\mathrm{m}^2`]. 
    
    Returns:
        ndarray: Photon count rate [:math:`\mathrm{s}^{-1}`]
    """

    # (Speed of light) * (Planck's constant)
    hc = 1.986446e-25  # h*c (kg*m**3/s**2)

    dlam = lam[1:] - lam[:-1]
    dlam = np.hstack([dlam, dlam[-1]])

    # interpolate filter throughout to HR grid
    T = np.interp(lam, wl, throughput)

    cphot = np.sum(flux * dlam * (lam * 1e-6) / hc * T * atel, axis=flux.ndim - 1)

    return cphot


def estimate_eclipse_snr(
    lam_filter,
    throughput,
    tint=36.4 * 60.0,
    nout=4.0,
    lammin=1.0,
    lammax=30.0,
    Tstar=2560.0,
    Tplan=400.0,
    Rs=0.12,
    Rp=1.086,
    d=12.2,
    atel=25.0,
    verbose=True,
    plot=True,
    thermal=True,
):
    """
    Estimate the signal-to-noise on the detection of secondary eclipses in
    JWST/MIRI photometric filters.

    Args:
        lam_filter (ndarray): Wavelengths of the filter [:math:`\mu \mathrm{m}`]
        throughput (ndarray): Filter throughput.
        tint (float): Integration time [:math:`\mathrm{s}`].
        nout (int): Out-of-transit time observed [transit durations].
        Tstar (float): Stellar effective temperature [:math:`\mathrm{K}`].
        Tplan (float): Planetary equilibrium temperature [:math:`\mathrm{K}`].
        Rs (float): Stellar radius [:math:`\mathrm{R_\odot}`].
        Rp (float): Planetary radius [:math:`\mathrm{R_\mathrm{Earth}}`].
        d (float): Distance to the planet [pc].
        atel (float): Aperture area [:math:`\mathrm{m}^2`].
        thermal (bool): Include thermal background.
    """

    # Generate high-res wavelength grid
    Nlam = 1000
    lam = np.linspace(lam_filter[0], lam_filter[-1], Nlam)
    dlam = lam[1:] - lam[:-1]
    dlam = np.hstack([dlam, dlam[-1]])

    # Calculate BB intensities for the star and planet [W/m^2/um/sr]
    Bstar = planck(Tstar, lam).value / 1e06
    Bplan = planck(Tplan, lam).value / 1e06

    # solid angle in steradians
    omega_star = np.pi * (Rs * u.Rsun.in_units(u.km) / (d * u.pc.in_units(u.km))) ** 2.0
    omega_planet = (
        np.pi * (Rp * u.Rearth.in_units(u.km) / (d * u.pc.in_units(u.km))) ** 2.0
    )

    # fluxes at earth [W/m^2/um]
    Fstar = Bstar * omega_star
    Fplan = Bplan * omega_planet
    Fback = jwst_background(lam)

    # Count STELLAR photons
    Nphot_star = tint * photon_rate(lam, Fstar, lam_filter, throughput, atel=atel)

    # Count PLANET photons
    Nphot_planet = tint * photon_rate(lam, Fplan, lam_filter, throughput, atel=atel)

    # Count BACKGROUND photons
    if thermal:
        Nphot_bg = tint * photon_rate(lam, Fback, lam_filter, throughput, atel=atel)
    else:
        Nphot_bg = np.zeros_like(Nphot_planet)

    # Calculate SNR on planet photons
    SNR = Nphot_planet / np.sqrt(
        (1 + 1.0 / nout) * Nphot_star
        + 1.0 / nout * Nphot_planet
        + (1 + 1.0 / nout) * Nphot_bg
    )

    return SNR


# Load the filter throughput
filter_name = "f444w"

# Load filter
filt = load_filter(name=f"{filter_name}")

#  Compute SNR on grid
_tplan = np.linspace(1000, 3500, 50)
_dist = np.linspace(10, 100, 50)

tplan_grid, dist_grid = np.meshgrid(_tplan, _dist)


def compute_snr_on_grid(Rp, Tstar=5000, atel=25.4):
    snr = np.zeros_like(tplan_grid)

    for i, Tplan in enumerate(_tplan):
        for j, d in enumerate(_dist):
            snr[i, j] = estimate_eclipse_snr(
                filt[0],
                filt[1],
                Tstar=Tstar,
                Tplan=Tplan,
                Rs=1.0,
                Rp=Rp,
                d=d,
                tint=4.0,
                atel=atel,
            )
    return snr


snr_jwst = compute_snr_on_grid(1 * u.Rjupiter.to(u.Rearth), Tstar=5000)
snr_luvoir = compute_snr_on_grid(1 * u.Rjupiter.to(u.Rearth), Tstar=5000, atel=155.0)

# snr_se_jwst = compute_snr_on_grid(1.5, Tstar=5000)
# snr_se_luvoir = compute_snr_on_grid(1.5, Tstar=5000, atel=155.)


fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)


levels1 = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 160]
levels2 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 160, 190, 220, 250]


cont = ax[0].contour(tplan_grid, dist_grid, snr_jwst.T, levels=levels1, colors=f"k")
ax[0].clabel(cont, inline=1, fontsize=10)

cont = ax[1].contour(tplan_grid, dist_grid, snr_luvoir.T, levels=levels2, colors=f"k")
ax[1].clabel(cont, inline=1, fontsize=10)


ax[0].set(ylabel="Distance [pc]")

for a in ax:
    a.set(xlabel=r"Planet equilibrium temperature [K]")

for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

ax[0].set_title(
    "Secondary eclipse SNR for a Jupiter size planet orbiting\na Sun like star with $T_\mathrm{eff}=5000$K, assuming a $5s$ integration\ntime using the F444W $4.5\mu m$ filter",
    fontsize=12,
    pad=20,
)
ax[1].set_title(
    "Same as the plot on the left, except assuming\n a telescope collecting area equal to that of LUVOIR-A",
    fontsize=12,
    pad=20,
)

fig.savefig("eclipse_snr.pdf", bbox_inches="tight")
