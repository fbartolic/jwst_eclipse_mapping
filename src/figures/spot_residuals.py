import numpy as np
import starry
import astropy.units as u
from scipy.optimize import brent
from matplotlib import colors
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def compute_simulated_lightcurve(
    t, map_star, map_planet, params_s, params_p, texp, radius_ratio=0.1,
):

    # Ratio of star and planet map *amplitudes* needs to be proportional to
    # (Rp/Rs)**2 so we need to multiply the planet map amplitude with that factor
    map_planet.amp *= (radius_ratio) ** 2

    # Initialize system
    star = starry.Primary(map_star, r=params_s["r"])

    planet = starry.Secondary(
        map_planet,
        r=radius_ratio * params_s["r"],
        porb=params_p["porb"],
        prot=params_p["porb"],
        t0=t0,
        inc=params_p["inc"],
        theta0=180,
    )

    sys = starry.System(star, planet, texp=(texp.to(u.d)).value, oversample=9, order=0)

    # Compute flux
    A = sys.design_matrix(t)
    x = np.concatenate([map_star.amp * map_star._y, map_planet.amp * map_planet._y])
    fsim = (A @ x[:, None]).reshape(-1)

    # Rescale the amplitude of the planet map back to its original value
    map_planet.amp *= 1 / (radius_ratio) ** 2

    return fsim, sys


def get_map_amplitude_given_constraint(
    radius_ratio=0.1,
    target_Fp_Fs=2e-03,
    ydeg=20,
    spot_contrast=-0.25,
    radius=30.0,
    lat=30.0,
    lon=30.0,
):
    map_star = starry.Map(1)
    Fs = map_star.flux()[0]

    def cost_fn(map_amplitude):
        map = starry.Map(ydeg)
        map.amp = map_amplitude

        # Add spot
        map.spot(
            contrast=spot_contrast,
            radius=radius,
            lat=lat,
            lon=lon,
            spot_smoothing=2.0 / ydeg,
        )
        Fp = map.flux()[0]

        # Planet dayside flux in selected filter
        Fp_Fs = (radius_ratio) ** 2 * Fp / Fs

        return (Fp_Fs - target_Fp_Fs) ** 2

    map_amplitude = brent(cost_fn, tol=1e-02, maxiter=20,)
    return map_amplitude


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg)
    map[1:, :] = x[1:] / x[0]
    map.amp = x[0]

    return map


def estimate_deviation_from_uniform_map(
    t, spot_contrast, spot_radius, spot_lat, spot_lon,
):

    # Optimize for planet map amplitude
    map_planet_amp = get_map_amplitude_given_constraint(
        radius_ratio=radius_ratio,
        target_Fp_Fs=Fp_Fs,
        spot_contrast=spot_contrast,
        radius=spot_radius,
        lat=spot_lat,
        lon=spot_lon,
    )

    map_planet = starry.Map(20)
    map_planet.amp = map_planet_amp
    map_planet.spot(
        contrast=spot_contrast,
        radius=spot_radius,
        lat=spot_lat,
        lon=spot_lon,
        spot_smoothing=2.0 / 20,
    )
    map_star = starry.Map(1)

    fsim_reference, sys = compute_simulated_lightcurve(
        t,
        map_star,
        get_lower_order_map(map_planet, ydeg=1),
        params_s,
        params_p,
        texp,
        radius_ratio=radius_ratio,
    )

    fsim, sys = compute_simulated_lightcurve(
        t, map_star, map_planet, params_s, params_p, texp, radius_ratio=radius_ratio
    )
    lc_norm = np.max(fsim_reference)
    fsim_reference = fsim_reference / lc_norm
    fsim = fsim / lc_norm

    residual = (fsim - fsim_reference) * 1e06

    return residual


# Set orbital parameters
params_s = {}
params_p = {}

params_s["m"] = 1 * u.Msun
params_s["r"] = 1 * u.Rsun

params_p["porb"] = 1 * u.d
a = (params_p["porb"].to(u.yr).value ** 2 * params_s["m"].value) ** (1 / 3.0) * u.au


def impact_parameter_to_inc(b):
    return np.arccos(b * params_s["r"] / a.to(u.Rsun)).to(u.deg)


# Set inclination as a function of impact parameter
params_p["inc"] = impact_parameter_to_inc(0.5)

# Set exposure time (unimportant for this figure)
texp = 5 * u.s

# Radius ratio and planet/star flux ratio
Fp_Fs = 1e-03
radius_ratio = 0.1

# Observation times
t0 = 0.5 * params_p["porb"]
t_ = np.linspace(-t0.value, +t0.value, int(0.5 * params_p["porb"].to(u.s) / texp))

# Select only times around eclipse
mask_ecl = np.abs(t_) < 0.04
t = t_[mask_ecl]

# Varying spot contrast, spot_radius = 30, b = 0.
spot_contrasts = -np.linspace(0.0, 0.25, 10)
resid_varying_contrast = [
    estimate_deviation_from_uniform_map(t, c, 30.0, 0.0, 0.0) for c in spot_contrasts
]

# Varying spot radii
spot_radii = np.linspace(0, 45.0, 10)
resid_varying_radii = [
    estimate_deviation_from_uniform_map(t, -0.15, r, 0.0, 0.0) for r in spot_radii
]

# Varying spot latitude
spot_lats = np.linspace(-30, 30, 10)
resid_varying_lats = [
    estimate_deviation_from_uniform_map(t, -0.15, 30.0, lat, 0.0) for lat in spot_lats
]


def make_broken_axis(ax):
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["left"].set_visible(False)

    d = 0.01
    kwargs = dict(transform=ax[0].transAxes, color="k", clip_on=False)
    ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax[1].transAxes)
    ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax[1].plot((-d, +d), (-d, +d), **kwargs)


fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.5})


# Varying contrast
norm1 = mpl.colors.Normalize(vmin=spot_contrasts[-1], vmax=spot_contrasts[0])
cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", ["black", "white"])

for i, c in enumerate(spot_contrasts):
    ax[0].plot(t * 24 * 60, resid_varying_contrast[i], color=cmap1(norm1(c)), lw=2.0)

# Varying spot size
norm2 = mpl.colors.Normalize(vmin=spot_radii[0], vmax=spot_radii[-1])
cmap2 = colors.LinearSegmentedColormap.from_list("mycmap", ["white", "black"])

for i, r in enumerate(spot_radii):
    ax[1].plot(t * 24 * 60, resid_varying_radii[i], color=cmap2(norm2(r)), lw=2.0)

# Varying spot latitude
norm3 = mpl.colors.TwoSlopeNorm(vmin=spot_lats[0], vmax=spot_lats[-1], vcenter=0)
cmap3 = colors.LinearSegmentedColormap.from_list("mycmap", ["black", "white", "black"])

for i, lat in enumerate(spot_lats):
    ax[2].plot(t * 24 * 60, resid_varying_lats[i], color=cmap3(norm3(lat)), lw=2.0)

# Colorbars
cax = []
for a in ax:
    divider = make_axes_locatable(a)
    cx = divider.append_axes("right", size="8%", pad=0.15)
    cax.append(cx)


plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1),
    cax=cax[0],
    ticks=-np.arange(0.0, 0.3, 0.05),
    aspect=10,
    label="spot contrast",
)


plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
    cax=cax[1],
    ticks=np.arange(0, 60, 15),
    aspect=10,
    label="spot radius",
)


plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3),
    cax=cax[2],
    ticks=np.arange(-30, 45, 15),
    aspect=10,
    label="spot latitude",
)


for a in ax:
    a.set_yticks(np.arange(-200, 400, 100))
    a.set(xlim=(25, 45), ylim=(-210, 210))
    a.grid(alpha=0.5)
    a.set_xticks(np.arange(25, 50, 5))


for a in ax[1:]:
    a.set(yticklabels=[])

ax[0].set(ylabel="Residuals w.r.t. an $l=1$ map\n[ppm]")


ax[0].set_title("$r=30^\circ,\, \mathrm{lat}=0^\circ$", pad=20)
ax[1].set_title("$c=-0.15,\,\mathrm{lat}=0^\circ$", pad=20)
ax[2].set_title("$c=-0.15,\,r=30^\circ$", pad=20)
fig.text(0.36, -0.08, "Time from eclipse center [minutes]")

fig.savefig("spot_residuals.pdf", bbox_inches="tight", dpi=100)
