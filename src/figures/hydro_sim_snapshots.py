import numpy as np
import yaml
import starry
import astropy.units as u

from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import (
    load_filter,
    planck,
    starry_intensity_to_bbtemp,
    simulation_snapshot_to_ylm,
)

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def compute_simulated_lightcurve(
    map_star, map_planet, params_s, params_p, filt, wavelength_grid, texp,
):
    # Interpolate filter throughput
    thr_interp = np.interp(wavelength_grid, filt[0], filt[1])

    # Ratio of star and planet map *ampliudes* needs to be proportional to
    # (Rp/Rs)**2 so we need to multiply the planet map amplitude with that factor
    radius_ratio = params_p["r"] * u.Rjupiter.to(u.Rsun) / params_s["r"]
    map_planet.amp *= radius_ratio ** 2

    # Generate observation times excluding transit
    porb = params_p["porb"] * u.d
    t0 = 0.5 * params_p["porb"] * u.d

    t_ = np.linspace(-t0.value, +t0.value, int(0.5 * porb.to(u.s) / texp))

    # Mask transit
    mask_tran = np.abs(t_) > 0.9
    t = t_[~mask_tran]

    # Initialize system
    star = starry.Primary(map_star, r=params_s["r"] * u.Rsun, m=params_s["m"] * u.Msun)

    planet = starry.Secondary(
        map_planet,
        r=params_p["r"] * (u.Rjupiter.to(u.Rsun)) * u.Rsun,
        porb=params_p["porb"] * u.d,
        prot=params_p["prot"] * u.d,
        t0=0.5 * params_p["porb"] * u.d,
        inc=params_p["inc"] * u.deg,
        theta0=180,
    )

    sys = starry.System(star, planet, texp=(texp.to(u.d)).value, oversample=9, order=0)

    # Compute flux
    A = sys.design_matrix(t)
    x = np.concatenate([map_star.amp * map_star._y, map_planet.amp * map_planet._y])
    fsim_spectral = np.tensordot(A, x, axes=1)

    wav_filt = filt[0]
    throughput = filt[1]

    # Interpolate filter throughput to map wavelength grid
    throughput_interp = np.interp(wavelength_grid, wav_filt, throughput)

    # Integrate flux over bandpass
    fsim = np.trapz(
        fsim_spectral * throughput_interp, axis=1, x=wavelength_grid * u.um.to(u.m)
    )

    # Rescale the amplitude of the planet map back to its original value
    map_planet.amp *= radius_ratio ** (-2.0)

    return t, fsim, sys


def initialize_featureless_map(T, wavelength_grid, ydeg=1):
    # Initialize star map
    map = starry.Map(ydeg=1, nw=len(wavelength_grid))
    Llam = (4 * np.pi) * np.pi * planck(T, wavelength_grid).value
    map.amp = Llam / 4
    return map


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg, nw=map.nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]

    return map


# System parameters
planet = "hd189"
filter_name = "f444w"

# Load orbital and system parameters
with open(
    f"../../data/system_parameters/{planet}/orbital_params_planet.yaml", "rb"
) as handle:
    params_p = yaml.safe_load(handle)
with open(
    f"../../data/system_parameters/{planet}/orbital_params_star.yaml", "rb"
) as handle:
    params_s = yaml.safe_load(handle)

# Load filter
filt = load_filter(name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(filt[0][mask][0], filt[0][mask][-1], 100)

# Set exposure time
texp = 5 * u.s

# Load simulation snapshots as starry maps
ydeg = 25

snapshots_ylm = [
    simulation_snapshot_to_ylm(
        f"../../data/hydro_snapshots_raw/T341_temp_{day}days.txt",
        wavelength_grid,
        ydeg=ydeg,
    )
    for day in [100, 106, 108, 109]
]


def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map


snapshots_maps = [initialize_map(ydeg, len(wavelength_grid), x) for x in snapshots_ylm]

snapshots_maps_quadrupole = [get_lower_order_map(map, ydeg=2) for map in snapshots_maps]


snapshots_maps_bbtemp = [
    starry_intensity_to_bbtemp(
        map.render(res=250, projection="Mollweide"), wavelength_grid
    )
    for map in snapshots_maps
]

snapshots_maps_quadrupole_bbtemp = [
    starry_intensity_to_bbtemp(
        map.render(res=250, projection="Mollweide"), wavelength_grid
    )
    for map in snapshots_maps_quadrupole
]

# Figure showing the snapshot maps in the Mollweide projection
fig, ax_named = plt.subplot_mosaic(
    """
    ABCD
    EFGH
    IIII
    """,
    figsize=(15, 5),
    gridspec_kw={"wspace": 0.05, "height_ratios": [10, 10, 1], "hspace": 0.1},
)
ax_named["I"].axis("off")  # all of this is a hack to position the colorbar properly
ax1 = [ax_named[l] for l in ["A", "B", "C", "D"]]
ax2 = [ax_named[l] for l in ["E", "F", "G", "H"]]


map = starry.Map(ydeg)
norm = colors.Normalize(vmin=1000, vmax=1250)

for i, im in enumerate(snapshots_maps):
    map.show(
        image=snapshots_maps_bbtemp[i],
        projection="Mollweide",
        cmap="OrRd",
        norm=norm,
        ax=ax1[i],
    )
    map.show(
        image=snapshots_maps_quadrupole_bbtemp[i],
        projection="Mollweide",
        cmap="OrRd",
        norm=norm,
        ax=ax2[i],
    )

labels = ["$t = 100$ days", "$t = 106$ days", "$t = 108$ days", "$t = 109$ days"]

plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="OrRd"),
    orientation="horizontal",
    label="blackbody temperature [K]",
    fraction=0.8,
)

for i, a in enumerate(ax1):
    a.set_title(labels[i])

fig.suptitle(
    "Hydrodynamics simulation snapshots at $l=25$ (top) and $l=2$ (bottom)",
    x=0.5,
    y=1.04,
    fontweight="bold",
    fontsize=16,
)
for a in ax1 + ax2:
    a.set_rasterization_zorder(0)

fig.savefig("hydro_sim_snapshots.pdf", bbox_inches="tight")

# Compute predicted light curves for each snapshot
map_star = initialize_featureless_map(params_s["Teff"], wavelength_grid)

# Simulate reference light curves
fsim_reference_list = []
fsim_list = []

for i in range(len(snapshots_maps_quadrupole)):
    t, fsim_reference, _ = compute_simulated_lightcurve(
        map_star,
        snapshots_maps_quadrupole[i],
        params_s,
        params_p,
        filt,
        wavelength_grid,
        texp,
    )
    t, fsim, sys = compute_simulated_lightcurve(
        map_star, snapshots_maps[i], params_s, params_p, filt, wavelength_grid, texp
    )

    lc_norm = np.max(fsim_reference)

    fsim_reference = fsim_reference / lc_norm
    fsim = fsim / lc_norm

    fsim_reference_list.append(fsim_reference)
    fsim_list.append(fsim)

fsim_mean = np.mean(fsim_list, axis=0)

# Make figure with different light curves
fig, ax = plt.subplots(
    2,
    2,
    figsize=(14, 8),
    gridspec_kw={"wspace": 0.1, "hspace": 0.4, "width_ratios": [2, 1]},
)

for i in range(len(fsim_list)):
    ax[0, 0].plot(t, (fsim_list[i] - 1) * 1e06, lw=2, alpha=0.8)
    ax[0, 1].plot(
        t * 24 * 60, (fsim_list[i] - 1) * 1e06, lw=2, alpha=0.8, label=labels[i]
    )

    ax[1, 0].plot(t, (fsim_list[i] - fsim_reference_list[i]) * 1e06, lw=2, alpha=0.8)
    ax[1, 1].plot(
        t * 24 * 60, (fsim_list[i] - fsim_reference_list[i]) * 1e06, lw=2, alpha=0.8
    )

ax[0, 0].plot(t, (fsim_mean - 1) * 1e06, "k--", alpha=0.5, lw=2)
ax[0, 1].plot(
    t * 24 * 60,
    (fsim_mean - 1) * 1e06,
    "k--",
    alpha=0.5,
    lw=2,
    label="mean flux\nacross epochs",
)


for a in ax[:, 1]:
    a.set(xlim=(-0.03 * 24 * 60, 0.03 * 24 * 60), yticklabels=[])

for a in ax[:, 0]:
    a.set(xlim=(-0.8, 0.8))

for a in ax.flatten():
    a.grid(alpha=0.5)

for a in ax[0]:
    a.set(xticklabels=[], yticks=np.arange(-1500, 250, 250))

for a in ax[1]:
    a.set(ylim=(-18, 18), yticks=np.arange(-15, 20, 5))

ax[0, 1].legend(bbox_to_anchor=(1.08, 0.12))
ax[0, 0].set_ylabel(r"$(F/F_\mathrm{max} - 1)\times 10^{6}$ [ppm]")
ax[1, 0].set_ylabel("Flux difference w.r.t.\n an $l=2$ map [ppm]")

ax[1, 0].set_xlabel("Time from eclipse center [days]")
ax[1, 1].set_xlabel("Time from eclipse center [minutes]")

ax[0, 0].set_title(
    r"Predicted fluxes (F444W $4.5\mu m$ filter) for hydrodynamics simulation snapshots at $l=25$",
    x=0.78,
    pad=20,
    fontweight="bold",
    fontsize=16,
)
ax[1, 0].set_title(
    r"Difference between predicted fluxes at $l=25$ and $l=2$ for the same snapshots",
    x=0.78,
    pad=20,
    fontweight="bold",
    fontsize=16,
)

fig.savefig("hydro_sim_snapshot_lightcurves.pdf", bbox_inches="tight", dpi=100)
