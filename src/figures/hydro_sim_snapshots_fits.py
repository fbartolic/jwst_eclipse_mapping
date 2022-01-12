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


def initialize_featureless_map(T_star, wavelength_grid, ydeg=1):
    # Initialize star map
    map_star = starry.Map(ydeg=1, nw=len(wavelength_grid))
    Llam = (4 * np.pi) * np.pi * planck(T_star, wavelength_grid).value
    map_star.amp = Llam / 4
    return map_star


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg, nw=map.nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]

    return map


def draw_sample_lightcurve(fsim, sigma=None, snr=None):
    eclipse_depth = np.max(fsim) - np.min(fsim)
    sigma = eclipse_depth / snr
    fobs = fsim + np.random.normal(0, sigma, size=len(t))
    return fobs, sigma * np.ones_like(fobs)


def compute_ps(ydeg, x):
    map = starry.Map(ydeg)
    map[1:, :] = x[1:] / x[0]
    map.amp = x[0]

    ps = np.array([np.sum(map[l, :] ** 2) / (1 + 2 * l) for l in range(ydeg)])
    return ps


def solve_linear_system(sys, A, fobs, ferr, L_sec):
    # Primary
    L_prim = np.ones(sys.primary.map.N)
    L_prim[1:] = 1e-02 ** 2
    L = np.concatenate([L_prim, L_sec])

    x_mean, cho_cov = starry.linalg.solve(A, fobs, C=ferr ** 2, L=L,)
    return x_mean, cho_cov


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
texp = 3.746710 * u.s

# Signal to noise ratio on the secondary eclipse depth
snr_ratios = [20, 100, 500]

# Load simulation snapshots as starry maps
ydeg = 25

snapshots_ylm = [
    simulation_snapshot_to_ylm(
        f"../data/hydro_snapshots_raw/T341_temp_{day}days.txt",
        wavelength_grid,
        ydeg=ydeg,
        temp_offset=-450,
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


map_star = initialize_featureless_map(params_s["Teff"], wavelength_grid)

# Compute light curves for each snapshot and for different signal to noise ratios
fsim_reference_list = []
fsim_list = []

fobs_list = {snr: [] for snr in snr_ratios}
ferr_list = {snr: [] for snr in snr_ratios}

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

    for snr in snr_ratios:
        fobs, ferr = draw_sample_lightcurve(fsim, snr=snr)
        fobs_list[snr].append(fobs)
        ferr_list[snr].append(ferr)

# Compute the design matrix
ydeg_inf = 6
A = sys.design_matrix(t)
A = A[:, : sys.primary.map.N + (ydeg_inf + 1) ** 2]

# Specify a power-law prior on the power spectrum of planet map
compute_l = lambda ydeg: np.concatenate(
    [np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)]
)

ls = compute_l(ydeg_inf)

gamma = -3.0  # power law index
sigma_sec = np.sqrt((ls + 1) ** gamma / (2 * ls + 1))
L_sec = 1e-03 ** 2 * sigma_sec ** 2

# Solve for the map coefficients
solutions = {snr: [] for snr in snr_ratios}

for i in range(4):
    for snr in snr_ratios:
        x_mean, cho_cov = solve_linear_system(
            sys, A, fobs_list[snr][i], ferr_list[snr][i], L_sec
        )
        solutions[snr].append((x_mean, cho_cov))


def inferred_intensity_to_bbtemp(I_planet_raw, filt, params_s, params_p):
    """
    Convert inferred starry intensity map to a BB temperature map.
    """
    wav_filt = filt[0]
    throughput = filt[1]

    # Star spectral radiance integrated over solid angle and bandpass
    I_star = np.pi * integrate_planck_over_filter(params_s["Teff"], filt)

    # Rescale the intensity of the planet map to physical units
    I_planet = (
        I_planet_raw
        * I_star
        * (params_s["r"] / (params_p["r"] * u.Rjupiter.to(u.Rsun))) ** 2
    )

    # Plot temperature map of the planet
    bbtemp_map_inf = np.copy(I_planet[:, :].value)

    for i in range(I_planet.shape[0]):
        for j in range(I_planet.shape[1]):
            bbtemp_map_inf[i, j] = inverse_integrate_planck_over_filter(
                I_planet[i, j].value, filt
            )
    return bbtemp_map_inf


# Render inferred and simulated maps
resol = 120
resol_samples = 50
maps_sim_rendered = [
    starry_intensity_to_bbtemp(
        m.render(res=resol, projection="Mollweide"), wavelength_grid
    )
    for m in snapshots_maps
]
bbtemp_inferred_rendered = {snr: [] for snr in snr_ratios}
bbtemp_inferred_samples_rendered = {snr: [] for snr in snr_ratios}
predicted_fluxes = {snr: [] for snr in snr_ratios}

# Save inferred maps
map = starry.Map(ydeg_inf)

for snr in snr_ratios:
    for i in range(len(solutions[snr])):
        x_mean, cho_cov = solutions[snr][i]

        # Flux
        f = (A @ x_mean[:, None]).reshape(-1)
        predicted_fluxes[snr].append(f)

        x_mean = x_mean[sys.primary.map.N :]
        cho_cov = cho_cov[sys.primary.map.N :, sys.primary.map.N :]

        # Mean
        map[1:, :] = x_mean[1:] / x_mean[0]
        map.amp = x_mean[0]
        temp = inferred_intensity_to_bbtemp(
            map.render(res=resol, projection="Mollweide"), filt, params_s, params_p
        )
        bbtemp_inferred_rendered[snr].append(temp)

        # Samples
        samples = []
        for s in range(4):
            v = np.random.normal(size=len(x_mean))
            x_sample = x_mean + (cho_cov @ v[:, None]).reshape(-1)
            map[1:, :] = x_sample[1:] / x_sample[0]
            temp = inferred_intensity_to_bbtemp(
                map.render(res=resol_samples, projection="Mollweide"),
                filt,
                params_s,
                params_p,
            )
            samples.append(temp)
        bbtemp_inferred_samples_rendered[snr].append(samples)

# Main figure
def initialize_figure():
    fig = plt.figure(figsize=(16, 14))

    gs = fig.add_gridspec(
        nrows=10,
        ncols=4 + 1 + 4 + 1 + 4 + 1 + 4,
        height_ratios=[3, 2.5, 3, 2, 2.5, 3, 2, 2.5, 3, 2],
        width_ratios=4 * [1] + [0.4] + 4 * [1] + [0.4] + 4 * [1] + [0.4] + 4 * [1],
        hspace=0.0,
        wspace=0.02,
    )

    # Axes for the simulated maps
    ax_sim_maps = [
        fig.add_subplot(gs[0, :4]),
        fig.add_subplot(gs[0, 5:9]),
        fig.add_subplot(gs[0, 10:14]),
        fig.add_subplot(gs[0, 15:19]),
    ]

    ax_text = [
        fig.add_subplot(gs[1, :]),
        fig.add_subplot(gs[4, :]),
        fig.add_subplot(gs[7, :]),
    ]

    # Axes for inferred maps
    ax_inf_maps = {snr: [] for snr in snr_ratios}
    ax_samples = {snr: [] for snr in snr_ratios}

    for idx, snr in zip([2, 5, 8], snr_ratios):
        ax_inf_maps[snr] = [
            fig.add_subplot(gs[idx, :4]),
            fig.add_subplot(gs[idx, 5:9]),
            fig.add_subplot(gs[idx, 10:14]),
            fig.add_subplot(gs[idx, 15:19]),
        ]

        # Axes for samples
        ax_s1 = [fig.add_subplot(gs[idx + 1, i]) for i in range(4)]
        ax_s2 = [fig.add_subplot(gs[idx + 1, 5 + i]) for i in range(4)]
        ax_s3 = [fig.add_subplot(gs[idx + 1, 10 + i]) for i in range(4)]
        ax_s4 = [fig.add_subplot(gs[idx + 1, 15 + i]) for i in range(4)]
        ax_samples[snr] = [ax_s1, ax_s2, ax_s3, ax_s4]

    return fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples


fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples = initialize_figure()

norm = colors.Normalize(vmin=1000, vmax=1300)

# Plot simulated map
map = starry.Map(25)

for i in range(4):
    map.show(
        image=maps_sim_rendered[i],
        ax=ax_sim_maps[i],
        cmap="OrRd",
        projection="Mollweide",
        norm=norm,
    )

# Text
# Build a rectangle in axes coords
left, width = 0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height

text_list = [
    "Inferred maps - S/N = 20",
    "Inferred maps - S/N = 100",
    "Inferred maps - S/N = 500",
]
for i, a in enumerate(ax_text):
    a.axis("off")
    a.text(
        0.5 * (left + right),
        0.3 * (bottom + top),
        text_list[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=a.transAxes,
        fontweight="bold",
        fontsize=16,
    )

# Plot inferred maps
for snr in snr_ratios:
    map = starry.Map(ydeg_inf)
    for i in range(len(solutions[snr])):
        map.show(
            image=bbtemp_inferred_rendered[snr][i],
            ax=ax_inf_maps[snr][i],
            cmap="OrRd",
            projection="Mollweide",
            norm=norm,
        )

        # Plot samples
        for s in range(4):
            map.show(
                image=bbtemp_inferred_samples_rendered[snr][i][s],
                ax=ax_samples[snr][i][s],
                cmap="OrRd",
                projection="Mollweide",
                norm=norm,
                grid=False,
            )

labels = ["$t = 100$ days", "$t = 106$ days", "$t = 108$ days", "$t = 109$ days"]

for i, a in enumerate(ax_sim_maps):
    a.set_title(labels[i])

# Colorbar
cax = fig.add_axes([0.42, 0.08, 0.2, 0.01])
plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="OrRd"),
    cax=cax,
    orientation="horizontal",
    label="blackbody temperature [K]",
    fraction=1.0,
)

fig.suptitle("Simulated maps", x=0.517, y=0.93, fontweight="bold", fontsize=16)

fig.savefig("hydro_sim_snapshots_fits.pdf", bbox_inches="tight", dpi=100)
