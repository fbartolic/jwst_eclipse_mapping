import numpy as np
import starry
import astropy.units as u
from scipy.optimize import brent
from scipy.stats import binned_statistic
from matplotlib import colors
from matplotlib import pyplot as plt

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def compute_simulated_lightcurve(
    map_star, map_planet, params_s, params_p, texp, radius_ratio=0.1,
):

    # Initialize star map
    map_star = starry.Map(ydeg=1)

    # Ratio of star and planet map *amplitudes* needs to be proportional to
    # (Rp/Rs)**2 so we need to multiply the planet map amplitude with that factor
    map_planet.amp *= (radius_ratio) ** 2

    # Generate observation times excluding transit
    porb = params_p["porb"]
    t0 = 0.5 * params_p["porb"]

    t_ = np.linspace(-t0.value, +t0.value, int(0.5 * porb.to(u.s) / texp))

    # Select only times around eclipse
    mask_ecl = np.abs(t_) < 0.45
    t = t_[mask_ecl]

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

    return t, fsim, sys


def get_map_amplitude_given_constraint(
    radius_ratio=0.08,
    target_Fp_Fs=2e-03,
    ydeg=25,
    spot_contrast=-0.25,
    radius=30.0,
    lat=30,
    lon=30.0,
):
    map_star = starry.Map(1)
    Fs = map_star.flux()[0]

    def cost_fn(map_amplitude, return_map=False):
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

    map_amplitude = brent(cost_fn, tol=1e-02, maxiter=30,)
    return map_amplitude


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


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg)
    map[1:, :] = x[1:] / x[0]
    map.amp = x[0]

    return map


def solve_linear_system(A, fobs, ferr, L_sec):
    # Primary
    L_prim = np.ones(sys.primary.map.N)
    L_prim[1:] = 1e-02 ** 2
    L = np.concatenate([L_prim, L_sec])

    x_mean, cho_cov = starry.linalg.solve(A, fobs, C=ferr ** 2, L=L,)
    return x_mean, cho_cov


# Set orbital parameters
params_s = {}
params_p = {}

params_s["m"] = 1 * u.Msun
params_s["r"] = 1 * u.Rsun

params_p["porb"] = 1 * u.d
a = (params_p["porb"].to(u.yr).value ** 2 * params_s["m"].value) ** (1 / 3.0) * u.au


def impact_parameter_to_inc(b):
    return np.arccos(b * params_s["r"] / a.to(u.Rsun)).to(u.deg)


params_p["inc"] = impact_parameter_to_inc(0.5)

# Set exposure time
texp = 5 * u.s

# Radius ratio and planet/star flux ratio
Fp_Fs = 1e-03
radius_ratio = 0.1

# Spot parameters
spot_radius = 30.0
spot_lat = 20.0
spot_lon = 15.0

# S/N on secondary eclipse depth
SNR = 20

# Spot contrasts
spot_contrasts = [-0.05, -0.15, -0.30]

# Create three maps with the same dayside flux and spots with different contrasts
map_star = starry.Map(1)

maps_sim = []

for spot_contrast in spot_contrasts:
    # Optimize for planet map amplitude
    map_planet_amp = get_map_amplitude_given_constraint(
        radius_ratio=radius_ratio,
        target_Fp_Fs=Fp_Fs,
        spot_contrast=spot_contrast,
        radius=spot_radius,
        lat=spot_lat,
        lon=spot_lon,
    )

    map_planet = starry.Map(25)
    map_planet.amp = map_planet_amp
    map_planet.spot(
        contrast=spot_contrast,
        radius=spot_radius,
        lat=spot_lat,
        lon=spot_lon,
        spot_smoothing=2.0 / 20,
    )

    maps_sim.append(map_planet)

# Simulate light curves for those maps and the same maps at l=1
fluxes_sim = []
fluxes_ref = []
fluxes_obs = []
fluxes_err = []

for i in range(3):
    t, fsim_reference, sys = compute_simulated_lightcurve(
        map_star,
        get_lower_order_map(maps_sim[i], ydeg=1),
        params_s,
        params_p,
        texp,
        radius_ratio=radius_ratio,
    )

    t, fsim, sys = compute_simulated_lightcurve(
        map_star, maps_sim[i], params_s, params_p, texp, radius_ratio=radius_ratio
    )

    # Normalize
    lc_norm = np.max(fsim_reference)
    fsim_reference = fsim_reference / lc_norm
    fsim = fsim / lc_norm

    fluxes_sim.append(fsim)
    fluxes_ref.append(fsim_reference)

    fobs, ferr = draw_sample_lightcurve(fsim, snr=SNR)

    fluxes_obs.append(fobs)
    fluxes_err.append(ferr)

# Compute the design matrix
ydeg_inf = 5
A = sys.design_matrix(t)
A = A[:, : sys.primary.map.N + (ydeg_inf + 1) ** 2]

# Specify a power-law prior on the power spectrum of planet map
compute_l = lambda ydeg: np.concatenate(
    [np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)]
)

ls = compute_l(ydeg_inf)

gamma = -1.0  # power law index
sigma_sec = np.sqrt((ls + 1) ** gamma / (2 * ls + 1))
L_sec = 5e-04 ** 2 * sigma_sec ** 2

solutions = []

for i in range(3):
    x_mean, cho_cov = solve_linear_system(A, fluxes_obs[i], fluxes_err[i], L_sec)
    solutions.append((x_mean, cho_cov))


def rescale_inferred_map_to_original_units(I_inferred):
    I_star = map_star.flux()[0]
    I_planet = I_inferred * I_star * radius_ratio ** (-2)
    return I_planet


resol = 150
maps_sim_rendered = [m.render(res=resol) for m in maps_sim]
maps_inferred_rendered = []
maps_inferred_samples_rendered = []
predicted_fluxes = []

# Save inferred maps
map = starry.Map(ydeg_inf)
for i in range(len(solutions)):
    x_mean, cho_cov = solutions[i]

    # Flux
    f = (A @ x_mean[:, None]).reshape(-1)
    predicted_fluxes.append(f)

    x_mean = x_mean[sys.primary.map.N :]
    cho_cov = cho_cov[sys.primary.map.N :, sys.primary.map.N :]

    # Mean
    map[1:, :] = x_mean[1:] / x_mean[0]
    map.amp = x_mean[0]
    I = map.render(res=resol)
    I_rescaled = rescale_inferred_map_to_original_units(I)
    maps_inferred_rendered.append(I_rescaled)

    # Samples
    samples = []
    for s in range(4):
        v = np.random.normal(size=len(x_mean))
        x_sample = x_mean + (cho_cov @ v[:, None]).reshape(-1)
        map[1:, :] = x_sample[1:] / x_sample[0]
        I = map.render(res=resol)
        I_rescaled = rescale_inferred_map_to_original_units(I)
        samples.append(I_rescaled)
    maps_inferred_samples_rendered.append(samples)


# Normalize all maps
map_norm = np.nanpercentile(maps_sim_rendered[-1], 99)

maps_sim_rendered = [m / map_norm for m in maps_sim_rendered]
maps_inferred_rendered = [m / map_norm for m in maps_inferred_rendered]
maps_inferred_samples_rendered = [m / map_norm for m in maps_inferred_samples_rendered]


def lon_lat_to_ortho(lon, lat):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)

    return x, y


def initialize_figure():
    fig = plt.figure(figsize=(12, 12))

    gs = fig.add_gridspec(
        nrows=6,
        ncols=4 + 1 + 4 + 1 + 4,
        height_ratios=[3, 1.0, 3, 1.0, 3, 2],
        hspace=0.2,
        wspace=-0.3,
    )

    # Axes for the simulated maps
    ax_sim_maps = [
        fig.add_subplot(gs[0, :4]),
        fig.add_subplot(gs[0, 5:9]),
        fig.add_subplot(gs[0, 10:14]),
    ]

    ax_text = fig.add_subplot(gs[1, :])

    # Axes for inferred maps
    ax_inf_maps = [
        fig.add_subplot(gs[2, :4]),
        fig.add_subplot(gs[2, 5:9]),
        fig.add_subplot(gs[2, 10:14]),
    ]

    # Axes for samples
    ax_s1 = [fig.add_subplot(gs[3, i]) for i in range(4)]
    ax_s2 = [fig.add_subplot(gs[3, 5 + i]) for i in range(4)]
    ax_s3 = [fig.add_subplot(gs[3, 10 + i]) for i in range(4)]
    ax_samples = [
        ax_s1,
        ax_s2,
        ax_s3,
    ]

    # Axes for light curves
    ax_lcs = [
        fig.add_subplot(gs[4, :4]),
        fig.add_subplot(gs[4, 5:9]),
        fig.add_subplot(gs[4, 10:14]),
    ]

    # Axes for residuals
    ax_res = [
        fig.add_subplot(gs[5, :4]),
        fig.add_subplot(gs[5, 5:9]),
        fig.add_subplot(gs[5, 10:14]),
    ]
    return fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples, ax_lcs, ax_res


# Initialize figure layout
fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples, ax_lcs, ax_res = initialize_figure()

norm = colors.Normalize(vmin=0.2, vmax=1.0)

# Plot simulated map
map = starry.Map(25)

for i in range(3):
    map.show(image=maps_sim_rendered[i], ax=ax_sim_maps[i], cmap="OrRd", norm=norm)

# Text
# Build a rectangle in axes coords
left, width = 0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height
ax_text.axis("off")
ax_text.text(
    0.5 * (left + right),
    0.5 * (bottom + top),
    "Inferred maps",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax_text.transAxes,
    fontweight="bold",
    fontsize=16,
)

# Plot inferred maps
map = starry.Map(ydeg_inf)
for i in range(len(solutions)):
    map.show(image=maps_inferred_rendered[i], ax=ax_inf_maps[i], cmap="OrRd", norm=norm)

    # Plot samples
    for s in range(4):
        map.show(
            image=maps_inferred_samples_rendered[i][s],
            ax=ax_samples[i][s],
            cmap="OrRd",
            norm=norm,
            grid=False,
        )

    x_spot, y_spot = lon_lat_to_ortho(spot_lon, spot_lat)

    ax_inf_maps[i].scatter(
        x_spot, y_spot, marker="x", color="black", s=45.0, alpha=0.4,
    )

# Plot light curves and residuals
for i, (fobs, ferr) in enumerate(zip(fluxes_obs, fluxes_err)):
    x_mean, cho_cov = solutions[i]

    # Posterior light curves
    for s in range(10):
        v = np.random.normal(size=len(x_mean))
        x_sample = x_mean + (cho_cov @ v[:, None]).reshape(-1)
        fpred = (A @ x_sample[:, None]).reshape(-1)
        ax_lcs[i].plot(t * 24 * 60, (fpred - 1) * 1e06, "C1-", alpha=0.2)

    ax_lcs[i].errorbar(
        t * 24 * 60,
        (fluxes_obs[i] - 1) * 1e06,
        ferr / 1e06,
        marker="o",
        linestyle="",
        alpha=0.04,
        color="black",
    )
    ax_res[i].errorbar(
        t * 24 * 60,
        (fluxes_obs[i] - fluxes_ref[i]) * 1e06,
        fluxes_err[i] / 1e06,
        marker="o",
        linestyle="",
        alpha=0.04,
        color="black",
    )

    # Plot binned data
    dur = (t[-1] - t[0]) * 24 * 60
    bin_width = 5
    nbins = int(dur / bin_width)
    bin_means, bin_edges, binnumber = binned_statistic(
        t * 24 * 60, (fobs - fluxes_ref[i]) * 1e06, bins=nbins
    )
    bin_centers = bin_edges[1:] - bin_width / 2
    ax_res[i].scatter(bin_centers, bin_means, color="k", marker="_", lw=2, alpha=0.7)

for a in ax_lcs + ax_res:
    a.set_xlim(-120, 120)
    a.grid(alpha=0.5)
    a.set_xticks(np.arange(-100, 150, 50))

for a in ax_lcs:
    a.set(xticklabels=[], yticks=np.arange(-1000, 250, 250), ylim=(-1250, 400))

for a in ax_res:
    a.set_ylim(-100, 100)
    a.set(yticks=np.arange(-300, 400, 100), ylim=(-340, 340))

for a in ax_lcs[1:] + ax_res[1:]:
    a.set(yticklabels=[])

for i in range(3):
    ax_sim_maps[i].set_title(f"c = {-spot_contrasts[i]:.2f}")

ax_lcs[0].set_ylabel(r"$(F/F_\mathrm{max} - 1)\times 10^{6}$" + "\n[ppm]")
ax_res[0].set_ylabel("Residuals w.r.t.\n an $l=1$ map\n[ppm]")

fig.text(
    0.37, 0.05, "Time from eclipse center [minutes]",
)
fig.suptitle("Simulated maps", x=0.517, y=0.95, fontweight="bold", fontsize=16)

fig.savefig("varying_contrast_spot.pdf", bbox_inches="tight", dpi=100)
