import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors

from scipy.special import legendre as P

import starry
import astropy.units as u

np.random.seed(42)
starry.config.lazy = False

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
    Add an azimuthally symmetric band to map.
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


def get_preimage(map_planet, b, ecc=0.0, Omega=0.0, include_phase_curves=False):
    map = starry.Map(map_planet.ydeg)
    map[1:, :] = map_planet[1:, :]
    map.amp = map_planet.amp

    # Initialize starry system
    Porb = 1 * u.d
    Mstar = 1 * u.Msun
    Rstar = 1 * u.Rsun
    a = (Porb.to(u.yr).value ** 2 * Mstar.value) ** (1 / 3.0) * u.au
    i = np.arccos(b * Rstar / a.to(u.Rsun))

    map_star = starry.Map(ydeg=0)
    map_star.amp = 1
    star = starry.Primary(
        map_star,
        r=Rstar,
    )

    planet = starry.Secondary(
        map,
        r=1 * u.Rjupiter.to(u.Rsun),
        porb=Porb,
        prot=Porb,
        inc=i.to(u.deg),
        Omega=Omega,
        ecc=ecc,
        theta0=180.0,
    )

    # Tidally locked planet -> equatorial plane same as orbital plane
    planet.map.inc = planet.inc
    planet.map.obl = planet.Omega

    sys = starry.System(star, planet)

    # Generate high cadence light excluding transit
    t0 = -Porb / 2
    texp = 5 * u.s
    delta_t = 0.4 * u.d
    npts = int((2 * delta_t.to(u.s)) / (texp))  # total number of data points
    t = np.linspace(t0.value - delta_t.value, t0.value + delta_t.value, npts)

    mask1 = t < t0.value + 0.07
    mask2 = t > t0.value - 0.07
    mask_ecl = np.logical_and(mask1, mask2)
    t_ = t[mask_ecl]

    if include_phase_curves:
        t_ = t

    A = sys.design_matrix(t_)
    x_com = np.concatenate([map_star._y * map_star.amp, map._y * map.amp])
    flux = (A @ x_com[:, None]).reshape(-1)
    ferr = 0.1 * np.ones_like(flux)
    #     fobs = flux + np.random.normal(0, ferr[0], size=len(flux))

    x_preimage, cho_cov = starry.linalg.solve(
        A, flux, C=ferr ** 2, L=0.5 ** 2, N=A.shape[1]
    )

    return x_preimage[1:]


# Single spot
ydeg = 20
map_planet1 = starry.Map(ydeg=ydeg)
map_planet1.spot(
    contrast=-8,
    radius=15,
    lat=0,
    lon=0.0,
)

# Ellipsoidal spot
X, Y = np.meshgrid(np.linspace(-180, 180, 400), np.linspace(-90, 90, 4300))
Z = 1e-02 * np.ones_like(X)
ellipse = lambda x, y, a, b: x ** 2 / a ** 2 + y ** 2 / b ** 2 < 1
mask = ellipse(X, Y, 30, 15)
Z[mask] = 1

map_planet2 = starry.Map(ydeg)
map_planet2.load(Z, smoothing=1.5 / 20, force_psd=True)

# Two spots equal longitude
map_planet3 = starry.Map(ydeg=ydeg)
map_planet3.spot(
    contrast=-8,
    radius=15,
    lat=30,
    lon=0.0,
)

map_planet3.spot(
    contrast=-4,
    radius=15,
    lat=-30,
    lon=0.0,
)

# Two spots equal latitude
map_planet4 = starry.Map(ydeg=ydeg)
map_planet4.spot(
    contrast=-8,
    radius=15,
    lat=0.0,
    lon=-30,
)

map_planet4.spot(
    contrast=-4,
    radius=15,
    lat=0.0,
    lon=30,
)

# Banded planet
map_planet5 = starry.Map(ydeg=ydeg)
map_planet5.spot(
    contrast=-5,
    radius=45,
    lat=90.0,
    lon=0,
)

map_planet5.spot(
    contrast=-5,
    radius=45,
    lat=-90.0,
    lon=0,
)

# Narrow bands
map_planet5 = add_band(map_planet5, 0.7, relative=False, sigma=0.1, lat=15)
map_planet5 = add_band(map_planet5, 0.7, relative=False, sigma=0.1, lat=-15)

map_planet6 = starry.Map(ydeg)
map_planet6.load("earth")

b_ = np.array([0.0, 0.3, 0.5, 0.7])
preim1_list = [get_preimage(map_planet1, b) for b in b_]
preim2_list = [get_preimage(map_planet2, b) for b in b_]
preim3_list = [get_preimage(map_planet3, b) for b in b_]
preim4_list = [get_preimage(map_planet4, b) for b in b_]
preim5_list = [get_preimage(map_planet5, b) for b in b_]
preim6_list = [get_preimage(map_planet6, b) for b in b_]


def make_plot(
    preim1_list,
    preim2_list,
    preim3_list,
    preim4_list,
    preim5_list,
    preim6_list,
):
    fig = plt.figure(figsize=(10, 9))

    nmaps = 6

    # Layout
    gs_sim = fig.add_gridspec(
        nrows=1,
        ncols=nmaps,
        bottom=0.74,
        left=0.12,
        right=0.98,
        wspace=0.1,
        width_ratios=[4, 4, 4, 4, 4, 4],
    )
    gs_inf = fig.add_gridspec(
        nrows=4,
        ncols=nmaps,
        bottom=0.02,
        top=0.64,
        left=0.12,
        right=0.98,
        hspace=0.15,
        wspace=0.1,
        width_ratios=[4, 4, 4, 4, 4, 4],
    )
    gs_geom = fig.add_gridspec(
        nrows=4,
        ncols=1,
        bottom=0.02,
        top=0.64,
        left=0.02,
        right=0.08,
        hspace=0.15,
        wspace=0.1,
    )

    ax_sim = [fig.add_subplot(gs_sim[0, i]) for i in range(nmaps)]
    ax_geom = [fig.add_subplot(gs_geom[i]) for i in range(4)]
    ax1 = [fig.add_subplot(gs_inf[i, 0]) for i in range(4)]
    ax2 = [fig.add_subplot(gs_inf[i, 1]) for i in range(4)]
    ax3 = [fig.add_subplot(gs_inf[i, 2]) for i in range(4)]
    ax4 = [fig.add_subplot(gs_inf[i, 3]) for i in range(4)]
    ax5 = [fig.add_subplot(gs_inf[i, 4]) for i in range(4)]
    ax6 = [fig.add_subplot(gs_inf[i, 5]) for i in range(4)]

    map = starry.Map(ydeg)
    resol = 150
    resol_mini = 50
    norm = colors.Normalize(vmin=-0.02)

    # Spot
    norm1 = colors.Normalize(vmin=0.3, vmax=2.0)
    map_planet1.show(ax=ax_sim[0], norm=norm1, cmap="OrRd", res=resol)
    for i, a in enumerate(ax1):
        x_preim = preim1_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm1, res=resol_mini)

    # Elipse
    norm2 = colors.Normalize(vmin=0.1, vmax=0.8)
    map_planet2.show(ax=ax_sim[1], norm=norm2, cmap="OrRd", res=resol)
    for i, a in enumerate(ax2):
        x_preim = preim2_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm2, res=resol_mini)

    # Spots at same latitude
    norm3 = colors.Normalize(vmin=0.3, vmax=2.0)
    map_planet3.show(ax=ax_sim[2], norm=norm3, cmap="OrRd", res=resol)
    for i, a in enumerate(ax3):
        x_preim = preim3_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm3, res=resol_mini)

    # Spots at the same longitude
    norm4 = norm3
    map_planet4.show(ax=ax_sim[3], norm=norm4, cmap="OrRd", res=resol)
    for i, a in enumerate(ax4):
        x_preim = preim4_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm4, res=resol_mini)

    # Banded planet
    norm5 = colors.Normalize(vmin=0.4, vmax=2.0)
    map_planet5.show(ax=ax_sim[4], cmap="OrRd", norm=norm5, res=resol)
    for i, a in enumerate(ax5):
        x_preim = preim5_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm5, res=resol_mini)

    # Earth
    norm6 = colors.Normalize(vmin=0.1, vmax=1.0)
    map_planet6.show(ax=ax_sim[5], cmap="OrRd", norm=norm6, res=resol)
    for i, a in enumerate(ax6):
        x_preim = preim6_list[i]
        map[1:, :] = x_preim[1:] / x_preim[0]
        map.amp = x_preim[0]
        map.show(ax=a, cmap="OrRd", norm=norm6, res=resol_mini)

    # Geometry
    for i in range(len(b_)):
        c1 = matplotlib.patches.Circle(
            (0, 0),
            radius=1,
            fill=True,
            facecolor="white",
            edgecolor="black",
            lw=0.8,
            alpha=0.8,
        )
        ax_geom[i].axhline(
            b_[i], color="black", linestyle="-", alpha=0.8, lw=2, zorder=-1
        )
        ax_geom[i].add_patch(c1)
        ax_geom[i].axis("off")
        ax_geom[i].text(-1.4, -2.0, f"b = {b_[i]}", fontsize=12)

    for a in ax_geom:
        a.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        a.set_aspect("equal")

    ax_geom[0].set_title("Impact\n parameter", fontsize=12)

    fig.text(
        0.53,
        0.92,
        "Simulated maps",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=16,
    )
    fig.text(
        0.53,
        0.67,
        "Recovered maps (noiseless observations)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=16,
    )

    fig.savefig("preimages.png", bbox_inches="tight", dpi=200)


make_plot(
    preim1_list,
    preim2_list,
    preim3_list,
    preim4_list,
    preim5_list,
    preim6_list,
)
