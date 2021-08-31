"""
Fit model.

Usage:
    fit_simulated_lc_from_hydro_snapshot.py <ydeg> <filter> <n_ecl> <n_int> 

Arguments:
    ydeg 
    filter 
    n_ecl 
    n_int 
"""
from docopt import docopt
import numpy as np
import starry
import astropy.units as u

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import *

import arviz as az
import pickle as pkl

from jwst_eclipse_mapping.utils import *

np.random.seed(42)
starry.config.lazy = False
starry.config.quiet = True

numpyro.enable_x64(True)
numpyro.set_host_device_count(4)


def model_num(lc, A_full, texp=1.02183 * u.s, ydeg_inf=8, prior="exponential"):
    map = starry.Map(ydeg_inf)
    lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
    npix = Y2P.shape[0]
    A = A_full[:, 4 : (ydeg_inf + 1) ** 2 + 4]

    def model(obs=np.array(lc["fobs"])):
        if prior == "exponential":
            pixel_scale = numpyro.sample("pixel_scale", dist.HalfNormal(1e-04))
            p = numpyro.sample(
                "p", dist.Exponential(1 / pixel_scale).expand([npix])
            )

        else:
            pixel_scale = numpyro.sample("pixel_scale", dist.HalfNormal(1e-04))
            p = numpyro.sample(
                "p", dist.HalfNormal(pixel_scale).expand([npix])
            )

        x = jnp.dot(P2Y, p)
        numpyro.deterministic("x", x)

        fp = jnp.dot(A, x[:, None]).reshape(-1)
        ln_fs_delta = numpyro.sample("ln_fs_delta", dist.Normal(-10, 5))
        fs = 1.0 - jnp.exp(ln_fs_delta)
        numpyro.deterministic("fs", fs)
        f = fs + fp

        numpyro.deterministic("fpred", f)
        numpyro.sample("obs", dist.Normal(f, np.array(lc["ferr"])), obs=obs)

    return model


def generate_simulated_lightcurve(
    map_planet, params_s, params_p, filt, wav_map, texp, snr=16
):
    # Filter througput interpolated to wav_map
    thr_interp = np.interp(wav_map, filt[0], filt[1])

    # Initialize star map
    map_star = starry.Map(ydeg=1, udeg=2, nw=len(wav_map))
    Llam = (4 * np.pi) * np.pi * planck(params_s["T"].value, wav_map,)
    map_star.amp = Llam / 4

    # Generate high cadence lightcurve  excluding transit
    delta_t = params_p["porb"] / 2 + 0.1 * u.d
    npts = int((2 * delta_t.to(u.s)) / (texp))  # total number of data points
    t = np.linspace(-delta_t.value, delta_t.value, npts)

    # Masks for eclipse, transit and phase curves
    mask_ecl = np.logical_and(t < 0.1, t > -0.1)
    mask_tran = np.abs(t) > 1.05
    mask_phase = ~np.logical_or(mask_ecl, mask_tran)

    t_ecl = t[mask_ecl]
    t_tran = t[mask_tran][::10]  # subsample for performance reasons
    t_phase = t[mask_phase][::100]

    t_combined = np.sort(np.concatenate([t_ecl, t_phase]))

    # Generate light curve
    fsim, fsim_unif, sys = compute_flux(
        t_combined,
        params_s,
        params_p,
        map_star,
        map_planet,
        filt,
        wav_map,
        texp=texp,
    )

    lc = generate_lightcurve(t_combined, fsim, fsim_unif, snr=snr,)

    return t_combined, fsim, fsim_unif, sys, lc


def compute_design_matrix(t, params_p, params_s, ydeg_inf):
    # Star map parameters
    star = starry.Primary(
        starry.Map(ydeg=1, udeg=2),
        r=params_s["r"].value,
        m=params_s["m"].value,
        length_unit=u.Rsun,
        mass_unit=u.Msun,
    )
    star.map[1] = params_s["u"][0]
    star.map[2] = params_s["u"][1]

    planet = starry.Secondary(
        starry.Map(ydeg=20, inc=params_p["inc"].value,),
        ecc=params_p["ecc"],
        omega=params_p["omega"].value,
        r=params_p["r"].value,
        porb=params_p["porb"].value,
        prot=params_p["prot"].value,
        t0=params_p["t0"].value,
        inc=params_p["inc"].value,
        theta0=180,
        length_unit=u.Rsun,
        angle_unit=u.deg,
        time_unit=u.d,
    )
    sys_fit = starry.System(star, planet, texp=(texp.to(u.d)).value)

    # Design matrix
    A_full = sys_fit.design_matrix(t)
    A = A_full[:, 4:]

    return A, A_full



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
    filter_name = model["PandeiaOutTrans"]["input"]["configuration"][
        "instrument"
    ]["filter"]

    return {"snr": SNR, "texp": texp, "filter_name": filter_name}


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    filter_name = args["<filter>"]
    n_ecl = int(args["<n_ecl>"])
    n_int = int(args["<n_int>"])

    path_to_pandexo_output = f"data/system_parameters/hd189_pandexo/{n_ecl}ecl_{filter_name}_{n_int}int.p"
    pandexo_params = load_params_from_pandexo_output(path_to_pandexo_output)

    texp = pandexo_params["texp"]
    snr = float(pandexo_params["snr"])

    # Load orbital and system parameters
    with open(
        "data/system_parameters/hd189_orbital_params_planet.p", "rb"
    ) as handle:
        params_p = pkl.load(handle)
    with open(
        "data/system_parameters/hd189_orbital_params_star.p", "rb"
    ) as handle:
        params_s = pkl.load(handle)

    # Load filter
    filt = load_filter(name=f"{filter_name}")
    mask = filt[1] > 0.002

    # Wavelength grid for starry map (should match filter range)
    wav_map = np.linspace(filt[0][mask][0], filt[0][mask][-1], 80)

    # Load simulation snapshots as starry maps
    sim1 = np.load("data/hydro_snapshots_raw/T42_temp_0.1bar_25days_ylm.npz")
    sim2 = np.load("data/hydro_snapshots_raw/T42_temp_0.1bar_100days_ylm.npz")
    sim3 = np.load("data/hydro_snapshots_raw/T42_temp_0.1bar_300days_ylm.npz")
    sim4 = np.load("data/hydro_snapshots_raw/T42_temp_0.1bar_500days_ylm.npz")

    # Interpolate onto a less dense wavelength grid for performance reasons
    ydeg_sim = 20
    x1_interp = np.stack(
        [
            np.interp(wav_map, sim1["wav_grid"], sim1["x"][i, :])
            for i in range((ydeg_sim + 1) ** 2)
        ]
    )
    x2_interp = np.stack(
        [
            np.interp(wav_map, sim2["wav_grid"], sim2["x"][i, :])
            for i in range((ydeg_sim + 1) ** 2)
        ]
    )
    x3_interp = np.stack(
        [
            np.interp(wav_map, sim3["wav_grid"], sim3["x"][i, :])
            for i in range((ydeg_sim + 1) ** 2)
        ]
    )
    x4_interp = np.stack(
        [
            np.interp(wav_map, sim4["wav_grid"], sim4["x"][i, :])
            for i in range((ydeg_sim + 1) ** 2)
        ]
    )

    def initialize_map(ydeg, nw, x):
        map = starry.Map(ydeg, nw=nw)
        map[1:, :, :] = x[1:, :] / x[0]
        map.amp = x[0]
        return map

    map1_sim = initialize_map(ydeg_sim, len(wav_map), x1_interp)
    map2_sim = initialize_map(ydeg_sim, len(wav_map), x2_interp)
    map3_sim = initialize_map(ydeg_sim, len(wav_map), x3_interp)
    map4_sim = initialize_map(ydeg_sim, len(wav_map), x4_interp)

    # Generate lightcurves
    t, fsim1, fsim1_unif, sys1, lc1 = generate_simulated_lightcurve(
        map1_sim, params_s, params_p, filt, wav_map, texp, snr=snr
    )
    t, fsim2, fsim2_unif, sys2, lc2 = generate_simulated_lightcurve(
        map2_sim, params_s, params_p, filt, wav_map, texp, snr=snr
    )
    t, fsim3, fsim3_unif, sys3, lc3 = generate_simulated_lightcurve(
        map3_sim, params_s, params_p, filt, wav_map, texp, snr=snr
    )
    t, fsim4, fsim4_unif, sys4, lc4 = generate_simulated_lightcurve(
        map4_sim, params_s, params_p, filt, wav_map, texp, snr=snr
    )

    # Fit models
    ydeg_inf = int(args["<ydeg>"])
    A, A_full = compute_design_matrix(t, params_p, params_s, ydeg_inf)

    map = starry.Map(ydeg_inf)
    lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
    npix = Y2P.shape[0]

    # Sampling parameters
    nsamples = 800
    nchains = 2
    init_vals = {
        "ln_fs_delta": -6.5,
        "p": 1e-03 * np.random.rand(npix),
        "pixel_scale": 0.00031178,
    }

    # t = 25 days
    model1 = model_num(lc1, A_full, ydeg_inf=ydeg_inf, prior="gaussian")
    samples1, samples1_az = fit_model_num(
        model1,
        init_vals=init_vals,
        nwarmup=500,
        nsamples=nsamples,
        nchains=nchains,
    )

    # t = 100 days
    model2 = model_num(lc2, A_full, ydeg_inf=ydeg_inf, prior="gaussian")
    samples2, samples2_az = fit_model_num(
        model2,
        init_vals=init_vals,
        nwarmup=500,
        nsamples=nsamples,
        nchains=nchains,
    )

    # t = 300 days
    model3 = model_num(lc3, A_full, ydeg_inf=ydeg_inf, prior="gaussian")
    samples3, samples3_az = fit_model_num(
        model3,
        init_vals=init_vals,
        nwarmup=500,
        nsamples=nsamples,
        nchains=nchains,
    )

    # t = 500 days
    model4 = model_num(lc4, A_full, ydeg_inf=ydeg_inf, prior="gaussian")
    samples4, samples4_az = fit_model_num(
        model4,
        init_vals=init_vals,
        nwarmup=500,
        nsamples=nsamples,
        nchains=nchains,
    )

    output_dir = f"data/output/hd189_{filter_name}_ydeg_{ydeg_inf}_necl_{n_ecl}_nint_{n_int}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Samples to disc
    samples_list = [samples1, samples2, samples3, samples4]
    samples_az_list = [samples1_az, samples2_az, samples3_az, samples4_az]

    # Light curves
    obs1_dict = {"t": t, "fsim": fsim1, "fsim_unif": fsim1_unif, "lc": lc1}
    obs2_dict = {"t": t, "fsim": fsim2, "fsim_unif": fsim2_unif, "lc": lc2}
    obs3_dict = {"t": t, "fsim": fsim3, "fsim_unif": fsim3_unif, "lc": lc3}
    obs4_dict = {"t": t, "fsim": fsim4, "fsim_unif": fsim4_unif, "lc": lc4}

    obs_dict_list = [obs1_dict, obs2_dict, obs3_dict, obs4_dict]

    with open(os.path.join(output_dir, "samples_list.pkl"), "wb") as handle:
        pkl.dump(samples_list, handle)
    with open(os.path.join(output_dir, "samples_az_list.pkl"), "wb") as handle:
        pkl.dump(samples_az_list, handle)
    with open(os.path.join(output_dir, "obs_dict_list.pkl"), "wb") as handle:
        pkl.dump(obs_dict_list, handle)
    with open(os.path.join(output_dir, "design_matrix.pkl"), "wb") as handle:
        pkl.dump(A_full, handle)