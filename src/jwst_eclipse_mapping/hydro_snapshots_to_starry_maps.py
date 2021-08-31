import numpy as np
import starry

from jwst_eclipse_mapping.utils import *

np.random.seed(42)
starry.config.lazy = False
starry.config.quiet = True


def simulation_snapshot_to_ylm(path, wav_map, ydeg=20, temp_offset=-450):
    data = np.loadtxt(path)

    nlat = 64
    nlon = 128

    lons = np.linspace(-180, 180, nlon)
    lats = np.linspace(-90, 90, nlat)

    lon_grid, grid = np.meshgrid(lons, lats)

    temp_grid = np.zeros_like(lon_grid)

    for i in range(nlat):
        for j in range(nlon):
            temp_grid[i, j] = data.reshape((nlat, nlon))[i, j]

    temp_grid = (
        np.roll(temp_grid, int(temp_grid.shape[1] / 2), axis=1) + temp_offset
    )

    x_list = []
    map_tmp = starry.Map(ydeg)

    # Evaluate at fewer points for performance reasons
    idcs = np.linspace(0, len(wav_map) - 1, 10).astype(int)
    for i in idcs:
        I_grid = np.pi * planck(temp_grid, wav_map[i])
        map_tmp.load(I_grid, force_psd=True)
        x_list.append(map_tmp._y * map_tmp.amp)

    # Interpolate to full grid
    x_ = np.vstack(x_list).T
    x_interp_list = [
        np.interp(wav_map, wav_map[idcs], x_[i, :]) for i in range(x_.shape[0])
    ]
    x = np.vstack(x_interp_list)

    return x


if __name__ == "__main__":
    # Wavelength grid for starry map (should match filter range)
    wav_map = np.linspace(2, 6, 200)

    # Load simulation snapshots as starry maps
    x1 = simulation_snapshot_to_ylm(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_25days.txt", wav_map
    )
    x2 = simulation_snapshot_to_ylm(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_100days.txt", wav_map
    )
    x3 = simulation_snapshot_to_ylm(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_300days.txt", wav_map
    )
    x4 = simulation_snapshot_to_ylm(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_500days.txt", wav_map
    )

    # Save Ylm coefficients ofthe multi-spectral maps
    np.savez(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_25days_ylm.npz",
        x=x1,
        wav_grid=wav_map,
    )
    np.savez(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_100days_ylm.npz",
        x=x2,
        wav_grid=wav_map,
    )
    np.savez(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_300days_ylm.npz",
        x=x3,
        wav_grid=wav_map,
    )
    np.savez(
        "data/hydro_snapshots_raw/T42_temp_0.1bar_500days_ylm.npz",
        x=x4,
        wav_grid=wav_map,
    )
