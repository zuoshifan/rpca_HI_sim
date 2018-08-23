import os
import numpy as np
import h5py
import healpy as hp
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if not config.conv_beam:
    pass
else:
    D = config.D
    nside = config.nside
    freq_low = 700.0
    freq_high = 800.0
    nfreq = 256

    out_dir = '../results/conv_beam/conv_%.1f/' % D
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    map_dir = '../sky_map/'
    ps_name = map_dir + 'sim_pointsource_%d_700_800_256.hdf5' % nside
    ga_name = map_dir + 'sim_galaxy_%d_700_800_256.hdf5' % nside
    cm_name = map_dir + 'sim_21cm_%d_700_800_256.hdf5' % nside
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:, 0, :]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:, 0, :]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:, 0, :]

    # all freq points
    freqs = np.linspace(freq_low, freq_high, nfreq)

    sm_name = 'smooth_%s_%d_700_800_256.hdf5'
    for mp, nm in zip([ ps_map, ga_map, cm_map ], [ 'pointsource', 'galaxy', '21cm' ]):
        sm = np.zeros_like(mp) # to save the smoothed map
        for fi in range(nfreq):
            fwhm = 1.22*3.0e8/(D*freqs[fi]*1.0e6) # radians
            sm[fi] = hp.smoothing(mp[fi], fwhm=fwhm)
        # save smoothed data to file
        with h5py.File(out_dir+(sm_name % (nm, nside)), 'w') as f:
            f.create_dataset('map', data=sm)
