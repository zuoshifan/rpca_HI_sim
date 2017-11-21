import os
import numpy as np
import h5py
import config


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    out_dir = '../results/corr_data/conv_%.1f/' % D
else:
    out_dir = '../results/corr_data/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    map_dir = '../results/conv_beam/conv_%.1f/' % D
    ps_name = map_dir + 'smooth_pointsource_256_700_800_256.hdf5'
    ga_name = map_dir + 'smooth_galaxy_256_700_800_256.hdf5'
    cm_name = map_dir + 'smooth_21cm_256_700_800_256.hdf5'
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:]
else:
    map_dir = '../sky_map/'
    ps_name = map_dir + 'sim_pointsource_256_700_800_256.hdf5'
    ga_name = map_dir + 'sim_galaxy_256_700_800_256.hdf5'
    cm_name = map_dir + 'sim_21cm_256_700_800_256.hdf5'
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:, 0, :]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:, 0, :]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:, 0, :]


# generate normal distributed noise
sigma = 2.0e-4 # make noise close to 21 cm signal
ns_map = sigma * np.random.standard_normal(size=cm_map.shape)

# no noise
fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

# plus noise
fgn_map = fg_map + ns_map # foreground + noise
ttn_map = tt_map + ns_map # total + noise

npix = ps_map.shape[-1]

# no noise
ga_ga_corr = np.dot(ga_map, ga_map.T) / npix
ga_ps_corr = np.dot(ga_map, ps_map.T) / npix
ga_cm_corr = np.dot(ga_map, cm_map.T) / npix
ga_ns_corr = np.dot(ga_map, ns_map.T) / npix
ps_ps_corr = np.dot(ps_map, ps_map.T) / npix
ps_cm_corr = np.dot(ps_map, cm_map.T) / npix
ps_ns_corr = np.dot(ps_map, ns_map.T) / npix
cm_cm_corr = np.dot(cm_map, cm_map.T) / npix
cm_ns_corr = np.dot(cm_map, ns_map.T) / npix
ns_ns_corr = np.dot(ns_map, ns_map.T) / npix

fg_fg_corr = np.dot(fg_map, fg_map.T) / npix
fg_cm_corr = np.dot(fg_map, cm_map.T) / npix
fg_ns_corr = np.dot(fg_map, ns_map.T) / npix

tt_tt_corr = np.dot(tt_map, tt_map.T) / npix

diff_corr = tt_tt_corr - (fg_fg_corr + cm_cm_corr)

# plus noise
ttn_ttn_corr = np.dot(ttn_map, ttn_map.T) / npix
diffn_corr = ttn_ttn_corr - (fg_fg_corr + cm_cm_corr + ns_ns_corr)
diffnn_corr = ttn_ttn_corr - (tt_tt_corr + ns_ns_corr)

corrs = {
          'ga_ga': ga_ga_corr,
          'ga_ps': ga_ps_corr,
          'ga_cm': ga_cm_corr,
          'ga_ns': ga_ns_corr,
          'ps_ps': ps_ps_corr,
          'ps_cm': ps_cm_corr,
          'ps_ns': ps_ns_corr,
          'cm_cm': cm_cm_corr,
          'cm_ns': cm_ns_corr,
          'ns_ns': ns_ns_corr,
          'fg_fg': fg_fg_corr,
          'fg_cm': fg_cm_corr,
          'fg_ns': fg_ns_corr,
          'tt_tt': tt_tt_corr,
          'diff': diff_corr,
          'ttn_ttn': ttn_ttn_corr,
          'diffn': diffn_corr,
          'diffnn': diffnn_corr,
        }


# save corr data to file
with h5py.File(out_dir + 'corr.hdf5', 'w') as f:
    for name, corr in corrs.items():
        f.create_dataset(name, data=corr)
