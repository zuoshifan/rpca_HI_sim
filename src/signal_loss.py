import os
import numpy as np
from scipy import linalg as la
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = True

if conv_beam:
    out_dir = '../signal_loss/conv/'
else:
    out_dir = '../signal_loss/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    map_dir = '../conv/'
    ps_name = map_dir + 'smooth_pointsource_256_700_800_256.hdf5'
    ga_name = map_dir + 'smooth_galaxy_256_700_800_256.hdf5'
    cm_name = map_dir + 'smooth_21cm_256_700_800_256.hdf5'
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:]
    with h5py.File('../decomp/conv/decomp.hdf5', 'r') as f:
        # R_tt = f['tt_tt'][:]
        R_HI_rpca = f['S'][:]
        # L = f['L'][:]
    with h5py.File('../corr_data/conv/corr.hdf5', 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['cm_cm'][:]
        # R_f = f['fg_fg'][:]
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
    with h5py.File('../decomp/no_conv/decomp.hdf5', 'r') as f:
        # R_tt = f['tt_tt'][:]
        R_HI_rpca = f['S'][:]
        # L = f['L'][:]
    with h5py.File('../corr_data/no_conv/corr.hdf5', 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['cm_cm'][:]
        # R_f = f['fg_fg'][:]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal


# PCA for R_tt
s, U = la.eigh(R_tt)

loss = []
rel_loss = []
R_HI_norm = la.norm(R_HI) # Frobenius norm of the HI cov
modes = range(2, 11)
for i in modes:
    pc = np.dot(np.dot(U[:, -i], U[:, -i].T), tt_map)
    pc_sum = np.dot(np.dot(U[:, -i:], U[:, -i:].T), tt_map)
    res = tt_map - pc_sum
    R_res = np.dot(res, res.T) / res.shape[-1] # frequency covariance of res

    # compute ||R_res - R_HI||_F
    diff_norm = la.norm(R_res - R_HI)
    loss.append(diff_norm)
    rel_loss.append(diff_norm / R_HI_norm)

loss_rpca = la.norm(R_HI_rpca - R_HI)
rel_loss_rpca = loss_rpca / R_HI_norm

# plot loss
plt.figure()
plt.semilogy(modes, loss)
plt.semilogy(modes, loss, 'ro', label='classical PCA')
plt.axhline(loss_rpca, color='g', label='robust PCA')
plt.xlim(1, 11)
plt.xlabel('modes')
plt.ylabel(r'$\Delta$')
plt.legend()
plt.savefig(out_dir + 'loss.png')
plt.close()


# plot relative loss
plt.figure()
plt.semilogy(modes, rel_loss)
plt.semilogy(modes, rel_loss, 'ro', label='classical PCA')
plt.axhline(rel_loss_rpca, color='g', label='robust PCA')
plt.xlim(1, 11)
plt.xlabel('modes')
plt.ylabel(r'$\Delta$')
plt.legend()
plt.savefig(out_dir + 'rel_loss.png')
plt.close()