import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
# from scipy import optimize
import h5py
import healpy
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    out_dir = '../results/pca_reconstruct/conv_%.1f/' % D
else:
    out_dir = '../results/pca_reconstruct/no_conv/'
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
    with h5py.File('../results/decomp/conv_%.1f/decomp.hdf5' % D, 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['S'][:]
        L = f['L'][:]
    with h5py.File('../results/corr_data/conv_%.1f/corr.hdf5' % D, 'r') as f:
        R_f = f['fg_fg'][:]
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
    with h5py.File('../results/decomp/no_conv/decomp.hdf5', 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['S'][:]
        L = f['L'][:]
    with h5py.File('../results/corr_data/no_conv/corr.hdf5', 'r') as f:
        R_f = f['fg_fg'][:]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

# PCA for R_tt
s, U = la.eigh(R_tt)

xinds = range(len(s))

# plot eigen values
plt.figure()
plt.semilogy(xinds, s[::-1])
plt.semilogy(xinds, s[::-1], 'ro', markersize=4.0)
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(-1, 256)
plt.ylabel('Eigen-values')
plt.savefig(out_dir + 'eig_val.png')
plt.close()

# plot eigen vectors
plt.figure()
plt.plot(xinds, U[:, -1])
plt.plot(xinds, U[:, -2])
plt.plot(xinds, U[:, -3])
plt.plot(xinds, U[:, -4])
plt.plot(xinds, U[:, -5])
plt.plot(xinds, U[:, -6])
plt.xlim(-1, 256)
plt.ylabel('Eigen-vector')
plt.savefig(out_dir + 'eig_vector.png')
plt.close()


cind = len(cm_map) / 2 # central frequency index
# plot principal components
for i in xrange(1, 7):
    pc = np.dot(np.dot(U[:, -i], U[:, -i].T), tt_map)
    pc_sum = np.dot(np.dot(U[:, -i:], U[:, -i:].T), tt_map)
    res = tt_map - pc_sum

    # # plot pc
    # plt.figure()
    # fig = plt.figure(1, figsize=(13, 5))
    # healpy.mollview(pc[cind], fig=1, title='')
    # healpy.graticule(verbose=False)
    # fig.savefig(out_dir + 'pc_%d.png' % i)
    # plt.close()

    # plot pc_sum
    plt.figure()
    fig = plt.figure(1)
    healpy.mollview(pc_sum[cind], fig=1, title='', min=0, max=50)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'pc_sum_%d.png' % i)
    plt.close()

    # plot the difference covariance of pc_sum
    plt.figure()
    plt.imshow(R_f - np.dot(pc_sum, pc_sum.T)/pc_sum.shape[-1])
    plt.colorbar()
    plt.savefig(out_dir + 'Rf_diff_%d.png' % i)
    plt.close()

    # plot res
    plt.figure()
    fig = plt.figure(1)
    healpy.mollview(res[cind], fig=1, title='')
    # healpy.mollview(res[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(res[cind], fig=1, title='', min=-0.0005, max=0.0005)
    # healpy.mollview(res[cind], fig=1, title='', min=-0.0004, max=0.0004)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'res_%d.png' % i)
    plt.close()

    rec_cm = res

    # plot difference map
    fig = plt.figure(1)
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.0005, max=0.0005)
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.0003, max=0.0003)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'diff_%d.png' % i)
    plt.close()

    # plot scatter
    plt.figure()
    plt.scatter(cm_map[cind], rec_cm[cind])
    # val = 0.002
    val = 0.001
    plt.xlim(-val, val)
    plt.ylim(-val, val)
    ref_line = np.linspace(-val, val, 100)
    plt.plot(ref_line, ref_line, 'k--')
    plt.savefig(out_dir + 'scatter_%d.png' % i)
    plt.close()


    # compute cl
    cl_sim = healpy.anafast(cm_map[cind]) # K^2
    cl_sim *= 1.0e6 # mK^2
    cl_est = healpy.anafast(rec_cm[cind]) # K^2
    cl_est *= 1.0e6 # mK^2
    cl_simxest = healpy.anafast(cm_map[cind], rec_cm[cind]) # K^2
    cl_simxest *= 1.0e6 # mK^2

    # plot cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    plt.plot(cl_simxest, label='cross', color='magenta')
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT}$ [mK${}^2$]')
    plt.savefig(out_dir + 'cl_%d.png' % i)
    plt.close()

    # plot transfer function cl_out / cl_in
    plt.figure()
    plt.plot(cl_est/cl_sim)
    plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$T_l$')
    plt.savefig(out_dir + 'Tl_%d.png' % i)
    plt.close()

    # normalize cl to l(l+1)Cl/2pi
    l = np.arange(len(cl_sim))
    factor = l*(l + 1) / (2*np.pi)
    cl_sim *= factor
    cl_est *= factor
    cl_simxest *= factor

    # plot normalized cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    plt.plot(cl_simxest, label='cross', color='magenta')
    plt.plot(cl_sim - cl_est, label='Residual', color='red')
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1) C_l^{TT}/2\pi$ [mK${}^2$]')
    plt.savefig(out_dir + 'cl_normalize_%d.png' % i)
    plt.close()
