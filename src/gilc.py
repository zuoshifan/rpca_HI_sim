import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = True

if conv_beam:
    out_dir = '../gilc_reconstruct/conv/'
else:
    out_dir = '../gilc_reconstruct/no_conv/'
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
        R_tt = f['tt_tt'][:]
        R_HI = f['S'][:]
        L = f['L'][:]
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
        R_tt = f['tt_tt'][:]
        R_HI = f['S'][:]
        L = f['L'][:]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

# with h5py.File('corr_data/corr.hdf5', 'r') as f:
#     R_f = f['fg_fg'][:]

s, U = la.eigh(R_HI)
R_HIh = np.dot(U*s**0.5, U.T)
R_HInh = np.dot(U*(1.0/s)**0.5, U.T)
s1, U1 = la.eigh(np.dot(np.dot(R_HInh, R_tt), R_HInh))
print (s1[::-1])[:50]

# plot eigen values
plt.figure()
plt.semilogy(range(len(s1)), s1[::-1])
plt.semilogy(range(len(s1)), s1[::-1], 'ro', markersize=4.0)
plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(-1, 256)
plt.ylabel('Eigen-values')
plt.savefig(out_dir + 'eig_val.png')
plt.close()

# bins = 201
cind = len(cm_map) / 2 # central frequency index

if conv_beam:
    # for D = 35.2
    # threshold = [ 200.0, 2.0, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1 ]
    # for D = 100.0
    threshold = [ 200.0, 2.0, 1.2, 1.15, 1.1 ]
else:
    # threshold = [ 1.0, 1.05, 1.1, 1.15, 1.2, 5.0e3 ]
    threshold = [ 1.0, 1.1, 1.2, 5.0e3 ]

Ri = la.inv(R_tt)
for td in threshold:
    # reconstruct 21cm map
    n = np.where(s1>td)[0][0]
    S = np.dot(R_HIh, U1[:, :n])
    STRiS = np.dot(S.T, np.dot(Ri, S))
    W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
    rec_cm = np.dot(W, tt_map)

    # plot reconstructed 21cm map
    fig = plt.figure(1)
    healpy.mollview(rec_cm[cind], fig=1, title='')
    # healpy.mollview(rec_cm[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(rec_cm[cind], fig=1, title='', min=-0.0003, max=0.0003)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'rec_cm_%.2f.png' % td)
    plt.close()

    # plot difference map
    fig = plt.figure(1)
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.0003, max=0.0003)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'diff_%.2f.png' % td)
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
    plt.savefig(out_dir + 'scatter_%.2f.png' % td)
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
    if td > 10.0:
        plt.ylim(0, 1.0e-10)
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT}$ [mK${}^2$]')
    plt.savefig(out_dir + 'cl_%.2f.png' % td)
    plt.close()

    # plot cross cl
    plt.figure()
    plt.plot(cl_simxest)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT, cross}$ [mK${}^2$]')
    plt.savefig(out_dir + 'xcl_%.2f.png' % td)
    plt.close()

    # plot transfer function cl_out / cl_in
    plt.figure()
    plt.plot(cl_est/cl_sim)
    plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$T_l$')
    plt.savefig(out_dir + 'Tl_%.2f.png' % td)
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
    plt.savefig(out_dir + 'cl_normalize_%.2f.png' % td)
    plt.close()

    # plot normalized cross cl
    plt.figure()
    plt.plot(cl_simxest)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1) C_l^{TT, cross}/2\pi$ [mK${}^2$]')
    plt.savefig(out_dir + 'xcl_normalize_%.2f.png' % td)
    plt.close()
