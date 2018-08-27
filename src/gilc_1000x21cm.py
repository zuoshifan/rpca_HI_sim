import os
import numpy as np
from scipy import linalg as la
import h5py
import config
import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D
nside = config.nside

if conv_beam:
    out_dir = '../results/gilc_reconstruct/conv_%.1f/' % D
else:
    out_dir = '../results/gilc_reconstruct/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    # with h5py.File('../results/decomp/conv_%.1f/decomp_1000x21cm.hdf5' % D, 'r') as f:
    with h5py.File('../results/corr_data/conv_%.1f/corr_1000x21cm.hdf5' % D, 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['cm_cm'][:]
else:
    # with h5py.File('../results/decomp/no_conv/decomp_1000x21cm.hdf5', 'r') as f:
    with h5py.File('../results/corr_data/no_conv/corr_1000x21cm.hdf5', 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['cm_cm'][:]


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
plt.xlabel('eigen modes', fontsize=14)
plt.ylabel('eigenvalues', fontsize=14)
plt.savefig(out_dir + 'eig_val_1000x21cm.png')
plt.close()
