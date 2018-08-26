import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
from rpca import RPCA
import config


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    out_dir = '../results/decomp/conv_%.1f/' % D
else:
    out_dir = '../results/decomp/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    in_dir = '../results/corr_data/conv_%.1f/' % D
else:
    out_dir = '../results/corr_data/no_conv/'
with h5py.File(in_dir+'corr.hdf5', 'r') as f:
    cm_cm_corr = f['cm_cm'][:]
    tt_tt_corr = f['tt_tt'][:]

# rpca = RPCA(tt_tt_corr, mu=1.0e8, lmbda=None)
rpca = RPCA(tt_tt_corr, mu=1.0e8, lmbda=None)
L, S, err = rpca.fit(tol=1.0e-14, max_iter=20000, iter_print=100, return_err=True)

print matrix_rank(L)
print matrix_rank(S)
print la.norm(cm_cm_corr - S, ord='fro') / la.norm(cm_cm_corr, ord='fro')
print np.allclose(L, L.T), np.allclose(S, S.T)

# save data to file
with h5py.File(out_dir + 'decomp.hdf5', 'w') as f:
    f.create_dataset('tt_tt', data=tt_tt_corr)
    f.create_dataset('cm_cm', data=cm_cm_corr)
    f.create_dataset('L', data=L)
    f.create_dataset('S', data=S)
