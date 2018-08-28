import os
import numpy as np
import h5py
import healpy
import config
from cora.signal.corr21cm import Corr21cm
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

if conv_beam:
    out_dir = '../results/plot/rmse/conv_%.1f/' % D
else:
    out_dir = '../results/plot/rmse/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    in_dir = '../results/gilc_reconstruct/conv_%.1f/' % D
else:
    in_dir = '../results/gilc_reconstruct/no_conv/'

fl = 'rec_cm_2.00.hdf5'
with h5py.File(in_dir + fl, 'r') as f:
    cm_map = f['cm_map'][:]
    rec_cm = f['rec_cm'][:]

cf = 750.0 # MHz
f0 = 1420.4 # MHz
z = f0 / cf - 1.0
c21 = Corr21cm()
Tb = c21.T_b(z)
print z, Tb

cind = 128
fig = plt.figure(1, figsize=(13, 5))
# healpy.mollview(rec_cm[cind] / cm_map[cind] - 1, fig=1, title='', min=-10, max=10)
healpy.mollview((rec_cm[cind] - cm_map[cind]) / (Tb + cm_map[cind]), fig=1, title='', min=-2, max=2)
# healpy.mollview((rec_cm[cind] - cm_map[cind]) / (Tb + cm_map[cind]), fig=1, title='', min=-1, max=1)
# healpy.mollview(2 * (rec_cm[cind] - cm_map[cind]) / (rec_cm[cind] + cm_map[cind]), fig=1, title='', min=-1, max=1)
healpy.graticule(verbose=False)
fig.savefig(out_dir + 'ratio_new_2.png')
plt.close()
