import os
import numpy as np
from scipy.stats import pearsonr
import config
import h5py
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
    in_dir_gilc = '../results/gilc_reconstruct/conv_%.1f/' % D
    in_dir_pca = '../results/pca_reconstruct/conv_%.1f/' % D
else:
    in_dir_gilc = '../results/gilc_reconstruct/no_conv/'
    in_dir_pca = '../results/pca_reconstruct/no_conv/'

fl = 'rec_cm_2.00.hdf5'
with h5py.File(in_dir_gilc + fl, 'r') as f:
    cm_map = f['cm_map'][:]
    rec_cm = f['rec_cm'][:]


nfreq, npix = cm_map.shape

freq_low = 700.0 # MHz
freq_high = 800.0 # MHz
# all freq points
freqs = np.linspace(freq_low, freq_high, nfreq)

r_gilc = np.zeros(nfreq) # save Pearson correlation coefficient
e_gilc = np.zeros(nfreq) # save root-mean-square error
for fi in range(nfreq):
    r_gilc[fi] = pearsonr(cm_map[fi], rec_cm[fi])[0]
    # e_gilc[fi] = np.sqrt(np.sum((cm_map[fi] - rec_cm[fi])**2) / npix)
    e_gilc[fi] = np.sqrt(np.sum((cm_map[fi] - rec_cm[fi])**2))

#######################################################

fl = 'rec_cm_4.hdf5'
with h5py.File(in_dir_pca + fl, 'r') as f:
    cm_map = f['cm_map'][:]
    rec_cm = f['rec_cm'][:]

r_pca = np.zeros(nfreq)
e_pca = np.zeros(nfreq)
for fi in range(nfreq):
    r_pca[fi] = pearsonr(cm_map[fi], rec_cm[fi])[0]
    # e_pca[fi] = np.sqrt(np.sum((cm_map[fi] - rec_cm[fi])**2) / npix)
    e_pca[fi] = np.sqrt(np.sum((cm_map[fi] - rec_cm[fi])**2))


# plot
plt.figure()
plt.plot(r_gilc, 'r', label='RPCA+GILC')
plt.plot(r_pca, 'g', label='classical PCA')
plt.xlim(-2, nfreq+2)
plt.legend(loc=0)
plt.xlabel('frequency points', fontsize=14)
# plt.ylabel(r'$\rho$')
plt.ylabel(r'$r$', fontsize=14)
plt.savefig(out_dir + 'r.png')
plt.close()

inds1 = np.arange(0, 20)
inds2 = np.arange(30, 221)
inds3 = np.arange(236, nfreq)
freqs1 = freqs[inds1]
freqs2 = freqs[inds2]
freqs3 = freqs[inds3]
plt.figure()
plt.subplot(311)
plt.plot(freqs1, r_gilc[inds1], 'r', label='RPCA+GILC')
plt.plot(freqs1, r_pca[inds1], 'g', label='classical PCA')
plt.legend(loc=0)
plt.xlim(freqs1[0], freqs1[-1])
plt.ylabel(r'$r$', fontsize=14)
plt.subplot(312)
plt.plot(freqs2, r_gilc[inds2], 'r', label='RPCA+GILC')
plt.plot(freqs2, r_pca[inds2], 'g', label='classical PCA')
plt.xlim(freqs2[0], freqs2[-1])
plt.ylabel(r'$r$', fontsize=14)
plt.subplot(313)
plt.plot(freqs3, r_gilc[inds3], 'r', label='RPCA+GILC')
plt.plot(freqs3, r_pca[inds3], 'g', label='classical PCA')
plt.xlim(freqs3[0], freqs3[-1])
plt.xlabel('frequency [MHz]', fontsize=14)
plt.ylabel(r'$r$', fontsize=14)
plt.savefig(out_dir + 'r1.png')
plt.close()

plt.figure()
plt.semilogy(r_gilc, 'r', label='RPCA+GILC')
plt.semilogy(r_pca, 'g', label='classical PCA')
plt.xlim(-2, nfreq+2)
plt.ylim(0.75, 1.0)
plt.legend(loc=0)
plt.xlabel('frequency points', fontsize=14)
# plt.ylabel(r'$\rho$')
plt.ylabel(r'$r$', fontsize=14)
plt.savefig(out_dir + 'r_log.png')
plt.close()

plt.figure()
plt.plot(r_gilc-r_pca)
plt.axhline(y=0.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(-2, nfreq+2)
plt.xlabel('frequency points', fontsize=14)
plt.ylabel(r'$r_{\rm{RPCA}} - r_{\rm{PCA}}$', fontsize=14)
plt.savefig(out_dir + 'r_diff.png')
plt.close()

plt.figure()
plt.plot(e_gilc, 'r', label='RPCA+GILC')
plt.plot(e_pca, 'g', label='classical PCA')
plt.xlim(-2, nfreq+2)
plt.legend(loc=0)
plt.xlabel('frequency points', fontsize=14)
plt.ylabel('RMSE [K]', fontsize=14)
plt.savefig(out_dir + 'e.png')
plt.close()

plt.figure()
plt.subplot(311)
plt.plot(freqs1, e_gilc[inds1], 'r', label='RPCA+GILC')
plt.plot(freqs1, e_pca[inds1], 'g', label='classical PCA')
plt.legend(loc=0)
plt.xlim(freqs1[0], freqs1[-1])
plt.ylabel('RMSE [K]', fontsize=14)
plt.subplot(312)
plt.plot(freqs2, e_gilc[inds2], 'r', label='RPCA+GILC')
plt.plot(freqs2, e_pca[inds2], 'g', label='classical PCA')
plt.xlim(freqs2[0], freqs2[-1])
plt.ylabel('RMSE [K]', fontsize=14)
plt.subplot(313)
plt.plot(freqs3, e_gilc[inds3], 'r', label='RPCA+GILC')
plt.plot(freqs3, e_pca[inds3], 'g', label='classical PCA')
plt.xlim(freqs3[0], freqs3[-1])
plt.xlabel('frequency [MHz]', fontsize=14)
plt.ylabel('RMSE [K]', fontsize=14)
plt.savefig(out_dir + 'e1.png')
plt.close()

plt.figure()
plt.semilogy(e_gilc, 'r', label='RPCA+GILC')
plt.semilogy(e_pca, 'g', label='classical PCA')
plt.xlim(-2, nfreq+2)
plt.ylim(0.02, 0.08)
plt.legend(loc=0)
plt.xlabel('frequency points', fontsize=14)
plt.ylabel('RMSE [K]', fontsize=14)
plt.savefig(out_dir + 'e_log.png')
plt.close()

plt.figure()
plt.plot(e_gilc-e_pca)
plt.axhline(y=0.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(-2, nfreq+2)
plt.xlabel('frequency points', fontsize=14)
# plt.ylabel(r'$\rm{RMSE}_{\rm{RPCA}} - \rm{RMSE}_{\rm{PCA}}$')
plt.ylabel('RMSE difference between RPCA and PCA [K]', fontsize=14)
plt.savefig(out_dir + 'e_diff.png')
plt.close()