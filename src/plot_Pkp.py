import os
import numpy as np
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
from matplotlib.ticker import ScalarFormatter



conv_beam = config.conv_beam
D = config.D
nside = config.nside

if conv_beam:
    out_dir = '../results/pk/conv_%.1f/' % D
else:
    out_dir = '../results/pk/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


with h5py.File(out_dir + 'Pkp.hdf5', 'r') as f:
    Pkps = f['Pkps'][:]
    kp = f['kp'][:]

# labels = [ 'Pkp', 'Pkp_gilc', 'Pkp_pca_4', 'Pkp_pca_6', 'Pkp_pca_8', 'Pkp_pca_10' ]
labels = [ 'input', 'RPCA+GILC', 'classical PCA 4 modes', 'classical PCA 6 modes', 'classical PCA 8 modes', 'classical PCA 10 modes' ]
colors = ['b', 'r', 'g', 'c', 'm', 'y']

# plot Pkp and Pkp_pca
fig, ax = plt.subplots()
for i in range(6):
    plt.loglog(kp, Pkps[i], colors[i], label=labels[i])
plt.legend(loc=0)
plt.xlim(0.01, 0.5)
# plt.xlim(0.01, 0.2)
plt.ylim(5, 200)
# plt.ylim(10, 1000)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
# ax.set_xticks([0.01, 0.02, 0.1, 0.2])
ax.set_yticks([10, 20, 100, 200])
# ax.set_yticks([10, 100, 200])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]', fontsize=14)
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]', fontsize=14)
plt.savefig(out_dir + 'Pkp_compare.png')
plt.close()


# # plot transfer function
# fig, ax = plt.subplots()
# for i in range(1, 6):
#     plt.semilogx(kp, Pkps[i]/Pkps[0], colors[i], label=labels[i])
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
# plt.xlim(0.01, 0.5)
# # plt.ylim(0.5, 1.03)
# plt.ylim(0.8, 1.01)
# ax.set_xticks([0.01, 0.05, 0.1, 0.5])
# ax.xaxis.set_major_formatter(ScalarFormatter())
# ax.yaxis.set_major_formatter(ScalarFormatter())
# plt.legend(loc=0)
# plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]', fontsize=14)
# plt.ylabel(r'$P_{21}^{\mathrm{recovered}}(k_\parallel) / P_{21}^{\mathrm{input}}(k_\parallel)$', fontsize=14)
# plt.savefig(out_dir + 'Tkp_compare.png')
# plt.close()
