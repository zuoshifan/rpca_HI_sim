import os
import numpy as np
from scipy import linalg as la
import h5py
from astropy.cosmology import Planck13 as cosmo
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
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

if conv_beam:
    map_dir = '../results/conv_beam/conv_%.1f/' % D
    ps_name = map_dir + 'smooth_pointsource_%d_700_800_256.hdf5' % nside
    ga_name = map_dir + 'smooth_galaxy_%d_700_800_256.hdf5' % nside
    cm_name = map_dir + 'smooth_21cm_%d_700_800_256.hdf5' % nside
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
else:
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
    with h5py.File('../results/decomp/no_conv/decomp.hdf5', 'r') as f:
        R_tt = f['tt_tt'][:]
        R_HI = f['S'][:]
        L = f['L'][:]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

freqs = np.linspace(700.0, 800.0, 256) # MHz
freq0 = 1420.4 # MHz
z = freq0 / freqs - 1.0 # redshift
# print z
# get comoving distance
cd = cosmo.comoving_distance(z).value # Mpc
# print cosmo.h
cd /= cosmo.h # Mpc / h


def ndft(x, f, k):
    """non-equispaced discrete Fourier transform"""
    # return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))
    return np.dot(f, np.exp(-1.0J * k * x[:, np.newaxis]))


dcd = np.abs(np.diff(cd)).min() # Mpc / h
cd_span = np.max(cd) - np.min(cd) # Mpc / h
kp_span = 2.0*np.pi / dcd # h Mpc^-1
dkp = 2.0*np.pi / cd_span # h Mpc^-1
N = int(kp_span / dkp)
print N
# kp = np.linspace(-kp_span, kp_span, 2*N) # h Mpc^-1
# kp = np.linspace(-3*kp_span, 3*kp_span, 6*N) # h Mpc^-1
# kp = np.linspace(0, 0.5*kp_span, N) # h Mpc^-1
N = 200
kp = np.logspace(-2.0, -0.3, N) # h Mpc^-1

Nf, Np = cm_map.shape
Pkp = np.zeros(N)
# Np = 1000 # for fast test
for pi in range(Np):
    gkp = (cd_span / N) * ndft(cd, cm_map[:, pi], kp) # K (h Mpc^-1)^-1
    # according to the definition <\delta(k) \delta(k')^*> = (2\pi)^3 \delta(k - k') P(k), but for 1D, <\delta(k) \delta(k')^*> = (2\pi) \delta(k - k') P(k)
    Pkp += np.abs(gkp)**2 / (2.0*np.pi) # K^2 (Mpc / h)
Pkp /= Np # K^2 (Mpc / h)
Pkp *= 1.0e6 # mK^2 (Mpc / h)

# plot Pkp
fig, ax = plt.subplots()
plt.loglog(kp, Pkp)
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv.png')
plt.close()


# ###########################################################################
# gilc reconstruct
s, U = la.eigh(R_HI)
R_HIh = np.dot(U*s**0.5, U.T)
R_HInh = np.dot(U*(1.0/s)**0.5, U.T)
s1, U1 = la.eigh(np.dot(np.dot(R_HInh, R_tt), R_HInh))
# print (s1[::-1])[:50]

# reconstruct 21cm map
if conv_beam:
    if D == 35.0:
        td = 1.7
    elif D == 100.0:
        td = 2.0 # the threshold
    elif D == 300.0:
        td = 1.2
    else:
        raise ValueError('Unsupported diameter D = %f' % D)
else:
    td = 1.2
n = np.where(s1>td)[0][0]
S = np.dot(R_HIh, U1[:, :n])
Ri = la.inv(R_tt)
STRiS = np.dot(S.T, np.dot(Ri, S))
W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
rec_cm = np.dot(W, tt_map)

Pkp_gilc = np.zeros(N)
for pi in range(Np):
    gkp = (cd_span / N) * ndft(cd, rec_cm[:, pi], kp) # K (h Mpc^-1)^-1
    Pkp_gilc += np.abs(gkp)**2 / (2.0*np.pi) # K^2 (Mpc / h)
Pkp_gilc /= Np # K^2 (Mpc / h)
Pkp_gilc *= 1.0e6 # mK^2 (Mpc / h)

# plot Pkp_gilc
fig, ax = plt.subplots()
plt.loglog(kp, Pkp_gilc)
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv_gilc.png')
plt.close()

# plot Pkp and Pkp_gilc
fig, ax = plt.subplots()
plt.loglog(kp, Pkp, label='input')
plt.loglog(kp, Pkp_gilc, label='recovered')
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend()
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv_gilc2.png')
plt.close()

# plot transfer function
fig, ax = plt.subplots()
plt.semilogx(kp, Pkp_gilc/Pkp)
plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(0.01, 0.5)
plt.ylim(0.5, 1.1)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}^{\mathrm{recovered}}(k_\parallel) / P_{21}^{\mathrm{input}}(k_\parallel)$')
plt.savefig(out_dir+'Tkp_gilc.png')
plt.close()


# ###########################################################################
# PCA for R_tt
s, U = la.eigh(R_tt)

Nm = 4 # number of modes to subtract
pc_sum = np.dot(np.dot(U[:, -Nm:], U[:, -Nm:].T), tt_map)
res = tt_map - pc_sum
rec_cm = res

Pkp_pca = np.zeros(N)
for pi in range(Np):
    gkp = (cd_span / N) * ndft(cd, rec_cm[:, pi], kp) # K (h Mpc^-1)^-1
    Pkp_pca += np.abs(gkp)**2 / (2.0*np.pi) # K^2 (Mpc / h)
Pkp_pca /= Np # K^2 (Mpc / h)
Pkp_pca *= 1.0e6 # mK^2 (Mpc / h)

# plot Pkp_pca
fig, ax = plt.subplots()
plt.loglog(kp, Pkp_pca)
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv_pca.png')
plt.close()

# plot Pkp and Pkp_pca
fig, ax = plt.subplots()
plt.loglog(kp, Pkp, label='input')
plt.loglog(kp, Pkp_pca, label='recovered')
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend()
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv_pca2.png')
plt.close()

# plot transfer function
fig, ax = plt.subplots()
plt.semilogx(kp, Pkp_pca/Pkp)
plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(0.01, 0.5)
plt.ylim(0.5, 1.1)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}^{\mathrm{recovered}}(k_\parallel) / P_{21}^{\mathrm{input}}(k_\parallel)$')
plt.savefig(out_dir+'Tkp_pca.png')
plt.close()


# ############################################################################
# combined plot of gilc and pca
# plot Pkp and Pkp_pca
fig, ax = plt.subplots()
plt.loglog(kp, Pkp, label='input')
plt.loglog(kp, Pkp_gilc, label='recovered by RPCA+GILC')
plt.loglog(kp, Pkp_pca, label='recovered by classical PCA')
plt.xlim(0.01, 0.5)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend()
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}(k_\parallel)$ [mK${}^2 h^{-1}$ Mpc]')
plt.savefig(out_dir+'Pkp_conv_combined.png')
plt.close()

# plot transfer function
fig, ax = plt.subplots()
plt.semilogx(kp, Pkp_gilc/Pkp, label='RPCA+GILC')
plt.semilogx(kp, Pkp_pca/Pkp, label='classical PCA')
plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(0.01, 0.5)
plt.ylim(0.5, 1.1)
ax.set_xticks([0.01, 0.05, 0.1, 0.5])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend()
plt.xlabel(r'$k_\parallel$ [$h$ Mpc${}^{-1}$]')
plt.ylabel(r'$P_{21}^{\mathrm{recovered}}(k_\parallel) / P_{21}^{\mathrm{input}}(k_\parallel)$')
plt.savefig(out_dir+'Tkp_combined.png')
plt.close()
