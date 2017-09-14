import os
import numpy as np
from scipy import linalg as la
import h5py
from astropy.cosmology import Planck13 as cosmo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


conv_beam = True
# conv_beam = False

if conv_beam:
    out_dir = '../pk/conv/'
else:
    out_dir = '../pk/no_conv/'
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

# # Fourier transform \hat{delta}_HI(kp)
# gkp = (cd_span / N) * ndft(cd, cm_map[:, 0], kp) # K (h Mpc^-1)^-1
# Pkp = np.abs(gkp)**2 / (2.0*np.pi) # K^2 (Mpc / h)
# Pkp *= 1.0e6 # mK^2 (Mpc / h)

Nf, Np = cm_map.shape
Pkp = np.zeros(N)
# Np = 1000 # for fast test
for pi in range(Np):
    gkp = (cd_span / N) * ndft(cd, cm_map[:, pi], kp) # K (h Mpc^-1)^-1
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


# gilc reconstruct
s, U = la.eigh(R_HI)
R_HIh = np.dot(U*s**0.5, U.T)
R_HInh = np.dot(U*(1.0/s)**0.5, U.T)
s1, U1 = la.eigh(np.dot(np.dot(R_HInh, R_tt), R_HInh))
# print (s1[::-1])[:50]

# reconstruct 21cm map
td = 2.0 # the threshold
n = np.where(s1>td)[0][0]
S = np.dot(R_HIh, U1[:, :n])
Ri = la.inv(R_tt)
STRiS = np.dot(S.T, np.dot(Ri, S))
W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
rec_cm = np.dot(W, tt_map)

Pkp_gilc = np.zeros(N)
# Np = 1000
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
plt.savefig(out_dir+'Tkp.png')
plt.close()