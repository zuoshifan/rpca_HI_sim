import os
import numpy as np
from scipy import linalg as la
import h5py
from astropy.cosmology import Planck13 as cosmo
from caput import mpiutil
import config


conv_beam = config.conv_beam
D = config.D
nside = config.nside

if conv_beam:
    out_dir = '../results/pk/conv_%.1f/' % D
else:
    out_dir = '../results/pk/no_conv/'
if not os.path.exists(out_dir):
    if mpiutil.rank0:
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


def compute_Pkp(sky_map, kp):
    N = kp.shape[0]
    Nf, Np = sky_map.shape
    Pkp = np.zeros(N)
    for pi in mpiutil.mpirange(Np):
        gkp = (cd_span / N) * ndft(cd, sky_map[:, pi], kp) # K (h Mpc^-1)^-1
        # according to the definition <\delta(k) \delta(k')^*> = (2\pi)^3 \delta(k - k') P(k), but for 1D, <\delta(k) \delta(k')^*> = (2\pi) \delta(k - k') P(k)
        Pkp += np.abs(gkp)**2 / (2.0*np.pi) # K^2 (Mpc / h)
    Pkp /= Np # K^2 (Mpc / h)
    Pkp *= 1.0e6 # mK^2 (Mpc / h)

    tmp = mpiutil.gather_array(Pkp.reshape(1, -1), axis=0, root=0)
    if mpiutil.rank0:
        print 'tmp shape: ', tmp.shape
        Pkp = np.sum(tmp, axis=0)
        print 'Pkp shape: ', Pkp.shape

    return Pkp # only rank0 has the correct Pkp


Pkp = compute_Pkp(cm_map, kp)

if mpiutil.rank0:
    # save all Pkp
    cnt = 0
    Pkps = np.zeros((6, N))
    Pkps[cnt] = Pkp
    cnt += 1


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

Pkp_gilc = compute_Pkp(rec_cm, kp)

if mpiutil.rank0:
    Pkps[cnt] = Pkp_gilc
    cnt += 1



# ###########################################################################
# PCA for R_tt
s, U = la.eigh(R_tt)

# Nm = 4 # number of modes to subtract
for Nm in [4, 6, 8, 10]:
    pc_sum = np.dot(np.dot(U[:, -Nm:], U[:, -Nm:].T), tt_map)
    res = tt_map - pc_sum
    rec_cm = res

    Pkp_pca = compute_Pkp(rec_cm, kp)

    if mpiutil.rank0:
        Pkps[cnt] = Pkp_pca
        cnt += 1

# save data to file
if mpiutil.rank0:
    with h5py.File(out_dir+'Pkp.hdf5', 'w') as f:
        f.create_dataset('kp', data=kp)
        f['kp'].attrs['unit'] = 'h Mpc^-1'
        f.create_dataset('Pkps', data=Pkps)
        f['Pkps'].attrs['unit'] = 'mK^2 (Mpc / h)'
        f['Pkps'].attrs['labels'] = 'Pkp, Pkp_gilc, Pkp_pca_4, Pkp_pca_6, Pkp_pca_8, Pkp_pca_10'


if mpiutil.rank0:
    print
    print 'Done!'