import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = True

if conv_beam:
    out_dir = '../plot/decomp/conv/'
else:
    out_dir = '../plot/decomp/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    in_dir = '../decomp/conv/'
else:
    in_dir = '../decomp/no_conv/'
with h5py.File(in_dir + 'decomp.hdf5', 'r') as f:
    tt_tt = f['tt_tt'][:]
    cm_cm = f['cm_cm'][:]
    L = f['L'][:]
    S = f['S'][:]

res = tt_tt - L - S
diff = cm_cm - S

# synthesized plot
plt.figure()
# f, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
f, axarr = plt.subplots(3, 2, sharex=True)
im = axarr[0, 0].imshow(tt_tt, origin='lower')
axarr[0, 0].set_adjustable('box-forced')
axarr[0, 0].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[0, 0])
im = axarr[0, 1].imshow(cm_cm, origin='lower')
axarr[0, 1].set_adjustable('box-forced')
axarr[0, 1].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[0, 1])
im = axarr[1, 0].imshow(L, origin='lower')
axarr[1, 0].set_adjustable('box-forced')
axarr[1, 0].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[1, 0])
im = axarr[1, 1].imshow(S, origin='lower')
axarr[1, 1].set_adjustable('box-forced')
axarr[1, 1].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[1, 1])
im = axarr[2, 0].imshow(res, origin='lower')
axarr[2, 0].set_adjustable('box-forced')
axarr[2, 0].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[2, 0])
im = axarr[2, 1].imshow(diff, origin='lower')
axarr[2, 1].set_adjustable('box-forced')
axarr[2, 1].autoscale(False)
# plt.axis('off')
plt.colorbar(im, ax=axarr[2, 1])
plt.savefig(out_dir + 'decomp.png')
plt.close()


#############################################################
# separate plot
# plot tt
plt.figure()
plt.imshow(tt_tt, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'tt_corr.png')
plt.close()

# plot cm
plt.figure()
plt.imshow(cm_cm, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'cm_corr.png')
plt.close()

# plot L
plt.figure()
plt.imshow(L, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'L.png')
plt.close()

# plot S
plt.figure()
plt.imshow(S, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'S.png')
plt.close()

# plot res
plt.figure()
plt.imshow(res, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'res.png')
plt.close()

# plot diff
plt.figure()
plt.imshow(diff, origin='lower')
plt.colorbar()
plt.savefig(out_dir + 'diff.png')
plt.close()