#!/usr/bin/env bash

# generate beam convolved sky maps
python convolve_beam.py
python convolve_beam_1000x21cm.py

# generate correlation matrices
python gen_corr.py
python gen_corr_1000x21cm.py

# RPCA decomposition
python decomp.py
python decomp_1000x21cm.py

# plot decomposition
python plot_decomp.py

# do gilc reconstruction
python gilc.py
python gilc_1000x21cm.py

# do pca reconstruction
python pca.py

# plot RMSE and r
python plot_rmse.py

# plot error map
python plot_error_map.py

# quantify signal loss
python signal_loss.py

# compute P(k_\parallel)
# python pk.py
mpiexec -n 12 python pk_mpi.py

# plot pk results
python plot_Pkp.py

echo done