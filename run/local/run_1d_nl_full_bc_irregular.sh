#!/bin/bash
# LOCAL quasi-1D flume - nl_full_bc_irregular
#   The Hs sea-state case that failed on the cluster, at 1-D cost
# SEQUENTIAL, and that is a MEASURED choice, not an oversight. Strong scaling on
# this exact 240x3 mesh (20.2k DOFs, 40 steps):
#     ranks    1(LU)     2       4       6      12
#     s/step    6.45   14.44   13.10   19.53   18.27
#     gmres      --    758     758     758     715-774
# The GMRES iteration count is RANK-INDEPENDENT (~760), so the cost is the weak
# Jacobi preconditioner, not the decomposition: a direct LU factorisation beats
# it 2-3x at this problem size. The 12 cores are used by running several CASES
# side by side instead (see run_all_1d.sh), which is strictly faster than
# 12-way-decomposing one case. Sequential also KEEPS the point gauges.
source "$(dirname "${BASH_SOURCE[0]}")/balfem_local.sh"

export BALFEM_WAVE_GEN=sea
export BALFEM_REGIME=nonlinear
export BALFEM_NL_PRESSURE=full
export BALFEM_FLAT_BED=1
export BALFEM_RELAX=1
export BALFEM_HS=0.02
export BALFEM_TP=1.6

# 12-rank sizing: 60 x 3 m, dx=0.25 (16 cells/lambda at kd=5.5), 16 periods
export BALFEM_MPI=0           # direct LU: measured 2-3x faster than any MPI split here
export BALFEM_LX=60.0
export BALFEM_NX=240
# no BALFEM_PERIODS here: boundary generation needs ~26 periods to fill the
# flume (45 m / c_g 1.25 m/s = 36 s); the script's transit-aware default covers it.
export BALFEM_SAVE_EVERY=10
export BALFEM_DIAG_EVERY=5

balfem_local_run examples/local_1d/run_flume_1d.jl
