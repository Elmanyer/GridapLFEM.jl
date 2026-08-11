#!/bin/bash
# LOCAL quasi-1D flume - nl_inner_flat
#   Isolates nonlinear advection from the nonlinear pressure
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
source "$(dirname "${BASH_SOURCE[0]}")/lfem_local.sh"

export LFEM_WAVE_GEN=inner
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=none
export LFEM_FLAT_BED=1

# 12-rank sizing: 60 x 3 m, dx=0.25 (16 cells/lambda at kd=5.5), 16 periods
export LFEM_MPI=0           # direct LU: measured 2-3x faster than any MPI split here
export LFEM_LX=60.0
export LFEM_NX=240
export LFEM_PERIODS=16
export LFEM_SAVE_EVERY=10
export LFEM_DIAG_EVERY=5

lfem_local_run examples/local_1d/run_flume_1d.jl
