#!/bin/bash
# LOCAL quasi-1D flume - nl_full_bc_irregular
#   The Hs sea-state case that failed on the cluster, at 1-D cost
# 12-rank MPI partition (12x1 in x). NOTE: the distributed driver has NO point
# gauges, so this case is judged from diagnostics.csv, not from gauge time series.
# (For a gauge-based measurement run it sequentially: LFEM_MPI=0.)
source "$(dirname "${BASH_SOURCE[0]}")/lfem_local.sh"

export LFEM_WAVE_GEN=sea
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=full
export LFEM_FLAT_BED=1
export LFEM_RELAX=1
export LFEM_HS=0.02
export LFEM_TP=1.6

# 12-rank sizing: 60 x 3 m, dx=0.25 (16 cells/lambda at kd=5.5), 16 periods
export LFEM_MPI=1
export LFEM_PX=12           # 12 x 1 = 12 ranks; LFEM_NX must divide by LFEM_PX
export LFEM_LX=60.0
export LFEM_NX=240
export LFEM_PERIODS=16
export LFEM_SAVE_EVERY=10
export LFEM_DIAG_EVERY=5

lfem_local_mpi 12 examples/local_1d/run_flume_1d.jl
