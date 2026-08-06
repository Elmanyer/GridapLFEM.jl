#!/bin/bash
# LOCAL quasi-1D flume - lin_inner_flat
#   Reference: cheapest complete run; linear dispersion sanity
# 12-rank MPI partition (12x1 in x). NOTE: the distributed driver has NO point
# gauges, so this case is judged from diagnostics.csv, not from gauge time series.
# (For a gauge-based measurement run it sequentially: LFEM_MPI=0.)
source "$(dirname "${BASH_SOURCE[0]}")/lfem_local.sh"

export LFEM_WAVE_GEN=inner
export LFEM_REGIME=linear
export LFEM_NL_PRESSURE=none
export LFEM_FLAT_BED=1

# 12-rank sizing: 60 x 3 m, dx=0.25 (16 cells/lambda at kd=5.5), 16 periods
export LFEM_MPI=1
export LFEM_PX=12           # 12 x 1 = 12 ranks; LFEM_NX must divide by LFEM_PX
export LFEM_LX=60.0
export LFEM_NX=240
export LFEM_PERIODS=16
export LFEM_SAVE_EVERY=10
export LFEM_DIAG_EVERY=5

lfem_local_mpi 12 examples/local_1d/run_flume_1d.jl
