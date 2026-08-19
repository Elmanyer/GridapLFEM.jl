#!/bin/bash
# LOCAL small 2-D - nl_full_plane_flat
#   Full nonlinear pressure in 2-D.
#   Cluster sibling: run/dist_small/run_nl_periodic_plane_flat_small.sh
# MPI on a 12-rank partition (4x3). NO point gauges - judge it from diagnostics.csv.
source "$(dirname "${BASH_SOURCE[0]}")/balfem_local.sh"

export BALFEM_WAVE_GEN=line
export BALFEM_REGIME=nonlinear
export BALFEM_NL_PRESSURE=full
export BALFEM_FLAT_BED=1

export BALFEM_PX=4
export BALFEM_PY=3            # 4*3 = 12 ranks
export BALFEM_LX=40.0
export BALFEM_LY=15.0
export BALFEM_NX=96           # 96/4 = 24 cells per rank in x
export BALFEM_NY=36           # 36/3 = 12 cells per rank in y
export BALFEM_PERIODS=10
export BALFEM_SAVE_EVERY=10
export BALFEM_DIAG_EVERY=5

balfem_local_mpi 12 examples/local_2d/run_small_2d.jl
