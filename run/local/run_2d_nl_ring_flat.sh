#!/bin/bash
# LOCAL small 2-D - nl_ring_flat
#   2-D radial spreading; exercises the solid-wall lateral BC.
#   Cluster sibling: run/dist_small/run_nl_ring_flat_small.sh
# MPI on a 12-rank partition (4x3). NO point gauges - judge it from diagnostics.csv.
source "$(dirname "${BASH_SOURCE[0]}")/lfem_local.sh"

export LFEM_WAVE_GEN=point
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=none
export LFEM_FLAT_BED=1

export LFEM_PX=4
export LFEM_PY=3            # 4*3 = 12 ranks
export LFEM_LX=40.0
export LFEM_LY=15.0
export LFEM_NX=96           # 96/4 = 24 cells per rank in x
export LFEM_NY=36           # 36/3 = 12 cells per rank in y
export LFEM_PERIODS=10
export LFEM_SAVE_EVERY=10
export LFEM_DIAG_EVERY=5

lfem_local_mpi 12 examples/local_2d/run_small_2d.jl
