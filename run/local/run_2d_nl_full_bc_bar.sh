#!/bin/bash
# LOCAL small 2-D - nl_full_bc_bar
#   Hardest deterministic case: BC + full pressure + bar.
#   Cluster sibling: run/dist_small/run_nl_bc_plane_varbed_small.sh
# MPI on a 12-rank partition (4x3). NO point gauges - judge it from diagnostics.csv.
source "$(dirname "${BASH_SOURCE[0]}")/lfem_local.sh"

export LFEM_WAVE_GEN=bc
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=full
export LFEM_FLAT_BED=0
export LFEM_RELAX=1
export LFEM_AWAVE=0.01

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
