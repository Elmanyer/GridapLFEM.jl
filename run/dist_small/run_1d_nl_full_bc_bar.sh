#!/bin/bash
# CLUSTER quasi-1D flume - nl_full_bc_bar
#   Shoaling over a submerged bar + full pressure: the hardest deterministic case
#SBATCH --job-name="LFEM_1d_nl_full_bc_bar"
#SBATCH --partition=rome
#SBATCH --time=11:59:00
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
# Memory: the rome node default (2 GB/core) was TESTED (2026-08) and the job was
# still OOM-killed. This request is therefore NOT provisional headroom for a
# compile spike -- it is required until the consumption is attributed; see
# building_files/LOCAL_VALIDATION_PLAN.md section 2.2. Sized so it FITS a rome
# node (256 GB / 128 cores): 32 ranks x 4 GB over 1 node(s) = 128 GB/node.
# Do NOT drop this on the argument that the sysimage removes the per-rank
# compile: that argument was tested and refuted.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid: a 1-D flume is partitioned in x only ---
export LFEM_MPI=1
export LFEM_PX=32           # 32 x 1 = 32 ranks; LFEM_NX must divide by LFEM_PX

export LFEM_WAVE_GEN=bc
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=full
export LFEM_FLAT_BED=0
export LFEM_RELAX=1

# cluster sizing: 200 x 3 m, dx=0.25, 60 periods
export LFEM_LX=200.0
export LFEM_NX=800
export LFEM_XBAR=100.0
export LFEM_PERIODS=60
export LFEM_SAVE_EVERY=25
export LFEM_DIAG_EVERY=10
export LFEM_PRINT_EVERY=10

lfem_run 32 examples/local_1d/run_flume_1d.jl
