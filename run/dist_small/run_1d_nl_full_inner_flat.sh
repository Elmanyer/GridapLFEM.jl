#!/bin/bash
# CLUSTER quasi-1D flume - nl_full_inner_flat
#   The {1,2,4,5} frozen-projection tier - the one that NaN'd on the cluster
#SBATCH --job-name="BALFEM_1d_nl_full_inner_flat"
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

source $HOME/GridapBALFEM.jl/run/balfem_env.sh

# --- MPI process grid: a 1-D flume is partitioned in x only ---
export BALFEM_MPI=1
export BALFEM_PX=32           # 32 x 1 = 32 ranks; BALFEM_NX must divide by BALFEM_PX

export BALFEM_WAVE_GEN=inner
export BALFEM_REGIME=nonlinear
export BALFEM_NL_PRESSURE=full
export BALFEM_FLAT_BED=1

# cluster sizing: 200 x 3 m, dx=0.25, 60 periods
export BALFEM_LX=200.0
export BALFEM_NX=800
export BALFEM_XBAR=100.0
export BALFEM_PERIODS=60
export BALFEM_SAVE_EVERY=25
export BALFEM_DIAG_EVERY=10
export BALFEM_PRINT_EVERY=10

balfem_run 32 examples/local_1d/run_flume_1d.jl
