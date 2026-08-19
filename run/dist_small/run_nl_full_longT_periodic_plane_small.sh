#!/bin/bash
# SMALL run — nonlinear full plane wave, flat, long T=3.0
#SBATCH --job-name="BALFEM_nl_longT_plane"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
# Memory: the rome node default (2 GB/core) was TESTED (2026-08) and the job was
# still OOM-killed. This request is therefore NOT provisional headroom for a
# compile spike -- it is required until the consumption is attributed; see
# building_files/OPEN_ITEMS.md section 1. Sized so it FITS a rome
# node (256 GB / 128 cores): 32 ranks x 4 GB over 1 node(s) = 128 GB/node.
# Do NOT drop this on the argument that the sysimage removes the per-rank
# compile: that argument was tested and refuted.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapBALFEM.jl/run/balfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to balfem_run) ---
export BALFEM_PX=8
export BALFEM_PY=4            # 8*4 = 32 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export BALFEM_TWAVE=3.0
# T=3.0 => lambda = 13.1 m, nearly 2x the base case. The domain and the sponges
# must scale with it: a 12 m sponge was 0.92 lambda (reflective), and a 50 m box
# cannot host two sponges of 1+ lambda plus a useful working region.
# 70 m with 14 m sponges = 1.07 lambda each, leaving 42 m = 3.2 lambda clean.
export BALFEM_LX=70.0
export BALFEM_NX=280           # dx=0.25 preserved; 280/8 = 35 cells per rank in x
export BALFEM_XWM=18.0         # source clear of the left sponge (14 m)
export BALFEM_SPONGE_L=14
export BALFEM_SPONGE_R=14
export BALFEM_PERIODS=10

balfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
