#!/bin/bash
# SMALL run — nonlinear full irregular sea, flat, Hs=0.2 (BASE)
#SBATCH --job-name="BALFEM_nl_irregular"
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
# (base config — no overrides needed)

balfem_run 32 examples/distributed_small/run_irregular_sea_small.jl
