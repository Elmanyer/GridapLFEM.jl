#!/bin/bash
# SMALL run — linear plane wave, flat, A=0.001
#SBATCH --job-name="LFEM_lin_plane"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=8
export LFEM_PY=4            # 8*4 = 32 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export LFEM_REGIME=linear
export LFEM_NL_PRESSURE=none
export LFEM_AWAVE=0.001

lfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
