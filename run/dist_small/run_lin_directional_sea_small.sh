#!/bin/bash
# SMALL run — linear directional sea, flat, Hs=0.2
#SBATCH --job-name="LFEM_lin_directional"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=8
export LFEM_PY=8            # 8*8 = 64 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export LFEM_REGIME=linear
export LFEM_NL_PRESSURE=none

lfem_run 64 examples/distributed_small/run_directional_sea_small.jl
