#!/bin/bash
# SMALL run — BC-generated nonlinear plane wave, variable bed (submerged bar), A=0.1
#SBATCH --job-name="LFEM_nl_bcplane_varbed"
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
export LFEM_REGIME=nonlinear
export LFEM_NL_PRESSURE=full
export LFEM_FLAT_BED=0      # variable bathymetry => submerged bar
export LFEM_AWAVE=0.1

lfem_run 32 examples/distributed_small/run_bc_plane_small.jl
