#!/bin/bash
# SMALL run — BC-generated linear plane wave, variable bed (submerged bar), A=0.001
#SBATCH --job-name="LFEM_lin_bcplane_varbed"
#SBATCH --partition=fat_rome
#SBATCH --time=119:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ---
export LFEM_PX=8
export LFEM_PY=4            # 8*4 = 32 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export LFEM_REGIME=linear
export LFEM_NL_PRESSURE=none
export LFEM_FLAT_BED=0      # variable bathymetry => submerged bar
export LFEM_AWAVE=0.001

mpiexecjl -n 32 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_bc_plane_small.jl
