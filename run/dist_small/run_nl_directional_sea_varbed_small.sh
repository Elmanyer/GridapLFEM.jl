#!/bin/bash
# SMALL run — nonlinear full directional sea over a bar, Hs=0.2
#SBATCH --job-name="LFEM_nl_directional_varbed"
#SBATCH --partition=fat_rome
#SBATCH --time=119:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ---
export LFEM_PX=8
export LFEM_PY=8            # 8*8 = 64 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export LFEM_FLAT_BED=0

mpiexecjl -n 64 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_directional_sea_small.jl
