#!/bin/bash
# SMALL run — nonlinear full plane wave, flat, A=0.1 (BASE)
#SBATCH --job-name="LFEM_nl_plane_flat"
#SBATCH --partition=rome
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
# (base config — no overrides needed)

mpiexecjl -n 32 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_periodic_plane_small.jl
