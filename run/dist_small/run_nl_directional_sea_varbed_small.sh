#!/bin/bash
# SMALL run — nonlinear full directional sea over a bar, Hs=0.2
#SBATCH --job-name="LFEM_nl_directional_varbed"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
# Memory headroom while the ranks may still compile. Sized so it FITS a rome
# node (256 GB / 128 cores): 64 ranks x 4 GB over 1 node(s) = 256 GB/node.
# Once a sysimage CONTAINING the solver is proven to remove the per-rank
# compile, drop this and let the 2 GB/core node default apply.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=8
export LFEM_PY=8            # 8*8 = 64 ranks

# --- Case-specific overrides (only what differs from the script's base) ---
export LFEM_FLAT_BED=0

lfem_run 64 examples/distributed_small/run_directional_sea_small.jl
