#!/bin/bash

#SBATCH --job-name="LFEM_ringwave"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
# Memory headroom while the ranks may still compile. Sized so it FITS a rome
# node (256 GB / 128 cores): 64 ranks x 4 GB over 1 node(s) = 256 GB/node.
# Once a sysimage CONTAINING the solver is proven to remove the per-rank
# compile, drop this and let the 2 GB/core node default apply.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=8
export LFEM_PY=8              # 8*8 = 64 ranks; mesh 200x200 -> 25x25 cells/rank

# --- Case-specific knobs (defaults shown; uncomment to override) ------------
# export LFEM_L=200           # basin side length [m]
# export LFEM_D=3.5           # still-water depth [m]
# export LFEM_TWAVE=1.6       # wave period [s]
# export LFEM_AWAVE=0.001     # amplitude [m]
# export LFEM_SPONGE=20       # sponge width, all 4 sides [m]
# export LFEM_MUMAX=10        # sponge strength
# export LFEM_PERIODS=20      # run length in wave periods
# export LFEM_NX=200; export LFEM_NY=200

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000 # raise GMRES cap if convergence warnings appear
# export LFEM_NL_TOL=1e-6

lfem_run 64 examples/distributed/run_ring_wave_dist.jl
