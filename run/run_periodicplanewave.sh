#!/bin/bash

#SBATCH --job-name="LFEM_periodicplanewave"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
# Memory headroom while the ranks may still compile. Sized so it FITS a rome
# node (256 GB / 128 cores): 128 ranks x 4 GB over 2 node(s) = 256 GB/node.
# Once a sysimage CONTAINING the solver is proven to remove the per-rank
# compile, drop this and let the 2 GB/core node default apply.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=32
export LFEM_PY=4            # 32*4 = 128 ranks; mesh 2000x100

# --- Case-specific knobs (periodic-width flume; defaults shown) --------------
#     The y-edges are periodic (set in the example) -> no lateral sponges.
# export LFEM_LX=400; export LFEM_LY=20   # domain size [m]
# export LFEM_D=3.5           # still-water depth [m]
# export LFEM_TWAVE=1.6       # wave period [s]
# export LFEM_AWAVE=0.001     # amplitude [m]
# export LFEM_XWM=40          # wavemaker x [m]
# export LFEM_SPONGE=40       # x-end sponge width [m]
# export LFEM_MUMAX=5         # sponge strength
# export LFEM_PERIODS=50      # run length in wave periods
# export LFEM_NX=2000; export LFEM_NY=100

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

lfem_run 128 examples/distributed/run_periodic_plane_wave_dist.jl
