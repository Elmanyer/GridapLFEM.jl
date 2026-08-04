#!/bin/bash

#SBATCH --job-name="LFEM_periodicplanewave"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
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
