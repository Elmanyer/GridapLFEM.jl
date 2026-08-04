#!/bin/bash

#SBATCH --job-name="LFEM_bathymetry"
#SBATCH --partition=rome
#SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
# Memory headroom while the ranks may still compile. Sized so it FITS a rome
# node (256 GB / 128 cores): 100 ranks x 4 GB over 2 node(s) = 200 GB/node.
# Once a sysimage CONTAINING the solver is proven to remove the per-rank
# compile, drop this and let the 2 GB/core node default apply.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=20
export LFEM_PY=5             # 20*5 = 100 ranks; mesh 1500x100 -> 75x20 cells/rank

# --- Case-specific knobs (plane wave shoaling over a tanh submerged bar) -----
# export LFEM_LX=300; export LFEM_LY=20   # domain size [m]
# export LFEM_D0=3.5          # offshore depth [m]
# export LFEM_HBAR=2.0        # bar height [m]
# export LFEM_XBAR=150        # bar centre x [m]
# export LFEM_WBAR=25         # bar half-width [m]
# export LFEM_TWAVE=2.5       # wave period [s]
# export LFEM_AWAVE=0.001     # amplitude [m]
# export LFEM_XWM=40          # wavemaker x [m]
# export LFEM_SPONGE=35       # sponge width [m]
# export LFEM_PERIODS=40      # run length in wave periods
# export LFEM_NX=1500; export LFEM_NY=100

# --- Sea-bed geometry (variable bed → ∇h terms ON) --------------------------
# export LFEM_FLAT_BED=0         # 0 = variable bathymetry (default for the bar); 1 = flat bed (∇h≡0)
# export LFEM_NL_PRESSURE=native # nonlinear pressure: none | native | full

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

lfem_run 100 examples/distributed/run_bathymetry_dist.jl
