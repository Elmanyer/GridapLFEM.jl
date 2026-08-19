#!/bin/bash

#SBATCH --job-name="BALFEM_bathymetry"
#SBATCH --partition=rome
#SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
# Memory: the rome node default (2 GB/core) was TESTED (2026-08) and the job was
# still OOM-killed. This request is therefore NOT provisional headroom for a
# compile spike -- it is required until the consumption is attributed; see
# building_files/OPEN_ITEMS.md section 1. Sized so it FITS a rome
# node (256 GB / 128 cores): 100 ranks x 4 GB over 2 node(s) = 200 GB/node.
# Do NOT drop this on the argument that the sysimage removes the per-rank
# compile: that argument was tested and refuted.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapBALFEM.%j.out
#SBATCH --error=GridapBALFEM.%j.err

source $HOME/GridapBALFEM.jl/run/balfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to balfem_run) ---
export BALFEM_PX=20
export BALFEM_PY=5             # 20*5 = 100 ranks; mesh 1500x100 -> 75x20 cells/rank

# --- Case-specific knobs (plane wave shoaling over a tanh submerged bar) -----
# export BALFEM_LX=300; export BALFEM_LY=20   # domain size [m]
# export BALFEM_D0=3.5          # offshore depth [m]
# export BALFEM_HBAR=2.0        # bar height [m]
# export BALFEM_XBAR=150        # bar centre x [m]
# export BALFEM_WBAR=25         # bar half-width [m]
# export BALFEM_TWAVE=2.5       # wave period [s]
# export BALFEM_AWAVE=0.001     # amplitude [m]
# export BALFEM_XWM=40          # wavemaker x [m]
# export BALFEM_SPONGE=35       # sponge width [m]
# export BALFEM_PERIODS=40      # run length in wave periods
# export BALFEM_NX=1500; export BALFEM_NY=100

# --- Sea-bed geometry (variable bed → ∇h terms ON) --------------------------
# export BALFEM_FLAT_BED=0         # 0 = variable bathymetry (default for the bar); 1 = flat bed (∇h≡0)
# export BALFEM_NL_PRESSURE=native # nonlinear pressure: none | native | full

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export BALFEM_LS_MAXITER=4000
# export BALFEM_NL_TOL=1e-6

balfem_run 100 examples/distributed/run_bathymetry_dist.jl
