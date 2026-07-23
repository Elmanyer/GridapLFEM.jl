#!/bin/bash

#SBATCH --job-name="LFEM_bathymetry"
#SBATCH --partition=fat_rome
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
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

# --- Slope-pressure physics (variable bed) — LINP on by default here --------
# export LFEM_LINP=1          # A/K linear slope-pressure package
# export LFEM_PFULL=1         # + full leading-pressure slope (P1L1+P2L2)
# export LFEM_NLP68=1         # + nonlinear pressure comps 6-8

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

mpiexecjl -n 100 julia --project=$HOME/GridapLFEM.jl $HOME/GridapLFEM.jl/examples/distributed/run_bathymetry_dist.jl
