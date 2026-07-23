#!/bin/bash

#SBATCH --job-name="LFEM_directionalsea"
#SBATCH --partition=fat_rome
#SBATCH --time=119:59:00
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
export LFEM_PX=16
export LFEM_PY=8            # 16*8 = 128 ranks; mesh 1200x600 -> 75x75 cells/rank

# --- Sea state (WaveSpec JONSWAP x cosine-power spreading — short-crested) ---
#     directional seas force y_wall_bc=false inside the example (lateral sponges)
# export LFEM_HS=0.002        # significant wave height [m]
# export LFEM_TP=1.6          # peak period [s]
# export LFEM_GAMMA=3.3       # JONSWAP peakedness
# export LFEM_NFREQ=21        # frequency samples (bins = nf-1)
# export LFEM_NTHETA=7        # angle samples (bins = n-1)
# export LFEM_SPREAD_STD=20   # spreading sigma_theta [deg]
# export LFEM_THETA_MAX=60    # angular truncation +/- [deg]
# export LFEM_SEED=20260723   # phase seed (reproducible, rank-deterministic)

# --- Domain / run length ----------------------------------------------------
# export LFEM_LX=400; export LFEM_LY=200  # domain size [m] (Ly wide: oblique paths)
# export LFEM_D=3.5           # still-water depth [m]
# export LFEM_SPONGE=40       # right + lateral sponge width [m]
# export LFEM_MUMAX=5         # sponge strength
# export LFEM_PERIODS=100     # run length in Tp units
# export LFEM_WRITE_W=0; export LFEM_WRITE_PRESSURE=0   # field output OFF (large runs)

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

mpiexecjl -n 128 julia --project=$HOME/GridapLFEM.jl $HOME/GridapLFEM.jl/examples/distributed/run_directional_sea_dist.jl
