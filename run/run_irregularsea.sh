#!/bin/bash

#SBATCH --job-name="LFEM_irregularsea"
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
export LFEM_PY=4             # 32*4 = 128 ranks; mesh 2000x100

# --- Sea state (WaveSpec JONSWAP, seeded phases — long-crested) --------------
# export LFEM_HS=0.002        # significant wave height [m] (linear regime)
# export LFEM_TP=1.6          # peak period [s]
# export LFEM_GAMMA=3.3       # JONSWAP peakedness
# export LFEM_NFREQ=21        # frequency samples (bins = nf-1)
# export LFEM_SEED=20260723   # phase seed (reproducible, rank-deterministic)
# export LFEM_FMIN_FAC=2.5    # fmin = 1/(FMIN_FAC*Tp)
# export LFEM_FMAX_FAC=0.75   # fmax = 1/(FMAX_FAC*Tp); keep kd(fmax) <= kd_app
# export LFEM_TRAMP=          # Hann ramp [s] (unset -> 2*Tp)

# --- Domain / run length ----------------------------------------------------
# export LFEM_LX=400; export LFEM_LY=20   # domain size [m]
# export LFEM_D=3.5           # still-water depth [m]  (kd_p~3.6 at Tp=1.6)
# export LFEM_SPONGE=40       # right-sponge width [m] (>= 4 peak wavelengths)
# export LFEM_MUMAX=5         # sponge strength
# export LFEM_PERIODS=200     # run length in Tp units
# export LFEM_WRITE_W=0; export LFEM_WRITE_PRESSURE=0   # field output OFF (large runs)

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

lfem_run 128 examples/distributed/run_irregular_sea_dist.jl
