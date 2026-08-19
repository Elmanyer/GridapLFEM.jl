#!/bin/bash

#SBATCH --job-name="BALFEM_irregularsea"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
# Memory: the rome node default (2 GB/core) was TESTED (2026-08) and the job was
# still OOM-killed. This request is therefore NOT provisional headroom for a
# compile spike -- it is required until the consumption is attributed; see
# building_files/LOCAL_VALIDATION_PLAN.md section 2.2. Sized so it FITS a rome
# node (256 GB / 128 cores): 128 ranks x 4 GB over 2 node(s) = 256 GB/node.
# Do NOT drop this on the argument that the sysimage removes the per-rank
# compile: that argument was tested and refuted.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapBALFEM.%j.out
#SBATCH --error=GridapBALFEM.%j.err

source $HOME/GridapBALFEM.jl/run/balfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to balfem_run) ---
export BALFEM_PX=32
export BALFEM_PY=4             # 32*4 = 128 ranks; mesh 2000x100

# --- Sea state (WaveSpec JONSWAP, seeded phases — long-crested) --------------
# export BALFEM_HS=0.002        # significant wave height [m] (linear regime)
# export BALFEM_TP=1.6          # peak period [s]
# export BALFEM_GAMMA=3.3       # JONSWAP peakedness
# export BALFEM_NFREQ=21        # frequency samples (bins = nf-1)
# export BALFEM_SEED=20260723   # phase seed (reproducible, rank-deterministic)
# export BALFEM_FMIN_FAC=2.5    # fmin = 1/(FMIN_FAC*Tp)
# export BALFEM_FMAX_FAC=0.75   # fmax = 1/(FMAX_FAC*Tp); keep kd(fmax) <= kd_app
# export BALFEM_TRAMP=          # Hann ramp [s] (unset -> 2*Tp)

# --- Domain / run length ----------------------------------------------------
# export BALFEM_LX=400; export BALFEM_LY=20   # domain size [m]
# export BALFEM_D=3.5           # still-water depth [m]  (kd_p~3.6 at Tp=1.6)
# export BALFEM_SPONGE=40       # right-sponge width [m] (>= 4 peak wavelengths)
# export BALFEM_MUMAX=5         # sponge strength
# export BALFEM_PERIODS=200     # run length in Tp units
# export BALFEM_WRITE_W=0; export BALFEM_WRITE_PRESSURE=0   # field output OFF (large runs)

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export BALFEM_LS_MAXITER=4000
# export BALFEM_NL_TOL=1e-6

balfem_run 128 examples/distributed/run_irregular_sea_dist.jl
