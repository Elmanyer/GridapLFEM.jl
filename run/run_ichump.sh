#!/bin/bash

#SBATCH --job-name="LFEM_ichump"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

# --- MPI process grid (PX*PY MUST equal the rank count given to lfem_run) ---
export LFEM_PX=4
export LFEM_PY=4              # 4*4 = 16 ranks; mesh 100x100 -> 25x25 cells/rank

# --- Case-specific knobs (Gaussian hump released from rest, CLOSED basin) ----
#     (x_wall_bc=true is forced inside the example — do not disable it)
# export LFEM_L=100           # basin side [m]
# export LFEM_D=3.5           # depth [m]
# export LFEM_A0=0.01         # hump amplitude [m]
# export LFEM_SIGMA0=4.0      # hump half-width [m]
# export LFEM_TFINAL=40       # final time [s]
# export LFEM_NX=100; export LFEM_NY=100

# --- Solver knobs (RungeKutta :SDIRK_2_2 defaults; bump if Newton stalls) ----
# export LFEM_LS_MAXITER=4000
# export LFEM_NL_TOL=1e-6

lfem_run 16 examples/distributed/run_ic_hump_dist.jl
