#!/bin/bash

#SBATCH --job-name="LFEM_ichump"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
# Memory: the rome node default (2 GB/core) was TESTED (2026-08) and the job was
# still OOM-killed. This request is therefore NOT provisional headroom for a
# compile spike -- it is required until the consumption is attributed; see
# building_files/LOCAL_VALIDATION_PLAN.md section 2.2. Sized so it FITS a rome
# node (256 GB / 128 cores): 16 ranks x 4 GB over 1 node(s) = 64 GB/node.
# Do NOT drop this on the argument that the sysimage removes the per-rank
# compile: that argument was tested and refuted.
#SBATCH --mem-per-cpu=4G
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
