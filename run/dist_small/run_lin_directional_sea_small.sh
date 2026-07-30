#!/bin/bash
# SMALL sea-state comparison run — linear directional sea, flat bed, Hs=0.2
#SBATCH --job-name="LFEMsea_lin_directional"
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
export LFEM_PX=8
export LFEM_PY=8            # 8*8 = 64 ranks; small 50x20 domain

# --- Sea state (defaults set in the .jl; override here) ----------------------
# export LFEM_HS=0.2; export LFEM_TP=2.0
# export LFEM_SEED=20260723
# export LFEM_RELAX=1; export LFEM_RELAX_W=6
# export LFEM_PERIODS=15

mpiexecjl -n 64 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_lin_directional_sea_small.jl
