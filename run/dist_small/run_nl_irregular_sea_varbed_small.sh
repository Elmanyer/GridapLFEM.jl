#!/bin/bash
# SMALL sea-state comparison run — nonlinear full-pressure irregular sea over a bar (variable bed), Hs=0.2
#SBATCH --job-name="LFEMsea_nl_irregular_varbed"
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
export LFEM_PX=8
export LFEM_PY=4            # 8*4 = 32 ranks; small 50x20 domain

# --- Sea state (defaults set in the .jl; override here) ----------------------
# export LFEM_HS=0.2; export LFEM_TP=2.0
# export LFEM_SEED=20260723
# export LFEM_RELAX=1; export LFEM_RELAX_W=6
# export LFEM_PERIODS=15

mpiexecjl -n 32 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_nl_irregular_sea_varbed_small.jl
