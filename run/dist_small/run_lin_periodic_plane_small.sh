#!/bin/bash
# SMALL observation run — CASE 1: linear · flat bed · y-periodic · A=0.001
#SBATCH --job-name="LFEMsmall_lin_periodic"
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
export LFEM_PY=4            # 8*4 = 32 ranks; small 50x20 domain, mesh 200x40

# --- Case knobs (small-domain defaults live in the .jl; override here) -------
# export LFEM_AWAVE=0.001    # small amplitude (linear reference)
# export LFEM_PERIODS=12     # run length in wave periods
# export LFEM_TFINAL=24      # or set final time directly [s]

mpiexecjl -n 32 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_lin_periodic_plane_small.jl
