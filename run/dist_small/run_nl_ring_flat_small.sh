#!/bin/bash
# SMALL observation run — CASE 4: nonlinear · full pressure · flat bed · ring wave · A=0.1
#SBATCH --job-name="LFEMsmall_nl_ring"
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
export LFEM_PX=4
export LFEM_PY=2            # 4*2 = 8 ranks; small 50x20 domain, 2-D mesh 200x80

# --- Case knobs (point source at centre; 4-side sponges) --------------------
# export LFEM_AWAVE=0.1
# export LFEM_SPONGE_X=6; export LFEM_SPONGE_Y=4; export LFEM_MUMAX=12
# export LFEM_PERIODS=12

mpiexecjl -n 8 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_nl_ring_flat_small.jl
