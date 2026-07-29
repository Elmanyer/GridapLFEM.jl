#!/bin/bash
# SMALL observation run — CASE 2: nonlinear · full pressure · flat bed · y-periodic · A=0.1
#SBATCH --job-name="LFEMsmall_nl_periodic_flat"
#SBATCH --partition=fat_rome
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ------------------------
export LFEM_PX=4
export LFEM_PY=1            # 4*1 = 4 ranks; small 50x20 domain, mesh 200x20

# --- Case knobs (big amplitude; soften if Newton stalls) --------------------
# export LFEM_AWAVE=0.1      # big amplitude (kA≈0.10, safe below breaking)
# export LFEM_PERIODS=12
# export LFEM_LS_MAXITER=4000

mpiexecjl -n 4 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_nl_periodic_plane_flat_small.jl
