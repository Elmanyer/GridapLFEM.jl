#!/bin/bash
# SMALL observation run — CASE 3: nonlinear · full pressure · VARIABLE bed · y-periodic · A=0.1
#SBATCH --job-name="LFEMsmall_nl_periodic_varbed"
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

# --- Case knobs (submerged bar; ∇h terms ON via flat_bed=false in the .jl) ---
# export LFEM_HBAR=1.5       # bar height (d: 3.5 -> 2.0 on the crest)
# export LFEM_XBAR=26; export LFEM_WBAR=6
# export LFEM_AWAVE=0.1
# export LFEM_PERIODS=12

mpiexecjl -n 32 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_nl_periodic_plane_varbed_small.jl
