#!/bin/bash
# SMALL comparison run — nonlinear, native NL-pressure, flat bed, A=0.1
#SBATCH --job-name="LFEMcmp_nl_native_periodic"
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

# --- Case config is hard-coded (env still overrides). Common knobs: ----------
# export LFEM_AWAVE=...      # wave amplitude [m]
# export LFEM_TWAVE=...      # wave period [s]
# export LFEM_PERIODS=12     # run length in periods

mpiexecjl -n 4 julia --project=$HOME/GridapLFEM.jl \
    $HOME/GridapLFEM.jl/examples/distributed_small/run_nl_native_periodic_plane_small.jl
