#!/bin/bash

#SBATCH --job-name="GridapLFEM"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

mpiexecjl -n 16 julia --project=$HOME/GridapLFEM.jl $HOME/GridapLFEM.jl/examples/distributed/run_plane_wave_alg_dist.jl
