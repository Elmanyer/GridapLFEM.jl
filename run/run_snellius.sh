#!/bin/bash

#SBATCH --job-name="GridapLFEM"
#SBATCH --partition=rome
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/compile/load_modules_snellius.sh

mpiexecjl -n 128 julia --project=$HOME/GridapLFEM.jl $HOME/GridapLFEM.jl/examples/distributed/run_plane_wave_alg_dist.jl
