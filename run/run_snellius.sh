#!/bin/bash

#SBATCH --job-name="GridapLFEM"
#SBATCH --partition=rome
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

lfem_run 128 examples/distributed/run_plane_wave_dist.jl
