#!/bin/bash

#SBATCH --job-name="GridapLFEM"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

export LFEM_CLUSTER=blue
source $HOME/GridapLFEM.jl/run/lfem_env.sh

lfem_run 16 examples/distributed/run_plane_wave_dist.jl
