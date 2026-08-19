#!/bin/bash

#SBATCH --job-name="GridapBALFEM"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=GridapBALFEM.%j.out
#SBATCH --error=GridapBALFEM.%j.err

export BALFEM_CLUSTER=blue
source $HOME/GridapBALFEM.jl/run/balfem_env.sh

balfem_run 16 examples/distributed/run_plane_wave_dist.jl
