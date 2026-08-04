#!/bin/bash

#SBATCH --job-name="GridapLFEM"
#SBATCH --partition=rome
#SBATCH --time=119:59:00
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
# Memory headroom while the ranks may still compile. Sized so it FITS a rome
# node (256 GB / 128 cores): 128 ranks x 4 GB over 2 node(s) = 256 GB/node.
# Once a sysimage CONTAINING the solver is proven to remove the per-rank
# compile, drop this and let the 2 GB/core node default apply.
#SBATCH --mem-per-cpu=4G
#SBATCH --output=GridapLFEM.%j.out
#SBATCH --error=GridapLFEM.%j.err

source $HOME/GridapLFEM.jl/run/lfem_env.sh

lfem_run 128 examples/distributed/run_plane_wave_dist.jl
