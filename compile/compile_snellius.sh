#!/bin/bash

#SBATCH --job-name="compile_GridapSWE"
#SBATCH --partition=rome
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=GridapSWE.%j.out
#SBATCH --error=GridapSWE.%j.err

source load_modules_snellius.sh

julia --project=../ compile.jl
