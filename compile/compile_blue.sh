#!/bin/bash

#SBATCH --job-name="compile_GridapLFEM"
#SBATCH --partition=memory
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=compile_GridapLFEM.%j.out
#SBATCH --error=compile_GridapLFEM.%j.err
#
# DelftBlue variant. NOTE: for DelftBlue you must also point set_preferences.jl at
# the DelftBlue OpenMPI libdir (the commented block in that file), then rebuild.

set -euo pipefail
PROJ=$HOME/GridapLFEM.jl
source "$PROJ/compile/load_modules_blue.sh"

julia --project="$PROJ" "$PROJ/compile/set_preferences.jl"
julia --project="$PROJ" -e 'using Pkg; haskey(Pkg.project().dependencies, "PackageCompiler") || Pkg.add("PackageCompiler")'
mpiexecjl --project="$PROJ" -n 1 julia --project="$PROJ" "$PROJ/compile/compile.jl"
