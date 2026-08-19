#!/bin/bash
#SBATCH --job-name="compile_GridapBALFEM"
#SBATCH --partition=rome
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=compile_GridapBALFEM.%j.out
#SBATCH --error=compile_GridapBALFEM.%j.err
#
# Build the GridapBALFEM system image on Snellius. Submit from the compile/ dir:
#     sbatch compile_snellius.sh
# Produces  $HOME/GridapBALFEM.jl/GridapBALFEM_sysimage.so  (~1 GB).
# Build on the SAME partition (rome) you run on, so the CPU target matches.

set -euo pipefail
PROJ=$HOME/GridapBALFEM.jl
source "$PROJ/compile/load_modules_snellius.sh"    # OpenMPI 5.0.3 + LD_LIBRARY_PATH

# 1) Pin MPI.jl to the SYSTEM OpenMPI (writes $PROJ/LocalPreferences.toml).
#    Without this the sysimage bakes in the bundled MPICH and crashes at launch.
julia --project="$PROJ" "$PROJ/compile/set_preferences.jl"

# 1b) Force a clean precompile with the new preference, then VERIFY that MPI binds
#     the SYSTEM OpenMPI. `set -e` aborts the build here if it still resolves to a
#     JLL artifact — far better than discovering it at launch.
julia --project="$PROJ" -e 'using Pkg; Pkg.precompile()'
julia --project="$PROJ" -e '
    using MPIPreferences
    @assert MPIPreferences.binary == "system" "MPIPreferences.binary=$(MPIPreferences.binary) (expected system)"
    using MPI
    v = MPI.MPI_LIBRARY_VERSION_STRING
    println("MPI bound at build: ", v)
    @assert occursin("Open MPI", v) "MPI did not bind system Open MPI: $v"'

# 2) One-time: ensure PackageCompiler is available in the project.
julia --project="$PROJ" -e 'using Pkg; haskey(Pkg.project().dependencies, "PackageCompiler") || Pkg.add("PackageCompiler")'

# 3) Build the sysimage. Single MPI rank so warmup.jl can trace the distributed
#    path too; the full cold compile happens here, once.
mpiexecjl --project="$PROJ" -n 1 julia --project="$PROJ" "$PROJ/compile/compile.jl"

# 4) Stamp the image with a hash of src/*.jl, so the launchers can detect later
#    that the solver was edited and the image is stale (run/balfem_env.sh,
#    balfem_check_sysimage_freshness). Without the stamp they fall back to a
#    coarser mtime comparison.
source "$PROJ/run/balfem_env.sh"
balfem_write_sysimage_stamp "$PROJ/GridapBALFEM_sysimage.so" "$PROJ/src"

echo "Sysimage built: $PROJ/GridapBALFEM_sysimage.so"
