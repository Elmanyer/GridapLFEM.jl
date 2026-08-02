# ==============================================================
#  compile.jl — build the GridapLFEM system image
#
#  GridapLFEM is loaded by include() (it is NOT a Pkg.add'd package), so we bake
#  the DEPENDENCY packages into the sysimage and let warmup.jl trace-compile the
#  LFE-M solver code (residual, Jacobians, time loops, reconstruction) on top of
#  them, for BOTH the sequential (CellField) and distributed (DistributedCellField)
#  code paths.
#
#  PREREQUISITE: LocalPreferences.toml must already pin MPI to the system OpenMPI,
#  otherwise the sysimage bakes in the bundled MPICH and crashes under the cluster
#  launcher. Run compile/set_preferences.jl first (compile_snellius.sh does this).
#
#  BUILD (single rank; MPI available so the distributed trace works):
#      mpiexecjl -n 1 julia --project=$HOME/GridapLFEM.jl compile/compile.jl
#  USE:
#      julia --project=$HOME/GridapLFEM.jl -J$HOME/GridapLFEM.jl/GridapLFEM_sysimage.so script.jl
# ==============================================================

using PackageCompiler

create_sysimage(
    [:Gridap, :GridapDistributed, :GridapSolvers, :PartitionedArrays,
     :MPI, :BlockArrays, :WaveSpec];
    sysimage_path             = joinpath(@__DIR__, "..", "GridapLFEM_sysimage.so"),
    precompile_execution_file = joinpath(@__DIR__, "warmup.jl"),
)
