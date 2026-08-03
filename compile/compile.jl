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

# ── MPI PREFLIGHT ─────────────────────────────────────────────────────────────
# The JLL-vs-system choice is frozen into the sysimage at THIS point (it is a
# precompile-time @static if on MPIPreferences.binary). Refuse to build unless
# MPI is actually bound to the SYSTEM OpenMPI in this process — otherwise we bake
# an OpenMPI_jll image that crashes at launch with `undefined symbol:
# opal_single_threaded`. `using MPI` here also forces MPI to be (re)compiled
# against the system binary before create_sysimage bakes it.
using MPIPreferences
MPIPreferences.binary == "system" || error("""
    MPIPreferences.binary = $(repr(MPIPreferences.binary))  (expected "system").
    LocalPreferences.toml is NOT pinning the system OpenMPI for this project.
    Fix: source compile/load_modules_snellius.sh, then
         julia --project=$(dirname(Base.active_project())) compile/set_preferences.jl
    and rebuild.""")

using MPI
const _LIBVER = MPI.MPI_LIBRARY_VERSION_STRING
occursin("Open MPI", _LIBVER) || error("""
    MPI bound the WRONG library at build time:
        $_LIBVER
    Expected system Open MPI (5.0.3). This usually means the OpenMPI module was
    not loaded (LD_LIBRARY_PATH missing its lib) or MPIPreferences.libmpi points
    at a JLL artifact instead of the cluster library.""")
@info "MPI preflight OK — baking sysimage against SYSTEM OpenMPI" MPIPreferences.binary libmpi=MPIPreferences.libmpi version=_LIBVER
# ──────────────────────────────────────────────────────────────────────────────

using PackageCompiler

create_sysimage(
    [:Gridap, :GridapDistributed, :GridapSolvers, :PartitionedArrays,
     :MPI, :BlockArrays, :WaveSpec];
    sysimage_path             = joinpath(@__DIR__, "..", "GridapLFEM_sysimage.so"),
    precompile_execution_file = joinpath(@__DIR__, "warmup.jl"),
)
