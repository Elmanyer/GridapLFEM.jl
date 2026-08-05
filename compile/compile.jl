# ==============================================================
#  compile.jl — build the GridapLFEM system image
#
#  GridapLFEM is a Julia PACKAGE (name+uuid in ../Project.toml), so it is baked
#  into the image alongside its dependencies, and warmup.jl then trace-compiles
#  the LFE-M solver kernels (residual, Jacobians, time loops, reconstruction) for
#  BOTH the sequential (CellField) and distributed (DistributedCellField) paths.
#
#  WHY THIS MATTERS (the bug this fixed): the solver used to be loaded with
#  `include(src/GridapLFEM.jl)`, creating a fresh `Main.GridapLFEM` in every
#  process. PackageCompiler only retains code belonging to the packages it bakes,
#  so the solver's types and — far more expensive — every Gridap FEM
#  specialisation keyed on them were recompiled in EVERY rank at EVERY run: the
#  image removed the library compile but never the application compile. Listing
#  :GridapLFEM below is what actually delivers "the ranks load, they do not
#  compile". (The 2026-08-04 exit-137 kills were first blamed on this; the
#  demonstrated cause was instead a GMRES cache over-allocation — see krylov_m
#  vs ls_maxiter in timeloop_dist.jl. Both were real and both are fixed.)
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

# ── PACKAGE PREFLIGHT ─────────────────────────────────────────────────────────
# The whole point of the build is to bake the SOLVER, not just its dependencies.
# If GridapLFEM is not resolvable as a package in this environment (e.g. the
# active project is not the package root, or name/uuid were lost from
# Project.toml), create_sysimage would either fail late or — worse — silently
# produce an image without the solver, reintroducing the per-rank compile + OOM.
# Fail loudly here instead.
let id = Base.identify_package("GridapLFEM")
    id === nothing && error("""
        GridapLFEM is not resolvable as a package from the active project
            $(Base.active_project())
        Expected `name = "GridapLFEM"` and a `uuid` in its Project.toml.
        Build with:  mpiexecjl --project=<GridapLFEM.jl> -n 1 julia --project=<GridapLFEM.jl> compile/compile.jl""")
    @info "package preflight OK — the solver will be baked into the image" uuid = id.uuid
end
using GridapLFEM   # must load cleanly before we try to bake it

using PackageCompiler

# include_transitive_dependencies=false is LOAD-BEARING for system MPI:
# with the default `true`, PackageCompiler does `using` on EVERY transitive
# Manifest dependency to bake it — including OpenMPI_jll, which is a dep of MPI
# that is NOT imported under MPIPreferences.binary="system". That force-load bakes
# OpenMPI_jll's initializer into the image; it then fires at startup, dlopen's the
# JLL artifact libmpi.so, and crashes with `undefined symbol: opal_single_threaded`.
# Disabling the transitive sweep bakes only the packages actually loaded (which use
# the system libmpi), so the JLL never enters the image. The explicitly listed
# packages + what they load are still baked; genuinely-unused deps just recompile
# on demand at runtime (negligible).
create_sysimage(
    [:GridapLFEM,                                    # the solver itself — load-bearing
     :Gridap, :GridapDistributed, :GridapSolvers, :PartitionedArrays,
     :MPI, :BlockArrays, :WaveSpec];
    sysimage_path                 = joinpath(@__DIR__, "..", "GridapLFEM_sysimage.so"),
    precompile_execution_file     = joinpath(@__DIR__, "warmup.jl"),
    include_transitive_dependencies = false,
)
