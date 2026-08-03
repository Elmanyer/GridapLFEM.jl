# ==============================================================
#  set_preferences.jl — pin MPI.jl to the cluster's SYSTEM OpenMPI
#
#  WHY: without this, MPI.jl uses its bundled JLL binary (MPICH). A sysimage
#  built in that state bakes in MPICH, and launching it under the cluster's
#  OpenMPI mpiexec/PMIx then crashes (implementation mismatch — the "continuous
#  run kills"). This writes LocalPreferences.toml so MPI.jl binds to the system
#  OpenMPI at BOTH build and run time.
#
#  RUN IT ONCE, with the SAME --project used to build the sysimage AND to launch
#  runs (so the binding lands in the right LocalPreferences.toml):
#      source compile/load_modules_snellius.sh
#      julia --project=$HOME/GridapLFEM.jl compile/set_preferences.jl
#
#  The libmpi path below MUST match the OpenMPI module in load_modules_snellius.sh
#  (Snellius: OpenMPI/5.0.3-GCC-13.3.0). If you change the module, change this.
# ==============================================================

using MPIPreferences, Pkg

### DelftBlue
#mpi_lib_dir = "/apps/arch/2024r1/software/linux-rhel8-cascadelake/gcc-11.3.0/openmpi-4.1.6-w6w5qi5ljesbctyoojlfialbynqt25jb/lib/"

# Snellius OpenMPI/5.0.3-GCC-13.3.0 (matches load_modules_snellius.sh)
mpi_lib_dir = "/sw/arch/RHEL9/EB_production/2024/software/OpenMPI/5.0.3-GCC-13.3.0/lib/"
libmpi = joinpath(mpi_lib_dir, "libmpi.so")

isfile(libmpi) || error(
    "libmpi.so not found in $mpi_lib_dir — load the OpenMPI module first " *
    "(`source compile/load_modules_snellius.sh`) and verify the path.")

# Use the OFFICIAL switch rather than a hand-rolled set_preferences! block. It
# dlopens the system libmpi, auto-detects the vendor/ABI, and writes ALL required
# keys (including the `_format` marker the hand-rolled block was missing — that
# omission makes MPIPreferences silently fall back to the JLL, which is exactly
# how a sysimage ends up baked against OpenMPI_jll). `force=true` overwrites any
# stale block.
MPIPreferences.use_system_binary(;
    library_names = [libmpi],
    mpiexec       = "mpiexec",
    force         = true,
)

# Rebuild MPI's precompile cache now so the system binding is in place BEFORE the
# sysimage build bakes it.
Pkg.precompile()
@info "MPIPreferences pinned to system OpenMPI via use_system_binary()" libmpi binary=MPIPreferences.binary abi=MPIPreferences.abi
