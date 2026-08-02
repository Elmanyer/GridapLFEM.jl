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

using Preferences, MPIPreferences, Pkg

### DelftBlue
#mpi_lib_dir = "/apps/arch/2024r1/software/linux-rhel8-cascadelake/gcc-11.3.0/openmpi-4.1.6-w6w5qi5ljesbctyoojlfialbynqt25jb/lib/"

# Snellius OpenMPI/5.0.3-GCC-13.3.0 (matches load_modules_snellius.sh)
mpi_lib_dir = "/sw/arch/RHEL9/EB_production/2024/software/OpenMPI/5.0.3-GCC-13.3.0/lib/"

isfile(joinpath(mpi_lib_dir, "libmpi.so")) || error(
    "libmpi.so not found in $mpi_lib_dir — load the OpenMPI module first " *
    "(`source compile/load_modules_snellius.sh`) and verify the path.")

# Manually pin the keys so MPI.jl never falls back to the JLL artifact.
set_preferences!(
    MPIPreferences,
    "binary"  => "system",
    "libmpi"  => joinpath(mpi_lib_dir, "libmpi.so"),
    "mpiexec" => "mpiexec",
    "abi"     => "OpenMPI",
    "preloads" => [
        joinpath(mpi_lib_dir, "libmpi_mpifh.so"),
        joinpath(mpi_lib_dir, "libmpi_usempif08.so"),
    ];
    force = true,
)

Pkg.instantiate()
@info "MPIPreferences pinned to system OpenMPI — LocalPreferences.toml written" mpi_lib_dir
