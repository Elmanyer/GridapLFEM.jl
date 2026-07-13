using Preferences, MPIPreferences
using Pkg

# Set MPIPreferences to use the system MPI binary
#MPIPreferences.use_system_binary()

### DelftBlue
mpi_lib_dir = "/apps/arch/2024r1/software/linux-rhel8-cascadelake/gcc-11.3.0/openmpi-4.1.6-w6w5qi5ljesbctyoojlfialbynqt25jb/lib/"

### Snellius
#mpi_lib_dir = "/sw/arch/RHEL9/EB_production/2024/software/OpenMPI/5.0.3-GCC-13.3.0/lib/"

# Manually set the keys to ensure Julia never looks at artifacts
set_preferences!(
    MPIPreferences,
    "binary" => "system",
    "libmpi" => joinpath(mpi_lib_dir, "libmpi.so"),
    "abi" => "OpenMPI",
    "preloads" => [
        joinpath(mpi_lib_dir, "libmpi_mpifh.so"),
        joinpath(mpi_lib_dir, "libmpi_usempif08.so")
    ];
    force = true
)

# Add specific branches of Gridap and GridapDistributed and GridapSolvers
Pkg.add([PackageSpec(name="Gridap", rev="distributed_AD_transient"),
         PackageSpec(name="GridapDistributed", rev="distributed_AD_transient"),
         PackageSpec(url="https://github.com/Elmanyer/GridapSolvers.jl.git", rev="fix_num_fields")
        ])
Pkg.instantiate()

# Execution in terminal
# julia --project=. set_preferences.jl

