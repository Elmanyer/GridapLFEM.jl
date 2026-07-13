
module load 2024r1 slurm openmpi julia

export MPI_ROOT=$(dirname $(mpicc --showme:libdirs | awk '{print $1}'))
export MPI_LIB_DIR="$MPI_ROOT/lib"
export LD_LIBRARY_PATH=$MPI_LIB_DIR:$LD_LIBRARY_PATH

export LD_PRELOAD="$MPI_LIB_DIR/libmpi.so"
export JULIA_MPI_BINARY=system
export JULIA_MPI_PATH=$MPI_ROOT
