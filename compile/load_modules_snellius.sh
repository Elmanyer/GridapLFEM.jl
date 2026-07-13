
module load 2024 Julia/1.10.4-linux-x86_64 OpenMPI/5.0.3-GCC-13.3.0

export MPI_LIB_DIR=$(mpicc --showme:libdirs | awk '{print $1}')
export LD_LIBRARY_PATH=$MPI_LIB_DIR:$LD_LIBRARY_PATH