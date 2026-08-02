#!/bin/bash
# ============================================================================
#  SMALL run WITH SYSTEM IMAGE — linear periodic plane wave, flat bed, A=0.001
#
#  Same case as run_lin_periodic_plane_small.sh, but launched against the
#  prebuilt sysimage (-J) so the 32 ranks LOAD the compiled code instead of each
#  JIT-recompiling it. This removes both the ~30-45 min per-rank compile and the
#  OOM that the simultaneous compiles caused.
#
#  PREREQUISITE — build the sysimage once (same partition, same OpenMPI module):
#      cd $HOME/GridapLFEM.jl/compile && sbatch compile_snellius.sh
#  It writes  $HOME/GridapLFEM.jl/GridapLFEM_sysimage.so.
#
#  The three things that MUST match between build and run (or MPI mismatches):
#    (1) the SAME OpenMPI module  → source load_modules_snellius.sh below,
#    (2) the SAME --project       → LocalPreferences.toml (system-MPI binding),
#    (3) the -J sysimage built with (1)+(2) active.
# ============================================================================
#SBATCH --job-name="LFEMsi_lin_plane"
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
PROJ=$HOME/GridapLFEM.jl
SYSIMG=$PROJ/GridapLFEM_sysimage.so

source "$PROJ/compile/load_modules_snellius.sh"    # (1) SAME OpenMPI used to build

[ -f "$SYSIMG" ] || { echo "ERROR: $SYSIMG not found — build it with compile/compile_snellius.sh"; exit 1; }

# --- MPI process grid (PX*PY MUST equal mpiexecjl -n) ---
export LFEM_PX=8
export LFEM_PY=4            # 8*4 = 32 ranks

# --- Case-specific physics overrides (only what differs from the script base) ---
export LFEM_REGIME=linear
export LFEM_NL_PRESSURE=none

# (2) --project = the project holding LocalPreferences.toml ; (3) -J the sysimage
mpiexecjl --project="$PROJ" -n 32 julia --project="$PROJ" \
    -J"$SYSIMG" \
    "$PROJ/examples/distributed_small/run_periodic_plane_small.jl"
