# ============================================================================
#  lfem_env.sh — shared launcher environment for every GridapLFEM SLURM script
#
#  Sourced (not executed) by every script in run/ and run/dist_small/. It does
#  the three things that are identical across all cases, and that MUST agree
#  with how the sysimage was built:
#
#    (1) load the SAME cluster modules used at build time (OpenMPI + Julia),
#    (2) resolve + verify the prebuilt sysimage GridapLFEM_sysimage.so,
#    (3) expose `lfem_run <nranks> <script.jl>` which launches the run with
#        the SAME --project (holds LocalPreferences.toml, i.e. the system-MPI
#        binding) and -J<sysimage>.
#
#  WHY THE SYSIMAGE. Without -J, every rank JIT-compiles the full Gridap FEM
#  stack independently: ~30-45 min of wall time per run, and a ~4-8 GB/rank
#  memory spike that OOM'd whole nodes at 32-128 ranks (which is why these jobs
#  used to need fat_rome). With -J the ranks mmap one shared, already-compiled
#  image: no compile, no spike, and the jobs fit the cheap rome/L1 budget at the
#  node-default 2 GB/core — do NOT add --mem-per-cpu to get memory back, that
#  only makes SLURM bill extra cores per rank.
#
#  Build the image once (same partition + same OpenMPI module as the runs):
#      cd $HOME/GridapLFEM.jl/compile && sbatch compile_snellius.sh
#  Rebuild it after any change to src/*.jl or a package/OpenMPI upgrade;
#  editing a launcher or its env vars needs no rebuild. Full walkthrough and
#  troubleshooting: compile/README.md.
#
#  USAGE in a launcher
#      source $HOME/GridapLFEM.jl/run/lfem_env.sh
#      export LFEM_PX=8; export LFEM_PY=4          # 8*4 = 32 ranks
#      export LFEM_REGIME=linear                   # case-specific overrides
#      lfem_run 32 examples/distributed_small/run_periodic_plane_small.jl
#
#  KNOBS (export before sourcing, or in the submit environment)
#      LFEM_PROJ         project root            (default $HOME/GridapLFEM.jl)
#      LFEM_CLUSTER      snellius | blue         (default snellius)
#      LFEM_SYSIMAGE     path to the .so         (default $LFEM_PROJ/GridapLFEM_sysimage.so)
#      LFEM_NO_SYSIMAGE  1 = run without -J      (fallback while the image is
#                                                 stale/rebuilding; the ranks
#                                                 then JIT-compile — expect the
#                                                 old cost and memory spike)
# ============================================================================

set -euo pipefail

: "${LFEM_PROJ:=$HOME/GridapLFEM.jl}"
: "${LFEM_CLUSTER:=snellius}"
: "${LFEM_SYSIMAGE:=$LFEM_PROJ/GridapLFEM_sysimage.so}"

# (1) the SAME OpenMPI module the sysimage was built against — a mismatch here
#     is what produces the launch-time MPI symbol errors.
source "$LFEM_PROJ/compile/load_modules_${LFEM_CLUSTER}.sh"

# ----------------------------------------------------------------------------
#  lfem_run <nranks> <julia-script> [script args...]
#    <julia-script> is relative to $LFEM_PROJ (absolute paths also accepted).
# ----------------------------------------------------------------------------
lfem_run() {
    local nranks="$1"; shift
    local script="$1"; shift
    [[ "$script" == /* ]] || script="$LFEM_PROJ/$script"

    [ -f "$script" ] || {
        echo "[lfem_env] ERROR: run script not found: $script" >&2
        exit 1
    }

    # (2) resolve the sysimage
    local -a jflag=()
    local simg_msg
    if [ "${LFEM_NO_SYSIMAGE:-0}" = "1" ]; then
        simg_msg="<disabled: LFEM_NO_SYSIMAGE=1 — ranks will JIT-compile>"
    else
        [ -f "$LFEM_SYSIMAGE" ] || {
            echo "[lfem_env] ERROR: sysimage not found: $LFEM_SYSIMAGE" >&2
            echo "[lfem_env]   build it once:  cd $LFEM_PROJ/compile && sbatch compile_${LFEM_CLUSTER}.sh" >&2
            echo "[lfem_env]   or set LFEM_NO_SYSIMAGE=1 to run without it (slow, memory-hungry)." >&2
            exit 1
        }
        jflag=(-J"$LFEM_SYSIMAGE")
        simg_msg="$LFEM_SYSIMAGE"
    fi

    echo "[lfem_env] project  : $LFEM_PROJ"
    echo "[lfem_env] cluster  : $LFEM_CLUSTER"
    echo "[lfem_env] sysimage : $simg_msg"
    echo "[lfem_env] ranks    : $nranks"
    echo "[lfem_env] script   : $script"

    # (3) --project must be the project holding LocalPreferences.toml, on BOTH
    #     mpiexecjl and julia; -J adds the image built with (1)+(2) active.
    mpiexecjl --project="$LFEM_PROJ" -n "$nranks" \
        julia --project="$LFEM_PROJ" ${jflag[@]+"${jflag[@]}"} "$script" "$@"
}
