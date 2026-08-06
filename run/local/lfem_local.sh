#!/bin/bash
# ==============================================================
#  lfem_local.sh — shared helper for LOCAL runs (this workstation, 12-rank partitions)
#
#  The local counterpart of run/lfem_env.sh. That helper is cluster-only: it
#  loads Snellius/DelftBlue modules and resolves the prebuilt system image.
#  Locally there is no sysimage and no scheduler, so this helper only has to
#  resolve the project, cap the rank count, and launch.
#
#  SOURCE IT, then call one of:
#      lfem_local_run  <script.jl>            # sequential (KEEPS point gauges)
#      lfem_local_mpi  <nranks> <script.jl>   # MPI, nranks ≤ LFEM_MAX_RANKS
#
#  Sequential is the default for anything that measures: the distributed driver
#  evaluates no point gauges (timeloop_dist.jl:21-22), so an MPI local run can
#  only be judged from the diagnostics CSV. Use lfem_local_mpi when exercising
#  the DISTRIBUTED path is the point of the run.
#
#  PARTITION SIZE. This workstation has 16 cores; a simulation is given a
#  12-rank partition, leaving 4 cores for the OS and for the analysis tooling.
#  The consequence is that only ONE simulation runs at a time — 12 ranks per
#  case is a deliberate trade of case throughput for per-case size/resolution,
#  not a free speed-up. (Two 6-rank partitions side by side would finish a
#  multi-case set sooner; a single 12-rank partition runs a bigger case.)
#
#  Knobs
#    LFEM_PROJ        project directory      (default: this file's ../..)
#    LFEM_MAX_RANKS   hard cap on ranks      (default 12 of this box's 16 cores)
#    JULIA            julia binary           (default: julia on PATH)
#    MPIEXECJL        MPI launcher           (default ~/.julia/bin/mpiexecjl)
#
#  Notes
#   * ALWAYS use mpiexecjl, never the system mpiexec: the latter fails on this
#     machine with a PMIx version mismatch.
#   * MPI_Finalize prints a benign OFI error from the WiFi NIC and exits 143.
#     lfem_local_mpi treats 143 as success — do not "fix" that by hiding real
#     failures: any other non-zero status is still reported.
#   * Julia buffers stdout hard when redirected, so stdbuf is used throughout;
#     otherwise a running job's log file looks empty until it exits.
# ==============================================================

_lfem_local_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LFEM_PROJ="${LFEM_PROJ:-$(cd "$_lfem_local_here/../.." && pwd)}"
LFEM_MAX_RANKS="${LFEM_MAX_RANKS:-12}"
JULIA="${JULIA:-julia}"
MPIEXECJL="${MPIEXECJL:-$HOME/.julia/bin/mpiexecjl}"

lfem_local_banner() {
    echo "--------------------------------------------------------------"
    echo " GridapLFEM local run"
    echo "   project : $LFEM_PROJ"
    echo "   script  : $1"
    echo "   mode    : $2"
    echo "   started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------"
}

# --- sequential ------------------------------------------------------------
lfem_local_run() {
    local script="$1"
    [ -n "$script" ] || { echo "lfem_local_run: no script given" >&2; return 2; }
    [ -f "$LFEM_PROJ/$script" ] || [ -f "$script" ] || {
        echo "lfem_local_run: script not found: $script" >&2; return 2; }
    [ -f "$script" ] || script="$LFEM_PROJ/$script"

    lfem_local_banner "$script" "sequential (gauges available)"
    stdbuf -oL -eL "$JULIA" --project="$LFEM_PROJ" "$script"
    local rc=$?
    echo "--- exit status $rc ---"
    return $rc
}

# --- MPI -------------------------------------------------------------------
lfem_local_mpi() {
    local n="$1"; local script="$2"
    [ -n "$n" ] && [ -n "$script" ] || {
        echo "lfem_local_mpi: usage lfem_local_mpi <nranks> <script.jl>" >&2; return 2; }
    if [ "$n" -gt "$LFEM_MAX_RANKS" ]; then
        echo "lfem_local_mpi: refusing $n ranks — this machine is capped at" \
             "$LFEM_MAX_RANKS (raise LFEM_MAX_RANKS deliberately if you mean it)" >&2
        return 2
    fi
    [ -f "$script" ] || script="$LFEM_PROJ/$script"
    [ -f "$script" ] || { echo "lfem_local_mpi: script not found: $2" >&2; return 2; }
    [ -x "$MPIEXECJL" ] || {
        echo "lfem_local_mpi: $MPIEXECJL not found. Install it with" >&2
        echo "    julia -e 'using MPI; MPI.install_mpiexecjl()'" >&2
        return 2; }

    lfem_local_banner "$script" "MPI, $n ranks (NO point gauges — read diagnostics.csv)"
    export LFEM_MPI=1
    stdbuf -oL -eL "$MPIEXECJL" --project="$LFEM_PROJ" -n "$n" \
        "$JULIA" --project="$LFEM_PROJ" "$script"
    local rc=$?
    # 143 = SIGTERM during MPI_Finalize: the documented benign OFI cleanup error
    # on this machine. The computation has already completed at that point.
    if [ "$rc" -eq 143 ]; then
        echo "--- exit status 143 (benign MPI_Finalize/OFI cleanup — treated as success) ---"
        return 0
    fi
    echo "--- exit status $rc ---"
    return $rc
}
