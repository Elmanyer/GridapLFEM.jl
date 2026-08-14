#!/bin/bash
#  GATE T (parallel) — does the distributed linear tolerance CAP the spatial error?
#  Runs the SAME finest-level case at ls_rtol 1e-13 and 1e-15 CONCURRENTLY (2x4=8 cores).
#  >1% change in |e| ⇒ GMRES is capping the error and EVERY distributed rate is VOID.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT=output/local/logs; mkdir -p "$OUT"
run_tol() {
    rt="$1"
    LFEM_CONV_PU="${PU:-3}" LFEM_CONV_DOMAIN=d1 LFEM_CONV_MODE=static \
    LFEM_CONV_LEVELS=1 LFEM_CONV_NX0="${NX0:-64}" LFEM_CONV_NY0=3 \
    LFEM_CONV_DIST=1 LFEM_CONV_PX=4 LFEM_CONV_PY=1 \
    LFEM_CONV_LSRTOL="$rt" LFEM_CONV_NLTOL="$rt" \
    LFEM_CONV_DT=1e-5 LFEM_CONV_NSTEPS=100 \
    LFEM_CONV_OUT="output/local/mms_conv_tol_$rt" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
        stdbuf -oL -eL "$HOME/.julia/bin/mpiexecjl" --project=. -n 4 \
        julia --project=. examples/local_mms/run_mms_matrix.jl > "$OUT/gateT_$rt.log" 2>&1
    echo "  ls_rtol=$rt : $(grep -oE 'e_eta=[0-9.e+-]+  e_u=[0-9.e+-]+' "$OUT/gateT_$rt.log" | tail -1)"
}
export -f run_tol; export OUT
echo "=== GATE T, 2 concurrent x 4 ranks, $(date '+%H:%M:%S') ==="
printf '%s\n' 1e-13 1e-15 | xargs -P 2 -I{} bash -c 'run_tol "$@"' _ {}
echo "--- verdict: >1% relative change between the two ⇒ TOLERANCE-CAPPED, rates VOID ---"
