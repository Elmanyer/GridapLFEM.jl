#!/bin/bash
# GATE T — does the distributed linear tolerance CAP the spatial error?
# Re-runs the finest level with tolerances tightened 2 further orders. If |e| moves
# by >1%, the study is tolerance-capped and its convergence rate is VOID.
# Sequential needs no such gate: LUSolver is direct and the linear regime makes
# Newton converge to round-off in one step.
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh; mkdir -p output/local/logs
for RT in 1e-13 1e-15; do
  echo "=== ls_rtol=$RT ==="
  export LFEM_CONV_PU="${PU:-4}" LFEM_CONV_DOMAIN=d1 LFEM_CONV_MODE=static
  export LFEM_CONV_LEVELS=1 LFEM_CONV_NX0="${NX0:-128}" LFEM_CONV_DIST=1
  export LFEM_CONV_PX=4 LFEM_CONV_PY=1 LFEM_CONV_LSRTOL="$RT" LFEM_CONV_NLTOL="$RT"
  export LFEM_CONV_DT=1e-5 LFEM_CONV_NSTEPS=100
  export LFEM_CONV_OUT="output/local/mms_conv_tol_$RT"
  lfem_local_mpi 4 examples/local_mms/run_mms_matrix.jl 2>&1 | tee "output/local/logs/conv_tolgate_$RT.log"
done
echo "Compare e_eta / e_u between the two: >1% relative change ⇒ TOLERANCE-CAPPED."
