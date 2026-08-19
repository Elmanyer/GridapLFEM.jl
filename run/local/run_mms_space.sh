#!/bin/bash
# ==============================================================
#  run_mms_space.sh — SPATIAL convergence of the linear flat-bed operator
#
#  Four levels, Q2 ⇒ expected slope 3 in L². The time step is small and FIXED so
#  the O(dt^2) error stays below the finest spatial error; if the last pairwise
#  rate sags, that is the signature of the temporal error taking over — lower
#  BALFEM_MMS_DT0 rather than believing the fitted slope.
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh
export BALFEM_MMS_MODE=space BALFEM_MMS_LEVELS="${LEVELS:-4}"
export BALFEM_MMS_NX0=6 BALFEM_MMS_NY0=4 BALFEM_MMS_DT0=2e-4 BALFEM_MMS_TFINAL=0.02
export BALFEM_MMS_OUT=output/local/mms_space
mkdir -p output/local/logs
balfem_local_run examples/local_mms/run_mms_convergence.jl 2>&1 | tee output/local/logs/mms_space.log
