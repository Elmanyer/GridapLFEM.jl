#!/bin/bash
# ==============================================================
#  run_mms_time.sh — TEMPORAL convergence of the linear flat-bed operator
#
#  Fixed fine mesh, dt halved four times. SDIRK_2_2 ⇒ expected slope 2.
#  Set BALFEM_MMS_SOLVER=theta to measure Crank-Nicolson instead (also 2).
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh
export BALFEM_MMS_MODE=time BALFEM_MMS_LEVELS="${LEVELS:-4}"
export BALFEM_MMS_NXF=24 BALFEM_MMS_NYF=16 BALFEM_MMS_DT0_T=8e-3 BALFEM_MMS_TFINAL_T=0.08
export BALFEM_MMS_SOLVER="${SOLVER:-sdirk}"
export BALFEM_MMS_OUT=output/local/mms_time
mkdir -p output/local/logs
balfem_local_run examples/local_mms/run_mms_convergence.jl 2>&1 | tee output/local/logs/mms_time.log
