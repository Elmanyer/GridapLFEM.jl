#!/bin/bash
# ==============================================================
#  run_mms_time.sh — TEMPORAL convergence of the linear flat-bed operator
#
#  Fixed fine mesh, dt halved four times. SDIRK_2_2 ⇒ expected slope 2.
#  Set LFEM_MMS_SOLVER=theta to measure Crank-Nicolson instead (also 2).
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh
export LFEM_MMS_MODE=time LFEM_MMS_LEVELS="${LEVELS:-4}"
export LFEM_MMS_NXF=24 LFEM_MMS_NYF=16 LFEM_MMS_DT0_T=8e-3 LFEM_MMS_TFINAL_T=0.08
export LFEM_MMS_SOLVER="${SOLVER:-sdirk}"
export LFEM_MMS_OUT=output/local/mms_time
mkdir -p output/local/logs
lfem_local_run examples/local_mms/run_mms_convergence.jl 2>&1 | tee output/local/logs/mms_time.log
