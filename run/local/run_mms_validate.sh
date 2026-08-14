#!/bin/bash
# ==============================================================
#  run_mms_validate.sh — the analytic-MMS gate suite (PRE-COMMIT CHECK)
#
#  Runs the cheap forcing gates first and ABORTS if they fail: if the forcing is
#  wrong there is no point measuring a convergence rate, and a rate measured
#  against a wrong forcing is actively misleading.
#
#  Cost: forcing gates seconds; convergence a few minutes (small meshes, LU).
#  USAGE: run/local/run_mms_validate.sh
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh
OUT=output/local/logs; mkdir -p "$OUT"

echo "=== 1/2  forcing gates (independent of the residual code) ==="
lfem_local_run test/test_mms_forcing.jl 2>&1 | tee "$OUT/mms_forcing.log"
if ! grep -q "0 FAIL" "$OUT/mms_forcing.log"; then
    echo
    echo "!!! FORCING GATES FAILED — not running the convergence study."
    echo "!!! A rate measured against a wrong forcing means nothing."
    exit 1
fi

echo
echo "=== 2/2  order of accuracy (the verification result) ==="
lfem_local_run test/test_mms_convergence.jl 2>&1 | tee "$OUT/mms_convergence.log"
