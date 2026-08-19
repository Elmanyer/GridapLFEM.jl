#!/bin/bash
# ==============================================================
#  run_mms_validate.sh — the analytic-MMS gate suite (PRE-COMMIT CHECK)
#
#  Runs the cheap forcing gates first and ABORTS if they fail: if the forcing is
#  wrong there is no point measuring a convergence rate, and a rate measured
#  against a wrong forcing is actively misleading.
#
#  COVERS BOTH THE LINEAR AND THE NONLINEAR MMS. The nonlinear forcing gates
#  (test_mms_forcing_nonlinear.jl) bracket each nonlinear model between two
#  already-verified ones — Model 4 at constant h must equal Model 3 exactly, and
#  the nonlinear-minus-linear gap must scale as ε² — so a term added at the wrong
#  amplitude order is caught in seconds instead of in a 30-minute rate study.
#
#  ⚠ ASSERT ON GATE OUTPUT, NEVER ON THE EXIT CODE. A test file guarded by
#  `if abspath(PROGRAM_FILE) == @__FILE__` prints its header and returns cleanly
#  having executed NOTHING. Every check below greps for the PASS/FAIL summary
#  line and treats a MISSING summary as a failure, which is why `0 FAIL` alone is
#  not sufficient — the line must exist.
#
#  Cost: forcing gates seconds (both files); convergence a few minutes for the
#  linear models, tens of minutes for the nonlinear ones (hence opt-in).
#  ENV:  WITH_NONLINEAR_RATES=1  also run test_mms_convergence_nonlinear.jl
#  USAGE: run/local/run_mms_validate.sh
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh
OUT=output/local/logs; mkdir -p "$OUT"

# gate <logfile> <description> — fails unless a "N PASS, 0 FAIL" summary is present
gate() {
    local log="$1" desc="$2"
    if ! grep -qE '[0-9]+ PASS,[[:space:]]+0 FAIL' "$log"; then
        echo
        echo "!!! $desc FAILED (or produced no gate output at all)."
        echo "!!! A rate measured against a wrong forcing means nothing."
        echo "!!! Log: $log"
        exit 1
    fi
}

echo "=== 1/3  LINEAR forcing gates (independent of the residual code) ==="
balfem_local_run test/test_mms_forcing.jl 2>&1 | tee "$OUT/mms_forcing.log"
gate "$OUT/mms_forcing.log" "linear forcing gates"

echo
echo "=== 2/3  NONLINEAR forcing gates (models 3 and 4, :none tier) ==="
balfem_local_run test/test_mms_forcing_nonlinear.jl 2>&1 | tee "$OUT/mms_forcing_nl.log"
gate "$OUT/mms_forcing_nl.log" "nonlinear forcing gates"

echo
echo "=== 3/3  order of accuracy (the verification result) ==="
balfem_local_run test/test_mms_convergence.jl 2>&1 | tee "$OUT/mms_convergence.log"

if [ "${WITH_NONLINEAR_RATES:-0}" != "0" ]; then
    echo
    echo "=== extra  nonlinear order of accuracy (models 3 and 4) ==="
    echo "    Tens of minutes. Model 4 is quasi-Newton-limited and carries a"
    echo "    raised nl_iter — if it still stalls, raise the BUDGET, never nl_tol."
    balfem_local_run test/test_mms_convergence_nonlinear.jl 2>&1 \
        | tee "$OUT/mms_convergence_nl.log"
fi
