#!/bin/bash
# ==============================================================
#  run_local_tests.sh — the LOCAL validation suite
#
#  Runs test/local/*.jl as INDEPENDENT Julia processes, up to $JOBS at a time.
#  Process-level concurrency (not MPI) is the right tool here: every test is
#  gauge-based and the distributed driver has no point gauges, so each test is
#  a sequential run and the cores are spent on running several tests at once.
#
#  The ~200 s FEM JIT compile is paid ONCE PER PROCESS, so a file that runs
#  several cases amortises it — which is why the tests are grouped by theme
#  rather than one case per file.
#
#  USAGE
#    test/local/run_local_tests.sh                # all tests, 4 concurrent
#    JOBS=6 test/local/run_local_tests.sh         # more concurrency
#    test/local/run_local_tests.sh reststate sponge     # a subset (substring match)
#    LFEM_TEST_PERIODS=8 test/local/run_local_tests.sh boundary   # a shorter gate
#
#  Logs land in $LOG_DIR (default output/local/logs); the exit status is
#  non-zero if any test failed.
# ==============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
JOBS="${JOBS:-4}"
LOG_DIR="${LOG_DIR:-$PROJ/output/local/logs}"
JULIA="${JULIA:-julia}"

# Ordered cheapest-first: reststate is ~1 min and fails loudly if the residual
# or the diagnostics are broken, so a mistake is caught before the long ones.
ALL_TESTS=(
    "test_reststate_1d.jl"
    "test_sponge_1d.jl"
    "test_relaxation_1d.jl"
    "test_2d_reduces_to_1d.jl"
    "test_boundary_modes_1d.jl"
)

# Optional substring filters from the command line.
TESTS=()
if [ "$#" -gt 0 ]; then
    for pat in "$@"; do
        for t in "${ALL_TESTS[@]}"; do
            [[ "$t" == *"$pat"* ]] && TESTS+=("$t")
        done
    done
    [ "${#TESTS[@]}" -eq 0 ] && { echo "no test matches: $*"; exit 2; }
else
    TESTS=("${ALL_TESTS[@]}")
fi

mkdir -p "$LOG_DIR"
echo "=============================================================="
echo " GridapLFEM local validation suite"
echo "   project     : $PROJ"
echo "   tests       : ${TESTS[*]}"
echo "   concurrency : $JOBS"
echo "   logs        : $LOG_DIR"
echo "=============================================================="

declare -A PIDS
start_all=$(date +%s)

for t in "${TESTS[@]}"; do
    # Throttle to $JOBS concurrent processes.
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    log="$LOG_DIR/${t%.jl}.log"
    echo "  [start] $t  → $log"
    # stdbuf keeps the log readable while the test runs: Julia buffers stdout
    # hard when it is redirected to a file.
    stdbuf -oL -eL "$JULIA" --project="$PROJ" "$HERE/$t" > "$log" 2>&1 &
    PIDS["$t"]=$!
done

fail=0
for t in "${TESTS[@]}"; do
    if wait "${PIDS[$t]}"; then
        printf "  [ OK ] %-30s\n" "$t"
    else
        printf "  [FAIL] %-30s  (see %s)\n" "$t" "$LOG_DIR/${t%.jl}.log"
        fail=$((fail+1))
    fi
done

elapsed=$(( $(date +%s) - start_all ))
echo "=============================================================="
printf " %d/%d tests passed in %dm%02ds\n" \
       "$(( ${#TESTS[@]} - fail ))" "${#TESTS[@]}" "$((elapsed/60))" "$((elapsed%60))"
echo "=============================================================="

# Surface each test's own summary line so the terminal shows the verdicts.
for t in "${TESTS[@]}"; do
    grep -h "Results:" "$LOG_DIR/${t%.jl}.log" 2>/dev/null | sed "s|^|  ${t%.jl}: |"
done

exit $(( fail > 0 ? 1 : 0 ))
