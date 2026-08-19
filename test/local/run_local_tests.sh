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
#    BALFEM_TEST_PERIODS=8 test/local/run_local_tests.sh boundary   # a shorter gate
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
echo " GridapBALFEM local validation suite"
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

#  VERDICT FROM GATE OUTPUT, NOT FROM THE EXIT CODE.
#  A clean exit is not evidence that a test ran: a file whose body sits behind a
#  guard, or that returns early, exits 0 having asserted nothing — which is
#  exactly how four broken tests were recorded as passing for three weeks. So a
#  log with NO PASS/FAIL lines is reported BLANK and counted as a failure, and a
#  log with any FAIL line fails even if the process exited 0.
fail=0
for t in "${TESTS[@]}"; do
    log="$LOG_DIR/${t%.jl}.log"
    rc=0; wait "${PIDS[$t]}" || rc=$?
    #  `grep -c` already prints 0 and exits 1 when there is no match, so the old
    #  `|| echo 0` appended a SECOND zero and the count became the two-line string
    #  "0\n0", which made every `[ "$np" -eq 0 ]` below abort with
    #  "integer expression expected". Use the count grep prints, and default only
    #  when the file is missing entirely.
    raw_p=$(grep -cE '^[[:space:]]*PASS\b' "$log" 2>/dev/null); raw_p=${raw_p:-0}
    raw_f=$(grep -cE '^[[:space:]]*FAIL\b' "$log" 2>/dev/null); raw_f=${raw_f:-0}

    #  A file may print FAIL lines that are NOT failures. test_boundary_modes_1d
    #  runs a NEGATIVE CONTROL whose gates are MEANT to fire, and scores them on a
    #  separate counter — it reported "16 PASS, 0 FAIL" while this runner called it
    #  a failure on 3 deliberate FAIL lines. So prefer the test's OWN verdict line
    #  when it emits one, and fall back to raw counting when it does not. The BLANK
    #  check below still keys off the RAW counts, so a file that asserts nothing is
    #  caught exactly as before.
    summary=$(grep -oE 'Results:[[:space:]]*[0-9]+ PASS,[[:space:]]*[0-9]+ FAIL' "$log" 2>/dev/null | tail -1)
    if [ -n "$summary" ]; then
        np=$(printf '%s' "$summary" | sed -E 's/.*Results:[[:space:]]*([0-9]+) PASS.*/\1/')
        nf=$(printf '%s' "$summary" | sed -E 's/.*,[[:space:]]*([0-9]+) FAIL.*/\1/')
    else
        np=$raw_p; nf=$raw_f
    fi
    if [ "$raw_p" -eq 0 ] && [ "$raw_f" -eq 0 ] && [ -z "$summary" ]; then
        printf "  [BLANK] %-28s  NO GATE OUTPUT — it asserted nothing (see %s)\n" "$t" "$log"
        fail=$((fail+1))
    elif [ "$nf" -gt 0 ]; then
        printf "  [FAIL] %-30s  %d pass / %d fail  (see %s)\n" "$t" "$np" "$nf" "$log"
        fail=$((fail+1))
    elif [ "$rc" -ne 0 ]; then
        printf "  [ERR ] %-30s  gates passed but exited %d  (see %s)\n" "$t" "$rc" "$log"
        fail=$((fail+1))
    else
        printf "  [ OK ] %-30s  %d gates\n" "$t" "$np"
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
