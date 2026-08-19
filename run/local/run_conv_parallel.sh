#!/bin/bash
# ==============================================================
#  run_conv_parallel.sh — the campaign as INDEPENDENT CONCURRENT PROCESSES
#
#  Each study (pairing × domain × time-mode) is a separate sequential Julia
#  process, so they are embarrassingly parallel. This replaces the single-process
#  serial sweep, which used 1 of 16 cores.
#
#  CONCURRENCY. Not 12. `building_files/CONFIGURATION.md` §7 measured three concurrent
#  Julia processes each running at ~1/3 solo speed — MEMORY BANDWIDTH is the
#  binding constraint, not core count, so throughput saturates early. These
#  studies are smaller (1.2k-5k DOFs coarse) and should scale further, but
#  JOBS=6 is the sensible starting point; raise it only on measured evidence.
#  Each Julia process also holds ~1.5-2 GB RSS, so JOBS=6 ≈ 12 GB of 30 GB.
#
#  USAGE
#    run/local/run_conv_parallel.sh                  # default job list, JOBS=6
#    JOBS=4 run/local/run_conv_parallel.sh
#    JOBS=6 STUDIES="d1:4:static d2:3:transient" run/local/run_conv_parallel.sh
#  A study spec is  <domain>:<p_u>:<mode>
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
JOBS="${JOBS:-6}"
OUT=output/local/logs; mkdir -p "$OUT" output/local/mms_conv

# Default: everything NOT already done by the serial Suite-A run
# (Q2/Q1 and Q3/Q2 in 1-D, static+transient, are complete).
STUDIES="${STUDIES:-d1:4:static d1:4:transient d2:2:static d2:2:transient d2:3:static d2:3:transient d2:4:static d2:4:transient}"

run_one() {
    spec="$1"; dom="${spec%%:*}"; rest="${spec#*:}"; pu="${rest%%:*}"; mode="${rest##*:}"
    tag="${dom}_Q${pu}_${mode}"
    # 2-D is far more expensive per level: fewer levels and a smaller base mesh.
    if [ "$dom" = "d1" ]; then lv=4; nx0=16; ny0=3; else lv=3; nx0=6; ny0=4; fi
    BALFEM_CONV_PU="$pu" BALFEM_CONV_DOMAIN="$dom" BALFEM_CONV_MODE="$mode" \
    BALFEM_CONV_LEVELS="$lv" BALFEM_CONV_NX0="$nx0" BALFEM_CONV_NY0="$ny0" \
    BALFEM_CONV_DIST=0 BALFEM_CONV_DT=1e-5 BALFEM_CONV_NSTEPS=100 \
    BALFEM_CONV_OUT="output/local/mms_conv/$tag" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
        stdbuf -oL -eL julia --project=. examples/local_mms/run_mms_matrix.jl \
        > "$OUT/conv_$tag.log" 2>&1
    rc=$?
    printf "%-22s exit=%d  %s\n" "$tag" "$rc" \
        "$(grep -oE 'fitted: .*' "$OUT/conv_$tag.log" | tail -1)"
}
export -f run_one
export OUT

echo "=== launching $(echo $STUDIES | wc -w) studies, JOBS=$JOBS, started $(date '+%H:%M:%S') ==="
t0=$(date +%s)
printf '%s\n' $STUDIES | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
t1=$(date +%s)
echo "=== all studies done in $(( (t1-t0)/60 )) min $(( (t1-t0)%60 )) s ==="
echo
echo "================= CAMPAIGN RESULTS ================="
for f in "$OUT"/conv_d*_Q*_*.log; do
    [ -e "$f" ] || continue
    printf "%-24s %s\n" "$(basename "$f" .log | sed 's/^conv_//')" \
        "$(grep -oE 'fitted: .*' "$f" | tail -1)"
done
