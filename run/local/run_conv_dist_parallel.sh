#!/bin/bash
# ==============================================================
#  run_conv_dist_parallel.sh — distributed studies, N CONCURRENT MPI JOBS
#
#  One 4-rank study uses 4 of 16 cores. This runs JOBS studies at once, each on
#  RANKS ranks: JOBS=3 x RANKS=4 = 12 cores.
#
#  MEMORY, NOT CORES, IS THE BINDING CONSTRAINT. Each rank holds ~1.2-1.9 GB, so
#  12 ranks is ~15-23 GB of this box's 30 GB. Do NOT raise JOBS*RANKS past 12
#  without checking `free -g`: an OOM kill mid-campaign loses every study in
#  flight, which costs far more than the idle cores save.
#
#  Partitions: 1-D studies use RANKS x 1 (the solution is y-invariant);
#              2-D studies use 2 x 2 and fewer levels (far costlier per level).
#
#  USAGE:  JOBS=3 RANKS=4 run/local/run_conv_dist_parallel.sh
#          STUDIES="d2:2:static d2:3:static" JOBS=3 run/local/...
#          (spec = domain:p_u:mode)
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
JOBS="${JOBS:-3}"; RANKS="${RANKS:-4}"
OUT=output/local/logs; mkdir -p "$OUT"
STUDIES="${STUDIES:-d1:3:transient d1:4:static d1:4:transient}"

run_one_dist() {
    spec="$1"
    dom="${spec%%:*}"; rest="${spec#*:}"; pu="${rest%%:*}"; mode="${rest##*:}"
    tag="dist_${dom}_Q${pu}_${mode}"
    if [ "$dom" = "d1" ]; then
        lv=4; nx0=16; ny0=3; px="$RANKS"; py=1
    else
        lv=3; nx0=6;  ny0=4; px=2;        py=2
    fi
    env BALFEM_CONV_PU="$pu" BALFEM_CONV_DOMAIN="$dom" BALFEM_CONV_MODE="$mode" \
        BALFEM_CONV_LEVELS="$lv" BALFEM_CONV_NX0="$nx0" BALFEM_CONV_NY0="$ny0" \
        BALFEM_CONV_DIST=1 BALFEM_CONV_PX="$px" BALFEM_CONV_PY="$py" \
        BALFEM_CONV_LSRTOL=1e-13 BALFEM_CONV_NLTOL=1e-13 \
        BALFEM_CONV_DT=1e-5 BALFEM_CONV_NSTEPS=100 \
        BALFEM_CONV_OUT="output/local/mms_conv/$tag" \
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
        stdbuf -oL -eL "$HOME/.julia/bin/mpiexecjl" --project=. -n $((px*py)) \
        julia --project=. examples/local_mms/run_mms_matrix.jl \
        > "$OUT/conv_$tag.log" 2>&1
    rc=$?   # 143 = the benign MPI_Finalize OFI error on this box
    printf "%-26s exit=%d  %s\n" "$tag" "$rc" \
        "$(grep -oE 'fitted: .*' "$OUT/conv_$tag.log" | tail -1)"
}
export -f run_one_dist; export OUT RANKS

echo "=== $(echo $STUDIES | wc -w) studies | JOBS=$JOBS x $RANKS ranks | $(date '+%H:%M:%S') ==="
free -g | sed -n 2p
t0=$(date +%s)
printf '%s\n' $STUDIES | xargs -P "$JOBS" -I{} bash -c 'run_one_dist "$@"' _ {}
echo "=== done in $(( ($(date +%s)-t0)/60 )) min ==="
for f in "$OUT"/conv_dist_*.log; do [ -e "$f" ] || continue
    printf "%-28s %s\n" "$(basename "$f" .log|sed 's/^conv_//')" \
        "$(grep -oE 'fitted: .*' "$f"|tail -1)"; done
