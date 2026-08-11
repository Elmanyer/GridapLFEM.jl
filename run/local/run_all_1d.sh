#!/bin/bash
# ==============================================================
#  run_all_1d.sh — the whole quasi-1D case set
#
#  Each case runs SEQUENTIALLY (direct LU — measured 2-3x faster than any MPI
#  split of this mesh, see any run_1d_*.sh header) and the cases run SIDE BY
#  SIDE, so the machine's cores are used by case-level parallelism rather than
#  by decomposing one small case 12 ways. JOBS defaults to 7 = the case count.
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
JOBS="${JOBS:-7}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
mkdir -p output/local/logs; rm -f output/local/logs/batch1d_summary.txt
for f in run/local/run_1d_*.sh; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  n=$(basename "$f" .sh)
  ( bash "$f" > "output/local/logs/${n}.log" 2>&1
    echo "$n exit=$?" >> output/local/logs/batch1d_summary.txt ) &
done
wait
echo "ALL 1D DONE" >> output/local/logs/batch1d_summary.txt
