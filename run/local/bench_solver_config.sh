#!/bin/bash
# ==============================================================
#  bench_solver_config.sh — which linear-solver configuration is actually best?
#
#  WHY THIS EXISTS. The local runs of 2026-08 showed the linear solve, not the
#  physics, is the cost of this solver:
#      2-D 96x36 (98.6k DOFs), 12 ranks : 451-517 GMRES iterations per solve
#      2-D ring (same mesh)             : 654-695
#      quasi-1D 240x3 (20.2k DOFs)      : 758-765  <-- SMALLER problem, MORE iters
#  and the count is RANK-INDEPENDENT (758 at 2, 4, 6 and 12 ranks), which is the
#  signature of a weak preconditioner rather than a bad decomposition. Newton
#  meanwhile converges in 1-2 iterations per stage, so essentially all the work
#  is inside GMRES.
#
#  Two levers are worth measuring before any more production runs:
#    (a) the PRECONDITIONER  - Jacobi (current) vs additive Schwarz with an exact
#        per-rank LU vs symmetric Gauss-Seidel;
#    (b) the LINEAR TOLERANCE - ls_rtol is 1e-9 while the Newton tolerance is
#        1e-6 and the observed step residual is ~1e-9, i.e. roughly three orders
#        of slack. Each order costs GMRES iterations.
#
#  The benchmark runs the SAME short case under each configuration and reports
#  iterations, wall time, and the resulting max|eta| so a cheaper configuration
#  can be rejected if it changes the answer.
#
#  USAGE
#    run/local/bench_solver_config.sh              # 12 ranks, ~30 steps per config
#    RANKS=4 STEPS=20 run/local/bench_solver_config.sh
#
#  Results: output/local/logs/bench_solver_config.csv  (+ a printed table)
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/lfem_local.sh

RANKS="${RANKS:-12}"
STEPS="${STEPS:-30}"
DT=0.04
TF=$(python3 -c "print($STEPS*$DT)")
OUT=output/local/logs
mkdir -p "$OUT"
CSV="$OUT/bench_solver_config.csv"
echo "config,precond,ls_rtol,ranks,steps,s_per_step,gmres_min,gmres_max,nl_per_step,max_eta,rss_mb" > "$CSV"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
# Fixed physical case for every configuration: the 2-D nonlinear plane wave.
export LFEM_WAVE_GEN=line LFEM_REGIME=nonlinear LFEM_NL_PRESSURE=full LFEM_FLAT_BED=1
export LFEM_LX=40.0 LFEM_LY=15.0 LFEM_NX=96 LFEM_NY=36
export LFEM_PX=4 LFEM_PY=3 LFEM_MPI=1
export LFEM_TFINAL="$TF" LFEM_DT="$DT"
export LFEM_SAVE_EVERY=0 LFEM_WRITE_W=0 LFEM_WRITE_PRESSURE=0
export LFEM_PRINT_EVERY=10 LFEM_DIAG_EVERY=5

# precond  ls_rtol   label
CONFIGS=(
  "jacobi  1e-9  baseline(current default)"
  "jacobi  1e-7  jacobi_looser"
  "jacobi  1e-6  jacobi_loosest"
  "schwarz 1e-9  schwarz_tight"
  "schwarz 1e-7  schwarz_looser"
  "gs      1e-9  gauss_seidel"
)

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; pc=$1; rt=$2; label=$3
  echo "=== $label : precond=$pc ls_rtol=$rt ==="
  export LFEM_PRECOND="$pc" LFEM_LS_RTOL="$rt"
  export LFEM_OUTDIR="output/local/bench_${pc}_${rt}"
  log="$OUT/bench_${pc}_${rt}.log"
  lfem_local_mpi "$RANKS" examples/local_2d/run_small_2d.jl > "$log" 2>&1
  rc=$?

  sps=$(grep -oE '\([0-9.]+ s/step average\)' "$log" | grep -oE '[0-9.]+' | head -1)
  gmin=$(grep -oE 'gmres [0-9]+/[0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | head -1)
  gmax=$(grep -oE 'gmres [0-9]+/[0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | tail -1)
  # per-step Newton iterations and the final amplitude come from the CSV, which
  # is written continuously and survives a truncated run
  read nlps meta rss <<<"$(python3 - "$LFEM_OUTDIR/diagnostics.csv" <<'PY'
import csv,sys
try:
    rows=list(csv.DictReader(open(sys.argv[1])))
    print("%.2f %.6e %.0f"%(sum(float(r['nl_iters']) for r in rows)/len(rows),
                            max(float(r['eta_max']) for r in rows),
                            max(float(r['rss_peak_mb']) for r in rows)))
except Exception:
    print("nan nan nan")
PY
)"
  echo "$label,$pc,$rt,$RANKS,$STEPS,${sps:-nan},${gmin:-nan},${gmax:-nan},$nlps,$meta,$rss" >> "$CSV"
  printf "    s/step=%-8s gmres=%s-%s  NL/step=%s  max|eta|=%s  (exit %d)\n" \
         "${sps:-?}" "${gmin:-?}" "${gmax:-?}" "$nlps" "$meta" "$rc"
done

echo
echo "=============================================================="
echo " solver-configuration benchmark — $CSV"
echo "=============================================================="
column -s, -t < "$CSV"
echo
echo "Read it as: fewer GMRES iterations AND unchanged max|eta| = a free win."
echo "A configuration that changes max|eta| beyond ~1e-3 relative is buying"
echo "speed with accuracy and must be rejected."
