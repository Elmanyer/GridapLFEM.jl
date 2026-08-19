#!/bin/bash
# ==============================================================
#  sweep_nlpressure_amplitude.sh — DOES nl_pressure=:full DO WHAT IT SHOULD?
#
#  WHY THIS EXISTS. `nl_pressure=:full` (the 𝓝{1,2,4,5} block) is the one piece
#  of solver physics the local campaign could NOT validate, and the reason is
#  resolution, not suspicion: at A=1e-3 switching it off changes max|eta| by
#  0.013 % (quasi-1D) / 0.094 % (2-D), which is below everything else's noise.
#  See building_files/TEST_SUITE.md §7 — "a test validates a term only if the test
#  can resolve that term's contribution".
#
#  THE EXPERIMENT. The block is O(A^3) against O(A) leading terms, so its
#  RELATIVE contribution scales as A^2. Ten-fold the amplitude and it should
#  grow a HUNDRED-fold, from ~0.1 % to ~10 % — far above any noise floor. That
#  turns an unresolvable difference into a falsifiable prediction:
#
#      r(A) = ( max|eta|_full - max|eta|_none ) / max|eta|_none
#      PREDICTION:  r(1e-2) / r(1e-3)  ~=  100      (i.e. slope 2 in log-log)
#
#  A 2x2 matrix measures it, with a built-in control: the A=1e-3 arm must
#  reproduce the archived pair (4.025e-3 vs 4.029e-3 => 0.099 %).
#
#  WHAT EACH OUTCOME MEANS
#    r ratio ~100        -> the block's magnitude is confirmed DYNAMICALLY, and
#                           the O(A^3) ordering asserted by test_nlpressure G2
#                           (which measures it by static amplitude scaling) is
#                           reproduced by the time-dependent solver.
#    r ratio ~1          -> the block is not actually O(A^3) in the running
#                           solver; something is mis-scaled. Investigate.
#    r ratio >>100       -> the block is growing faster than cubic => likely an
#                           instability, not a physical contribution.
#    r(1e-2) ~ 0         -> the block is effectively inert at ANY amplitude,
#                           i.e. it is not being assembled. That would be a real
#                           defect and the most valuable outcome of the sweep.
#
#  NOTE ON WHAT THIS IS NOT. This measures the block's MAGNITUDE and SCALING,
#  not the correctness of its coefficients. Only the analytic MMS staged to this
#  tier can do the latter (building_files/OPEN_ITEMS.md §6). Do not report a
#  passing ratio as "nl_pressure=:full is validated".
#
#  CONFIGURATION. Flat bed + interior line source, deliberately: the flat case
#  has no wave-to-bathymetry transit requirement, so the differential starts
#  accumulating immediately and a 6-period run suffices. (The bar variant is
#  richer -- it also activates the grad-h half of the block -- but the wave
#  needs ~5 periods just to REACH the bar. Run BALFEM_FLAT_BED=0 as a follow-up
#  once the flat result is in, with more periods.)
#
#  Mesh is 64x24 (dx=dy=0.625, ~6.4 cells/wavelength) rather than the 96x36 of
#  the observation cases: 2.25x cheaper, and this is a DIFFERENTIAL measurement
#  in which both arms share the mesh, so the ratio is what matters, not the
#  absolute resolution. Tolerances are the validated production defaults
#  (nl_tol=1e-5, ls_rtol=1e-5); they cancel in the differential anyway, and the
#  effect sought (0.1-10 %) is orders above the 3.8e-5 tolerance sensitivity.
#
#  COST: ~3 h per :full arm, ~2.5 h per :none arm => ~11 h for the four.
#  Cases are ordered so the DECISIVE pair (A=1e-2) finishes first: if you stop
#  the sweep early you still have the headline result.
#
#  USAGE
#    run/local/sweep_nlpressure_amplitude.sh                  # all four
#    PERIODS=4 run/local/sweep_nlpressure_amplitude.sh        # cheaper
#    BALFEM_FLAT_BED=0 PERIODS=12 run/local/... .sh             # the bar variant
#
#  Results: output/local/logs/sweep_nlpressure_amplitude.csv (+ printed analysis)
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh

RANKS="${RANKS:-12}"
PERIODS="${PERIODS:-6}"
OUT=output/local/logs
mkdir -p "$OUT"
CSV="$OUT/sweep_nlpressure_amplitude.csv"
echo "case,A_wave,nl_pressure,flat_bed,periods,steps,s_per_step,gmres_min,gmres_max,nl_per_step,max_eta,rss_mb,exit" > "$CSV"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
# ---- the fixed case: everything except A_wave and nl_pressure is held ----
export BALFEM_WAVE_GEN=line BALFEM_REGIME=nonlinear
export BALFEM_FLAT_BED="${BALFEM_FLAT_BED:-1}"
export BALFEM_LX=40.0 BALFEM_LY=15.0 BALFEM_NX=64 BALFEM_NY=24
export BALFEM_PX=4 BALFEM_PY=3 BALFEM_MPI=1
export BALFEM_DT=0.04 BALFEM_PERIODS="$PERIODS"
export BALFEM_SAVE_EVERY=0 BALFEM_WRITE_W=0 BALFEM_WRITE_PRESSURE=0
export BALFEM_PRINT_EVERY=20 BALFEM_DIAG_EVERY=5
export BALFEM_PRECOND=jacobi          # tolerances: validated defaults, not overridden

#  A_wave   nl_pressure   label
CONFIGS=(
  "0.01   full   bigA_full"
  "0.01   none   bigA_none"
  "0.001  full   smallA_full"
  "0.001  none   smallA_none"
)

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; aw=$1; nlp=$2; label=$3
  echo "=================================================================="
  echo "=== $label : A_wave=$aw  nl_pressure=$nlp  flat_bed=$BALFEM_FLAT_BED"
  echo "=================================================================="
  export BALFEM_AWAVE="$aw" BALFEM_NL_PRESSURE="$nlp"
  export BALFEM_OUTDIR="output/local/sweep_nlp_${label}"
  log="$OUT/sweep_nlp_${label}.log"
  balfem_local_mpi "$RANKS" examples/local_2d/run_small_2d.jl > "$log" 2>&1
  rc=$?

  sps=$(grep -oE '\([0-9.]+ s/step average\)' "$log" | grep -oE '[0-9.]+' | head -1)
  gmin=$(grep -oE 'gmres [0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | head -1)
  gmax=$(grep -oE 'gmres [0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | tail -1)
  steps=$(python3 -c "print(int(round($PERIODS*1.6/0.04)))")
  read nlps meta rss <<<"$(python3 - "$BALFEM_OUTDIR/diagnostics.csv" <<'PY'
import csv,sys
try:
    rows=list(csv.DictReader(open(sys.argv[1])))
    print("%.2f %.9e %.0f"%(sum(float(r['nl_iters']) for r in rows)/len(rows),
                            max(float(r['eta_max']) for r in rows),
                            max(float(r['rss_peak_mb']) for r in rows)))
except Exception:
    print("nan nan nan")
PY
)"
  echo "$label,$aw,$nlp,$BALFEM_FLAT_BED,$PERIODS,$steps,${sps:-nan},${gmin:-nan},${gmax:-nan},$nlps,$meta,$rss,$rc" >> "$CSV"
  printf "    s/step=%-8s gmres=%s-%s  NL/step=%s  max|eta|=%s  (exit %d)\n" \
         "${sps:-?}" "${gmin:-?}" "${gmax:-?}" "$nlps" "$meta" "$rc"
done

echo
echo "=============================================================="
echo " nl_pressure x amplitude sweep — $CSV"
echo "=============================================================="
column -s, -t < "$CSV"

# ---- the analysis the sweep exists for ----------------------------------
python3 - "$CSV" <<'PY'
import csv, sys
rows = {r['case']: r for r in csv.DictReader(open(sys.argv[1]))}
def val(c):
    try:    return float(rows[c]['max_eta'])
    except: return float('nan')

print("\n--- the differential r(A) = (full - none)/none ---")
res = {}
for tag, A in (("bigA", 1e-2), ("smallA", 1e-3)):
    f, n = val(f"{tag}_full"), val(f"{tag}_none")
    if f == f and n == n and n != 0:
        r = (f - n)/n
        res[A] = r
        print(f"  A={A:<7g} none={n:.9e}  full={f:.9e}   r = {r:+.4e}  ({r*100:+.4f} %)")
    else:
        print(f"  A={A:<7g} INCOMPLETE (a run did not produce a usable CSV)")

if len(res) == 2:
    rb, rs = res[1e-2], res[1e-3]
    print("\n--- the A^2 prediction ---")
    print(f"  measured ratio r(1e-2)/r(1e-3) = {rb/rs:.1f}      PREDICTED ~100")
    if rs != 0 and rb/rs > 0:
        import math
        slope = math.log10(abs(rb/rs))/math.log10(10.0)
        print(f"  implied log-log slope          = {slope:.2f}      PREDICTED 2.00")
    print("\n  ~100 (slope 2) => O(A^3) block confirmed dynamically.")
    print("  ~1            => not O(A^3) in the running solver: INVESTIGATE.")
    print("  r(1e-2) ~ 0   => block inert at any amplitude: LIKELY NOT ASSEMBLED.")
    print("\n  Control: the A=1e-3 arm should reproduce the archived flat line pair")
    print("  (4.025e-3 vs 4.029e-3 => r = +9.9e-4). Mesh differs (64x24 vs 96x36),")
    print("  so expect the RATIO to match, not the absolute amplitudes.")
    print("\n  REMINDER: this measures MAGNITUDE and SCALING, not coefficient")
    print("  correctness. Only the analytic MMS can do that.")
PY
