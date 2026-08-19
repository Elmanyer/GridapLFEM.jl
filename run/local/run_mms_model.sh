#!/bin/bash
# ==============================================================
#  run_mms_model.sh — MODEL validation + convergence, linear flat-bed BALFE-M
#
#  DISTINCT FROM run_mms_space/time.sh. Those force the solution with an analytic
#  source and verify that the discretised OPERATOR is the intended one (code
#  verification). This one runs with NO FORCING from an exact standing mode of the
#  linear flat-bed system, so the dynamics are the MODEL's own: the phase is set by
#  the model dispersion relation ω = k·Cm(k), and a wrong dispersion appears as a
#  phase drift that does not converge away under refinement.
#
#  Both are needed. A forced test never exercises free evolution; an unforced test
#  cannot isolate individual operator terms.
#
#  ENV:  MODE=space|time   LEVELS=n   NMODE=n (basin mode number)
#  USAGE: run/local/run_mms_model.sh
# ==============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source run/local/balfem_local.sh
mkdir -p output/local/logs output/local/mms_model

MODE="${MODE:-space}"; LEVELS="${LEVELS:-3}"; NMODE="${NMODE:-1}"
cat > /tmp/_balfem_mms_model.jl <<'JL'
using GridapBALFEM, Printf
mode   = Symbol(get(ENV, "MODE", "space"))
levels = parse(Int, get(ENV, "LEVELS", "3"))
nmode  = parse(Int, get(ENV, "NMODE", "1"))
println("#"^70)
println("#  MODEL VALIDATION — linear flat-bed BALFE-M, exact standing mode n=$nmode")
println("#  NO FORCING: the solver evolves the model's own dynamics.")
println("#"^70)
r = mode == :space ?
    run_model_refinement(:space; levels=levels, nx0=8, ny0=4, dt0=2e-4,
                         T_final=0.02, n_mode=nmode) :
    run_model_refinement(:time;  levels=levels, nx_fine=32, ny_fine=8, dt0=8e-3,
                         T_final=0.08, n_mode=nmode)
@printf("\n>>> MODEL %s ORDER: p_eta=%.4f  p_u=%.4f  (expected %.1f)\n",
        uppercase(String(mode)), r.p_eta, r.p_u, r.expected)
open("output/local/mms_model/model_convergence_$(mode).csv","w") do io
    println(io, "param,e_eta,e_u,p_eta,p_u,expected")
    for i in eachindex(r.param)
        @printf(io, "%.8g,%.10e,%.10e,%.4f,%.4f,%.1f\n",
                r.param[i], r.e_eta[i], r.e_u[i], r.p_eta, r.p_u, r.expected)
    end
end
JL
MODE="$MODE" LEVELS="$LEVELS" NMODE="$NMODE" \
    balfem_local_run /tmp/_balfem_mms_model.jl 2>&1 | tee "output/local/logs/mms_model_${MODE}.log"
