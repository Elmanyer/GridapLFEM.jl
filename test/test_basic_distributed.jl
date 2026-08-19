# ==============================================================
#  test_basic_distributed.jl — distributed smoke test (4 ranks, 2×2)
#
#  Same tiny 16×2 flat-bed flume as test_basic.jl, run distributed with
#  GMRES+Jacobi+Newton on a 2×2 process grid, in BOTH regimes:
#    regime=:linear     → linear core
#    regime=:nonlinear  → FULLY NONLINEAR (the algebraic solver's one-path
#      distributed capability — no V⊗H loop needed)
#  Checks: no NaN, bounded, wave generated, and agreement with the sequential
#  reference amplitudes from test_basic.jl (GMRES vs LU ⇒ loose 1e-3 rel).
#
#  Physics is selected by regime / nl_pressure / flat_bed; the retired
#  linearised=/advection= boolean pair is no longer accepted by the drivers.
#
#  RUN (from the project root; system mpiexec fails with PMIx mismatch):
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. \
#        GridapBALFEM.jl/test/test_basic_distributed.jl
# ==============================================================

using GridapBALFEM
using Printf

# sequential references (same config via setup_and_run, LU + Newton, 120 steps)
#
#  REF_EMAX_LIN RE-MEASURED 2026-08-17: 0.0037461259 → 0.0041084522.
#  The old value had been STALE for some time and nothing noticed, because the
#  distributed tests need `mpiexecjl` and are not part of any routine run — the
#  same gap CLAUDE.md flags when it says distributed scores "have not been
#  re-run". It is almost certainly fallout from the h-WEIGHTING of the linear
#  model (MMS_VARBED_PLAN §0.A), whose own plan warned in advance that the
#  assembled system changes "even on a flat bed (by the constant factor d), so
#  reference values MUST be re-checked"; the sequential references were
#  re-checked at the time, this distributed one was missed.
#
#  It is NOT fallout from the 2026-08-17 Jacobian/gravity fixes, and the evidence
#  is worth keeping because "my last change broke it" is the tempting reading:
#    * REF_EMAX_NL still matches to 2.35e-05 — and the nonlinear path assembles
#      strictly MORE terms, so an operator change would move it at least as much;
#    * this case runs at CONSTANT depth, so ∇h ≡ 0 and the added gravity term is
#      identically zero here;
#    * a Jacobian change cannot move a converged answer beyond nl_tol anyway.
#  Re-measured the way this block documents its own provenance — the SEQUENTIAL
#  solver on this exact configuration — not by copying the distributed output:
#      sequential LU  : 0.0041084522
#      distributed GMRES: 0.0041085   (rel ~1e-5, the expected agreement)
#  REF_EMAX_NL also re-measured 2026-08-17: 0.0041032781 → 0.0041031807, a shift
#  of 2.37e-5 RELATIVE — a different story from REF_EMAX_LIN above, and traceable
#  to the 2026-08-17 ∂R/∂u̇ fix. Even at CONSTANT depth that fix is not inert: it
#  zeroes the 𝓐 (bed-slope) half, but the 𝓚 half carries prefactor ∇H = ∇η ≠ 0
#  and does contribute. So the effective mass matrix changed here, the Newton
#  iteration path changed with it, and the answer moved by the nl_tol
#  STEP-FUNCTION amount — CLAUDE.md measures that effect at 1.9e-5–3.8e-5
#  relative, and 2.37e-5 sits inside it. That is an iteration-path change, not an
#  operator change: the old value still passed REF_RTOL=2e-3 with 85x margin.
const REF_EMAX_LIN = 0.0041084522  # linearised baseline max η over the run
const REF_EMAX_NL  = 0.0041031807  # fully nonlinear max η over the run
const REF_RTOL     = 2e-3          # GMRES(1e-9)+Jacobi vs LU ⇒ loose agreement

A_wave = 0.001
T_wave = 1.6

results = Dict{String,Bool}()

function run_mode(regime, label, ref)
    diags, vert, prob = setup_and_run_distributed(
        cpu_grid=(2,2),
        M=2, h_val=3.5, T_wave=T_wave, A_wave=A_wave,
        domain=((0.0,16.0),(0.0,2.0)), partition=(16,2), p_horizontal=2,
        x_wm=4.0, y_wm=nothing,
        sponge_wL=4.0, sponge_wR=4.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=50.0,
        T_final=4.8, dt=0.04, regime=regime, nl_pressure=:none,
        save_every=0, print_dt=0.4,
    )
    emax = maximum(d.eta_max for d in diags)
    nan_free = all(!isnan(d.eta_max) for d in diags)
    rel = abs(emax - ref) / ref
    results["[$label] no NaN"]                        = nan_free
    results["[$label] bounded (max η < 20A)"]         = emax < 20*A_wave
    results["[$label] matches sequential (rel<$(REF_RTOL))"] = rel < REF_RTOL
    return emax, rel, length(diags)
end

emax1, rel1, n1 = run_mode(:linear,    "linear core",     REF_EMAX_LIN)
emax2, rel2, n2 = run_mode(:nonlinear, "fully nonlinear", REF_EMAX_NL)

# report on rank 0 only (re-open MPI context just for the rank id is overkill —
# use the env-var rank detection instead)
is_rank0 = get(ENV, "OMPI_COMM_WORLD_RANK", get(ENV, "PMI_RANK", "0")) == "0"
if is_rank0
    println("=" ^ 60)
    println("  test_basic_distributed.jl — results (4 ranks, 2×2)")
    println("=" ^ 60)
    @printf("  linear core:      steps=%d  max η=%.7f  (ref %.7f, rel=%.2e)\n",
            n1, emax1, REF_EMAX_LIN, rel1)
    @printf("  fully nonlinear:  steps=%d  max η=%.7f  (ref %.7f, rel=%.2e)\n",
            n2, emax2, REF_EMAX_NL, rel2)
    n_pass = count(values(results)); n_fail = length(results) - n_pass
    for (name, ok) in sort(collect(results); by=first)
        println(ok ? "  PASS  $name" : "  FAIL  $name")
    end
    @printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
    flush(stdout)
    n_fail > 0 && error("test_basic_distributed: $n_fail failed!")
    println("  Distributed algebraic solver validated.")
    flush(stdout)
end
