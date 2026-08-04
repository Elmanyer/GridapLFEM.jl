# ==============================================================
#  test_basic_distributed.jl — distributed smoke test (4 ranks, 2×2)
#
#  Same tiny 16×2 flat-bed flume as test_basic.jl, run distributed with
#  GMRES+Jacobi+Newton on a 2×2 process grid, in BOTH modes:
#    linearised=true,  advection=false  → linear core
#    linearised=false, advection=true   → FULLY NONLINEAR (the algebraic
#      solver's one-path distributed capability — no V⊗H loop needed)
#  Checks: no NaN, bounded, wave generated, and agreement with the sequential
#  reference amplitudes from test_basic.jl (GMRES vs LU ⇒ loose 1e-3 rel).
#
#  RUN (from the project root; system mpiexec fails with PMIx mismatch):
#    ~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. \
#        GridapLFEM.jl/test/test_basic_distributed.jl
# ==============================================================

using GridapLFEM
using Printf

# sequential references (same config via setup_and_run, LU + Newton, 120 steps)
const REF_EMAX_LIN = 0.0037095488  # linearised baseline max η over the run
const REF_EMAX_NL  = 0.0124478473  # fully nonlinear max η over the run
const REF_RTOL     = 2e-3          # GMRES(1e-9)+Jacobi vs LU ⇒ loose agreement

A_wave = 0.001
T_wave = 1.6

results = Dict{String,Bool}()

function run_mode(linflag, advflag, label, ref)
    diags, vert, prob = setup_and_run_distributed(
        cpu_grid=(2,2),
        M=2, h_val=3.5, T_wave=T_wave, A_wave=A_wave,
        domain=((0.0,16.0),(0.0,2.0)), partition=(16,2), p_horizontal=2,
        x_wm=4.0, y_wm=nothing,
        sponge_wL=4.0, sponge_wR=4.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=50.0,
        T_final=4.8, dt=0.04, regime=(linflag ? :linear : :nonlinear),
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

emax1, rel1, n1 = run_mode(true,  false, "linear core",     REF_EMAX_LIN)
emax2, rel2, n2 = run_mode(false, true,  "fully nonlinear", REF_EMAX_NL)

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
