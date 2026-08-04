# ==============================================================
#  test_basic.jl — smoke test (SMALL & FAST)
#
#  Port of ../../LFE-M_2D_solver/test/test_basic_lfem2D.jl to the algebraic
#  package. Tiny flat-bed plane-wave flume (16×2 m, 16×2 Q2 cells, M=2,
#  3 periods), run in BOTH modes:
#    linearised=true, advection=false → validated linear baseline
#    linearised=false, advection=true → fully nonlinear target
#  Checks: no NaN, bounded amplitude, wave generated.
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_basic.jl
# ==============================================================

using GridapLFEM
using Printf

println("=" ^ 60)
println("  test_basic.jl — algebraic solver smoke (small)")
println("=" ^ 60)

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond; println("  PASS  $name"); n_pass += 1
    else;    println("  FAIL  $name"); n_fail += 1; end
end

A_wave = 0.001
T_wave = 1.6                 # d = 3.5 ⇒ kd = 5.5, λ ≈ 4 m

function run_mode(linflag, advflag, label)
    println("\n--- $label (linearised=$linflag, advection=$advflag) ---")
    diags, vert, prob = setup_and_run(
        M=2, h_val=3.5, T_wave=T_wave, A_wave=A_wave,
        domain=((0.0,16.0),(0.0,2.0)), partition=(16,2), p_horizontal=2,
        x_wm=4.0, y_wm=nothing,
        sponge_wL=4.0, sponge_wR=4.0, sponge_wB=0.0, sponge_wT=0.0, mu_max=50.0,
        T_final=4.8, dt=0.04, regime=(linflag ? :linear : :nonlinear),
        gauges=[(8.0, 1.0)], save_every=0,
    )
    emax = maximum(d.eta_max for d in diags)
    nan_free = all(!isnan(d.eta_max) for d in diags)
    gvals = [d.gauge_vals[1] for d in diags if !isempty(d.gauge_vals)]
    gamp  = isempty(gvals) ? 0.0 : maximum(abs.(gvals))
    @printf("  steps=%d  max η=%.5f m  gauge amp=%.5f m\n", length(diags), emax, gamp)
    check("[$label] no NaN", nan_free)
    check("[$label] bounded (max η < 20·A = $(20A_wave))", emax < 20*A_wave)
    check("[$label] wave generated (gauge amp > 0.02·A)", gamp > 0.02*A_wave)
end

run_mode(true,  false, "linearised baseline")
run_mode(false, true,  "fully nonlinear")

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_basic: $n_fail failed!") :
             println("  Algebraic solver smoke passed.")
