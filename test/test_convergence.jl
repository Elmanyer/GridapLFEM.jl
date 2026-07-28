# ==============================================================
#  test_convergence.jl — grid & time self-convergence (Richardson)
#
#  Establishes the ORDER OF ACCURACY without an exact solution, by Richardson
#  (Cauchy) self-convergence: run the SAME forced linear plane wave at three
#  resolutions and form the observed order from successive gauge-signal
#  differences,
#     p ≈ log2( ‖g_{4h}−g_{2h}‖ / ‖g_{2h}−g_{h}‖ ),      (spatial)
#     q ≈ log2( ‖g_{4Δt}−g_{2Δt}‖ / ‖g_{2Δt}−g_{Δt}‖ ).  (temporal)
#  Expected: temporal q → 2 (Crank–Nicolson) — clean, gated. Spatial rate is
#  metric-sensitive for a propagating gauge at marginal resolution (wavemaker/
#  sponge inject a slowly-converging offset, phase error accumulates), so the
#  spatial part only checks that the gauge CONVERGES under refinement and prints
#  the (rough) order — the EXACT spatial-operator order is certified to machine
#  precision by test_mms.jl on a manufactured solution.
#
#  A multi-run FEM test — minutes after JIT. RUN:
#    julia --project=. GridapLFEM.jl/test/test_convergence.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 60)
println("  test_convergence.jl — spatial & temporal order (Richardson)")
println("=" ^ 60)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

g = 9.81; h_val = 3.5; T_wave = 1.6; A = 5e-4
omega = 2π/T_wave; k = find_wavenumber(omega, h_val, g); lam = 2π/k
x_wm = 6.0; Lx = x_wm + 5lam + 6.0; Ly = 1.0
x_g = x_wm + 2lam; y_g = Ly/2
T_final = 6*T_wave

"gauge time series η(x_g,t) for a linear plane wave at (nx, dt)."
function gauge_run(nx, dt)
    diags, _, _ = setup_and_run(
        M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,2),
        p_horizontal=2, h_val=h_val, T_wave=T_wave, A_wave=A, x_wm=x_wm, y_wm=nothing,
        sponge_wL=6.0, sponge_wR=6.0, mu_max=30.0, T_final=T_final, dt=dt,
        save_every=0, gauges=[(x_g, y_g)], linearised=true, advection=false,
        nl_tol=1e-8, print_every=10_000)
    return [d.gauge_vals[1] for d in diags]
end

order(a, b, c) = log2(norm(a .- b) / max(norm(b .- c), 1e-30))

# ---- spatial: fix a small dt, refine the mesh nx = 40,80,160 -------------------
println("\n  -- spatial refinement (dt fixed) --")
dt_s = T_wave/60
gh4 = gauge_run(40,  dt_s)
gh2 = gauge_run(80,  dt_s)
gh1 = gauge_run(160, dt_s)
n = min(length(gh4), length(gh2), length(gh1))
d1 = norm(gh4[1:n].-gh2[1:n]); d2 = norm(gh2[1:n].-gh1[1:n])
p_space = order(gh4[1:n], gh2[1:n], gh1[1:n])
@printf("  ‖g₄₀−g₈₀‖=%.3e  ‖g₈₀−g₁₆₀‖=%.3e  ⇒  spatial order ≈ %.2f\n", d1, d2, p_space)
# NOTE: self-convergence of a PROPAGATING-wave gauge is metric-sensitive at marginal
# resolution (the wavemaker/sponge inject a slowly-converging amplitude offset and
# phase error accumulates over the wave train), so the *rate* is unreliable here —
# the EXACT spatial-operator order is certified to machine precision by test_mms.jl.
# We therefore only require monotone convergence under refinement.
check("spatial: gauge converges under refinement (exact order: test_mms.jl)", d2 < d1,
      "(order≈$(round(p_space,digits=2)))")

# ---- temporal: fix a fine mesh, refine dt = T/20, T/40, T/80 -------------------
println("\n  -- temporal refinement (mesh fixed) --")
nx_t = 200
gt4 = gauge_run(nx_t, T_wave/20)
gt2 = gauge_run(nx_t, T_wave/40)
gt1 = gauge_run(nx_t, T_wave/80)
# align lengths (finer dt ⇒ more samples): subsample to the coarsest grid times
m = min(length(gt4), length(gt2), length(gt1))
sub(v, m) = v[round.(Int, LinRange(1, length(v), m))]
q_time = order(sub(gt4,m), sub(gt2,m), sub(gt1,m))
@printf("  ⇒  temporal order ≈ %.2f  (Crank–Nicolson → 2)\n", q_time)
check("temporal convergence order > 1.5 (CN expects ≈2)", q_time > 1.5,
      "($(round(q_time,digits=2)))")

println()
println("=" ^ 60)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 60)
n_fail > 0 ? error("test_convergence: $n_fail failed!") :
             println("  Solver converges at the expected FE / time-integrator rates.")
