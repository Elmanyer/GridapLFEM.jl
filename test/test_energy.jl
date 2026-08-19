# ==============================================================
#  test_energy.jl — non-dissipativity / conservation in a closed basin (SMALL)
#
#  Companion to test_conservation.jl (which checks EXACT mass conservation).
#  A closed, inviscid basin (all walls solid, NO wavemaker, NO sponge) with a
#  first-mode standing-wave IC should oscillate FOREVER with a stationary
#  amplitude: Crank–Nicolson is non-dissipative, so a correct discretisation
#  neither damps nor amplifies the free oscillation. We therefore test the
#  physically unambiguous signature of energy conservation — amplitude
#  preservation of the antinode signal over several periods — which needs no
#  explicit (and, for a non-hydrostatic model, subtle) energy functional.
#  A decaying envelope ⇒ spurious numerical dissipation; a growing envelope ⇒
#  instability. We also report a period-averaged hydrostatic energy as a
#  secondary diagnostic (its NET drift over the run should be tiny).
#
#  RUN:  julia --project=. GridapBALFEM.jl/test/test_energy.jl
# ==============================================================

using GridapBALFEM
using Gridap
using LinearAlgebra, Printf

println("=" ^ 60)
println("  test_energy.jl — closed-basin non-dissipativity")
println("=" ^ 60)

n_pass = 0; n_fail = 0
check(name, cond, extra="") = (global n_pass, n_fail;
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1))

g = 9.81; h_val = 1.0; L = 2.0; Ly = 0.5
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0]); Nσ = vert.N_dof
Mt = alg_to_tensor2(vert.Mmat); Bt = alg_to_tensor2(vert.B)

k = π/L; kd = k*h_val
Meff = vert.Mmat .- (kd^2).*vert.B
T_th = 2π/(k*sqrt(g*h_val*dot(vert.Phi, Meff \ vert.Phi)))
@printf("  standing wave: T_th=%.4f s  (basin %.1f×%.1f m)\n", T_th, L, Ly)

domain = ((0.0, L), (0.0, Ly)); partition = (24, 4); p_horizontal = 2
model, trian = build_horizontal_model(domain, partition)
dΩh = Measure(trian, 2*p_horizontal + 2)
U, V = build_fe_spaces(model, p_horizontal, Nσ; y_wall_bc=:wall, x_wall_bc=true)   # closed basin

prob = build_problem(vert; g=g, h_bathy=x -> h_val, regime=:linear,
                     mu_sponge=x -> 0.0, wm_src=(x, t) -> 0.0)

A0 = 0.005
eta0(x) = A0 * cos(π * x[1] / L)                # antinodes at x=0 and x=L
u0 = make_initial_conditions(U, Nσ; eta0_func=eta0)

# corrected linear wave energy (informational): kinetic uses the effective mass
#   K = ½∫(𝖴x·M𝖴x + 𝖴y·M𝖴y) − ½d²∫(𝖣·B𝖣),   𝖣=∇·𝖴 stacked (B⪯0 ⇒ −term ≥0)
function energy(uh)
    η = uh[1]; Ux = uh[2]; Uy = uh[3]; DU = alg_dx(Ux) + alg_dy(Uy)
    pe = 0.5*g*sum(∫( η*η )*dΩh)
    ke = 0.5*sum(∫( (Ux ⋅ alg_mul(Mt, Ux)) + (Uy ⋅ alg_mul(Mt, Uy)) )*dΩh) -
         0.5*h_val^2*sum(∫( DU ⋅ alg_mul(Bt, DU) )*dΩh)
    return pe + ke
end
E0 = energy(u0)

dt = T_th/80; Tf = 6*T_th
op     = build_ode_operator(prob, U, V, trian, dΩh)
#  ⚠ solver_type=:theta IS LOAD-BEARING — do not drop it to "use the default".
#  This test measures NON-DISSIPATIVITY of the spatial operator. The default
#  integrator is RungeKutta(:SDIRK_2_2), which is L-stable, i.e. dissipative BY
#  DESIGN — running this test on it measures the INTEGRATOR, not the model, and
#  the gate below then fails for a reason that has nothing to do with the solver.
#  Measured 2026-08-16 on this exact case, 6 periods:
#      SDIRK_2_2 (L-stable) : net ΔE/E₀ = −1.34e-2   ← fails the <1% gate
#      θ = 0.5  (CN)        : net ΔE/E₀ = +3.65e-14  ← machine precision
#  Eleven orders of magnitude: the model conserves energy to round-off, and CN is
#  the only integrator here that can show it. Predicted when the default was
#  switched on 2026-07-23 ("in physical tension with L-stable SDIRK").
solver = build_ode_solver(dt; nl_tol=1e-8, solver_type=:theta)
odesol = solve(solver, op, 0.0, Tf, u0)

xg = VectorValue(0.05, Ly/2)                     # near the x=0 antinode
ts = Float64[]; ug = Float64[]; Es = Float64[E0]; emax = 0.0
for (t_n, u_n) in odesol
    global emax
    push!(ts, t_n); push!(ug, u_n[1](xg)); push!(Es, energy(u_n))
    emax = max(emax, maximum(abs.(get_free_dof_values(u_n[1]))))
end

# peak amplitude in the first vs last period (envelope stationarity)
first_win = findall(t -> t <= T_th, ts)
last_win  = findall(t -> t >= Tf - T_th, ts)
A_first = maximum(abs.(ug[first_win])); A_last = maximum(abs.(ug[last_win]))
env_drift = A_last/A_first - 1.0
net_E     = (Es[end] - E0)/abs(E0)
@printf("  periods=%.0f  A_first=%.5f  A_last=%.5f  env drift=%.2f%%  net ΔE/E₀=%.2e\n",
        Tf/T_th, A_first, A_last, 100*env_drift, net_E)

check("amplitude preserved (|A_last/A_first−1| < 3%)", abs(env_drift) < 0.03,
      "($(round(100env_drift,digits=2))%)")
check("no growth (net energy drift < 1%)", abs(net_E) < 0.01,
      "($(round(100net_E,digits=3))%)")
check("bounded / no NaN", !isnan(emax) && emax < 5*A0)

println("=" ^ 60)
n_fail > 0 ? error("test_energy: $n_fail failed!") :
             println("  Closed-basin oscillation is non-dissipative (energy conserved).")
