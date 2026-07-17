# ==============================================================
#  test_conservation.jl — mass conservation (SMALL & FAST)
#
#  Port of ../../LFE-M_2D_solver/test/test_conservation_lfem2D.jl.
#  Closed basin (all 4 walls solid — x_wall_bc MANDATORY for IC problems),
#  NO wavemaker, NO sponge, initial Gaussian η hump. The continuity weak form
#  with no-flux BCs conserves ∫η dΩ exactly (test ψ≡const ⇒ d/dt∫η = 0);
#  verify the drift stays at solver tolerance while the hump sloshes.
#  Nonlinear advection ON (unlike the old solver, the plain Gridap path
#  handles it — no owned V⊗H loop).
#
#  RUN:  julia --project=. GridapLFEM.jl/test/test_conservation.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Gridap
using Printf

println("=" ^ 60)
println("  test_conservation.jl — mass conservation (closed basin)")
println("=" ^ 60)

g = 9.81; d_val = 1.0
vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
Nσ = vert.N_dof

domain = ((0.0, 10.0), (0.0, 2.0)); partition = (20, 4); fe_order = 2
model, trian = build_horizontal_model(domain, partition)
dO = Measure(trian, 2*fe_order + 2)
U, V = build_fe_spaces(model, fe_order, Nσ; y_wall_bc=true, x_wall_bc=true)

prob = build_problem(vert; g=g, d_func=x -> d_val,
    linearised=true, advection=true,
    mu_sponge=x -> 0.0, wm_src=(x, t) -> 0.0)

A0 = 0.01; xc = 5.0; wd = 1.0
eta0(x) = A0 * exp(-((x[1] - xc)^2) / wd^2)
u0 = make_initial_conditions(U, Nσ; eta0_func=eta0)

mass(uh) = sum(∫(uh[1]) * dO)
M0 = mass(u0)
@printf("  Initial mass ∫η = %.8e   (Nσ=%d, %d×%d cells)\n", M0, Nσ, partition...)

op     = build_ode_operator(prob, U, V, trian, dO)
solver = build_ode_solver(0.02)
odesol = solve(solver, op, 0.0, 2.0, u0)

masses = Float64[M0]; emax = 0.0; nst = 0
for (t_n, u_n) in odesol
    global emax, nst
    nst += 1
    push!(masses, mass(u_n))
    emax = max(emax, maximum(abs.(get_free_dof_values(u_n[1]))))
end

drift = maximum(abs.(masses .- M0)) / abs(M0)
@printf("  steps=%d  max|Δ∫η|/∫η = %.3e  max η=%.5f m\n", nst, drift, emax)

n_fail = 0
drift < 1e-6            ? println("  PASS  mass conserved (drift < 1e-6)") :
                          (println("  FAIL  mass drift $drift"); global n_fail += 1)
(!isnan(emax) && emax < 1.0) ? println("  PASS  bounded / no NaN") :
                          (println("  FAIL  unbounded ($emax)"); global n_fail += 1)

println("=" ^ 60)
n_fail > 0 ? error("test_conservation: $n_fail failed!") :
             println("  Mass conservation verified.")
